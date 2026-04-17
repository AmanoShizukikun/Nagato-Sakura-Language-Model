from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import math

import torch


TensorLikeKV = Union[torch.Tensor, "QuantizedTensor"]
CacheEntryLike = Union[Tuple[TensorLikeKV, TensorLikeKV], "KVCacheEntry"]


@dataclass
class KVQuantizationConfig:
    enabled: bool = False
    kv_bits: int = 32
    group_size: int = 64
    use_residual_sign: bool = False

    def should_quantize(self) -> bool:
        return self.enabled and self.kv_bits <= 8


@dataclass
class QuantizedTensor:
    q: torch.Tensor
    scale: torch.Tensor
    shape: Tuple[int, ...]
    bits: int
    group_size: int
    residual_sign: Optional[torch.Tensor] = None

    def dequantize(self, target_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if self.bits >= 16:
            return self.q.to(target_dtype).reshape(self.shape)

        qmax = float((1 << (self.bits - 1)) - 1)
        scale = self.scale.to(torch.float32).unsqueeze(-1)
        values = self.q.to(torch.float32) * scale

        if self.residual_sign is not None:
            # QJL-inspired residual sign correction: add one-bit residual direction.
            direction = self.residual_sign.to(torch.float32).mul(2.0).sub(1.0)
            correction = direction * (scale / (2.0 * max(qmax, 1.0)))
            values = values + correction

        flat = values.reshape(-1)
        total_size = 1
        for dim in self.shape:
            total_size *= dim
        flat = flat[:total_size]
        return flat.reshape(self.shape).to(target_dtype)

    def memory_bytes(self) -> int:
        total = self.q.numel() * self.q.element_size() + self.scale.numel() * self.scale.element_size()
        if self.residual_sign is not None:
            total += self.residual_sign.numel() * self.residual_sign.element_size()
        return int(total)


@dataclass
class KVCacheEntry:
    key: TensorLikeKV
    value: TensorLikeKV
    bits: int

    def as_tensors(self, target_dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        key_tensor = self.key.dequantize(target_dtype) if isinstance(self.key, QuantizedTensor) else self.key.to(target_dtype)
        value_tensor = self.value.dequantize(target_dtype) if isinstance(self.value, QuantizedTensor) else self.value.to(target_dtype)
        return key_tensor, value_tensor

    def memory_bytes(self) -> int:
        return tensor_memory_bytes(self.key) + tensor_memory_bytes(self.value)


def _reshape_groups(flat_values: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int]:
    num_values = flat_values.numel()
    if num_values == 0:
        return flat_values.reshape(0, group_size), 0

    num_groups = int(math.ceil(num_values / group_size))
    padded_len = num_groups * group_size
    if padded_len != num_values:
        pad = torch.zeros(padded_len - num_values, device=flat_values.device, dtype=flat_values.dtype)
        flat_values = torch.cat([flat_values, pad], dim=0)
    return flat_values.reshape(num_groups, group_size), num_values


def quantize_tensor(
    tensor: torch.Tensor,
    bits: int,
    group_size: int = 64,
    use_residual_sign: bool = False,
) -> QuantizedTensor:
    if bits >= 16:
        return QuantizedTensor(
            q=tensor.detach().clone(),
            scale=torch.ones(1, dtype=torch.float16, device=tensor.device),
            shape=tuple(tensor.shape),
            bits=bits,
            group_size=max(1, group_size),
            residual_sign=None,
        )

    if bits < 2 or bits > 8:
        raise ValueError(f"Unsupported quantization bits: {bits}")

    group_size = max(1, group_size)
    flat = tensor.detach().to(torch.float32).reshape(-1)
    grouped, num_values = _reshape_groups(flat, group_size)

    qmax = float((1 << (bits - 1)) - 1)
    scale = grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / max(qmax, 1.0)
    q = torch.round(grouped / scale).clamp(-qmax, qmax).to(torch.int8)

    residual_sign = None
    if use_residual_sign:
        recon = q.to(torch.float32) * scale
        residual_sign = (grouped >= recon).to(torch.uint8)

    if q.numel() > num_values:
        q = q.reshape(-1)[: math.ceil(num_values / group_size) * group_size].reshape(-1, group_size)
        if residual_sign is not None:
            residual_sign = residual_sign.reshape(-1)[: math.ceil(num_values / group_size) * group_size].reshape(-1, group_size)

    return QuantizedTensor(
        q=q,
        scale=scale.squeeze(-1).to(torch.float16),
        shape=tuple(tensor.shape),
        bits=bits,
        group_size=group_size,
        residual_sign=residual_sign,
    )


def dequantize_tensor(value: TensorLikeKV, target_dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, QuantizedTensor):
        return value.dequantize(target_dtype)
    return value.to(target_dtype)


def quantize_kv_pair(
    key: torch.Tensor,
    value: torch.Tensor,
    config: KVQuantizationConfig,
) -> CacheEntryLike:
    if not config.should_quantize():
        return (key, value)

    key_q = quantize_tensor(
        key,
        bits=config.kv_bits,
        group_size=config.group_size,
        use_residual_sign=config.use_residual_sign,
    )
    value_q = quantize_tensor(
        value,
        bits=config.kv_bits,
        group_size=config.group_size,
        use_residual_sign=config.use_residual_sign,
    )
    return KVCacheEntry(key=key_q, value=value_q, bits=config.kv_bits)


def restore_kv_pair(
    cache_entry: CacheEntryLike,
    target_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(cache_entry, KVCacheEntry):
        return cache_entry.as_tensors(target_dtype)

    if not isinstance(cache_entry, tuple) or len(cache_entry) != 2:
        raise ValueError("Invalid cache entry format")

    return dequantize_tensor(cache_entry[0], target_dtype), dequantize_tensor(cache_entry[1], target_dtype)


def cache_seq_len(cache_entry: CacheEntryLike) -> int:
    key_tensor, _ = restore_kv_pair(cache_entry, target_dtype=torch.float32)
    return int(key_tensor.shape[-2])


def tensor_memory_bytes(value: TensorLikeKV) -> int:
    if isinstance(value, QuantizedTensor):
        return value.memory_bytes()
    return int(value.numel() * value.element_size())


def estimate_kv_cache_bytes(entries: Optional[Iterable[CacheEntryLike]]) -> int:
    if entries is None:
        return 0

    total = 0
    for entry in entries:
        if isinstance(entry, KVCacheEntry):
            total += entry.memory_bytes()
        elif isinstance(entry, tuple) and len(entry) == 2:
            total += tensor_memory_bytes(entry[0]) + tensor_memory_bytes(entry[1])
    return total


def kv_cache_summary(entries: Optional[List[CacheEntryLike]]) -> Dict[str, Any]:
    total_bytes = estimate_kv_cache_bytes(entries)
    num_layers = len(entries) if entries else 0
    seq_len = cache_seq_len(entries[0]) if entries else 0
    return {
        "layers": num_layers,
        "seq_len": seq_len,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 ** 2),
    }
