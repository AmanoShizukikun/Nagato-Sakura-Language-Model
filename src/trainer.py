import os
import sys
import json
import time
import warnings
import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from functools import partial
from collections import defaultdict
import contextlib

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset as TorchDataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from datasets import Dataset
import numpy as np
import wandb
from tqdm import tqdm

from .nagato_sakura_model import NagatoSakuraForCausalLM, NSConfig
from .logger import setup_logging, SystemMonitor, CSVMetricsWriter
from .tokenizer import TokenizerManager
from .data_utils import smart_collate_fn, EarlyStoppingCallback, pretokenize_supervised_dataset
from .checkpoint_manager import CheckpointPolicy, CheckpointManager

# 混合精度處理
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    class GradScaler:
        def __init__(self, *args, **kwargs): pass
        def scale(self, loss): return loss
        def step(self, optimizer): optimizer.step()
        def update(self): pass
        def unscale_(self, optimizer): pass
        def is_enabled(self): return False
    def autocast(enabled=False, dtype=None):
        return contextlib.nullcontext()


class _NoOpMetricsWriter:
    def log_step_metrics(self, row: Dict[str, Any]):
        return None

    def log_eval_metrics(self, row: Dict[str, Any]):
        return None

    def log_event(self, row: Dict[str, Any]):
        return None

    def log_checkpoint_metrics(self, row: Dict[str, Any]):
        return None

    def log_training_summary(self, row: Dict[str, Any]):
        return None


class _DistributedEvalSampler(Sampler[int]):
    """分散式評估取樣器，不補齊且不重複樣本。"""

    def __init__(self, dataset: TorchDataset, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        dataset_size = len(self.dataset)
        if dataset_size <= self.rank:
            return 0
        return ((dataset_size - 1 - self.rank) // self.num_replicas) + 1


def _has_eval_dataloader(eval_dataloader: Optional[DataLoader]) -> bool:
    return eval_dataloader is not None


class AdvancedNagatoSakuraTrainer:
    """長門櫻訓練器"""
    
    def __init__(
        self,
        model_config: Optional[NSConfig],
        output_dir: str = "nagato_sakura_output",
        device: Optional[str] = None,
        use_wandb: bool = False,
        project_name: str = "nagato-sakura",
        precision: str = "auto",
        is_distributed: bool = False,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        
        # 基礎設置
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rank = int(rank)
        self.local_rank = int(local_rank)
        self.world_size = max(1, int(world_size))
        self.is_distributed = bool(is_distributed and self.world_size > 1)
        self.is_main_process = (not self.is_distributed) or self.rank == 0

        if self.is_distributed and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        
        # 初始化日誌
        if self.is_main_process:
            self.logger = setup_logging(str(self.output_dir))
        else:
            self.logger = logging.getLogger(f"{__name__}.rank{self.rank}")
            self.logger.handlers.clear()
            self.logger.propagate = False
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.WARNING)

        self.logger.info(f"使用設備: {self.device}")
        self.logger.info(
            f"分散式狀態 - enabled: {self.is_distributed}, rank: {self.rank}, "
            f"local_rank: {self.local_rank}, world_size: {self.world_size}"
        )

        self.requested_precision = str(precision).strip().lower()
        self.precision_profile = self._resolve_precision_profile(self.requested_precision)
        self.logger.info(
            "精度策略 - requested: %s, actual: %s, model_dtype: %s, autocast_dtype: %s, grad_scaler: %s",
            self.requested_precision,
            self.precision_profile["actual_precision"],
            self.precision_profile["model_dtype"],
            self.precision_profile["autocast_dtype"],
            self.precision_profile["use_grad_scaler"],
        )
        
        # 系統監控
        self.system_monitor = SystemMonitor()
        if self.is_main_process:
            self.system_monitor.log_system_status(self.logger)
        self.metrics_writer = CSVMetricsWriter(str(self.output_dir)) if self.is_main_process else _NoOpMetricsWriter()
        
        # 模型相關
        self.model_config = model_config
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer_manager = TokenizerManager(self.output_dir / "tokenizer.json")
        
        # 訓練狀態
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        self.training_history = defaultdict(list)
        self.loss_ema: Optional[float] = None
        self.best_checkpoint_path: Optional[Path] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.eval_short_max_tokens = 64
        self.eval_medium_max_tokens = 256
        self._scheduler_total_steps: Optional[int] = None
        self._scheduler_warmup_steps: Optional[int] = None
        best_model_dir = self.output_dir / "best_model"
        if (best_model_dir / "model.pt").exists():
            self.best_checkpoint_path = best_model_dir
        
        # WandB集成
        self.use_wandb = bool(use_wandb and self.is_main_process)
        if self.use_wandb:
            try:
                wandb.init(project=project_name, dir=str(self.output_dir))
                self.logger.info("WandB 初始化成功")
            except Exception as e:
                self.logger.warning(f"WandB 初始化失敗: {e}")
                self.use_wandb = False
        
        # 其他組件
        self.early_stopping = None
        
        # 嘗試加載現有分詞器
        try:
            self.tokenizer_manager.load_tokenizer()
        except Exception as e:
            self.logger.info(f"未找到現有分詞器或加載失敗: {e}")

    def _resolve_precision_profile(self, requested_precision: str) -> Dict[str, Any]:
        requested = requested_precision.lower()
        valid_modes = {"auto", "fp32", "fp16", "bf16"}
        if requested not in valid_modes:
            raise ValueError(f"不支援的 precision 模式: {requested_precision}")

        if self.device.type != "cuda":
            return {
                "actual_precision": "fp32",
                "use_autocast": False,
                "autocast_dtype": None,
                "model_dtype": torch.float32,
                "use_grad_scaler": False,
            }

        bf16_supported = bool(
            hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        )

        if requested == "auto":
            if bf16_supported:
                actual = "bf16"
            elif AMP_AVAILABLE:
                actual = "fp16"
            else:
                actual = "fp32"
        elif requested == "bf16":
            if bf16_supported:
                actual = "bf16"
            else:
                fallback = "fp16" if AMP_AVAILABLE else "fp32"
                self.logger.warning(f"當前設備不支援 bf16，回退至 {fallback}")
                actual = fallback
        elif requested == "fp16":
            if AMP_AVAILABLE:
                actual = "fp16"
            else:
                self.logger.warning("目前環境不支援 AMP fp16，回退至 fp32")
                actual = "fp32"
        else:
            actual = "fp32"

        use_autocast = AMP_AVAILABLE and actual in {"fp16", "bf16"}
        autocast_dtype = None
        if actual == "bf16":
            autocast_dtype = torch.bfloat16
        elif actual == "fp16":
            autocast_dtype = torch.float16

        model_dtype = torch.bfloat16 if actual == "bf16" else torch.float32
        use_grad_scaler = bool(actual == "fp16" and AMP_AVAILABLE)

        return {
            "actual_precision": actual,
            "use_autocast": use_autocast,
            "autocast_dtype": autocast_dtype,
            "model_dtype": model_dtype,
            "use_grad_scaler": use_grad_scaler,
        }

    def _autocast_context(self):
        if not self.precision_profile["use_autocast"] or self.device.type != "cuda":
            return contextlib.nullcontext()
        autocast_dtype = self.precision_profile.get("autocast_dtype")
        if autocast_dtype is None:
            return autocast(enabled=True)
        return autocast(enabled=True, dtype=autocast_dtype)

    def _unwrap_model(self) -> NagatoSakuraForCausalLM:
        if isinstance(self.model, DDP):
            return self.model.module  # type: ignore[return-value]
        return self.model  # type: ignore[return-value]

    def _all_ranks_true(self, condition: bool) -> bool:
        if not self.is_distributed:
            return condition
        cond_tensor = torch.tensor(1 if condition else 0, device=self.device, dtype=torch.int32)
        dist.all_reduce(cond_tensor, op=dist.ReduceOp.MIN)
        return bool(cond_tensor.item())

    def _sync_partial_accumulation_gradients(self):
        if not self.is_distributed or self.model is None:
            return
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(self.world_size)

    def _finalize_accumulation_window(
        self,
        optimizer,
        scheduler,
        scaler,
        *,
        accumulated_steps: int,
        gradient_accumulation_steps: int,
        max_grad_norm: float,
    ) -> Dict[str, Any]:
        if accumulated_steps <= 0:
            return {
                "applied": False,
                "invalid_grad": False,
                "runtime_error": None,
                "grad_norm": None,
            }

        if self.model is None:
            raise RuntimeError("模型尚未初始化")

        try:
            scaler.unscale_(optimizer)

            if accumulated_steps < gradient_accumulation_steps:
                correction = gradient_accumulation_steps / max(1, accumulated_steps)
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_grad_norm,
            )
            grad_norm_value = float(grad_norm)

            if not self._all_ranks_true(math.isfinite(grad_norm_value)):
                if self.is_main_process:
                    self.logger.warning(f"檢測到無效梯度範數: {grad_norm}")
                    self.metrics_writer.log_event({
                        "epoch": self.current_epoch + 1,
                        "global_step": self.global_step,
                        "event_type": "invalid_grad_norm",
                        "severity": "WARNING",
                        "message": "grad norm is nan/inf",
                        "value": str(grad_norm),
                        "accumulated_steps": accumulated_steps,
                    })
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                return {
                    "applied": False,
                    "invalid_grad": True,
                    "runtime_error": None,
                    "grad_norm": grad_norm_value,
                }

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            return {
                "applied": True,
                "invalid_grad": False,
                "runtime_error": None,
                "grad_norm": grad_norm_value,
            }
        except RuntimeError as error:
            if self.is_main_process:
                self.logger.warning(f"梯度累積後 optimizer step 失敗: {error}")
                self.metrics_writer.log_event({
                    "epoch": self.current_epoch + 1,
                    "global_step": self.global_step,
                    "event_type": "optimizer_step_failed",
                    "severity": "WARNING",
                    "message": "optimizer step failed after accumulation",
                    "value": str(error),
                    "accumulated_steps": accumulated_steps,
                })
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            return {
                "applied": False,
                "invalid_grad": False,
                "runtime_error": error,
                "grad_norm": None,
            }

    def _resolve_resume_checkpoint_path(self, resume_checkpoint: str) -> Optional[Path]:
        raw_path = str(resume_checkpoint).strip()
        if not raw_path:
            return None

        candidate = Path(raw_path).expanduser()
        candidates: List[Path] = [candidate]
        if not candidate.is_absolute():
            candidates.append(self.output_dir / candidate)

        for path in candidates:
            normalized = path.parent if path.is_file() else path
            if normalized.exists() and normalized.is_dir():
                return normalized

        return None

    def _align_scheduler_to_global_step(self, scheduler, reason: str) -> None:
        target_last_epoch = max(-1, int(self.global_step) - 1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                scheduler.step(epoch=target_last_epoch)

            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else None
            if current_lr is not None:
                self.logger.info(
                    f"已對齊學習率調度器({reason}) - global_step: {self.global_step}, "
                    f"last_epoch: {target_last_epoch}, lr: {current_lr:.2e}"
                )
            else:
                self.logger.info(
                    f"已對齊學習率調度器({reason}) - global_step: {self.global_step}, "
                    f"last_epoch: {target_last_epoch}"
                )
        except Exception as e:
            total_steps = getattr(scheduler, "total_steps", None)
            if isinstance(total_steps, int) and total_steps > 0:
                capped_last_epoch = min(target_last_epoch, total_steps - 1)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        scheduler.step(epoch=capped_last_epoch)
                    current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else None
                    if current_lr is not None:
                        self.logger.info(
                            f"已對齊學習率調度器({reason}/capped) - global_step: {self.global_step}, "
                            f"last_epoch: {capped_last_epoch}, lr: {current_lr:.2e}"
                        )
                    else:
                        self.logger.info(
                            f"已對齊學習率調度器({reason}/capped) - global_step: {self.global_step}, "
                            f"last_epoch: {capped_last_epoch}"
                        )
                    return
                except Exception as fallback_error:
                    self.logger.warning(
                        f"學習率調度器對齊失敗({reason})，且 capped fallback 失敗: {fallback_error}"
                    )
                    return

            self.logger.warning(f"學習率調度器對齊失敗({reason}): {e}")

    def _verify_scheduler_alignment(self, scheduler) -> None:
        expected_last_epoch = max(-1, int(self.global_step) - 1)
        actual_last_epoch = int(getattr(scheduler, "last_epoch", -1))
        if actual_last_epoch != expected_last_epoch:
            self.logger.warning(
                f"學習率調度器步數不一致: expected_last_epoch={expected_last_epoch}, "
                f"actual_last_epoch={actual_last_epoch}, global_step={self.global_step}"
            )
            self._align_scheduler_to_global_step(scheduler, reason="step_mismatch")

    def initialize_model(self):
        """初始化模型"""
        if self.model_config is None:
            raise ValueError("模型配置未設置")
        if self.tokenizer_manager.transformers_tokenizer is None:
            raise ValueError("分詞器未初始化")
        
        # 更新配置
        self.model_config.vocab_size = self.tokenizer_manager.transformers_tokenizer.vocab_size
        self.model_config.pad_token_id = self.tokenizer_manager.transformers_tokenizer.pad_token_id
        self.model_config.bos_token_id = self.tokenizer_manager.transformers_tokenizer.bos_token_id
        self.model_config.eos_token_id = self.tokenizer_manager.transformers_tokenizer.eos_token_id
        
        # 創建模型
        try:
            base_model = NagatoSakuraForCausalLM(self.model_config)
            base_model = base_model.to(
                device=self.device,
                dtype=self.precision_profile["model_dtype"],
            )

            if self.is_distributed:
                if self.device.type == "cuda":
                    self.model = DDP(
                        base_model,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        find_unused_parameters=False,
                    )
                else:
                    self.model = DDP(base_model, find_unused_parameters=False)
            else:
                self.model = base_model

            model_for_stats = self._unwrap_model()

            param_stats = model_for_stats.get_parameter_stats()
            total_params = param_stats["total_params"]
            trainable_params = param_stats["trainable_params"]
            embedding_params = param_stats["embedding_params"]
            non_embedding_params = param_stats["non_embedding_params"]

            self.logger.info("模型初始化完成")
            self.logger.info(f"模型參數量: {total_params/1e6:.2f}M")
            self.logger.info(
                f"參數組成: Embedding {embedding_params/1e6:.2f}M "
                f"+ 非Embedding {non_embedding_params/1e6:.2f}M"
            )

            tokenizer_path = self.tokenizer_manager.tokenizer_path
            if tokenizer_path.exists():
                tokenizer_size_bytes = tokenizer_path.stat().st_size
                tokenizer_size_mb = tokenizer_size_bytes / (1024 ** 2)
                tokenizer_ratio = (tokenizer_size_bytes / max(1, param_stats["parameter_memory_bytes"])) * 100.0
                self.logger.info(
                    f"Tokenizer檔案: {tokenizer_size_mb:.2f}MB "
                    f"(約 {tokenizer_ratio:.2f}% 參數記憶體)"
                )
            else:
                self.logger.info("Tokenizer檔案: 未找到（不影響模型參數統計）")

            # 詳細拆分放在 DEBUG，避免預設輸出過於冗長。
            self.logger.debug(f"可訓練參數: {trainable_params/1e6:.2f}M")
            self.logger.debug(
                f"Embedding矩陣: {param_stats['vocab_size']} x {param_stats['hidden_size']}"
            )
            if param_stats["lm_head_tied_with_embedding"]:
                self.logger.debug("LM Head參數: 0 (與Embedding共享權重)")
            else:
                self.logger.debug(f"LM Head參數: {param_stats['lm_head_params']/1e6:.2f}M")
            self.logger.debug(f"非Embedding參數: {non_embedding_params/1e6:.2f}M")
            self.logger.debug(f"參數記憶體估算(目前dtype): {param_stats['parameter_memory_gb']:.2f}GB")

            if getattr(self.model_config, "gradient_checkpointing", False):
                model_for_stats.model.enable_gradient_checkpointing()
                self.logger.info("已根據配置啟用梯度檢查點")
            
            if self.use_wandb:
                wandb.config.update({
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "embedding_params": embedding_params,
                    "non_embedding_params": non_embedding_params,
                    "model_size_gb": param_stats["parameter_memory_gb"],
                })
                
        except Exception as e:
            self.logger.error(f"模型初始化失敗: {e}")
            raise

    @staticmethod
    def _clean_optional_text(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        text = value.strip()
        if text.lower() in {"", "none", "null", "n/a", "nan"}:
            return ""
        return text

    @classmethod
    def _normalize_supervised_item(cls, item: Dict[str, Any]) -> Optional[Dict[str, str]]:
        if not isinstance(item, dict):
            return None

        candidates = [
            (item.get("instruction"), item.get("input"), item.get("output")),
            (item.get("zh_instruction"), item.get("zh_input"), item.get("zh_output")),
            (item.get("en_instruction"), item.get("en_input"), item.get("en_output")),
            (item.get("prompt"), "", item.get("completion")),
        ]

        for instruction_raw, input_raw, output_raw in candidates:
            if not isinstance(instruction_raw, str) or not isinstance(output_raw, str):
                continue

            instruction = cls._clean_optional_text(instruction_raw)
            output = output_raw.strip()
            input_text = cls._clean_optional_text(input_raw)

            if not instruction or not output:
                continue

            return {
                "instruction": instruction,
                "input": input_text,
                "output": output,
            }

        return None

    @classmethod
    def _sample_signature(cls, item: Dict[str, Any]) -> Tuple[str, str]:
        """將樣本轉為可比較的唯一鍵。"""
        normalized = cls._normalize_supervised_item(item)
        if not normalized:
            return "", ""

        instruction = normalized["instruction"]
        input_text = normalized["input"]
        output = normalized["output"]
        prompt_text = f"{instruction} {input_text}".strip() if input_text else instruction
        return prompt_text, output

    @staticmethod
    def _load_entries_from_jsonl(file_path: Path) -> List[Any]:
        rows: List[Any] = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    rows.append(json.loads(raw))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{file_path} 第 {line_no} 行 JSONL 解析失敗: {e}") from e
        return rows

    @classmethod
    def _load_entries_from_file(cls, file_path: Path) -> List[Any]:
        suffix = file_path.suffix.lower()
        if suffix == ".jsonl":
            return cls._load_entries_from_jsonl(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        if isinstance(loaded, list):
            return loaded
        if isinstance(loaded, dict):
            return [loaded]
        raise ValueError(f"{file_path} 格式錯誤：預期為 list/dict")

    def _load_supervised_data_file(self, data_file: str, dataset_name: str) -> List[Dict[str, str]]:
        """加載並驗證 instruction/input/output 監督數據（支援檔案或資料夾）。"""
        data_path = Path(data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"{dataset_name}來源不存在: {data_file}")

        source_files: List[Path] = []
        if data_path.is_dir():
            for pattern in ("*.json", "*.jsonl"):
                source_files.extend(data_path.rglob(pattern))

            unique_files: Dict[str, Path] = {}
            for path in source_files:
                unique_files[str(path.resolve())] = path
            source_files = [unique_files[k] for k in sorted(unique_files.keys())]

            if not source_files:
                raise ValueError(f"{dataset_name}資料夾沒有可用的 json/jsonl 文件: {data_file}")
        else:
            source_files = [data_path]

        valid_data: List[Dict[str, str]] = []
        invalid_count = 0
        failed_files: List[str] = []

        for source_file in source_files:
            try:
                loaded_data = self._load_entries_from_file(source_file)
            except Exception as e:
                failed_files.append(f"{source_file}: {e}")
                continue

            for item in loaded_data:
                normalized_item = self._normalize_supervised_item(item)
                if not normalized_item:
                    invalid_count += 1
                    continue
                valid_data.append(normalized_item)

        if failed_files:
            self.logger.warning(
                f"{dataset_name}有 {len(failed_files)} 個文件讀取失敗，將略過:\n" + "\n".join(failed_files[:10])
            )
            if len(failed_files) > 10:
                self.logger.warning(f"另有 {len(failed_files) - 10} 個失敗文件未列出")

        if not valid_data:
            raise ValueError(f"{dataset_name}沒有有效樣本: {data_file}")

        self.logger.info(
            f"{dataset_name}載入完成 - 來源檔案: {len(source_files)}, 有效: {len(valid_data)}, 格式錯誤: {invalid_count}"
        )
        return valid_data

    def _count_overlapping_samples(
        self,
        train_data: List[Dict[str, str]],
        eval_data: List[Dict[str, str]],
    ) -> int:
        """統計 train/eval 重疊樣本數（不做剔除）。"""
        eval_signatures = {self._sample_signature(item) for item in eval_data}
        overlap_count = 0
        for item in train_data:
            if self._sample_signature(item) in eval_signatures:
                overlap_count += 1
        return overlap_count

    def prepare_data_and_tokenizer(
        self,
        training_data_file: str,
        target_vocab_size: int,
        force_retrain_tokenizer: bool = False,
        tokenizer_min_frequency: int = 2,
        eval_data_file: Optional[str] = None,
        tokenizer_train_max_samples: int = 0,
        tokenizer_num_threads: int = 0,
    ) -> Tuple[List[Dict[str, str]], Optional[List[Dict[str, str]]]]:
        """準備訓練/評估數據並訓練分詞器。"""

        self.logger.info(f"從 {training_data_file} 加載訓練數據...")
        train_data = self._load_supervised_data_file(training_data_file, "訓練數據")
        fixed_eval_data: Optional[List[Dict[str, str]]] = None

        if eval_data_file:
            self.logger.info(f"從 {eval_data_file} 加載固定評估集...")
            fixed_eval_data = self._load_supervised_data_file(eval_data_file, "固定評估集")
            overlap_count = self._count_overlapping_samples(train_data, fixed_eval_data)

            self.logger.info(
                f"固定評估集模式 - 訓練集保留: {len(train_data)}, "
                f"評估集: {len(fixed_eval_data)}, 與訓練集重疊: {overlap_count}"
            )
            self.metrics_writer.log_event({
                "epoch": 0,
                "global_step": self.global_step,
                "event_type": "fixed_eval_dataset",
                "severity": "INFO",
                "message": "fixed eval dataset enabled and overlap intentionally kept",
                "value": json.dumps({
                    "train_before": len(train_data),
                    "train_after": len(train_data),
                    "eval_size": len(fixed_eval_data),
                    "overlap_count": overlap_count,
                    "overlap_removed": False,
                }, ensure_ascii=False),
            })

        # 分詞器使用完整訓練集，固定評估集可與訓練集重疊用於擬合監控。
        self.tokenizer_manager.prepare_tokenizer(
            train_data,
            target_vocab_size,
            force_retrain_tokenizer,
            min_frequency=tokenizer_min_frequency,
            max_training_samples=tokenizer_train_max_samples,
            num_threads=tokenizer_num_threads,
        )

        return train_data, fixed_eval_data

    def create_datasets(
        self,
        data_list: List[Dict[str, str]],
        eval_split_ratio: float = 0.01,
        fixed_eval_data: Optional[List[Dict[str, str]]] = None,
        pretokenize_batch_size: int = 1024,
        pretokenize_num_proc: Optional[int] = None,
        use_pretokenize_cache: bool = True,
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """創建訓練和評估數據集"""
        
        self.logger.info(f"創建數據集，總數據量: {len(data_list)}")
        
        if not data_list:
            raise ValueError("數據列表為空")

        if self.tokenizer_manager.transformers_tokenizer is None:
            raise ValueError("分詞器尚未初始化")
        if self.model_config is None:
            raise ValueError("模型配置尚未初始化")

        tokenizer = self.tokenizer_manager.transformers_tokenizer
        max_seq_length = self.model_config.max_position_embeddings
        pretokenize_batch_size = max(1, int(pretokenize_batch_size))

        if pretokenize_num_proc is None:
            cpu_count = os.cpu_count() or 0
            if sys.platform == "win32":
                pretokenize_num_proc = 1
            else:
                pretokenize_num_proc = min(8, max(2, cpu_count // 2)) if cpu_count else 1
        else:
            pretokenize_num_proc = max(1, int(pretokenize_num_proc))
            if sys.platform == "win32":
                pretokenize_num_proc = 1

        tokenizer_path = self.tokenizer_manager.tokenizer_path
        tokenizer_stamp = "missing"
        if tokenizer_path.exists():
            stat = tokenizer_path.stat()
            tokenizer_stamp = f"{stat.st_size}-{stat.st_mtime_ns}"

        cache_root = self.output_dir / "cache" / "pretokenized"
        self.logger.info(
            "Pretokenize 設定 - batch_size: %s, num_proc: %s, cache: %s",
            pretokenize_batch_size,
            pretokenize_num_proc,
            "啟用" if use_pretokenize_cache else "停用",
        )

        def _pretokenize_rows(rows: List[Dict[str, Any]], desc: str, split_name: str) -> Dataset:
            try:
                tokenized_dataset = pretokenize_supervised_dataset(
                    rows,
                    tokenizer=tokenizer,
                    max_seq_length=max_seq_length,
                    desc=desc,
                    batch_size=pretokenize_batch_size,
                    num_proc=pretokenize_num_proc,
                    cache_dir=cache_root,
                    cache_namespace=f"{split_name}-{tokenizer_stamp}-{max_seq_length}",
                    use_cache=use_pretokenize_cache,
                    return_dataset=True,
                )
                if len(tokenized_dataset) <= 0:
                    raise ValueError(f"{desc} produced no valid samples")
                return tokenized_dataset
            except Exception as e:
                self.logger.warning(
                    f"{desc} 失敗，回退為即時分詞路徑: {e}"
                )
                return Dataset.from_list(rows)

        if fixed_eval_data is not None:
            if not fixed_eval_data:
                raise ValueError("固定評估集為空，請檢查 eval_data_file")
            if eval_split_ratio > 0:
                self.logger.warning("已提供固定評估集，將忽略 eval_split_ratio")

            train_dataset = _pretokenize_rows(data_list, "Pretokenize train", "train")
            eval_dataset = _pretokenize_rows(fixed_eval_data, "Pretokenize eval", "eval")
            self.logger.info(
                f"使用固定評估集 - 訓練集: {len(train_dataset)}, 評估集: {len(eval_dataset)}"
            )
            return train_dataset, eval_dataset
        
        # 創建完整數據集
        dataset = Dataset.from_list(data_list)
        
        # 分割數據集
        if eval_split_ratio > 0 and eval_split_ratio < 1.0 and len(dataset) > 1:
            split_datasets = dataset.train_test_split(
                test_size=eval_split_ratio, 
                shuffle=True, 
                seed=42
            )
            train_dataset = _pretokenize_rows(
                list(split_datasets["train"]),
                "Pretokenize train",
                "train",
            )
            eval_dataset = _pretokenize_rows(
                list(split_datasets["test"]),
                "Pretokenize eval",
                "eval",
            )
            
            self.logger.info(f"數據集分割完成 - 訓練集: {len(train_dataset)}, 評估集: {len(eval_dataset)}")
            return train_dataset, eval_dataset
        else:
            train_dataset = _pretokenize_rows(data_list, "Pretokenize train", "train")
            self.logger.info(f"使用完整數據集進行訓練: {len(train_dataset)}")
            return train_dataset, None

    def setup_training_components(
        self,
        train_dataset: Dataset,
        batch_size: int,
        num_epochs: int,
        learning_rate: float,
        weight_decay: float,
        gradient_accumulation_steps: int,
        warmup_ratio: float,
        lr_scheduler_type: str,
        scheduler_target_epochs: Optional[int] = None,
        dataloader_num_workers: Optional[int] = None,
        dataloader_prefetch_factor: int = 2,
        dataloader_persistent_workers: bool = True,
        dataloader_drop_last: bool = True,
    ):
        """設置訓練組件"""
        
        # 數據加載器
        collate_fn = partial(
            smart_collate_fn,
            tokenizer=self.tokenizer_manager.transformers_tokenizer,
            max_seq_length=self.model_config.max_position_embeddings,
            pack_sequences=True
        )
        
        if dataloader_num_workers is None or dataloader_num_workers < 0:
            if sys.platform == "win32":
                num_workers = 0
            else:
                cpu_count = os.cpu_count() or 2
                num_workers = min(8, max(2, cpu_count // 2))
        else:
            num_workers = max(0, int(dataloader_num_workers))

        train_sampler: Optional[DistributedSampler] = None
        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=bool(dataloader_drop_last),
            )

        train_loader_kwargs: Dict[str, Any] = {
            "dataset": train_dataset,
            "batch_size": batch_size,
            "collate_fn": collate_fn,
            "shuffle": train_sampler is None,
            "sampler": train_sampler,
            "num_workers": num_workers,
            "pin_memory": self.device.type == 'cuda',
            "drop_last": bool(dataloader_drop_last),
        }
        if num_workers > 0:
            train_loader_kwargs["persistent_workers"] = bool(dataloader_persistent_workers)
            train_loader_kwargs["prefetch_factor"] = max(1, int(dataloader_prefetch_factor))

        train_dataloader = DataLoader(**train_loader_kwargs)
        
        # 優化器（使用更先進的配置）
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay
        )
        
        # 學習率調度器
        effective_epochs = max(1, int(scheduler_target_epochs or num_epochs))
        total_steps = max(1, len(train_dataloader) // max(1, gradient_accumulation_steps) * effective_epochs)
        warmup_steps = int(total_steps * warmup_ratio) if warmup_ratio > 0 else 0
        self._scheduler_total_steps = int(total_steps)
        self._scheduler_warmup_steps = int(warmup_steps)
        
        if lr_scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif lr_scheduler_type == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                total_steps=total_steps,
                pct_start=warmup_ratio,
                anneal_strategy='cos'
            )
        else:  # linear
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        
        # 混合精度
        scaler = GradScaler(enabled=bool(self.precision_profile["use_grad_scaler"] and self.device.type == 'cuda'))
        global_effective_batch = batch_size * gradient_accumulation_steps * (self.world_size if self.is_distributed else 1)
        
        self.logger.info(f"訓練組件設置完成:")
        self.logger.info(f"  批次大小: {batch_size}")
        self.logger.info(f"  梯度累積步數: {gradient_accumulation_steps}")
        self.logger.info(f"  有效批次大小(全域): {global_effective_batch}")
        self.logger.info(f"  總訓練步數: {total_steps}")
        self.logger.info(f"  預熱步數: {warmup_steps}")
        self.logger.info(f"  調度器目標Epoch: {effective_epochs}")
        self.logger.info(f"  學習率調度器: {lr_scheduler_type}")
        self.logger.info(f"  DataLoader workers: {num_workers}")
        self.logger.info(f"  DataLoader drop_last: {bool(dataloader_drop_last)}")
        self.logger.info(f"  混合精度: {'啟用' if scaler.is_enabled() else '禁用'}")
        
        return train_dataloader, optimizer, scheduler, scaler, train_sampler

    def train_epoch(
        self,
        train_dataloader: DataLoader,
        train_sampler: Optional[DistributedSampler],
        optimizer,
        scheduler,
        scaler: GradScaler,
        gradient_accumulation_steps: int,
        max_grad_norm: float,
        log_steps_per_epoch: int = 10,
        metrics_log_interval_steps: int = 50,
    ) -> Dict[str, Any]:
        """訓練一個epoch並回傳統計資訊"""

        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        accumulated_steps = 0
        invalid_batches = 0
        invalid_loss_count = 0
        invalid_grad_count = 0
        skipped_update_count = 0
        suspicious_label_ratio_count = 0
        epoch_tokens = 0
        metrics_window_tokens = 0
        optimizer_updates = 0
        loss_bucket = torch.zeros((), device=self.device)
        pending_ddp_grad_sync = False

        epoch_start_time = time.time()
        optimizer.zero_grad(set_to_none=True)

        total_steps = len(train_dataloader)
        log_interval_steps = max(1, total_steps // max(1, log_steps_per_epoch))
        metrics_log_interval_steps = max(1, metrics_log_interval_steps)

        if train_sampler is not None:
            train_sampler.set_epoch(self.current_epoch)

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
            disable=not self.is_main_process,
        )

        for step, batch in enumerate(progress_bar):
            local_batch_valid = batch["input_ids"] is not None
            if not self._all_ranks_true(local_batch_valid):
                invalid_batches += 1
                continue
            if batch["input_ids"] is None:
                invalid_batches += 1
                continue

            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            batch_valid_label_ratio = float(batch.get("valid_label_ratio", 0.0) or 0.0)
            if batch_valid_label_ratio > 0.0 and (batch_valid_label_ratio < 0.01 or batch_valid_label_ratio > 0.95):
                suspicious_label_ratio_count += 1
            batch_tokens = int((labels != -100).sum().item())
            epoch_tokens += batch_tokens
            metrics_window_tokens += batch_tokens

            is_last_batch = step == (total_steps - 1)
            should_sync_gradients = (
                (accumulated_steps + 1) >= gradient_accumulation_steps or is_last_batch
            )
            sync_context = contextlib.nullcontext()
            using_no_sync = (
                self.is_distributed
                and hasattr(self.model, "no_sync")
                and not should_sync_gradients
            )
            if using_no_sync:
                sync_context = self.model.no_sync()

            with sync_context:
                with self._autocast_context():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                    )
                    raw_loss = outputs.loss

                local_loss_valid = bool(raw_loss is not None and torch.isfinite(raw_loss.detach()).item())
                if not self._all_ranks_true(local_loss_valid):
                    invalid_loss_count += 1
                    if self.is_main_process:
                        self.logger.warning(f"檢測到無效損失，跳過此批次: {raw_loss}")
                        self.metrics_writer.log_event({
                            "epoch": self.current_epoch + 1,
                            "global_step": self.global_step,
                            "event_type": "invalid_loss",
                            "severity": "WARNING",
                            "message": "loss is nan/inf or missing",
                            "value": str(raw_loss),
                        })
                    accumulated_steps = 0
                    loss_bucket = torch.zeros((), device=self.device)
                    metrics_window_tokens = 0
                    skipped_update_count += 1
                    pending_ddp_grad_sync = False
                    optimizer.zero_grad(set_to_none=True)
                    del outputs
                    continue

                loss_bucket = loss_bucket + raw_loss.detach()
                scaled_loss = raw_loss / gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                pending_ddp_grad_sync = using_no_sync

            num_batches += 1
            accumulated_steps += 1

            del outputs, raw_loss, scaled_loss

            if accumulated_steps >= gradient_accumulation_steps:
                loss_sum_value = float(loss_bucket.item())
                loss_val = loss_sum_value / max(1, accumulated_steps)
                step_result = self._finalize_accumulation_window(
                    optimizer,
                    scheduler,
                    scaler,
                    accumulated_steps=accumulated_steps,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    max_grad_norm=max_grad_norm,
                )

                accumulated_steps = 0
                loss_bucket = torch.zeros((), device=self.device)
                pending_ddp_grad_sync = False
                grad_norm_value = step_result["grad_norm"]

                if step_result["runtime_error"] is not None:
                    skipped_update_count += 1
                    continue

                if step_result["invalid_grad"]:
                    invalid_grad_count += 1
                    skipped_update_count += 1
                    metrics_window_tokens = 0
                    continue

                epoch_loss += loss_sum_value
                self.global_step += 1
                optimizer_updates += 1

                current_lr = scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

                if self.loss_ema is None:
                    self.loss_ema = loss_val
                else:
                    self.loss_ema = 0.98 * self.loss_ema + 0.02 * loss_val

                progress_bar.set_postfix({
                    "Loss": f"{loss_val:.4f}",
                    "LR": f"{current_lr:.2e}",
                    "Step": self.global_step
                })

                if self.is_main_process and self.global_step % log_interval_steps == 0:
                    self.logger.info(
                        f"Step {self.global_step}: Loss={avg_loss:.4f}, "
                        f"LR={current_lr:.2e}, GradNorm={grad_norm_value:.4f}"
                    )

                    if self.use_wandb:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/loss_ema": self.loss_ema,
                            "train/learning_rate": current_lr,
                            "train/grad_norm": grad_norm_value,
                            "global_step": self.global_step,
                        })

                if self.is_main_process and self.global_step % metrics_log_interval_steps == 0:
                    elapsed = max(1e-6, time.time() - epoch_start_time)
                    tokens_per_sec = epoch_tokens / elapsed
                    gpu_memory_mb = 0.0
                    if self.device.type == "cuda":
                        gpu_memory_mb = torch.cuda.memory_allocated() / 1024**2

                    system_info = self.system_monitor.get_system_info()
                    self.metrics_writer.log_step_metrics({
                        "epoch": self.current_epoch + 1,
                        "global_step": self.global_step,
                        "batch_idx": step,
                        "train_loss": avg_loss,
                        "loss_ema": self.loss_ema,
                        "learning_rate": current_lr,
                        "grad_norm": grad_norm_value,
                        "batch_tokens": metrics_window_tokens,
                        "tokens_per_sec": tokens_per_sec,
                        "invalid_batches": invalid_batches,
                        "gpu_memory_mb": gpu_memory_mb,
                        "gpu_memory_percent": system_info.get("gpu_memory_percent", ""),
                        "cpu_percent": system_info.get("cpu_percent", ""),
                        "ram_percent": system_info.get("memory_percent", ""),
                    })
                    metrics_window_tokens = 0

                if self.is_main_process and self.global_step % (log_interval_steps * 5) == 0:
                    self.system_monitor.log_system_status(self.logger)

        if accumulated_steps > 0:
            if pending_ddp_grad_sync:
                self._sync_partial_accumulation_gradients()
            loss_sum_value = float(loss_bucket.item())
            step_result = self._finalize_accumulation_window(
                optimizer,
                scheduler,
                scaler,
                accumulated_steps=accumulated_steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
            )
            if step_result["applied"]:
                epoch_loss += loss_sum_value
                self.global_step += 1
                optimizer_updates += 1
            else:
                if step_result["invalid_grad"]:
                    invalid_grad_count += 1
                skipped_update_count += 1

        if num_batches == 0:
            raise RuntimeError(
                "本 epoch 沒有任何有效批次，訓練已停止。請檢查分詞器編碼、數據格式與 collate_fn。"
            )

        total_seen_batches = num_batches + invalid_batches
        invalid_ratio = 0.0
        if total_seen_batches > 0:
            invalid_ratio = invalid_batches / total_seen_batches
            if invalid_ratio > 0.3:
                if self.is_main_process:
                    self.logger.warning(
                        f"偵測到較高無效批次比例: {invalid_ratio:.1%} ({invalid_batches}/{total_seen_batches})"
                    )
                    self.metrics_writer.log_event({
                        "epoch": self.current_epoch + 1,
                        "global_step": self.global_step,
                        "event_type": "high_invalid_batch_ratio",
                        "severity": "WARNING",
                        "message": "invalid batch ratio exceeded 30%",
                        "value": invalid_ratio,
                    })

        if suspicious_label_ratio_count > 0 and self.is_main_process:
            self.logger.warning(
                f"偵測到可疑標籤比例批次: {suspicious_label_ratio_count}/{total_seen_batches} "
                f"(可能存在 prompt/label 邊界或資料格式異常)"
            )
            self.metrics_writer.log_event({
                "epoch": self.current_epoch + 1,
                "global_step": self.global_step,
                "event_type": "suspicious_label_ratio",
                "severity": "WARNING",
                "message": "valid label ratio is out of safe range for some batches",
                "value": suspicious_label_ratio_count,
            })

        avg_epoch_loss = epoch_loss / num_batches
        return {
            "avg_loss": avg_epoch_loss,
            "loss_sum": epoch_loss,
            "num_batches": num_batches,
            "invalid_batches": invalid_batches,
            "invalid_ratio": invalid_ratio,
            "invalid_loss_count": invalid_loss_count,
            "invalid_grad_count": invalid_grad_count,
            "skipped_update_count": skipped_update_count,
            "suspicious_label_ratio_count": suspicious_label_ratio_count,
            "optimizer_updates": optimizer_updates,
            "tokens_seen": epoch_tokens,
        }

    @staticmethod
    def _get_length_bucket(token_count: int, short_max: int = 64, medium_max: int = 256) -> str:
        if token_count <= short_max:
            return "short"
        if token_count <= medium_max:
            return "medium"
        return "long"

    @staticmethod
    def _finalize_bucket_metrics(raw_bucket_stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        bucket_metrics: Dict[str, Dict[str, Any]] = {}
        for name, stats in raw_bucket_stats.items():
            tokens = int(stats["tokens"])
            samples = int(stats["samples"])
            if tokens > 0:
                loss = float(stats["loss_sum"] / tokens)
                perplexity = float(math.exp(min(20.0, loss)))
            else:
                loss = None
                perplexity = None
            bucket_metrics[name] = {
                "samples": samples,
                "tokens": tokens,
                "loss": loss,
                "perplexity": perplexity,
            }
        return bucket_metrics

    @staticmethod
    def _flatten_bucket_metrics(bucket_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        flattened: Dict[str, Any] = {}
        for bucket_name in ("short", "medium", "long"):
            item = bucket_metrics.get(bucket_name, {})
            flattened[f"{bucket_name}_samples"] = item.get("samples", 0)
            flattened[f"{bucket_name}_tokens"] = item.get("tokens", 0)
            flattened[f"{bucket_name}_loss"] = item.get("loss", "")
            flattened[f"{bucket_name}_perplexity"] = item.get("perplexity", "")
        return flattened

    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, Any]:
        """評估模型"""
        if eval_dataloader is None:
            return {}
        
        self.model.eval()
        eval_start_time = time.time()
        total_nll_sum = 0.0
        total_tokens = 0
        num_batches = 0
        bucket_raw = {
            "short": {"loss_sum": 0.0, "tokens": 0, "samples": 0},
            "medium": {"loss_sum": 0.0, "tokens": 0, "samples": 0},
            "long": {"loss_sum": 0.0, "tokens": 0, "samples": 0},
        }
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="評估中", leave=False, disable=not self.is_main_process):
                if batch["input_ids"] is None:
                    continue
                
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                
                with self._autocast_context():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                    )
                logits = outputs.logits

                if logits is None:
                    del outputs
                    continue

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                if shift_logits.numel() == 0 or shift_labels.numel() == 0:
                    del outputs, logits, shift_logits, shift_labels
                    continue

                token_losses = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view_as(shift_labels)
                valid_mask = shift_labels != -100

                if not valid_mask.any():
                    del outputs, logits, shift_logits, shift_labels, token_losses, valid_mask
                    continue

                valid_token_losses = token_losses[valid_mask]
                if torch.isnan(valid_token_losses).any() or torch.isinf(valid_token_losses).any():
                    if self.is_main_process:
                        self.metrics_writer.log_event({
                            "epoch": self.current_epoch + 1,
                            "global_step": self.global_step,
                            "event_type": "invalid_eval_token_loss",
                            "severity": "WARNING",
                            "message": "eval token loss has nan/inf",
                            "value": "",
                        })
                    del outputs, logits, shift_logits, shift_labels, token_losses, valid_mask, valid_token_losses
                    continue

                batch_loss_sum = float(valid_token_losses.sum().item())
                batch_tokens = int(valid_mask.sum().item())

                total_nll_sum += batch_loss_sum
                total_tokens += batch_tokens
                num_batches += 1

                sample_token_counts = valid_mask.sum(dim=1)
                sample_loss_sums = (token_losses * valid_mask).sum(dim=1)
                for idx in range(sample_token_counts.shape[0]):
                    sample_tokens = int(sample_token_counts[idx].item())
                    if sample_tokens <= 0:
                        continue
                    sample_loss_sum = float(sample_loss_sums[idx].item())
                    bucket_name = self._get_length_bucket(
                        sample_tokens,
                        short_max=self.eval_short_max_tokens,
                        medium_max=self.eval_medium_max_tokens,
                    )
                    bucket_raw[bucket_name]["loss_sum"] += sample_loss_sum
                    bucket_raw[bucket_name]["tokens"] += sample_tokens
                    bucket_raw[bucket_name]["samples"] += 1
                
                # 釋放評估時的計算圖與張量
                del outputs, logits, shift_logits, shift_labels, token_losses, valid_mask, valid_token_losses

        if self.is_distributed:
            totals_tensor = torch.tensor(
                [total_nll_sum, float(total_tokens), float(num_batches)],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(totals_tensor, op=dist.ReduceOp.SUM)
            total_nll_sum = float(totals_tensor[0].item())
            total_tokens = int(totals_tensor[1].item())
            num_batches = int(totals_tensor[2].item())

            for bucket_name, stats in bucket_raw.items():
                bucket_tensor = torch.tensor(
                    [
                        float(stats["loss_sum"]),
                        float(stats["tokens"]),
                        float(stats["samples"]),
                    ],
                    device=self.device,
                    dtype=torch.float64,
                )
                dist.all_reduce(bucket_tensor, op=dist.ReduceOp.SUM)
                stats["loss_sum"] = float(bucket_tensor[0].item())
                stats["tokens"] = int(bucket_tensor[1].item())
                stats["samples"] = int(bucket_tensor[2].item())
        
        bucket_metrics = self._finalize_bucket_metrics(bucket_raw)

        if num_batches == 0 or total_tokens == 0:
            return {
                "eval_loss": float('inf'),
                "perplexity": float('inf'),
                "total_tokens": 0,
                "eval_time_sec": time.time() - eval_start_time,
                "bucket_metrics": bucket_metrics,
                "bucket_boundaries": {
                    "short_max_tokens": self.eval_short_max_tokens,
                    "medium_max_tokens": self.eval_medium_max_tokens,
                },
            }
        
        avg_loss = total_nll_sum / total_tokens
        perplexity = float(math.exp(min(20.0, avg_loss)))
        
        return {
            "eval_loss": avg_loss,
            "perplexity": perplexity,
            "total_tokens": total_tokens,
            "eval_time_sec": time.time() - eval_start_time,
            "bucket_metrics": bucket_metrics,
            "bucket_boundaries": {
                "short_max_tokens": self.eval_short_max_tokens,
                "medium_max_tokens": self.eval_medium_max_tokens,
            },
        }

    def save_checkpoint(
        self,
        checkpoint_name: str,
        optimizer,
        scheduler,
        scaler: Optional[GradScaler] = None,
        is_best: bool = False,
        checkpoint_reasons: Optional[List[str]] = None,
        eval_loss: Optional[float] = None,
    ) -> Optional[Path]:
        """保存檢查點"""
        if not self.is_main_process:
            return None

        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 模型狀態
            model_to_save = self._unwrap_model()
            torch.save(model_to_save.state_dict(), checkpoint_dir / "model.pt")
            
            # 配置
            with open(checkpoint_dir / "config.json", 'w', encoding='utf-8') as f:
                json.dump(vars(self.model_config), f, indent=2, ensure_ascii=False)
            
            # 分詞器
            if self.tokenizer_manager.tokenizer_object:
                self.tokenizer_manager.tokenizer_object.save(str(checkpoint_dir / "tokenizer.json"))
            
            # 訓練狀態
            training_state = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'best_eval_loss': self.best_eval_loss,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scheduler_total_steps': self._scheduler_total_steps,
                'scheduler_warmup_steps': self._scheduler_warmup_steps,
                'training_history': dict(self.training_history)
            }
            
            if scaler and scaler.is_enabled():
                training_state['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(training_state, checkpoint_dir / "training_state.pt")
            
            # 如果是最佳模型，額外保存
            if is_best:
                best_model_dir = self.output_dir / "best_model"
                if best_model_dir.exists():
                    import shutil
                    shutil.rmtree(best_model_dir)
                import shutil
                shutil.copytree(checkpoint_dir, best_model_dir)
                self.best_checkpoint_path = checkpoint_dir
            
            self.logger.info(f"檢查點已保存: {checkpoint_dir}")
            self.metrics_writer.log_checkpoint_metrics({
                "epoch": self.current_epoch + 1,
                "global_step": self.global_step,
                "checkpoint_name": checkpoint_name,
                "checkpoint_path": str(checkpoint_dir),
                "eval_loss": eval_loss,
                "is_best": is_best,
                "reasons": "|".join(checkpoint_reasons or []),
            })
            return checkpoint_dir
            
        except Exception as e:
            self.logger.error(f"保存檢查點失敗: {e}")
            return None

    def load_checkpoint(self, checkpoint_dir: str, optimizer, scheduler, 
                       scaler: Optional[GradScaler] = None) -> bool:
        """加載檢查點"""
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            self.logger.warning(f"檢查點目錄不存在: {checkpoint_dir}")
            return False
        
        try:
            # 加載訓練狀態
            state_path = checkpoint_path / "training_state.pt"
            if state_path.exists():
                state = torch.load(state_path, map_location=self.device)
                self.current_epoch = state.get('epoch', 0)
                self.global_step = state.get('global_step', 0)
                self.best_eval_loss = state.get('best_eval_loss', float('inf'))
                self.training_history = defaultdict(list, state.get('training_history', {}))

                optimizer_state = state.get('optimizer_state_dict')
                if optimizer_state is not None:
                    optimizer.load_state_dict(optimizer_state)
                else:
                    self.logger.warning("檢查點缺少 optimizer 狀態，將使用新優化器狀態")

                scheduler_state = state.get('scheduler_state_dict')
                if scheduler_state is not None:
                    try:
                        scheduler.load_state_dict(scheduler_state)
                    except Exception as e:
                        self.logger.warning(f"加載 scheduler 狀態失敗，改為按 global_step 對齊: {e}")
                        self._align_scheduler_to_global_step(scheduler, reason="load_state_failed")
                else:
                    self.logger.warning("檢查點缺少 scheduler 狀態，改為按 global_step 對齊")
                    self._align_scheduler_to_global_step(scheduler, reason="state_missing")

                saved_scheduler_total_steps = state.get('scheduler_total_steps')
                current_scheduler_total_steps = self._scheduler_total_steps
                if (
                    saved_scheduler_total_steps is not None
                    and current_scheduler_total_steps is not None
                    and int(saved_scheduler_total_steps) != int(current_scheduler_total_steps)
                ):
                    self.logger.warning(
                        f"檢查點 scheduler_total_steps({saved_scheduler_total_steps}) "
                        f"與當前設定({current_scheduler_total_steps})不一致，將按 global_step 對齊"
                    )
                    self._align_scheduler_to_global_step(scheduler, reason="total_steps_mismatch")
                else:
                    self._verify_scheduler_alignment(scheduler)

                if scaler and 'scaler_state_dict' in state:
                    scaler.load_state_dict(state['scaler_state_dict'])
            
            # 加載模型
            model_path = checkpoint_path / "model.pt"
            if model_path.exists():
                model_to_load = self._unwrap_model()
                model_to_load.load_state_dict(torch.load(model_path, map_location=self.device))
                self.best_checkpoint_path = checkpoint_path
            
            self.logger.info(f"檢查點加載成功: {checkpoint_dir}")
            self.logger.info(f"恢復至 Epoch {self.current_epoch}, Step {self.global_step}")
            if self.is_distributed:
                dist.barrier()
            return True
            
        except Exception as e:
            self.logger.error(f"加載檢查點失敗: {e}")
            return False

    def _load_best_model_weights(self) -> bool:
        """回滾到最佳模型權重"""
        best_model_dir = self.output_dir / "best_model"
        best_model_path = best_model_dir / "model.pt"

        target_path = None
        if best_model_path.exists():
            target_path = best_model_path
        elif self.best_checkpoint_path is not None:
            candidate = self.best_checkpoint_path / "model.pt"
            if candidate.exists():
                target_path = candidate

        if target_path is None:
            return False

        try:
            model_to_load = self._unwrap_model()
            model_to_load.load_state_dict(torch.load(target_path, map_location=self.device))
            return True
        except Exception as e:
            self.logger.warning(f"回滾最佳模型失敗: {e}")
            return False

    def find_latest_checkpoint(self) -> Optional[str]:
        """查找最新檢查點"""
        checkpoint_pattern = "checkpoint-*"
        checkpoints = [p for p in self.output_dir.glob(checkpoint_pattern) if p.is_dir()]
        
        if not checkpoints:
            return None

        epoch_checkpoints = [p for p in checkpoints if p.name.startswith("checkpoint-epoch-")]
        candidate_checkpoints = epoch_checkpoints if epoch_checkpoints else checkpoints
        
        # 按修改時間排序
        latest_checkpoint = max(candidate_checkpoints, key=lambda x: x.stat().st_mtime)
        return str(latest_checkpoint)

    def _export_training_summary(
        self,
        run_status: str,
        epochs_planned: int,
        epochs_completed: int,
        run_start_time: float,
        run_aggregates: Dict[str, Any],
        eval_history: List[Dict[str, Any]],
        final_eval_metrics: Optional[Dict[str, Any]],
        checkpoint_overview: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        run_time_sec = float(time.time() - run_start_time)

        finite_eval_records = [
            r for r in eval_history
            if r.get("eval_loss") is not None and math.isfinite(float(r.get("eval_loss")))
        ]
        best_record = min(finite_eval_records, key=lambda x: float(x["eval_loss"])) if finite_eval_records else None

        best_bucket_metrics = (best_record or {}).get("bucket_metrics", {})
        best_bucket_flat = self._flatten_bucket_metrics(best_bucket_metrics)

        final_eval_loss = None
        final_perplexity = None
        if final_eval_metrics:
            final_eval_loss = final_eval_metrics.get("eval_loss")
            final_perplexity = final_eval_metrics.get("perplexity")
        elif eval_history:
            final_eval_loss = eval_history[-1].get("eval_loss")
            final_perplexity = eval_history[-1].get("perplexity")

        summary = {
            "run_status": run_status,
            "epochs_planned": int(epochs_planned),
            "epochs_completed": int(epochs_completed),
            "global_step": int(self.global_step),
            "best_eval_loss": best_record.get("eval_loss") if best_record else None,
            "best_eval_epoch": best_record.get("epoch") if best_record else None,
            "final_eval_loss": final_eval_loss,
            "final_perplexity": final_perplexity,
            "total_tokens_seen": int(run_aggregates.get("tokens_seen", 0)),
            "invalid_loss_count": int(run_aggregates.get("invalid_loss_count", 0)),
            "invalid_grad_count": int(run_aggregates.get("invalid_grad_count", 0)),
            "skipped_update_count": int(run_aggregates.get("skipped_update_count", 0)),
            "run_time_sec": round(run_time_sec, 3),
            "best_checkpoint_path": (checkpoint_overview or {}).get("best_checkpoint_path", ""),
            "latest_checkpoint_path": (checkpoint_overview or {}).get("latest_checkpoint_path", ""),
            "best_bucket_metrics": best_bucket_metrics,
            "bucket_boundaries": {
                "short_max_tokens": self.eval_short_max_tokens,
                "medium_max_tokens": self.eval_medium_max_tokens,
            },
            "checkpoint_overview": checkpoint_overview or {},
            "eval_history": eval_history,
        }

        now_tag = time.strftime("%Y%m%d_%H%M%S")
        latest_json = metrics_dir / "training_summary_latest.json"
        timed_json = metrics_dir / f"training_summary_{now_tag}.json"

        with open(latest_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(timed_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        csv_row = {
            "run_status": summary["run_status"],
            "epochs_planned": summary["epochs_planned"],
            "epochs_completed": summary["epochs_completed"],
            "global_step": summary["global_step"],
            "best_eval_loss": summary["best_eval_loss"],
            "best_eval_epoch": summary["best_eval_epoch"],
            "final_eval_loss": summary["final_eval_loss"],
            "final_perplexity": summary["final_perplexity"],
            "total_tokens_seen": summary["total_tokens_seen"],
            "invalid_loss_count": summary["invalid_loss_count"],
            "invalid_grad_count": summary["invalid_grad_count"],
            "skipped_update_count": summary["skipped_update_count"],
            "run_time_sec": summary["run_time_sec"],
            "best_checkpoint_path": summary["best_checkpoint_path"],
            "latest_checkpoint_path": summary["latest_checkpoint_path"],
            "best_short_loss": best_bucket_flat.get("short_loss", ""),
            "best_medium_loss": best_bucket_flat.get("medium_loss", ""),
            "best_long_loss": best_bucket_flat.get("long_loss", ""),
            "best_short_perplexity": best_bucket_flat.get("short_perplexity", ""),
            "best_medium_perplexity": best_bucket_flat.get("medium_perplexity", ""),
            "best_long_perplexity": best_bucket_flat.get("long_perplexity", ""),
        }
        self.metrics_writer.log_training_summary(csv_row)

        return summary

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        batch_size: int = 4,
        num_epochs: int = 100,
        learning_rate: float = 5e-5,
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.01,
        lr_scheduler_type: str = "cosine",
        warmup_ratio: float = 0.03,
        max_grad_norm: float = 1.0,
        log_interval: int = 1,
        save_interval_epochs: int = 10,
        early_stopping_patience: int = 60,
        early_stopping_monitor: str = "eval_loss",
        early_stopping_min_delta: Optional[float] = None,
        early_stopping_warmup_epochs: int = 8,
        resume_from_checkpoint: bool = True,
        resume_checkpoint: Optional[str] = None,
        eval_interval_epochs: int = 5,
        metrics_log_interval_steps: int = 100,
        save_best_k: int = 3,
        save_latest_k: int = 2,
        save_on_improve_delta: float = 0.001,
        cleanup_old_checkpoints: bool = False,
        scheduler_target_epochs: Optional[int] = None,
        eval_short_max_tokens: int = 64,
        eval_medium_max_tokens: int = 256,
        dataloader_num_workers: Optional[int] = None,
        dataloader_prefetch_factor: int = 2,
        dataloader_persistent_workers: bool = True,
        dataloader_drop_last: bool = True,
    ):
        """主訓練函數"""

        self.logger.info("***** 開始訓練 *****")
        self.eval_short_max_tokens = max(1, int(eval_short_max_tokens))
        self.eval_medium_max_tokens = max(self.eval_short_max_tokens + 1, int(eval_medium_max_tokens))

        early_stopping_monitor = str(early_stopping_monitor).strip().lower()
        if early_stopping_monitor not in {"train_loss", "eval_loss"}:
            raise ValueError(f"不支援的 early_stopping_monitor: {early_stopping_monitor}")
        early_stopping_warmup_epochs = max(0, int(early_stopping_warmup_epochs))
        effective_early_stopping_delta = (
            max(0.0, float(early_stopping_min_delta))
            if early_stopping_min_delta is not None
            else max(0.0, float(save_on_improve_delta))
        )

        if early_stopping_patience > 0:
            self.early_stopping = EarlyStoppingCallback(
                patience=early_stopping_patience,
                min_delta=effective_early_stopping_delta,
                mode='min'
            )
            self.logger.info(
                f"早停配置 - monitor: {early_stopping_monitor}, patience: {early_stopping_patience}, "
                f"min_delta: {effective_early_stopping_delta}, warmup_epochs: {early_stopping_warmup_epochs}"
            )

        if scheduler_target_epochs is None and early_stopping_patience > 0:
            scheduler_target_epochs = min(num_epochs, max(30, early_stopping_patience * 2))

        self.checkpoint_manager = CheckpointManager(
            output_dir=self.output_dir,
            policy=CheckpointPolicy(
                save_interval_epochs=save_interval_epochs,
                keep_best_k=save_best_k,
                keep_latest_k=save_latest_k,
                save_on_improve_delta=save_on_improve_delta,
                cleanup_old_checkpoints=cleanup_old_checkpoints,
            ),
            logger=self.logger,
        )

        train_dataloader, optimizer, scheduler, scaler, train_sampler = self.setup_training_components(
            train_dataset,
            batch_size,
            num_epochs,
            learning_rate,
            weight_decay,
            gradient_accumulation_steps,
            warmup_ratio,
            lr_scheduler_type,
            scheduler_target_epochs=scheduler_target_epochs,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_prefetch_factor=dataloader_prefetch_factor,
            dataloader_persistent_workers=dataloader_persistent_workers,
            dataloader_drop_last=dataloader_drop_last,
        )

        eval_dataloader = None
        if eval_dataset:
            collate_fn = partial(
                smart_collate_fn,
                tokenizer=self.tokenizer_manager.transformers_tokenizer,
                max_seq_length=self.model_config.max_position_embeddings,
                pack_sequences=False
            )

            eval_sampler: Optional[Sampler[int]] = None
            if self.is_distributed:
                eval_sampler = _DistributedEvalSampler(
                    eval_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                )

            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=batch_size * 2,
                collate_fn=collate_fn,
                shuffle=False,
                sampler=eval_sampler,
                num_workers=0,
                pin_memory=self.device.type == 'cuda'
            )

        if resume_from_checkpoint:
            latest_checkpoint: Optional[str] = None
            if self.is_main_process:
                if resume_checkpoint:
                    resolved_resume_checkpoint = self._resolve_resume_checkpoint_path(resume_checkpoint)
                    if resolved_resume_checkpoint is None:
                        self.logger.warning(
                            f"指定的 resume_checkpoint 不存在或不可用: {resume_checkpoint}"
                        )
                    else:
                        latest_checkpoint = str(resolved_resume_checkpoint)
                else:
                    latest_checkpoint = self.find_latest_checkpoint()
            if self.is_distributed:
                broadcast_objects: List[Optional[str]] = [latest_checkpoint]
                dist.broadcast_object_list(broadcast_objects, src=0)
                latest_checkpoint = broadcast_objects[0]
            if latest_checkpoint and self.load_checkpoint(latest_checkpoint, optimizer, scheduler, scaler):
                self.logger.info(f"從檢查點恢復訓練: {latest_checkpoint}")
            elif resume_checkpoint and self.is_main_process:
                self.logger.warning("無法從指定檢查點恢復，將從當前狀態開始訓練")

        early_stopped = False
        run_status = "running"
        run_start_time = time.time()
        epochs_completed = 0
        final_eval_metrics: Optional[Dict[str, Any]] = None
        eval_history: List[Dict[str, Any]] = []
        run_aggregates = {
            "tokens_seen": 0,
            "invalid_loss_count": 0,
            "invalid_grad_count": 0,
            "skipped_update_count": 0,
        }

        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                self.logger.info(f"=== Epoch {epoch + 1}/{num_epochs} ===")

                epoch_stats = self.train_epoch(
                    train_dataloader,
                    train_sampler,
                    optimizer,
                    scheduler,
                    scaler,
                    gradient_accumulation_steps,
                    max_grad_norm,
                    log_steps_per_epoch=10,
                    metrics_log_interval_steps=metrics_log_interval_steps,
                )

                if self.is_distributed:
                    stats_tensor = torch.tensor(
                        [
                            float(epoch_stats.get("loss_sum", 0.0)),
                            float(epoch_stats.get("num_batches", 0)),
                            float(epoch_stats.get("tokens_seen", 0)),
                            float(epoch_stats.get("invalid_loss_count", 0)),
                            float(epoch_stats.get("invalid_grad_count", 0)),
                            float(epoch_stats.get("skipped_update_count", 0)),
                        ],
                        device=self.device,
                        dtype=torch.float64,
                    )
                    dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

                    global_num_batches = int(stats_tensor[1].item())
                    epoch_loss = (
                        float(stats_tensor[0].item() / global_num_batches)
                        if global_num_batches > 0
                        else float("inf")
                    )
                    run_aggregates["tokens_seen"] += int(stats_tensor[2].item())
                    run_aggregates["invalid_loss_count"] += int(stats_tensor[3].item())
                    run_aggregates["invalid_grad_count"] += int(stats_tensor[4].item())
                    run_aggregates["skipped_update_count"] += int(stats_tensor[5].item())
                else:
                    epoch_loss = float(epoch_stats["avg_loss"])
                    run_aggregates["tokens_seen"] += int(epoch_stats.get("tokens_seen", 0))
                    run_aggregates["invalid_loss_count"] += int(epoch_stats.get("invalid_loss_count", 0))
                    run_aggregates["invalid_grad_count"] += int(epoch_stats.get("invalid_grad_count", 0))
                    run_aggregates["skipped_update_count"] += int(epoch_stats.get("skipped_update_count", 0))

                self.training_history['train_loss'].append(epoch_loss)
                epochs_completed += 1

                should_log_details = self.is_main_process and ((log_interval <= 1) or ((epoch + 1) % log_interval == 0))
                if should_log_details:
                    self.logger.info(f"Epoch {epoch + 1} 完成 - 平均損失: {epoch_loss:.4f}")

                had_anomaly = (epoch_stats["invalid_loss_count"] > 0) or (epoch_stats["invalid_grad_count"] > 0)

                run_eval = _has_eval_dataloader(eval_dataloader) and (
                    eval_interval_epochs <= 1
                    or (epoch + 1) % eval_interval_epochs == 0
                    or epoch == 0
                    or epoch == num_epochs - 1
                )

                eval_loss = None
                perplexity = None
                is_best = False
                previous_best = self.best_eval_loss

                if run_eval:
                    eval_metrics = self.evaluate(eval_dataloader)
                    eval_loss = eval_metrics.get('eval_loss', float('inf'))
                    perplexity = eval_metrics.get('perplexity', float('inf'))
                    eval_time_sec = eval_metrics.get('eval_time_sec', 0.0)
                    bucket_metrics = eval_metrics.get("bucket_metrics", {})
                    bucket_metrics_flat = self._flatten_bucket_metrics(bucket_metrics)

                    self.training_history['eval_loss'].append(eval_loss)
                    self.training_history['perplexity'].append(perplexity)
                    eval_history.append({
                        "epoch": epoch + 1,
                        "global_step": self.global_step,
                        "eval_loss": eval_loss,
                        "perplexity": perplexity,
                        "total_tokens": eval_metrics.get("total_tokens", 0),
                        "eval_time_sec": eval_time_sec,
                        "bucket_metrics": bucket_metrics,
                    })

                    improved = eval_loss < previous_best
                    is_best = improved
                    if improved:
                        self.best_eval_loss = eval_loss
                        if should_log_details:
                            self.logger.info(f"🎉 新的最佳模型! 評估損失: {eval_loss:.4f}")

                    if should_log_details:
                        self.logger.info(f"評估結果 - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")

                    self.metrics_writer.log_eval_metrics({
                        "epoch": epoch + 1,
                        "global_step": self.global_step,
                        "eval_loss": eval_loss,
                        "perplexity": perplexity,
                        "total_tokens": eval_metrics.get("total_tokens", 0),
                        "eval_time_sec": eval_time_sec,
                        "improved": improved,
                        "is_best": is_best,
                        **bucket_metrics_flat,
                    })

                early_stop_score: Optional[float] = None
                if self.early_stopping:
                    if early_stopping_monitor == "train_loss":
                        early_stop_score = float(epoch_loss)
                    elif eval_loss is not None and math.isfinite(float(eval_loss)):
                        early_stop_score = float(eval_loss)

                early_stop_triggered = False
                if self.early_stopping and early_stop_score is not None:
                    if (epoch + 1) <= early_stopping_warmup_epochs:
                        if should_log_details:
                            self.logger.info(
                                f"早停預熱中({epoch + 1}/{early_stopping_warmup_epochs}) - "
                                f"monitor: {early_stopping_monitor}, score: {early_stop_score:.6f}"
                            )
                    else:
                        if self.is_main_process:
                            early_stop_triggered = self.early_stopping(early_stop_score)
                            if should_log_details:
                                self.logger.info(
                                    f"早停監控({early_stopping_monitor}={early_stop_score:.6f}) - "
                                    f"patience剩餘: {self.early_stopping.patience_remaining}/"
                                    f"{self.early_stopping.patience}"
                                )
                elif self.early_stopping and should_log_details and early_stopping_monitor == "eval_loss":
                    self.logger.info("早停監控(eval_loss) - 本epoch無評估結果，略過")

                if self.is_distributed:
                    stop_tensor = torch.tensor(1 if early_stop_triggered else 0, device=self.device, dtype=torch.int32)
                    dist.broadcast(stop_tensor, src=0)
                    early_stop_triggered = bool(stop_tensor.item())

                if early_stop_triggered:
                    early_stopped = True
                    run_status = "early_stopped"
                    if self.is_main_process:
                        self.logger.info(f"早停觸發! monitor={early_stopping_monitor}, score={early_stop_score:.6f}")
                        self.metrics_writer.log_event({
                            "epoch": epoch + 1,
                            "global_step": self.global_step,
                            "event_type": "early_stopping",
                            "severity": "INFO",
                            "message": f"early stopping triggered on {early_stopping_monitor}",
                            "value": early_stop_score,
                        })

                save_reasons = self.checkpoint_manager.should_save(
                    epoch=epoch,
                    num_epochs=num_epochs,
                    eval_loss=eval_loss,
                    previous_best_eval_loss=previous_best,
                    had_anomaly=had_anomaly,
                )
                if is_best and "best" not in save_reasons:
                    save_reasons.append("best")
                if early_stopped and "early_stop" not in save_reasons:
                    save_reasons.append("early_stop")

                if save_reasons:
                    checkpoint_name = f"checkpoint-epoch-{epoch + 1}"
                    checkpoint_path = self.save_checkpoint(
                        checkpoint_name,
                        optimizer,
                        scheduler,
                        scaler,
                        is_best=is_best,
                        checkpoint_reasons=save_reasons,
                        eval_loss=eval_loss,
                    )
                    if checkpoint_path is not None:
                        self.checkpoint_manager.register_checkpoint(
                            checkpoint_path=checkpoint_path,
                            checkpoint_name=checkpoint_name,
                            epoch=epoch + 1,
                            eval_loss=eval_loss,
                            is_best=is_best,
                            reasons=save_reasons,
                        )

                if self.use_wandb:
                    log_data = {
                        "epoch": epoch + 1,
                        "train/epoch_loss": epoch_loss,
                        "train/invalid_loss_count": epoch_stats["invalid_loss_count"],
                        "train/invalid_grad_count": epoch_stats["invalid_grad_count"],
                        "train/skipped_update_count": epoch_stats["skipped_update_count"],
                    }
                    if eval_loss is not None:
                        log_data.update({
                            "eval/loss": eval_loss,
                            "eval/perplexity": perplexity,
                        })
                    wandb.log(log_data)

                if early_stopped:
                    break

            if run_status == "running":
                run_status = "completed"

            self.logger.info("***** 訓練完成 *****")

            if early_stopped and self._load_best_model_weights():
                self.logger.info("早停後已回滾到最佳模型權重")

            final_checkpoint_path = self.save_checkpoint(
                "final_model",
                optimizer,
                scheduler,
                scaler,
                is_best=False,
                checkpoint_reasons=["final_export"],
                eval_loss=self.best_eval_loss if math.isfinite(self.best_eval_loss) else None,
            )
            if final_checkpoint_path is not None and self.checkpoint_manager is not None:
                self.checkpoint_manager.register_checkpoint(
                    checkpoint_path=final_checkpoint_path,
                    checkpoint_name="final_model",
                    epoch=self.current_epoch + 1,
                    eval_loss=self.best_eval_loss if math.isfinite(self.best_eval_loss) else None,
                    is_best=False,
                    reasons=["final_export"],
                )

            if _has_eval_dataloader(eval_dataloader):
                final_eval_metrics = self.evaluate(eval_dataloader)
                self.logger.info(f"最終評估結果: {final_eval_metrics}")

        except KeyboardInterrupt:
            run_status = "interrupted"
            self.logger.info("訓練被用戶中斷")
            self.metrics_writer.log_event({
                "epoch": self.current_epoch + 1,
                "global_step": self.global_step,
                "event_type": "keyboard_interrupt",
                "severity": "INFO",
                "message": "training interrupted by user",
                "value": "",
            })
            self.save_checkpoint(
                f"checkpoint-interrupted-step-{self.global_step}",
                optimizer,
                scheduler,
                scaler,
                is_best=False,
                checkpoint_reasons=["interrupted"],
                eval_loss=None,
            )
        except Exception as e:
            run_status = "failed"
            self.metrics_writer.log_event({
                "epoch": self.current_epoch + 1,
                "global_step": self.global_step,
                "event_type": "training_exception",
                "severity": "ERROR",
                "message": str(e),
                "value": "",
            })
            self.logger.error(f"訓練過程中發生錯誤: {e}", exc_info=True)
            raise
        finally:
            checkpoint_overview = {}
            if self.is_main_process and self.checkpoint_manager is not None:
                try:
                    checkpoint_overview = self.checkpoint_manager.export_index(self.output_dir / "metrics")
                except Exception as e:
                    self.logger.warning(f"checkpoint 索引匯出失敗: {e}")

            if self.is_main_process:
                try:
                    self._export_training_summary(
                        run_status=run_status,
                        epochs_planned=num_epochs,
                        epochs_completed=epochs_completed,
                        run_start_time=run_start_time,
                        run_aggregates=run_aggregates,
                        eval_history=eval_history,
                        final_eval_metrics=final_eval_metrics,
                        checkpoint_overview=checkpoint_overview,
                    )
                except Exception as e:
                    self.logger.warning(f"訓練摘要匯出失敗: {e}")

            if self.use_wandb:
                wandb.finish()