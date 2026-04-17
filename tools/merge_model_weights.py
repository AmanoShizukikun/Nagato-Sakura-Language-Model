#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""合併兩個 NagatoSakura 模型權重，輸出新的模型資料夾。

預設採用線性插值：merged = alpha * model_a + (1 - alpha) * model_b
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nagato_sakura_model import NagatoSakuraForCausalLM, NSConfig  # noqa: E402


WEIGHT_CANDIDATES = (
    "pytorch_model.bin",
    "model.pt",
    "model.safetensors",
)


def _resolve_model_dir(path: Path) -> Path:
    if path.is_file():
        return path.parent
    return path


def _find_weight_file(model_path: Path) -> Path:
    if model_path.is_file():
        return model_path

    for candidate in WEIGHT_CANDIDATES:
        file_path = model_path / candidate
        if file_path.exists():
            return file_path

    raise FileNotFoundError(
        f"找不到模型權重檔，預期其中之一: {', '.join(WEIGHT_CANDIDATES)}"
    )


def _load_state_dict(weight_path: Path) -> Dict[str, torch.Tensor]:
    if weight_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "讀取 model.safetensors 需要 safetensors 套件，請先安裝後再重試"
            ) from exc

        return load_file(str(weight_path))

    state_dict = torch.load(weight_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(f"權重檔格式不正確: {weight_path}")
    return state_dict


def _load_model_config(model_dir: Path) -> NSConfig:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置檔: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config_dict = json.load(file)

    if not isinstance(config_dict, dict):
        raise TypeError(f"配置檔格式不正確: {config_path}")

    return NSConfig.from_dict(config_dict)


def _compare_state_shapes(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
) -> List[str]:
    problems: List[str] = []
    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())

    missing_in_b = sorted(keys_a - keys_b)
    missing_in_a = sorted(keys_b - keys_a)
    if missing_in_b:
        problems.append(f"model_b 缺少 {len(missing_in_b)} 個權重鍵，例如: {missing_in_b[:5]}")
    if missing_in_a:
        problems.append(f"model_a 缺少 {len(missing_in_a)} 個權重鍵，例如: {missing_in_a[:5]}")

    shared_keys = sorted(keys_a & keys_b)
    shape_mismatches: List[str] = []
    for key in shared_keys:
        tensor_a = state_a[key]
        tensor_b = state_b[key]
        if tensor_a.shape != tensor_b.shape:
            shape_mismatches.append(f"{key}: {tuple(tensor_a.shape)} vs {tuple(tensor_b.shape)}")

    if shape_mismatches:
        preview = shape_mismatches[:10]
        problems.append(
            f"存在 {len(shape_mismatches)} 個權重形狀不一致，例如: {preview}"
        )

    return problems


def _merge_state_dicts(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for key in state_a.keys():
        tensor_a = state_a[key]
        tensor_b = state_b[key]
        if torch.is_floating_point(tensor_a) or torch.is_complex(tensor_a):
            merged_tensor = tensor_a.to(torch.float32) * alpha + tensor_b.to(torch.float32) * (1.0 - alpha)
            merged[key] = merged_tensor.to(dtype=tensor_a.dtype)
        else:
            merged[key] = tensor_a.clone()
    return merged


def _copy_tokenizer_if_present(source_dirs: Iterable[Path], output_dir: Path) -> Path | None:
    for source_dir in source_dirs:
        tokenizer_file = source_dir / "tokenizer.json"
        if tokenizer_file.exists():
            target = output_dir / "tokenizer.json"
            shutil.copy2(tokenizer_file, target)
            return target
    return None


def merge_models(model_a_path: Path, model_b_path: Path, output_dir: Path, alpha: float) -> Dict[str, str]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必須介於 0.0 到 1.0 之間")

    model_a_dir = _resolve_model_dir(model_a_path)
    model_b_dir = _resolve_model_dir(model_b_path)

    config = _load_model_config(model_a_dir)
    _ = _load_model_config(model_b_dir)

    weight_a = _find_weight_file(model_a_path)
    weight_b = _find_weight_file(model_b_path)
    state_a = _load_state_dict(weight_a)
    state_b = _load_state_dict(weight_b)

    problems = _compare_state_shapes(state_a, state_b)
    if problems:
        raise ValueError("\n".join(problems))

    output_dir.mkdir(parents=True, exist_ok=True)

    merged_state = _merge_state_dicts(state_a, state_b, alpha)

    model = NagatoSakuraForCausalLM(config)
    model.load_state_dict(merged_state, strict=True)
    model.save_pretrained(output_dir)

    tokenizer_copy = _copy_tokenizer_if_present((model_a_dir, model_b_dir), output_dir)

    summary = {
        "output_dir": str(output_dir),
        "model_a": str(model_a_path),
        "model_b": str(model_b_path),
        "alpha": f"{alpha:.6f}",
        "tokenizer_copied": str(bool(tokenizer_copy)),
        "config_source": str(model_a_dir),
    }

    with open(output_dir / "merge_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="合併兩個 NagatoSakura 模型權重")
    parser.add_argument("--model_a", type=str, required=True, help="第一個模型資料夾或權重檔")
    parser.add_argument("--model_b", type=str, required=True, help="第二個模型資料夾或權重檔")
    parser.add_argument("--output_dir", type=str, required=True, help="輸出模型資料夾")
    parser.add_argument("--alpha", type=float, default=0.5, help="model_a 權重比例，model_b 比例為 1-alpha")
    args = parser.parse_args()

    try:
        summary = merge_models(
            model_a_path=Path(args.model_a),
            model_b_path=Path(args.model_b),
            output_dir=Path(args.output_dir),
            alpha=float(args.alpha),
        )
        print("模型合併完成")
        print(f"輸出目錄: {summary['output_dir']}")
        print(f"alpha: {summary['alpha']}")
        print(f"tokenizer 已複製: {summary['tokenizer_copied']}")
    except Exception as exc:
        print(f"模型合併失敗: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()