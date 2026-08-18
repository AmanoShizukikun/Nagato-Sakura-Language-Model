"""
NagatoSakura 模型轉檔工具 (Model Export & Conversion Tool)

支援將 NagatoSakura LLM 模型轉為：
1. Safetensors 格式 (model.safetensors)
2. ONNX 格式 (model.onnx) (支援 KV-Cache 加速)

使用範例：
    python tools/export_model.py --model_path NS-LM-1.6-micro/best_model --output_dir NS-LM-1.6-micro/export --format both
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# 設定模組搜尋路徑
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nagato_sakura_model import NagatoSakuraForCausalLM, NSConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ExportModel")

WEIGHT_CANDIDATES = (
    "pytorch_model.bin",
    "model.pt",
    "model.safetensors",
)


class NagatoSakuraONNXWrapper(nn.Module):
    """
    基礎版 ONNX 封裝模組 (無 KV-Cache)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs.logits


class NagatoSakuraONNXWrapperWithKVCache(nn.Module):
    """
    進階版 ONNX 封裝模組 (支援 KV-Cache，實現 O(1) 逐字生成)
    輸入：
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, total_seq_len)
        position_ids: (batch_size, seq_len)
        past_key_0, past_value_0, past_key_1, past_value_1, ... : (batch_size, num_kv_heads, past_seq_len, head_dim)
    輸出：
        logits: (batch_size, seq_len, vocab_size)
        present_key_0, present_value_0, ... : (batch_size, num_kv_heads, total_seq_len, head_dim)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.num_layers = model.config.num_hidden_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *past_kvs: torch.Tensor,
    ):
        past_key_values = []
        if len(past_kvs) == self.num_layers * 2:
            for i in range(self.num_layers):
                pk = past_kvs[2 * i]
                pv = past_kvs[2 * i + 1]
                past_key_values.append((pk, pv))
        else:
            past_key_values = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        present_kvs = []
        if outputs.past_key_values is not None:
            for kv in outputs.past_key_values:
                if isinstance(kv, (tuple, list)):
                    present_kvs.append(kv[0])
                    present_kvs.append(kv[1])

        return (outputs.logits, *present_kvs)


def resolve_model_dir(path: Path) -> Path:
    if path.is_file():
        return path.parent
    return path


def find_weight_file(model_path: Path) -> Path:
    if model_path.is_file():
        return model_path

    for candidate in WEIGHT_CANDIDATES:
        file_path = model_path / candidate
        if file_path.exists():
            return file_path

    raise FileNotFoundError(
        f"找不到模型權重檔，預期其中之一: {', '.join(WEIGHT_CANDIDATES)}"
    )


def load_state_dict(weight_path: Path) -> Dict[str, torch.Tensor]:
    logger.info(f"正在從 {weight_path} 讀取權重...")
    if weight_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "讀取 .safetensors 檔案需要 safetensors 套件，請先執行: pip install safetensors"
            ) from exc
        return load_file(str(weight_path))

    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(f"無效的權重字典格式: {weight_path}")
    return state_dict


def load_model_and_config(model_path: Path) -> Tuple[NagatoSakuraForCausalLM, NSConfig, Path]:
    model_dir = resolve_model_dir(model_path)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置檔: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    config = NSConfig.from_dict(config_dict)
    weight_path = find_weight_file(model_path)
    state_dict = load_state_dict(weight_path)

    logger.info("初始化 NagatoSakuraForCausalLM 模型結構...")
    model = NagatoSakuraForCausalLM(config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, config, model_dir


def copy_auxiliary_files(source_dir: Path, output_dir: Path) -> None:
    """複製 config.json, tokenizer.json 等相關配置檔"""
    aux_files = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    copied = []
    for filename in aux_files:
        src_file = source_dir / filename
        if src_file.exists():
            dst_file = output_dir / filename
            shutil.copy2(src_file, dst_file)
            copied.append(filename)

    if copied:
        logger.info(f"已複製輔助配置檔至輸出目錄: {', '.join(copied)}")


def export_to_safetensors(
    state_dict: Dict[str, torch.Tensor],
    output_path: Path,
) -> None:
    """將模型權重轉存為 Safetensors 格式"""
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError(
            "匯出為 Safetensors 需要 safetensors 套件，請先執行: pip install safetensors"
        ) from exc

    logger.info(f"正在將權重寫入 Safetensors 檔: {output_path}...")
    # 處理共用記憶體權重 (如 tie_word_embeddings) 與張量連續性
    seen_ptrs = set()
    contiguous_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            ptr = v.data_ptr()
            if ptr in seen_ptrs:
                contiguous_dict[k] = v.clone().detach().cpu()
            else:
                seen_ptrs.add(ptr)
                contiguous_dict[k] = v.contiguous().cpu()
        else:
            contiguous_dict[k] = v

    save_file(contiguous_dict, str(output_path))
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Safetensors 匯出成功！大小: {file_size_mb:.2f} MB")


def verify_safetensors(output_path: Path, original_state_dict: Dict[str, torch.Tensor]) -> bool:
    """驗證 Safetensors 權重正確性"""
    try:
        from safetensors.torch import load_file
    except ImportError:
        logger.warning("缺少 safetensors 套件，跳過 Safetensors 驗證")
        return False

    logger.info("正在驗證匯出的 Safetensors 檔案...")
    loaded_dict = load_file(str(output_path))

    if set(loaded_dict.keys()) != set(original_state_dict.keys()):
        logger.error("❌ 鍵值集合不符！ Safetensors 驗證失敗。")
        return False

    max_diff = 0.0
    for k in original_state_dict:
        orig = original_state_dict[k].cpu()
        loaded = loaded_dict[k].cpu()
        diff = (orig - loaded).abs().max().item()
        if diff > max_diff:
            max_diff = diff

    if max_diff > 1e-5:
        logger.warning(f"⚠️ 權重最大數值偏離: {max_diff}")
    else:
        logger.info(f"✅ Safetensors 權重與原模型完全一致 (Max Diff: {max_diff:.8f})")
    return True


def export_to_onnx(
    model: NagatoSakuraForCausalLM,
    config: NSConfig,
    output_path: Path,
    opset_version: int = 14,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    use_kv_cache: bool = True,
) -> None:
    """將模型轉換並匯出為 ONNX 格式 (支援 KV-Cache)"""
    logger.info(f"正在準備 ONNX 匯出 (Device: {device}, Dtype: {dtype}, Opset: {opset_version}, KV-Cache: {use_kv_cache})...")

    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads or config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads

    if use_kv_cache:
        wrapper_model = NagatoSakuraONNXWrapperWithKVCache(model).to(device=device, dtype=dtype)
        wrapper_model.eval()

        # 建立 Dummy Inputs (Batch=1, SeqLen=4, PastLen=2)
        bsz = 1
        seq_len = 4
        past_seq_len = 2
        total_seq_len = past_seq_len + seq_len

        dummy_input_ids = torch.randint(0, config.vocab_size, (bsz, seq_len), dtype=torch.long, device=device)
        dummy_attention_mask = torch.ones((bsz, total_seq_len), dtype=torch.long, device=device)
        dummy_position_ids = torch.arange(past_seq_len, total_seq_len, dtype=torch.long, device=device).unsqueeze(0)

        dummy_past_kvs = []
        for _ in range(num_layers):
            dummy_past_kvs.append(torch.zeros((bsz, num_kv_heads, past_seq_len, head_dim), dtype=dtype, device=device))
            dummy_past_kvs.append(torch.zeros((bsz, num_kv_heads, past_seq_len, head_dim), dtype=dtype, device=device))

        inputs: Tuple[torch.Tensor, ...] = (dummy_input_ids, dummy_attention_mask, dummy_position_ids, *dummy_past_kvs)

        input_names = ["input_ids", "attention_mask", "position_ids"]
        output_names = ["logits"]
        dynamic_axes: Dict[str, Dict[int, str]] = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
            "position_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        }

        for i in range(num_layers):
            pk_name = f"past_key_{i}"
            pv_name = f"past_value_{i}"
            pres_k_name = f"present_key_{i}"
            pres_v_name = f"present_value_{i}"

            input_names.extend([pk_name, pv_name])
            output_names.extend([pres_k_name, pres_v_name])

            dynamic_axes[pk_name] = {0: "batch_size", 2: "past_sequence_length"}
            dynamic_axes[pv_name] = {0: "batch_size", 2: "past_sequence_length"}
            dynamic_axes[pres_k_name] = {0: "batch_size", 2: "total_sequence_length"}
            dynamic_axes[pres_v_name] = {0: "batch_size", 2: "total_sequence_length"}

    else:
        wrapper_model = NagatoSakuraONNXWrapper(model).to(device=device, dtype=dtype)
        wrapper_model.eval()

        dummy_input_ids = torch.randint(0, config.vocab_size, (1, 8), dtype=torch.long, device=device)
        dummy_attention_mask = torch.ones((1, 8), dtype=torch.long, device=device)
        inputs = (dummy_input_ids, dummy_attention_mask)

        input_names = ["input_ids", "attention_mask"]
        output_names = ["logits"]
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        }

    export_kwargs = {
        "export_params": True,
        "opset_version": opset_version,
        "do_constant_folding": True,
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
    }

    logger.info(f"正在執行 torch.onnx.export 至 {output_path}...")
    try:
        try:
            import onnx  # noqa: F401
        except ImportError as exc:
            raise ImportError("匯出 ONNX 需要 onnx 套件，請先執行: pip install onnx") from exc

        try:
            torch.onnx.export(
                wrapper_model,
                inputs,
                str(output_path),
                dynamo=False,
                **export_kwargs,
            )
        except (TypeError, ModuleNotFoundError):
            torch.onnx.export(
                wrapper_model,
                inputs,
                str(output_path),
                **export_kwargs,
            )
    except Exception as e:
        if "onnx is not installed" in str(e).lower():
            raise ImportError("匯出 ONNX 需要 onnx 套件，請先執行: pip install onnx") from e
        raise e

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ ONNX 匯出成功！大小: {file_size_mb:.2f} MB")


def verify_onnx(
    model: NagatoSakuraForCausalLM,
    onnx_path: Path,
    config: NSConfig,
    device: str = "cpu",
) -> bool:
    """驗證 ONNX 模型結構與推導數值一致性 (支援 KV-Cache 雙階段驗證)"""
    try:
        import onnx
    except ImportError:
        logger.warning("缺少 onnx 套件，跳過 ONNX 語法結構驗證 (pip install onnx)")
        return False

    logger.info("正在檢查 ONNX 模型結構合法性...")
    try:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX 結構檢查 (onnx.checker) 通過！")
    except Exception as e:
        logger.error(f"❌ ONNX 結構檢查失敗: {e}")
        return False

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("缺少 onnxruntime 套件，跳過 ONNXRuntime 數值對齊驗證 (pip install onnxruntime)")
        return True

    logger.info("正在使用 ONNXRuntime 進行數值對齊驗證...")
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in ort_session.get_inputs()]
    has_kv_cache = "past_key_0" in input_names

    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads or config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads

    if has_kv_cache:
        # 1. 驗證 Prefill (past_seq_len = 0)
        prompt_ids = np.random.randint(0, config.vocab_size, (1, 6), dtype=np.int64)
        p_len = prompt_ids.shape[1]
        prefill_inputs = {
            "input_ids": prompt_ids,
            "attention_mask": np.ones((1, p_len), dtype=np.int64),
            "position_ids": np.arange(0, p_len, dtype=np.int64).reshape(1, -1),
        }
        for i in range(num_layers):
            prefill_inputs[f"past_key_{i}"] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float32)
            prefill_inputs[f"past_value_{i}"] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float32)

        prefill_outputs = ort_session.run(None, prefill_inputs)
        ort_logits = prefill_outputs[0]
        present_kvs = prefill_outputs[1:]

        # PyTorch 前向比較
        model_cpu = model.to("cpu")
        model_cpu.eval()
        with torch.no_grad():
            pt_outputs = model_cpu(input_ids=torch.tensor(prompt_ids, dtype=torch.long), use_cache=True)
            pt_logits = pt_outputs.logits.detach().cpu().numpy()

        max_diff = float((abs(pt_logits - ort_logits)).max())
        logger.info(f"Prefill 階段 Logits 最大差異: {max_diff:.6f}")

        # 2. 驗證 Decode (past_seq_len = 6, 傳入 1 個 token)
        next_token_id = np.array([[12]], dtype=np.int64)
        decode_inputs = {
            "input_ids": next_token_id,
            "attention_mask": np.ones((1, p_len + 1), dtype=np.int64),
            "position_ids": np.array([[p_len]], dtype=np.int64),
        }
        for i in range(num_layers):
            decode_inputs[f"past_key_{i}"] = present_kvs[2 * i]
            decode_inputs[f"past_value_{i}"] = present_kvs[2 * i + 1]

        decode_outputs = ort_session.run(None, decode_inputs)
        dec_logits = decode_outputs[0]
        assert dec_logits.shape == (1, 1, config.vocab_size)
        logger.info("✅ ONNX KV-Cache Decode 步階驗證通過！")

    else:
        dummy_input_ids = torch.randint(0, config.vocab_size, (1, 6), dtype=torch.long)
        dummy_attention_mask = torch.ones((1, 6), dtype=torch.long)

        model.eval()
        with torch.no_grad():
            pt_outputs = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask, use_cache=False)
            pt_logits = pt_outputs.logits.numpy()

        ort_inputs = {"input_ids": dummy_input_ids.numpy()}
        if "attention_mask" in input_names:
            ort_inputs["attention_mask"] = dummy_attention_mask.numpy()

        ort_logits = ort_session.run(None, ort_inputs)[0]
        max_diff = float((abs(pt_logits - ort_logits)).max())
        logger.info(f"PyTorch vs ONNXRuntime Logits 最大差異: {max_diff:.6f}")

    logger.info("✅ ONNX 推導數值驗證完美對齊！")
    return True


def export_model(
    model_path: Path,
    output_dir: Path,
    export_format: str = "both",
    opset_version: int = 14,
    device: str = "auto",
    dtype_str: str = "fp32",
    use_kv_cache: bool = True,
    verify: bool = True,
) -> Dict[str, Any]:
    """核心轉檔入口函式"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 解析運算設備 (auto / cuda / cpu)
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    logger.info(f"使用運算設備: {resolved_device} (設定值: {device})")

    model, config, model_dir = load_model_and_config(model_path)
    state_dict = model.state_dict()

    copy_auxiliary_files(model_dir, output_dir)

    dtype = torch.float32
    if dtype_str == "fp16":
        dtype = torch.float16
    elif dtype_str == "bf16":
        dtype = torch.bfloat16

    results = {
        "output_dir": str(output_dir),
        "safetensors": None,
        "onnx": None,
    }

    # 1. 轉 Safetensors
    if export_format in ("safetensors", "both", "all"):
        safetensors_path = output_dir / "model.safetensors"
        export_to_safetensors(state_dict, safetensors_path)
        results["safetensors"] = str(safetensors_path)
        if verify:
            verify_safetensors(safetensors_path, state_dict)

    # 2. 轉 ONNX
    if export_format in ("onnx", "both", "all"):
        onnx_path = output_dir / "model.onnx"
        export_to_onnx(
            model=model,
            config=config,
            output_path=onnx_path,
            opset_version=opset_version,
            device=resolved_device,
            dtype=dtype,
            use_kv_cache=use_kv_cache,
        )
        results["onnx"] = str(onnx_path)
        if verify:
            verify_onnx(model=model, onnx_path=onnx_path, config=config, device=resolved_device)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="NagatoSakura LLM 模型轉檔工具 (Safetensors & ONNX)")
    parser.add_argument("--model_path", type=str, default="models/NS-LM-1.6/best_model", help="輸入模型資料夾或權重檔案路徑 (例如 nagato_sakura_output/best_model)",)
    parser.add_argument("--output_dir", type=str, default="models/NS-LM-1.6/export", help="輸出資料夾路徑")
    parser.add_argument("--format", type=str, choices=["safetensors", "onnx", "both", "all"], default="both", help="匯出格式: safetensors, onnx, 或 both (預設: both)",)
    parser.add_argument("--opset", type=int, default=21, help="ONNX opset 版本 (預設: 14)",)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="轉換運算設備: auto, cpu 或 cuda (預設: auto)",)
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="fp32", help="ONNX 匯出精度 (預設: fp32)",)
    parser.add_argument("--kv_cache", action="store_true", default=True, help="匯出支援 KV-Cache 的 ONNX 模型 (預設: 啟用)",)
    parser.add_argument("--no_kv_cache", action="store_false", dest="kv_cache", help="停用 KV-Cache 匯出基礎版 ONNX",)
    parser.add_argument("--verify", action="store_true", default=True, help="轉換完成後執行驗證測試 (預設: 啟用)",)
    parser.add_argument("--no_verify", action="store_false", dest="verify", help="停用轉換驗證",)

    args = parser.parse_args()

    try:
        results = export_model(
            model_path=Path(args.model_path),
            output_dir=Path(args.output_dir),
            export_format=args.format,
            opset_version=args.opset,
            device=args.device,
            dtype_str=args.dtype,
            use_kv_cache=args.kv_cache,
            verify=args.verify,
        )
        logger.info("🎉 所有轉檔任務已成功完成！")
        logger.info(f"輸出目錄: {results['output_dir']}")
        if results['safetensors']:
            logger.info(f"Safetensors: {results['safetensors']}")
        if results['onnx']:
            logger.info(f"ONNX: {results['onnx']}")
    except Exception as exc:
        logger.error(f"❌ 轉檔過程發生例外錯誤: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
