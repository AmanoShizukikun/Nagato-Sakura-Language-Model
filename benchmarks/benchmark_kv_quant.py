import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.nagato_sakura_model import NSConfig, NagatoSakuraForCausalLM  # noqa: E402
from src.tokenizer import TokenizerManager  # noqa: E402
from src.kv_cache import kv_cache_summary  # noqa: E402


def _resolve_weight_file(model_path: Path) -> Optional[Path]:
    for name in ("model.pt", "pytorch_model.bin", "model.safetensors"):
        candidate = model_path / name
        if candidate.exists():
            return candidate
    return None


def _auto_tokenizer_path(model_path: Path) -> Path:
    tokenizer_path = model_path / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError("找不到 tokenizer.json，請使用 --tokenizer_path 指定")
    return tokenizer_path


def load_model_and_tokenizer(
    model_path: Path,
    tokenizer_path: Path,
    device: torch.device,
    quantize_kv_cache: bool,
    kv_cache_bits: int,
    kv_quant_group_size: int,
    kv_residual_sign_correction: bool,
    num_key_value_heads: Optional[int],
) -> tuple[NagatoSakuraForCausalLM, Any]:
    with open(model_path / "config.json", "r", encoding="utf-8") as f:
        config_data = json.load(f)

    config = NSConfig.from_dict(config_data)
    if num_key_value_heads is not None:
        config.num_key_value_heads = num_key_value_heads

    config.quantize_kv_cache = quantize_kv_cache
    config.kv_cache_bits = kv_cache_bits
    config.kv_quant_group_size = kv_quant_group_size
    config.kv_residual_sign_correction = kv_residual_sign_correction

    tokenizer_manager = TokenizerManager(tokenizer_path)
    tokenizer_manager.load_tokenizer()
    tokenizer = tokenizer_manager.transformers_tokenizer

    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id

    model = NagatoSakuraForCausalLM(config)

    weight_file = _resolve_weight_file(model_path)
    if weight_file is None:
        raise FileNotFoundError("找不到模型權重，預期 model.pt / pytorch_model.bin / model.safetensors")

    state_dict = torch.load(weight_file, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, tokenizer


def benchmark_generation(
    model: NagatoSakuraForCausalLM,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int,
) -> Dict[str, float]:
    latencies: List[float] = []
    throughputs: List[float] = []
    generated_counts: List[int] = []
    peak_mems: List[float] = []

    bos_token = tokenizer.bos_token or "<s>"

    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(
                f"{bos_token}{prompt}",
                add_special_tokens=False,
                return_tensors="pt",
            ).to(device)
            attention_mask = torch.ones_like(input_ids)

            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

            start = time.perf_counter()
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + max_new_tokens,
                do_sample=False,
                top_k=1,
                top_p=1.0,
                temperature=1.0,
                use_cache=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = max(time.perf_counter() - start, 1e-6)

            generated = int(outputs.shape[1] - input_ids.shape[1])
            tps = generated / elapsed

            latencies.append(elapsed)
            throughputs.append(tps)
            generated_counts.append(generated)

            if device.type == "cuda":
                peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
            else:
                peak_mem = 0.0
            peak_mems.append(peak_mem)

    return {
        "avg_latency_sec": float(sum(latencies) / max(len(latencies), 1)),
        "avg_tokens_per_sec": float(sum(throughputs) / max(len(throughputs), 1)),
        "avg_generated_tokens": float(sum(generated_counts) / max(len(generated_counts), 1)),
        "avg_peak_memory_mb": float(sum(peak_mems) / max(len(peak_mems), 1)),
    }


def evaluate_perplexity(
    model: NagatoSakuraForCausalLM,
    tokenizer,
    eval_data: List[Dict[str, str]],
    device: torch.device,
    max_samples: int,
    max_seq_len: int,
) -> Dict[str, float]:
    losses: List[float] = []
    bos = tokenizer.bos_token or "<s>"
    eos = tokenizer.eos_token or "</s>"

    with torch.no_grad():
        for sample in eval_data[:max_samples]:
            prompt = str(sample.get("prompt", "")).strip()
            completion = str(sample.get("completion", "")).strip()
            if not prompt or not completion:
                continue

            text = f"{bos}{prompt}{completion}{eos}"
            input_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            ).to(device)
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            if outputs.loss is not None and torch.isfinite(outputs.loss):
                losses.append(float(outputs.loss.item()))

    if not losses:
        return {"eval_loss": float("inf"), "perplexity": float("inf")}

    avg_loss = sum(losses) / len(losses)
    ppl = float(torch.exp(torch.tensor(avg_loss)).item())
    return {"eval_loss": float(avg_loss), "perplexity": ppl}


def collect_cache_snapshot(
    model: NagatoSakuraForCausalLM,
    tokenizer,
    device: torch.device,
    prompt: str,
) -> Dict[str, Any]:
    bos_token = tokenizer.bos_token or "<s>"
    input_ids = tokenizer.encode(
        f"{bos_token}{prompt}",
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    return kv_cache_summary(outputs.past_key_values)


def load_prompts(prompts_file: Optional[Path], limit: int) -> List[str]:
    default_prompts = [
        "請用三句話介紹長門櫻的性格設定。",
        "幫我寫一段 8 行內的 Python 函數，計算字串中每個字元的頻率。",
        "請把以下句子改寫成更自然的繁體中文：I am building a compact language model.",
        "你認為在 24GB GPU 上訓練長上下文模型最關鍵的技巧是什麼？",
        "請用條列整理量化帶來的優點與風險。",
    ]

    if prompts_file is None:
        return default_prompts[:limit]

    with open(prompts_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("prompts_file 必須是 JSON list")

    prompts = [str(x) for x in data if str(x).strip()]
    if not prompts:
        raise ValueError("prompts_file 內沒有有效 prompt")
    return prompts[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="0.5.0 KV 量化 benchmark")
    parser.add_argument("--model_path", type=str, required=True, help="模型資料夾路徑")
    parser.add_argument("--tokenizer_path", type=str, help="分詞器路徑，預設 model_path/tokenizer.json")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="每個 prompt 生成 token 上限")
    parser.add_argument("--num_prompts", type=int, default=5, help="benchmark prompt 數量")
    parser.add_argument("--prompts_file", type=str, help="自定義 prompt 清單(JSON list)")
    parser.add_argument("--num_key_value_heads", type=int, help="覆寫 GQA key/value 頭數")

    parser.add_argument("--baseline_kv_bits", type=int, default=32, choices=[3, 4, 8, 16, 32])
    parser.add_argument("--quant_kv_bits", type=int, default=4, choices=[3, 4, 8, 16, 32])
    parser.add_argument("--kv_quant_group_size", type=int, default=64)
    parser.add_argument("--kv_residual_sign_correction", action="store_true")

    parser.add_argument("--eval_data_file", type=str, help="可選：評估資料（prompt/completion list）")
    parser.add_argument("--eval_max_samples", type=int, default=64)
    parser.add_argument("--eval_max_seq_len", type=int, default=1024)

    parser.add_argument("--output_json", type=str, default="benchmarks/benchmark_kv_quant_results.json")

    args = parser.parse_args()

    model_path = Path(args.model_path)
    tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path else _auto_tokenizer_path(model_path)
    prompts = load_prompts(Path(args.prompts_file) if args.prompts_file else None, args.num_prompts)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    baseline_model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device=device,
        quantize_kv_cache=False,
        kv_cache_bits=args.baseline_kv_bits,
        kv_quant_group_size=args.kv_quant_group_size,
        kv_residual_sign_correction=False,
        num_key_value_heads=args.num_key_value_heads,
    )

    quant_model, _ = load_model_and_tokenizer(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device=device,
        quantize_kv_cache=True,
        kv_cache_bits=args.quant_kv_bits,
        kv_quant_group_size=args.kv_quant_group_size,
        kv_residual_sign_correction=args.kv_residual_sign_correction,
        num_key_value_heads=args.num_key_value_heads,
    )

    baseline_metrics = benchmark_generation(
        model=baseline_model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )
    quant_metrics = benchmark_generation(
        model=quant_model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    baseline_cache = collect_cache_snapshot(baseline_model, tokenizer, device, prompts[0])
    quant_cache = collect_cache_snapshot(quant_model, tokenizer, device, prompts[0])

    result: Dict[str, Any] = {
        "device": str(device),
        "num_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "config": {
            "baseline_kv_bits": args.baseline_kv_bits,
            "quant_kv_bits": args.quant_kv_bits,
            "kv_quant_group_size": args.kv_quant_group_size,
            "kv_residual_sign_correction": args.kv_residual_sign_correction,
            "num_key_value_heads": args.num_key_value_heads,
        },
        "baseline": baseline_metrics,
        "quantized": quant_metrics,
        "cache_baseline": baseline_cache,
        "cache_quantized": quant_cache,
    }

    if args.eval_data_file:
        with open(args.eval_data_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        if isinstance(eval_data, list):
            result["eval_baseline"] = evaluate_perplexity(
                model=baseline_model,
                tokenizer=tokenizer,
                eval_data=eval_data,
                device=device,
                max_samples=args.eval_max_samples,
                max_seq_len=args.eval_max_seq_len,
            )
            result["eval_quantized"] = evaluate_perplexity(
                model=quant_model,
                tokenizer=tokenizer,
                eval_data=eval_data,
                device=device,
                max_samples=args.eval_max_samples,
                max_seq_len=args.eval_max_seq_len,
            )

    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator

    result["comparison"] = {
        "speedup_x": _safe_ratio(
            result["quantized"]["avg_tokens_per_sec"],
            result["baseline"]["avg_tokens_per_sec"],
        ),
        "latency_ratio": _safe_ratio(
            result["quantized"]["avg_latency_sec"],
            result["baseline"]["avg_latency_sec"],
        ),
        "peak_memory_ratio": _safe_ratio(
            result["quantized"]["avg_peak_memory_mb"],
            result["baseline"]["avg_peak_memory_mb"],
        ),
        "cache_bytes_ratio": _safe_ratio(
            result["cache_quantized"]["total_bytes"],
            result["cache_baseline"]["total_bytes"],
        ),
    }

    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("===== KV Quant Benchmark =====")
    print(f"Baseline tokens/s: {result['baseline']['avg_tokens_per_sec']:.2f}")
    print(f"Quantized tokens/s: {result['quantized']['avg_tokens_per_sec']:.2f}")
    print(f"Speedup (x): {result['comparison']['speedup_x']:.3f}")
    print(f"Peak memory ratio: {result['comparison']['peak_memory_ratio']:.3f}")
    print(f"Cache bytes ratio: {result['comparison']['cache_bytes_ratio']:.3f}")
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
