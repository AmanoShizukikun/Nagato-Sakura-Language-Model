import argparse
import csv
import importlib
from pathlib import Path
from typing import Dict, List


def _ensure_matplotlib():
    try:
        return importlib.import_module("matplotlib.pyplot")
    except Exception as e:
        raise RuntimeError(
            "找不到 matplotlib。請先安裝：pip install matplotlib"
        ) from e


def _read_csv(file_path: Path) -> List[Dict[str, str]]:
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _plot_step_metrics(step_rows: List[Dict[str, str]], output_dir: Path):
    if not step_rows:
        return

    plt = _ensure_matplotlib()

    steps = [_to_float(r.get("global_step")) for r in step_rows]
    losses = [_to_float(r.get("train_loss")) for r in step_rows]
    loss_ema = [_to_float(r.get("loss_ema")) for r in step_rows]
    grad_norm = [_to_float(r.get("grad_norm")) for r in step_rows]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150)

    axes[0].plot(steps, losses, label="train_loss", linewidth=1.2)
    axes[0].plot(steps, loss_ema, label="loss_ema", linewidth=1.2)
    axes[0].set_title("Train Loss Curve")
    axes[0].set_xlabel("Global Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, grad_norm, label="grad_norm", linewidth=1.2, color="#d94f30")
    axes[1].set_title("Gradient Norm Curve")
    axes[1].set_xlabel("Global Step")
    axes[1].set_ylabel("Grad Norm")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "train_loss_and_grad.png")
    plt.close(fig)


def _plot_eval_metrics(eval_rows: List[Dict[str, str]], output_dir: Path):
    if not eval_rows:
        return

    plt = _ensure_matplotlib()

    epochs = [_to_float(r.get("epoch")) for r in eval_rows]
    eval_loss = [_to_float(r.get("eval_loss")) for r in eval_rows]
    ppl = [_to_float(r.get("perplexity")) for r in eval_rows]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150)

    axes[0].plot(epochs, eval_loss, marker="o", linewidth=1.2, color="#1f77b4")
    axes[0].set_title("Eval Loss by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Eval Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, ppl, marker="o", linewidth=1.2, color="#2ca02c")
    axes[1].set_title("Perplexity by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "eval_loss_and_perplexity.png")
    plt.close(fig)


def _plot_system_metrics(step_rows: List[Dict[str, str]], output_dir: Path):
    if not step_rows:
        return

    plt = _ensure_matplotlib()

    steps = [_to_float(r.get("global_step")) for r in step_rows]
    gpu_mem = [_to_float(r.get("gpu_memory_percent")) for r in step_rows]
    cpu_pct = [_to_float(r.get("cpu_percent")) for r in step_rows]
    ram_pct = [_to_float(r.get("ram_percent")) for r in step_rows]

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.plot(steps, gpu_mem, label="GPU Mem %", linewidth=1.2)
    ax.plot(steps, cpu_pct, label="CPU %", linewidth=1.2)
    ax.plot(steps, ram_pct, label="RAM %", linewidth=1.2)
    ax.set_title("System Utilization")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Percent")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "system_utilization.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from CSV metrics")
    parser.add_argument("--metrics_dir", type=str, default="NS-LLM-0.8/metrics", help="metrics 目錄")
    parser.add_argument("--output_dir", type=str, default=None, help="圖表輸出目錄，預設為 metrics/plots")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir) if args.output_dir else metrics_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    step_rows = _read_csv(metrics_dir / "step_metrics.csv")
    eval_rows = _read_csv(metrics_dir / "eval_metrics.csv")

    _plot_step_metrics(step_rows, output_dir)
    _plot_eval_metrics(eval_rows, output_dir)
    _plot_system_metrics(step_rows, output_dir)

    print(f"圖表已輸出到: {output_dir}")


if __name__ == "__main__":
    main()
