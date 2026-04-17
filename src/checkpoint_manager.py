import math
import shutil
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class CheckpointPolicy:
    """檢查點保存策略"""

    save_interval_epochs: int = 5
    keep_best_k: int = 3
    keep_latest_k: int = 2
    save_on_improve_delta: float = 0.001
    cleanup_old_checkpoints: bool = False


class CheckpointManager:
    """管理 checkpoint 保存決策與保留策略"""

    def __init__(self, output_dir: Path, policy: CheckpointPolicy, logger):
        self.output_dir = output_dir
        self.policy = policy
        self.logger = logger
        self.records: List[Dict[str, Any]] = []

    def should_save(
        self,
        epoch: int,
        num_epochs: int,
        eval_loss: Optional[float],
        previous_best_eval_loss: float,
        had_anomaly: bool = False,
    ) -> List[str]:
        reasons: List[str] = []

        if self.policy.save_interval_epochs > 0 and (epoch + 1) % self.policy.save_interval_epochs == 0:
            reasons.append("interval")

        if eval_loss is not None and math.isfinite(eval_loss):
            if not math.isfinite(previous_best_eval_loss):
                reasons.append("first_eval")
            elif (previous_best_eval_loss - eval_loss) >= self.policy.save_on_improve_delta:
                reasons.append("improved")

        if had_anomaly:
            reasons.append("anomaly")

        if (epoch + 1) >= num_epochs:
            reasons.append("final")

        return reasons

    def register_checkpoint(
        self,
        checkpoint_path: Path,
        checkpoint_name: str,
        epoch: int,
        eval_loss: Optional[float],
        is_best: bool,
        reasons: List[str],
    ):
        self.records.append(
            {
                "path": str(checkpoint_path),
                "name": checkpoint_name,
                "epoch": int(epoch),
                "eval_loss": float(eval_loss) if eval_loss is not None and math.isfinite(eval_loss) else None,
                "is_best": bool(is_best),
                "reasons": list(reasons),
            }
        )

        if self.policy.cleanup_old_checkpoints:
            self._cleanup_old_checkpoints()

    def get_best_checkpoint_path(self) -> Optional[Path]:
        candidates = [r for r in self.records if r["eval_loss"] is not None]
        if not candidates:
            return None
        best = min(candidates, key=lambda x: x["eval_loss"])
        return Path(best["path"])

    def _cleanup_old_checkpoints(self):
        checkpoint_dirs = [p for p in self.output_dir.glob("checkpoint-*") if p.is_dir()]
        if not checkpoint_dirs:
            return

        keep_paths = set()

        if self.records:
            keep_paths.add(Path(self.records[-1]["path"]).resolve())

        latest_sorted = sorted(self.records, key=lambda r: r["epoch"], reverse=True)
        for record in latest_sorted[: max(0, self.policy.keep_latest_k)]:
            keep_paths.add(Path(record["path"]).resolve())

        best_records = [r for r in self.records if r["eval_loss"] is not None]
        best_sorted = sorted(best_records, key=lambda r: r["eval_loss"])
        for record in best_sorted[: max(0, self.policy.keep_best_k)]:
            keep_paths.add(Path(record["path"]).resolve())

        for ckpt_dir in checkpoint_dirs:
            resolved = ckpt_dir.resolve()
            if resolved in keep_paths:
                continue
            try:
                shutil.rmtree(ckpt_dir)
                self.logger.info(f"已清理舊檢查點: {ckpt_dir}")
            except Exception as e:
                self.logger.warning(f"清理檢查點失敗 {ckpt_dir}: {e}")

    @staticmethod
    def _parse_epoch_from_name(name: str) -> Optional[int]:
        parts = name.split("-")
        if not parts:
            return None
        tail = parts[-1]
        return int(tail) if tail.isdigit() else None

    @staticmethod
    def _folder_size_mb(path: Path) -> float:
        total = 0
        try:
            for p in path.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
        except Exception:
            return 0.0
        return total / (1024 ** 2)

    def export_index(self, metrics_dir: Path) -> Dict[str, Any]:
        metrics_dir.mkdir(parents=True, exist_ok=True)

        all_dirs: List[Path] = [p for p in self.output_dir.glob("checkpoint-*") if p.is_dir()]
        for special in ["best_model", "final_model"]:
            special_dir = self.output_dir / special
            if special_dir.is_dir():
                all_dirs.append(special_dir)

        unique_dirs: List[Path] = []
        seen = set()
        for d in all_dirs:
            key = str(d.resolve())
            if key in seen:
                continue
            seen.add(key)
            unique_dirs.append(d)

        record_map = {str(Path(r["path"]).resolve()): r for r in self.records}
        rows: List[Dict[str, Any]] = []

        for d in sorted(unique_dirs, key=lambda p: p.stat().st_mtime, reverse=True):
            resolved = str(d.resolve())
            record = record_map.get(resolved)

            row = {
                "checkpoint_name": d.name,
                "checkpoint_path": str(d),
                "epoch": record.get("epoch") if record else self._parse_epoch_from_name(d.name),
                "eval_loss": record.get("eval_loss") if record else None,
                "is_best": bool(record.get("is_best")) if record else (d.name == "best_model"),
                "reasons": "|".join(record.get("reasons", [])) if record else "",
                "size_mb": round(self._folder_size_mb(d), 3),
                "modified_time": datetime.utcfromtimestamp(d.stat().st_mtime).isoformat(timespec="seconds") + "Z",
                "has_model": (d / "model.pt").exists(),
                "has_state": (d / "training_state.pt").exists(),
            }
            rows.append(row)

        csv_path = metrics_dir / "checkpoint_index.csv"
        headers = [
            "checkpoint_name",
            "checkpoint_path",
            "epoch",
            "eval_loss",
            "is_best",
            "reasons",
            "size_mb",
            "modified_time",
            "has_model",
            "has_state",
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        valid_eval_rows = [r for r in rows if r.get("eval_loss") is not None and math.isfinite(float(r["eval_loss"]))]
        best_row = min(valid_eval_rows, key=lambda x: float(x["eval_loss"])) if valid_eval_rows else None
        latest_row = rows[0] if rows else None

        overview = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "total_checkpoints": len(rows),
            "total_size_mb": round(sum(float(r.get("size_mb", 0.0)) for r in rows), 3),
            "best_checkpoint_path": best_row.get("checkpoint_path") if best_row else "",
            "best_eval_loss": best_row.get("eval_loss") if best_row else None,
            "latest_checkpoint_path": latest_row.get("checkpoint_path") if latest_row else "",
        }

        json_payload = {
            "overview": overview,
            "entries": rows,
        }

        with open(metrics_dir / "checkpoint_index.json", "w", encoding="utf-8") as f:
            json.dump(json_payload, f, ensure_ascii=False, indent=2)

        return overview
