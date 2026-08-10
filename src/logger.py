import csv
import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import psutil
import torch
from tqdm import tqdm

try:
    import GPUtil
except ImportError:
    GPUtil = None


import copy


class ColoredFormatter(logging.Formatter):
    """彩色日誌格式器（使用 copy.copy 避免就地竄改共享 LogRecord）"""

    COLORS = {
        "DEBUG": "\033[36m",  # 青色
        "INFO": "\033[32m",  # 綠色
        "WARNING": "\033[33m",  # 黃色
        "ERROR": "\033[31m",  # 紅色
        "CRITICAL": "\033[35m",  # 紫色
    }
    RESET = "\033[0m"

    def format(self, record):
        record = copy.copy(record)
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


class TqdmConsoleHandler(logging.StreamHandler):
    """使用 tqdm.write 輸出，避免日誌破壞進度條。"""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(output_dir: str, log_level: str = "INFO"):
    """設置增強的日誌系統"""
    
    os.environ["TQDM_NCOLS"] = "115"
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 創建根日誌記錄器
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # 清除現有處理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 文件處理器
    file_handler = logging.FileHandler(log_dir / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8")
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 控制台處理器
    console_handler = TqdmConsoleHandler(sys.stdout)
    console_formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    for quiet_logger_name in ["datasets", "urllib3", "filelock", "transformers"]:
        logging.getLogger(quiet_logger_name).setLevel(logging.WARNING)

    return logging.getLogger(__name__)


class CSVMetricsWriter:
    """結構化訓練指標寫入器（CSV）"""

    SCHEMAS = {
        "step_metrics.csv": [
            "timestamp",
            "epoch",
            "global_step",
            "batch_idx",
            "train_loss",
            "loss_ema",
            "learning_rate",
            "grad_norm",
            "batch_tokens",
            "tokens_per_sec",
            "invalid_batches",
            "gpu_memory_mb",
            "gpu_memory_percent",
            "cpu_percent",
            "ram_percent",
        ],
        "eval_metrics.csv": [
            "timestamp",
            "epoch",
            "global_step",
            "eval_loss",
            "perplexity",
            "total_tokens",
            "eval_time_sec",
            "improved",
            "is_best",
            "short_samples",
            "short_tokens",
            "short_loss",
            "short_perplexity",
            "medium_samples",
            "medium_tokens",
            "medium_loss",
            "medium_perplexity",
            "long_samples",
            "long_tokens",
            "long_loss",
            "long_perplexity",
        ],
        "events.csv": [
            "timestamp",
            "epoch",
            "global_step",
            "event_type",
            "severity",
            "message",
            "value",
        ],
        "checkpoint_metrics.csv": [
            "timestamp",
            "epoch",
            "global_step",
            "checkpoint_name",
            "checkpoint_path",
            "eval_loss",
            "is_best",
            "reasons",
        ],
        "training_summary.csv": [
            "timestamp",
            "run_status",
            "epochs_planned",
            "epochs_completed",
            "global_step",
            "best_eval_loss",
            "best_eval_epoch",
            "final_eval_loss",
            "final_perplexity",
            "total_tokens_seen",
            "invalid_loss_count",
            "invalid_grad_count",
            "skipped_update_count",
            "run_time_sec",
            "best_checkpoint_path",
            "latest_checkpoint_path",
            "best_short_loss",
            "best_medium_loss",
            "best_long_loss",
            "best_short_perplexity",
            "best_medium_perplexity",
            "best_long_perplexity",
        ],
    }

    def __init__(self, output_dir: str):
        self.metrics_dir = Path(output_dir) / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_headers()

    def _ensure_headers(self):
        for filename, headers in self.SCHEMAS.items():
            file_path = self.metrics_dir / filename
            if not file_path.exists() or file_path.stat().st_size == 0:
                with open(file_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()

    def _write_row(self, filename: str, row: Dict[str, Any]):
        headers = self.SCHEMAS[filename]
        payload = {k: row.get(k, "") for k in headers}
        if not payload.get("timestamp"):
            payload["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

        with self._lock:
            with open(self.metrics_dir / filename, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writerow(payload)

    def log_step_metrics(self, row: Dict[str, Any]):
        self._write_row("step_metrics.csv", row)

    def log_eval_metrics(self, row: Dict[str, Any]):
        self._write_row("eval_metrics.csv", row)

    def log_event(self, row: Dict[str, Any]):
        self._write_row("events.csv", row)

    def log_checkpoint_metrics(self, row: Dict[str, Any]):
        self._write_row("checkpoint_metrics.csv", row)

    def log_training_summary(self, row: Dict[str, Any]):
        self._write_row("training_summary.csv", row)


class SystemMonitor:
    """系統資源監控器（支援指定 PyTorch Device 與 Multi-GPU 視角）"""

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.gpu_available = torch.cuda.is_available() and self.device.type == "cuda"
        self.memory_history = deque(maxlen=100)
        self.gpu_memory_history = deque(maxlen=100)

    def get_system_info(self) -> Dict[str, Any]:
        """獲取系統資訊"""
        
        info = {
            # 非阻塞取樣，避免訓練步驟被監控卡住
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        }

        if self.gpu_available:
            try:
                device_idx = self.device.index if self.device.index is not None else 0
                allocated_bytes = torch.cuda.memory_allocated(device_idx)
                reserved_bytes = torch.cuda.memory_reserved(device_idx)
                total_bytes = torch.cuda.get_device_properties(device_idx).total_memory
                allocated_gb = allocated_bytes / (1024**3)
                reserved_gb = reserved_bytes / (1024**3)
                total_gb = total_bytes / (1024**3)
                gpu_percent = (reserved_bytes / total_bytes) * 100 if total_bytes > 0 else 0.0
                info.update(
                    {
                        "gpu_memory_percent": gpu_percent,
                        "gpu_memory_used_gb": reserved_gb,
                        "gpu_memory_allocated_gb": allocated_gb,
                        "gpu_memory_total_gb": total_gb,
                        "gpu_device_index": device_idx,
                        "gpu_load": 0.0,
                        "gpu_temperature": 0.0,
                    }
                )
                if GPUtil:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus and device_idx < len(gpus):
                            gpu = gpus[device_idx]
                            info.update(
                                {"gpu_temperature": gpu.temperature, "gpu_load": gpu.load * 100}
                            )
                    except Exception:
                        pass
            except Exception:
                pass

        return info

    def log_system_status(self, logger):
        """記錄系統狀態"""
        
        info = self.get_system_info()
        logger.info(
            f"系統狀態 - CPU: {info['cpu_percent']:.1f}%, "
            f"RAM: {info['memory_percent']:.1f}% "
            f"({info['memory_used_gb']:.1f}/{info['memory_total_gb']:.1f}GB)"
        )
        if "gpu_memory_percent" in info:
            logger.info(
                f"GPU[{info.get('gpu_device_index', 0)}]狀態 - 記憶體: {info['gpu_memory_percent']:.1f}% "
                f"({info['gpu_memory_used_gb']:.1f}/{info['gpu_memory_total_gb']:.1f}GB), "
                f"負載: {info.get('gpu_load', 0.0):.1f}%, 溫度: {info.get('gpu_temperature', 0.0)}°C"
            )
