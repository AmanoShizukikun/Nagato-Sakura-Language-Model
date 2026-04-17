import os
import sys
import time
import logging
import csv
import threading
from pathlib import Path
from collections import deque
from typing import Dict, Any
from datetime import datetime

import torch
import psutil
from tqdm import tqdm
try:
    import GPUtil
except ImportError:
    GPUtil = None


class ColoredFormatter(logging.Formatter):
    """彩色日誌格式器"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 綠色
        'WARNING': '\033[33m',  # 黃色
        'ERROR': '\033[31m',    # 紅色
        'CRITICAL': '\033[35m', # 紫色
    }
    RESET = '\033[0m'
    
    def format(self, record):
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
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 創建根日誌記錄器
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除現有處理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 文件處理器
    file_handler = logging.FileHandler(
        log_dir / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log",
        encoding='utf-8'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 控制台處理器（彩色 + tqdm相容）
    console_handler = TqdmConsoleHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
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
            payload["timestamp"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

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
    """系統資源監控器"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.memory_history = deque(maxlen=100)
        self.gpu_memory_history = deque(maxlen=100)
    
    def get_system_info(self) -> Dict[str, Any]:
        """獲取系統信息"""
        info = {
            # 非阻塞取樣，避免訓練步驟被監控卡住
            'cpu_percent': psutil.cpu_percent(interval=None),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        }
        
        if self.gpu_available and GPUtil:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    info.update({
                        'gpu_memory_percent': gpu.memoryUtil * 100,
                        'gpu_memory_used_gb': gpu.memoryUsed / 1024,
                        'gpu_memory_total_gb': gpu.memoryTotal / 1024,
                        'gpu_temperature': gpu.temperature,
                        'gpu_load': gpu.load * 100
                    })
            except Exception:
                pass
        
        return info
    
    def log_system_status(self, logger):
        """記錄系統狀態"""
        info = self.get_system_info()
        logger.info(f"系統狀態 - CPU: {info['cpu_percent']:.1f}%, "
                   f"RAM: {info['memory_percent']:.1f}% "
                   f"({info['memory_used_gb']:.1f}/{info['memory_total_gb']:.1f}GB)")
        
        if 'gpu_memory_percent' in info:
            logger.info(f"GPU狀態 - 記憶體: {info['gpu_memory_percent']:.1f}% "
                       f"({info['gpu_memory_used_gb']:.1f}/{info['gpu_memory_total_gb']:.1f}GB), "
                       f"負載: {info['gpu_load']:.1f}%, 溫度: {info['gpu_temperature']}°C")