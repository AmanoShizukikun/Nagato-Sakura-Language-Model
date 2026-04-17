import os
import sys
import time
import logging
from pathlib import Path
from collections import deque
from typing import Dict, Any

import torch
import psutil
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
    
    # 控制台處理器（彩色）
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)


class SystemMonitor:
    """系統資源監控器"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.memory_history = deque(maxlen=100)
        self.gpu_memory_history = deque(maxlen=100)
    
    def get_system_info(self) -> Dict[str, Any]:
        """獲取系統信息"""
        info = {
            'cpu_percent': psutil.cpu_percent(interval=1),
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