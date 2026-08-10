import argparse
import json
import logging
import os
import random
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
import torch
from transformers import PreTrainedTokenizerFast

# 忽略非必要警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 自定義模型與 Tokenizer 導入
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    from src.nagato_sakura_model import ConversationHistory, NagatoSakuraForCausalLM, NSConfig
    from src.tokenizer import TokenizerManager
except ImportError as e:
    print(f"錯誤：無法導入自定義模型。錯誤詳情: {e}")
    sys.exit(1)


# ==============================================================================
# 工具函數與組態類別
# ==============================================================================


def _fix_utf8_artifacts(text: str) -> str:
    """
    修復因 Tokenizer 特殊單字節 (如 Token 124) 導致解碼出來的 \ufffd () 為中間點 ·

    Args:
        text (str): 待處理的文字字串。

    Returns:
        str: 替換替換字元後的文字。
    """
    
    if not text:
        return text
    
    return text.replace("\ufffd", "·")


@dataclass
class InferenceConfig:
    """推理配置資料類別"""

    model_path: str
    tokenizer_path: str
    device: str = "auto"
    max_length: Optional[int] = None
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    do_sample: bool = True
    base_seed: int = -1
    silent_mode: bool = True
    quantize_kv_cache: Optional[bool] = None
    kv_cache_bits: Optional[int] = None
    kv_quant_group_size: Optional[int] = None
    kv_residual_sign_correction: Optional[bool] = None
    kv_decode_mode: Optional[str] = None
    num_key_value_heads: Optional[int] = None
    stateless_chat: bool = False
    weight_quantization: bool = False
    weight_quant_bits: int = 8
    weight_quant_group_size: int = 128
    weight_quant_embeddings: bool = False
    weight_quant_lm_head: bool = False
    weight_quant_mode: str = "auto"
    use_compile: bool = False
    compile_mode: str = "default"
    dtype: str = "auto"


class ColoredFormatter(logging.Formatter):
    """帶有 Terminal ANSI 色彩高亮日誌格式器"""

    COLORS = {
        "DEBUG": "\033[36m",  # 青色
        "INFO": "\033[32m",  # 綠色
        "WARNING": "\033[33m",  # 黃色
        "ERROR": "\033[31m",  # 紅色
        "CRITICAL": "\033[35m",  # 紫色
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """設置增強的日誌系統"""
    
    os.environ["TQDM_NCOLS"] = "115"
    for quiet_logger in ["werkzeug", "urllib3", "filelock", "transformers"]:
        logging.getLogger(quiet_logger).setLevel(logging.WARNING)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logging.getLogger(__name__)


def _log_system_environment(logger: logging.Logger) -> None:
    """記錄系統環境資訊"""
    
    logger.info("════════════════ [階段 1/2] 🌸 長門櫻推理引擎初始化 ════════════════")
    logger.info(f"環境基礎 - Python: {sys.version.split()[0]} | PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"GPU[0] 設備 - {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB)")
        logger.info(f"CUDA 版本 - {torch.version.cuda}")
    else:
        logger.info("CUDA 加速 - 不可用 (使用 CPU 推理)")


class RandomSeedManager:
    """隨機種子管理器 - 確保線程安全且每次生成使用不同的隨機種子"""

    def __init__(self, base_seed: int = -1):
        """
        初始化種子管理器。

        Args:
            base_seed (int): 基礎種子值，-1 表示使用隨機種子。
        """
        
        if base_seed == -1:
            self.base_seed = int(time.time() * 1000) % (2**32)
        else:
            self.base_seed = base_seed
        self.counter = 0
        self.lock = threading.Lock()

    def get_new_seed(self) -> int:
        """獲取新的隨機種子"""
        
        with self.lock:
            new_seed = (self.base_seed + self.counter + random.randint(1, 10000)) % (2**32)
            self.counter += 1
            return new_seed


# ==============================================================================
# 核心推理引擎
# ==============================================================================


class NagatoSakuraInference:
    """長門櫻模型流式推理器核心類別"""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.logger = setup_logging()
        self.seed_manager = RandomSeedManager(config.base_seed)
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
            
        self.logger.info(f"使用設備: {self.device}")
        self.logger.info(f"基礎種子: {self.seed_manager.base_seed}")
        self.model: Optional[NagatoSakuraForCausalLM] = None
        self.tokenizer_manager: Optional[TokenizerManager] = None
        self.tokenizer: Optional[PreTrainedTokenizerFast] = None
        self._load_model_and_tokenizer()

    def _log_tokenizer_health(self) -> None:
        """檢測 Tokenizer UTF-8 Byte Fallback 與未知字元處理能力健康度"""
        
        if self.tokenizer is None:
            return
        model_unk_token = None
        model_byte_fallback = None
        
        if self.tokenizer_manager is not None and self.tokenizer_manager.tokenizer_object is not None:
            try:
                payload = json.loads(self.tokenizer_manager.tokenizer_object.to_str())
                model = payload.get("model", {}) if isinstance(payload, dict) else {}
                if isinstance(model, dict):
                    model_unk_token = model.get("unk_token")
                    model_byte_fallback = model.get("byte_fallback")
            except Exception:
                pass
            
        issues = []
        if model_unk_token != "<unk>":
            issues.append(f"model.unk_token={model_unk_token!r}")
            
        if model_byte_fallback is not True:
            issues.append(f"model.byte_fallback={model_byte_fallback!r}")
            
        try:
            probe_text = "混合 Mixed 中文English日本語😀"
            probe_ids = self.tokenizer.encode(probe_text, add_special_tokens=False)
            probe_decoded = self.tokenizer.decode(probe_ids, skip_special_tokens=True)
            if "\ufffd" in probe_decoded:
                issues.append("UTF-8 probe decode produced replacement char")
        except Exception as e:
            issues.append(f"UTF-8 probe failed: {e}")
            
        if issues:
            self.logger.warning(
                "檢測到 legacy tokenizer 風險，可能在新字元/emoji 場景出現亂碼。"
                "建議以 --force_retrain_tokenizer --no_resume 開新 run 重建 tokenizer。"
                f" 詳細: {'; '.join(issues)}"
            )
        else:
            self.logger.info("Tokenizer UTF-8 健檢通過。")

    def _load_model_and_tokenizer(self) -> None:
        """載入 Tokenizer、模型組態檔與權重"""

        try:
            self.tokenizer_manager = TokenizerManager(Path(self.config.tokenizer_path))
            self.tokenizer_manager.load_tokenizer()
            self.tokenizer = self.tokenizer_manager.transformers_tokenizer
            self.logger.info(f"分詞器就緒: {self.config.tokenizer_path} (詞彙量 {len(self.tokenizer)})")
            self._log_tokenizer_health()
            model_dir = Path(self.config.model_path)
            if not (model_dir / "config.json").exists() and (model_dir / "best_model" / "config.json").exists():
                model_dir = model_dir / "best_model"

            config_path = model_dir / "config.json"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)
                model_config = NSConfig.from_dict(config_dict)
            else:
                self.logger.warning("未找到配置文件，使用默認配置")
                model_config = NSConfig(vocab_size=len(self.tokenizer))
                
            model_config.vocab_size = len(self.tokenizer)
            model_config.pad_token_id = self.tokenizer.pad_token_id
            model_config.bos_token_id = self.tokenizer.bos_token_id
            model_config.eos_token_id = self.tokenizer.eos_token_id
            model_config.unk_token_id = self.tokenizer.unk_token_id
            if getattr(self.config, "num_key_value_heads", None) is None and hasattr(model_config, "num_key_value_heads"):
                self.config.num_key_value_heads = model_config.num_key_value_heads
                
            if getattr(self.config, "quantize_kv_cache", None) is None and hasattr(model_config, "quantize_kv_cache"):
                self.config.quantize_kv_cache = model_config.quantize_kv_cache
                
            if getattr(self.config, "kv_cache_bits", None) is None and hasattr(model_config, "kv_cache_bits"):
                self.config.kv_cache_bits = model_config.kv_cache_bits
                
            if getattr(self.config, "kv_quant_group_size", None) is None and hasattr(model_config, "kv_quant_group_size"):
                self.config.kv_quant_group_size = model_config.kv_quant_group_size
                
            if getattr(self.config, "kv_residual_sign_correction", None) is None and hasattr(model_config, "kv_residual_sign_correction"):
                self.config.kv_residual_sign_correction = model_config.kv_residual_sign_correction

            if getattr(self.config, "kv_decode_mode", None) is None and hasattr(model_config, "kv_decode_mode"):
                self.config.kv_decode_mode = model_config.kv_decode_mode

            if self.config.num_key_value_heads is not None:
                model_config.num_key_value_heads = self.config.num_key_value_heads
                
            if self.config.quantize_kv_cache is not None:
                model_config.quantize_kv_cache = self.config.quantize_kv_cache
                
            if self.config.kv_cache_bits is not None:
                model_config.kv_cache_bits = self.config.kv_cache_bits
                
            if self.config.kv_quant_group_size is not None:
                model_config.kv_quant_group_size = self.config.kv_quant_group_size
                
            if self.config.kv_residual_sign_correction is not None:
                model_config.kv_residual_sign_correction = self.config.kv_residual_sign_correction

            if getattr(self.config, "kv_decode_mode", None) is not None:
                model_config.kv_decode_mode = self.config.kv_decode_mode

            model_max_length = int(getattr(model_config, "max_position_embeddings", 512))
            if self.config.max_length is None or int(self.config.max_length) <= 0:
                self.config.max_length = model_max_length
            else:
                self.config.max_length = min(int(self.config.max_length), model_max_length)

            if self.config.max_new_tokens is None or int(self.config.max_new_tokens) <= 0:
                self.config.max_new_tokens = max(16, self.config.max_length // 2)

            self.logger.info(f"推理上下文上限: {self.config.max_length} (模型 config.json 上限: {model_max_length})")
            self.logger.info(f"單輪最大生成 Token 上限: {self.config.max_new_tokens} (預設為上下文長度的一半)")
            self.logger.info(f"加載模型: {self.config.model_path}")
            self.model = NagatoSakuraForCausalLM(model_config)
            weight_files = ["model.pt", "pytorch_model.bin", "model.safetensors"]
            model_loaded = False
            for weight_file in weight_files:
                model_file_path = model_dir / weight_file
                if model_file_path.exists():
                    try:
                        state_dict = torch.load(model_file_path, map_location=self.device, weights_only=True)
                        self.model.load_state_dict(state_dict)
                        self.logger.info(f"模型權重從 {model_file_path} 加載完成")
                        model_loaded = True
                        break
                    except Exception as e:
                        self.logger.warning(f"從 {model_file_path} 加載權重失敗: {e}")
                        continue

            target_dtype = torch.float32
            if self.config.dtype == "auto":
                if self.device.type == "cuda":
                    if torch.cuda.is_bf16_supported():
                        target_dtype = torch.bfloat16
                    else:
                        target_dtype = torch.float16
                else:
                    target_dtype = torch.float32
            elif self.config.dtype == "bf16":
                target_dtype = torch.bfloat16
            elif self.config.dtype == "fp16":
                target_dtype = torch.float16
            elif self.config.dtype == "fp32":
                target_dtype = torch.float32

            self.model.to(device=self.device, dtype=target_dtype)
            self.model.eval()
            self.logger.info(f"模型權重與 KV Cache 計算精度已轉為: {target_dtype}")
            if self.config.weight_quantization:
                self.logger.info(
                    f"[量化] 啟動權重量化 — INT{self.config.weight_quant_bits}"
                    + (
                        f" (group_size={self.config.weight_quant_group_size})"
                        if self.config.weight_quant_bits == 4
                        else ""
                    )
                )
                try:
                    q_info = self.model.quantize_weights(
                        bits=self.config.weight_quant_bits,
                        group_size=self.config.weight_quant_group_size,
                        quantize_embeddings=self.config.weight_quant_embeddings,
                        quantize_lm_head=self.config.weight_quant_lm_head,
                        mode=self.config.weight_quant_mode,
                    )
                    self.logger.info(
                        f"[量化完成] 量化層數: {q_info.get('layers', 0)}, "
                        f"節省記憶體: {q_info.get('savings_mb', 0.0):.1f} MB, "
                        f"壓縮比: {q_info.get('compression_ratio', 1.0):.2f}x"
                    )
                except Exception as e:
                    self.logger.error(f"[量化失敗] {e}，使用原始模型繼續推理")
                    
            param_stats = self.model.get_parameter_stats()
            self.logger.info(f"模型參數量: {param_stats['total_params'] / 1e6:.2f}M")
            self.logger.info(f"參數組成: Embedding {param_stats['embedding_params'] / 1e6:.2f}M + 非Embedding {param_stats['non_embedding_params'] / 1e6:.2f}M")
            if param_stats.get("weight_quantization"):
                bits_list = param_stats.get("weight_quant_bits", [])
                bits_str = "/".join(str(b) for b in bits_list) if bits_list else "?"
                savings = param_stats.get("weight_quant_savings_mb", 0.0)
                layers = param_stats.get("weight_quant_layers", 0)
                self.logger.info(f"權重量化狀態: INT{bits_str} 已啟用 ({layers} 層，估算節省 {savings:.1f} MB)")
            else:
                self.logger.info("權重量化狀態: 未啟用（使用原始精度）")

            if self.config.use_compile:
                try:
                    self.model.enable_torch_compile(mode=self.config.compile_mode, dynamic=True)
                except Exception as e:
                    self.logger.warning(f"torch.compile 啟用失敗: {e}")

            tokenizer_file = Path(self.config.tokenizer_path)
            if tokenizer_file.exists():
                tokenizer_size_bytes = tokenizer_file.stat().st_size
                tokenizer_size_mb = tokenizer_size_bytes / (1024**2)
                tokenizer_ratio = (tokenizer_size_bytes / max(1, param_stats["parameter_memory_bytes"])) * 100.0
                self.logger.info(f"Tokenizer檔案: {tokenizer_size_mb:.2f}MB (約 {tokenizer_ratio:.2f}% 參數記憶體)")
            else:
                self.logger.info("Tokenizer檔案: 未找到（不影響模型參數統計）")

            self.logger.debug(f"可訓練參數: {param_stats['trainable_params'] / 1e6:.2f}M")
            self.logger.debug(f"Embedding矩陣: {param_stats['vocab_size']} x {param_stats['hidden_size']}")
            if param_stats["lm_head_tied_with_embedding"]:
                self.logger.debug("LM Head參數: 0 (與Embedding共享權重)")
            else:
                self.logger.debug(f"LM Head參數: {param_stats['lm_head_params'] / 1e6:.2f}M")
                
            self.logger.debug(f"參數記憶體估算(目前dtype): {param_stats['parameter_memory_gb']:.2f}GB")

        except Exception as e:
            self.logger.error(f"加載模型或分詞器失敗: {e}")
            raise

    def _prepare_input(self, prompt: str) -> torch.Tensor:
        """
        將提示詞格式化並編碼為 Tensor 格式。

        Args:
            prompt (str): 使用者輸入提示文字。

        Returns:
            torch.Tensor: Shape (1, seq_len) 的 Tensor。
        """
        
        bos = self.tokenizer.bos_token or "<s>"
        normalized_prompt = str(prompt).strip()
        formatted_prompt = f"{bos}{normalized_prompt}\n" if normalized_prompt else bos
        input_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False, return_tensors="pt")
        return input_ids.to(self.device)

    def stream_generate(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        流式生成響應 - 記憶體與 UTF-8 增量輸出優化版本。

        Args:
            prompt (str): 輸入提示。
            max_new_tokens (int): 最大新生成 token 數。
            **kwargs: 包含 temperature, top_k, top_p 等生成參數。

        Yields:
            dict: 包含增量文字 (delta)、狀態與完整響應 (full_response) 的字典。
        """
        
        if not self.model or not self.tokenizer:
            raise RuntimeError("模型或分詞器未初始化")

        try:
            max_context_len = int(
                kwargs.get("max_length")
                or self.config.max_length
                or getattr(self.model.config, "max_position_embeddings", 512)
            )
            input_ids = self._prepare_input(prompt)
            if input_ids.shape[1] >= max_context_len - 10:
                yield {"delta": "輸入太長，請縮短後重試。", "finished": True, "error": True}
                return
            current_seed = self.seed_manager.get_new_seed()
            available_tokens = max_context_len - input_ids.shape[1] - 10
            if available_tokens <= 0:
                yield {"delta": "可用上下文長度不足，請縮短輸入或提高 max_length。","finished": True,"error": True,}
                return

            effective_max_tokens = min(max(1, int(max_new_tokens)), max(1, int(available_tokens)))
            generation_params = {
                "max_new_tokens": effective_max_tokens,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_k": kwargs.get("top_k", self.config.top_k),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
                "do_sample": kwargs.get("do_sample", self.config.do_sample),
                "generation_seed": current_seed,
            }
            self.logger.debug(f"使用種子: {current_seed}, 最大token數: {effective_max_tokens}")
            full_response = ""
            last_memory_report = 0
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated() / 1024**2
                self.logger.debug(f"初始GPU內存: {initial_memory:.1f}MB")

            with torch.no_grad():
                for step, output in enumerate(self.model.stream_generate(input_ids=input_ids, **generation_params)):
                    if (not self.config.silent_mode and torch.cuda.is_available() and step > 0 and step % 100 == 0):
                        current_memory = torch.cuda.memory_allocated() / 1024**2
                        if current_memory > last_memory_report + 100:
                            self.logger.debug(f"步驟 {step}: GPU內存 {current_memory:.1f}MB (+{current_memory - initial_memory:.1f}MB)")
                            last_memory_report = current_memory
                            total_memory = (torch.cuda.get_device_properties(0).total_memory / 1024**2)
                            usage_rate = current_memory / total_memory
                            if usage_rate > 0.85:
                                self.logger.error(f"GPU內存使用率過高: {usage_rate * 100:.1f}%")
                                torch.cuda.empty_cache()

                    if output["finished"]:
                        if torch.cuda.is_available():
                            final_memory = torch.cuda.memory_allocated() / 1024**2
                            self.logger.debug(f"最終GPU內存: {final_memory:.1f}MB")
                            torch.cuda.empty_cache()

                        if "generated_ids" in output and output["generated_ids"] is not None:
                            try:
                                final_new_tokens = output["generated_ids"][:, input_ids.shape[1] :]
                                if final_new_tokens.shape[1] > 0:
                                    final_decoded = self.tokenizer.decode(final_new_tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                                    final_decoded = _fix_utf8_artifacts(final_decoded)
                                    _REPL = "\ufffd"
                                    if _REPL in final_decoded:
                                        final_decoded = final_decoded.rstrip(_REPL)
                                    if len(final_decoded) >= len(full_response):
                                        full_response = final_decoded
                            except Exception:
                                pass

                        yield {
                            "delta": "",
                            "finished": True,
                            "tokens_generated": output.get("tokens_generated", 0),
                            "stop_reason": output.get("stop_reason", "completed"),
                            "full_response": full_response,
                            "error": False,
                        }
                        break

                    if "generated_ids" in output and output["generated_ids"] is not None:
                        try:
                            new_tokens = output["generated_ids"][:, input_ids.shape[1] :]
                            if new_tokens.shape[1] > 0:
                                current_response = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                                current_response = _fix_utf8_artifacts(current_response)
                                _REPL = "\ufffd"
                                if _REPL in current_response:
                                    stable_response = current_response.rstrip(_REPL)
                                else:
                                    stable_response = current_response

                                if len(stable_response) > len(full_response):
                                    delta = stable_response[len(full_response) :]
                                    full_response = stable_response
                                    if delta:
                                        yield {"delta": delta, "finished": False, "full_response": full_response, "error": False}

                            del new_tokens

                        except Exception as e:
                            self.logger.warning(f"解碼token失敗: {e}")
                            continue

        except Exception as e:
            self.logger.error(f"流式生成錯誤: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            yield {"delta": f"生成失敗: {str(e)}", "finished": True, "error": True}

    def single_inference(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        """
        單次推理（非交互式）。

        Args:
            prompt (str): 輸入提示。
            max_new_tokens (int): 最大新生成 token 數。
            **kwargs: 其他生成參數。

        Returns:
            str: 生成的完整響應。
        """
        
        full_response = ""
        for output in self.stream_generate(prompt, max_new_tokens, **kwargs):
            if output["finished"]:
                if output.get("error", False):
                    return f"生成失敗: {output['delta']}"
                return output.get("full_response", full_response)

            if output["delta"] and not output.get("error", False):
                full_response += output["delta"]

        return full_response

    def interactive_chat(self) -> None:
        """交互式 Terminal CLI 流式對話迴圈"""
        print("🌸 長門櫻流式推理程序")
        print("=" * 60)
        print("💡 提示: 這是流式輸出模式，您將看到即時生成的回應")
        print("   - 輸入 'quit' 或 'exit' 退出")
        print("   - 輸入 '/help' 查看命令")
        print("   - 輸入 '/config' 查看當前配置")
        print("   - 輸入 '/temp' 切換臨時聊天模式（每輪新對話）")
        print(f"   - 基礎種子: {self.seed_manager.base_seed}")
        print(f"   - 臨時聊天模式: {'開啟' if self.config.stateless_chat else '關閉'}")
        print("   - 每次對話都會使用不同的隨機種子")
        print("=" * 60)
        conversation_history = ConversationHistory()
        while True:
            try:
                user_input = input("\n👤 用戶: ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "退出", "結束"]:
                    print("\n🌸 長門櫻: 謝謝您的陪伴，再見！")
                    break

                if user_input.startswith("/"):
                    self._handle_command(user_input, conversation_history)
                    continue

                print("\n🌸 長門櫻: ", end="", flush=True)
                start_time = time.time()
                full_response = ""
                try:
                    max_new_tokens = max(1, int(self.config.max_new_tokens))
                    for output in self.stream_generate(
                        user_input,
                        max_new_tokens=max_new_tokens,
                        temperature=self.config.temperature,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p,
                        repetition_penalty=self.config.repetition_penalty,
                        do_sample=self.config.do_sample,
                    ):
                        if output["finished"]:
                            end_time = time.time()
                            if not output.get("error", False):
                                full_response = output.get("full_response", full_response)
                                if not self.config.stateless_chat:
                                    conversation_history.add_turn("user", user_input)
                                    conversation_history.add_turn("assistant", full_response)
                                print(
                                    f"\n⏱️  生成時間: {end_time - start_time:.2f}秒，"
                                    f"{output.get('tokens_generated', 0)} tokens"
                                )
                            break

                        if output["delta"] and not output.get("error", False):
                            print(output["delta"], end="", flush=True)
                            full_response += output["delta"]

                except Exception as e:
                    print(f"\n❌ 生成過程中出錯: {e}")
                    self.logger.error(f"流式生成錯誤: {e}")

            except KeyboardInterrupt:
                print("\n\n🌸 長門櫻: 檢測到中斷，再見！")
                break
            
            except Exception as e:
                print(f"\n❌ 錯誤: {e}")
                self.logger.error(f"交互過程中發生錯誤: {e}")

        self._save_conversation_history(conversation_history)

    def _handle_command(self, command: str, conversation_history: ConversationHistory) -> None:
        """處理交互模式下的特殊斜線命令"""
        
        cmd_parts = command[1:].split()
        cmd = cmd_parts[0].lower()
        if cmd == "help":
            print("""
🌸 可用命令:
/help - 顯示此幫助
/config - 顯示當前配置
/temp [on|off] - 切換臨時聊天模式（每輪新對話）
/clear - 清空對話歷史
/history - 顯示對話歷史
/memory - 顯示內存使用情況
/cleanup - 手動清理GPU內存
/verbose - 切換詳細模式（顯示內存監控）
/set <param> <value> - 設置生成參數
    可設置參數: temperature, top_k, top_p, repetition_penalty, max_length, max_new_tokens
/seed <seed> - 設置基礎種子（-1為隨機）
""")
        elif cmd == "config":
            print(f"""
🔧 當前配置:
- 最大長度: {self.config.max_length}
- 單輪最大生成token: {self.config.max_new_tokens}
- 溫度: {self.config.temperature}
- Top-k: {self.config.top_k}
- Top-p: {self.config.top_p}
- 重複懲罰: {self.config.repetition_penalty}
- 採樣: {self.config.do_sample}
- 設備: {self.device}
- 基礎種子: {self.seed_manager.base_seed}
- KV量化: {self.config.quantize_kv_cache if self.config.quantize_kv_cache is not None else "依模型配置"}
- KV位寬: {self.config.kv_cache_bits if self.config.kv_cache_bits is not None else "依模型配置"}
- KV分組: {self.config.kv_quant_group_size if self.config.kv_quant_group_size is not None else "依模型配置"}
- 殘差符號修正: {self.config.kv_residual_sign_correction if self.config.kv_residual_sign_correction is not None else "依模型配置"}
- num_key_value_heads: {self.config.num_key_value_heads if self.config.num_key_value_heads is not None else "依模型配置"}
- 權重量化: {"INT" + str(self.config.weight_quant_bits) + " (已啟用)" if self.config.weight_quantization else "未啟用"}
- 臨時聊天模式: {"開啟" if self.config.stateless_chat else "關閉"}
- 靜默模式: {"開啟" if self.config.silent_mode else "關閉"}
- 隨機種子模式: 每次對話使用不同種子
""")
        elif cmd == "verbose":
            self.config.silent_mode = not self.config.silent_mode
            mode_str = "關閉" if self.config.silent_mode else "開啟"
            print(f"✅ 詳細模式已{mode_str}")
            if not self.config.silent_mode:
                print("💡 現在會顯示內存監控信息")
            else:
                print("💡 內存監控信息已隱藏，輸出更乾淨")
        elif cmd == "temp":
            if len(cmd_parts) >= 2:
                option = cmd_parts[1].strip().lower()
                if option in {"on", "true", "1", "yes", "enable"}:
                    self.config.stateless_chat = True
                elif option in {"off", "false", "0", "no", "disable"}:
                    self.config.stateless_chat = False
                else:
                    print(f"❌ 未知選項: {cmd_parts[1]}，可用 on/off")
                    return
            else:
                self.config.stateless_chat = not self.config.stateless_chat

            if self.config.stateless_chat:
                conversation_history.clear()
            mode_text = "開啟" if self.config.stateless_chat else "關閉"
            print(f"✅ 臨時聊天模式已{mode_text}")
            if self.config.stateless_chat:
                print("💡 目前每一輪都會從空白對話開始，不保留上一輪內容")
            else:
                print("💡 已恢復一般多輪對話模式")
        elif cmd == "memory":
            self._show_memory_info()
        elif cmd == "cleanup":
            self._cleanup_memory()
        elif cmd == "clear":
            conversation_history.clear()
            print("✅ 對話歷史已清空")
        elif cmd == "history":
            if conversation_history.turns:
                print("\n📜 對話歷史:")
                for i, turn in enumerate(conversation_history.turns[-10:], 1):
                    role_icon = "👤" if turn.role == "user" else "🌸"
                    print(f"  {i}. {role_icon} {turn.role}: {turn.content[:50]}...")
            else:
                print("📜 暫無對話歷史")
        elif cmd == "seed" and len(cmd_parts) >= 2:
            try:
                new_seed = int(cmd_parts[1])
                self.seed_manager = RandomSeedManager(new_seed)
                self.config.base_seed = new_seed
                print(f"✅ 基礎種子已設置為 {self.seed_manager.base_seed}")
                print("💡 每次對話仍會使用不同的隨機種子")
            except ValueError:
                print(f"❌ 無效的種子值: {cmd_parts[1]}")
        elif cmd == "set" and len(cmd_parts) >= 3:
            param = cmd_parts[1]
            try:
                value = float(cmd_parts[2]) if "." in cmd_parts[2] else int(cmd_parts[2])
                if param == "temperature":
                    self.config.temperature = max(0.1, min(2.0, value))
                elif param == "top_k":
                    self.config.top_k = max(1, min(100, int(value)))
                elif param == "top_p":
                    self.config.top_p = max(0.1, min(1.0, value))
                elif param == "repetition_penalty":
                    self.config.repetition_penalty = max(1.0, min(2.0, value))
                elif param == "max_length":
                    model_max = (int(getattr(self.model.config, "max_position_embeddings", 8192)) if self.model else 8192)
                    self.config.max_length = max(50, min(model_max, int(value)))
                    self.config.max_new_tokens = min(int(self.config.max_new_tokens), max(1, self.config.max_length - 10))
                elif param == "max_new_tokens":
                    max_limit = max(1, int(self.config.max_length or 512) - 10)
                    self.config.max_new_tokens = max(1, min(max_limit, int(value)))
                else:
                    print(f"❌ 未知參數: {param}")
                    return
                print(f"✅ {param} 已設置為 {value}")
            except ValueError:
                print(f"❌ 無效的值: {cmd_parts[2]}")
        else:
            print(f"❌ 未知命令: {command}")

    def _show_memory_info(self) -> None:
        """顯示 GPU 與模型參數記憶體佔用狀況"""
        
        print("\n💾 內存使用情況:")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            print("🔧 GPU內存:")
            print(f"   - 已分配: {allocated:.1f}MB")
            print(f"   - 已保留: {reserved:.1f}MB")
            print(f"   - 總容量: {total:.1f}MB")
            print(f"   - 使用率: {allocated / total * 100:.1f}%")
            if allocated / total > 0.8:
                print("⚠️  警告: GPU內存使用率過高，建議執行 /cleanup")
        else:
            print("❌ CUDA不可用")

        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            param_memory = total_params * 4 / 1024**2
            print(f"📊 模型參數: {total_params / 1e6:.2f}M ({param_memory:.1f}MB FP32 等效)")
            try:
                q_info = self.model.get_quantization_info()
                if q_info["is_quantized"]:
                    bits_str = "/".join(str(b) for b in q_info["bits"]) if q_info["bits"] else "?"
                    print(f"權重量化: INT{bits_str} ({q_info['quantized_layers']} 層量化)")
                    print(f"實際佔用: {q_info['quantized_mb']:.1f}MB (估算節省 {q_info['estimated_savings_mb']:.1f}MB)")
                else:
                    print("權重量化: 未啟用 (使用原始精度)")
            except Exception:
                pass

    def _cleanup_memory(self) -> None:
        """手動主動釋放 CUDA 快取記憶體"""
        
        print("🧹 正在清理GPU內存...")
        if torch.cuda.is_available():
            before = torch.cuda.memory_allocated() / 1024**2
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            after = torch.cuda.memory_allocated() / 1024**2
            print(f"✅ 內存清理完成: {before:.1f}MB → {after:.1f}MB (釋放 {before - after:.1f}MB)")
        else:
            print("❌ CUDA不可用，無法清理GPU內存")

    def _save_conversation_history(self, history: ConversationHistory) -> None:
        """保存對話歷史至 JSON 檔案"""
        
        if not history.turns:
            return
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(history.to_dict(), f, ensure_ascii=False, indent=2)
            self.logger.info(f"對話歷史已保存: {filename}")
        except Exception as e:
            self.logger.error(f"保存對話歷史失敗: {e}")


# ==============================================================================
# Web UI 啟動與輔助函數
# ==============================================================================


def _append_common_web_args(command: List[str], args: argparse.Namespace) -> None:
    """附加通用 Web 參數到子程序命令列。"""

    def append_arg(name: str, value: Any) -> None:
        if value is not None:
            command.extend([name, str(value)])

    append_arg("--model_path", args.model_path)
    append_arg("--tokenizer_path", args.tokenizer_path)
    append_arg("--device", args.device)
    append_arg("--max_length", args.max_length)
    append_arg("--max_new_tokens", args.max_new_tokens)
    append_arg("--temperature", args.temperature)
    append_arg("--top_k", args.top_k)
    append_arg("--top_p", args.top_p)
    append_arg("--repetition_penalty", args.repetition_penalty)
    append_arg("--seed", args.seed)
    append_arg("--history_rounds", args.history_rounds)
    append_arg("--kv_cache_bits", args.kv_cache_bits)
    append_arg("--kv_quant_group_size", args.kv_quant_group_size)
    append_arg("--num_key_value_heads", args.num_key_value_heads)
    append_arg("--log_level", getattr(args, "log_level", "INFO"))
    if args.no_sample:
        command.append("--no_sample")
        
    if args.verbose:
        command.append("--verbose")
        
    if args.stateless_chat:
        command.append("--stateless_chat")
        
    if args.quantize_kv_cache:
        command.append("--quantize_kv_cache")
        
    if args.kv_residual_sign_correction:
        command.append("--kv_residual_sign_correction")
        
    weight_quant = getattr(args, "weight_quantization", False)
    if weight_quant and weight_quant != "False":
        command.append("--weight_quantization")
        
    append_arg("--weight_quant_bits", getattr(args, "weight_quant_bits", None))
    append_arg("--weight_quant_group_size", getattr(args, "weight_quant_group_size", None))
    append_arg("--weight_quant_mode", getattr(args, "weight_quant_mode", None))
    if getattr(args, "weight_quant_embeddings", False):
        command.append("--weight_quant_embeddings")
        
    if getattr(args, "weight_quant_lm_head", False):
        command.append("--weight_quant_lm_head")


def launch_flask_web_demo(args: argparse.Namespace, logger: logging.Logger) -> int:
    """啟動 Flask Web Demo（tools/web_demo_flask.py）。"""
    
    web_demo_path = Path(__file__).resolve().parent / "tools" / "web_demo_flask.py"
    if not web_demo_path.exists():
        logger.error(f"找不到 Flask Web Demo 腳本: {web_demo_path}")
        return 1

    try:
        import flask
    except ImportError:
        logger.error("未安裝 flask。請先執行: pip install flask")
        return 1

    command = [sys.executable, str(web_demo_path),"--web_host",str(args.web_host),"--web_port",str(args.web_port),]
    _append_common_web_args(command, args)
    logger.info(f"啟動 Flask Web Demo: http://{args.web_host}:{args.web_port}")
    try:
        result = subprocess.run(command, check=False)
        return int(result.returncode)
    except KeyboardInterrupt:
        logger.info("已接收終止訊號 (Ctrl+C)，Web 服務已安全結束。")
        return 0
    except Exception as e:
        logger.error(f"啟動 Flask Web Demo 失敗: {e}")
        return 1


def launch_web_demo(args: argparse.Namespace, logger: logging.Logger) -> int:
    """啟動 Web UI。"""
    return launch_flask_web_demo(args, logger)


# ==============================================================================
# CLI 主程式入口
# ==============================================================================


def main() -> None:
    """主函數：解析 CLI 命令並啟動單次推理、CLI 對話或 Web 介面"""
    
    parser = argparse.ArgumentParser(description="長門櫻模型流式推理程序 (NagatoSakura LLM Inference Engine)")

    # 基本與模式設定
    req_group = parser.add_argument_group("基本與模式設定 (Basic & Mode Options)")
    req_group.add_argument("--model_path", type=str, default="NS-LM-1.4/best_model", help="模型路徑")
    req_group.add_argument("--tokenizer_path", type=str, help="分詞器路徑（如果未指定，將在模型路徑中查找）")
    req_group.add_argument("--mode",type=str,default="web", choices=["interactive", "single", "web"], help="推理模式: interactive=交互式對話, single=單次推理, web=啟動Web介面")

    # 單次推理與生成長度
    gen_len_group = parser.add_argument_group("生成長度設定 (Generation Length Options)")
    gen_len_group.add_argument("--prompt", type=str, help="單次推理的輸入提示")
    gen_len_group.add_argument("--max_new_tokens", type=int, default=0, help="最大新生成 token 數（0=自動設為模型上下文長度的一半）")
    gen_len_group.add_argument("--max_length", type=int, default=0, help="推理上下文上限（0=自動使用模型 config.json 的 max_position_embeddings）")

    # 解碼與超參數
    hyper_group = parser.add_argument_group("解碼超參數 (Sampling & Generation Hyperparameters)")
    hyper_group.add_argument("--temperature", type=float, default=0.7, help="溫度參數")
    hyper_group.add_argument("--top_k", type=int, default=50, help="Top-k參數")
    hyper_group.add_argument("--top_p", type=float, default=0.9, help="Top-p參數")
    hyper_group.add_argument("--repetition_penalty", type=float, default=1.0, help="重複懲罰")
    hyper_group.add_argument("--no_sample", action="store_true", help="禁用採樣（使用貪婪解碼）")

    # 高級覆寫選項
    override_group = parser.add_argument_group("高級模型覆寫選項 (Advanced Overrides)")
    override_group.add_argument("--quantize_kv_cache", action="store_true", default=None, help="強制啟用 KV cache 量化（覆寫模型 config.json）")
    override_group.add_argument("--no_quantize_kv_cache", action="store_false", dest="quantize_kv_cache", help="強制關閉 KV cache 量化（覆寫模型 config.json）")
    override_group.add_argument("--kv_cache_bits", type=int, choices=[3, 4, 8, 16, 32], help="手動覆寫：KV cache 位寬")
    override_group.add_argument("--kv_quant_group_size", type=int, help="手動覆寫：KV 量化分組大小")
    override_group.add_argument("--kv_residual_sign_correction", action="store_true", default=None, help="手動覆寫：啟用 1-bit 殘差符號修正")
    override_group.add_argument("--no_kv_residual_sign_correction", action="store_false", dest="kv_residual_sign_correction", help="手動覆寫：禁用 1-bit 殘差符號修正")
    override_group.add_argument("--kv_decode_mode", type=str, choices=["fast", "low_ram"], default="low_ram", help="KV 解碼模式: fast=啟用影子快取加速 ($O(1)$ 單步), low_ram=極致省顯存模式")
    override_group.add_argument("--num_key_value_heads", type=int, help="手動覆寫：GQA key/value 頭數")

    # 權重量化選項
    override_group.add_argument("--weight_quantization", action="store_true", help="啟用推理時權重量化 (Post-Training Quantization)，預設已啟用")
    override_group.add_argument("--weight_quant_bits", type=int,choices=[4, 8], default=8, help="權重量化位寬：8=INT8, 4=INT4 (預設: 8)",)
    override_group.add_argument("--weight_quant_group_size", type=int, default=128, help="INT4 group-wise 量化分組大小 (預設: 128)",)
    override_group.add_argument("--weight_quant_mode", type=str, default="auto", choices=["auto", "dynamic", "compressed"], help="權重量化路徑: auto=自動選擇, dynamic=qnnpack INT8, compressed=自定義")
    override_group.add_argument("--weight_quant_embeddings", action="store_true", help="是否也量化嵌入層 (embed_tokens)")
    override_group.add_argument("--weight_quant_lm_head", action="store_true", help="是否也量化 lm_head 輸出層")
    override_group.add_argument("--use_compile", action="store_true", default=True, help="啟用 torch.compile 加速模型前向傳播 (Try-Except 容錯保護)")
    override_group.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"], help="torch.compile 模式 (預設: default)")
    override_group.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"], help="推理權重與 KV Cache 計算精度 (預設: auto)")

    # 種子與系統
    system_group = parser.add_argument_group("種子與對話系統設定 (System & Seed Options)")
    system_group.add_argument("--seed", type=int, default=-1, help="基礎種子（-1為隨機種子）")
    system_group.add_argument("--history_rounds", type=int, default=3, help="Web 對話保留歷史輪次")
    system_group.add_argument("--verbose", action="store_true", help="啟用詳細輸出（包括內存監控）")
    system_group.add_argument("--stateless_chat", action="store_true", default=True, help="臨時聊天模式：每一輪都視為新對話")
    system_group.add_argument("--device", type=str, default="auto", help="指定設備")
    system_group.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日誌級別")
    system_group.add_argument("--web_host", type=str, default="127.0.0.1", help="Web UI 監聽位址（mode=web）")
    system_group.add_argument("--web_port", type=int, default=8501, help="Web UI 埠號（mode=web）")

    args = parser.parse_args()

    logger = setup_logging(args.log_level)
    _log_system_environment(logger)

    try:
        if not args.tokenizer_path:
            model_dir = Path(args.model_path)
            tokenizer_path = model_dir / "tokenizer.json"
            if tokenizer_path.exists():
                args.tokenizer_path = str(tokenizer_path)
            else:
                logger.error("未找到分詞器文件，請指定 --tokenizer_path")
                return

        if args.mode == "web":
            exit_code = launch_web_demo(args, logger)
            if exit_code != 0:
                sys.exit(exit_code)
            return

        config = InferenceConfig(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            max_length=(None if int(args.max_length) <= 0 else int(args.max_length)),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=not args.no_sample,
            base_seed=args.seed,
            silent_mode=not args.verbose,
            quantize_kv_cache=args.quantize_kv_cache,
            kv_cache_bits=args.kv_cache_bits,
            kv_quant_group_size=args.kv_quant_group_size,
            kv_residual_sign_correction=args.kv_residual_sign_correction,
            kv_decode_mode=args.kv_decode_mode if args.kv_decode_mode else None,
            num_key_value_heads=args.num_key_value_heads,
            stateless_chat=args.stateless_chat,
            weight_quantization=bool(args.weight_quantization),
            weight_quant_bits=args.weight_quant_bits,
            weight_quant_group_size=args.weight_quant_group_size,
            weight_quant_embeddings=args.weight_quant_embeddings,
            weight_quant_lm_head=args.weight_quant_lm_head,
            weight_quant_mode=getattr(args, "weight_quant_mode", "auto"),
            use_compile=bool(args.use_compile),
            compile_mode=args.compile_mode,
            dtype=args.dtype,
        )

        inference = NagatoSakuraInference(config)

        if args.mode == "interactive":
            inference.interactive_chat()
        elif args.mode == "single":
            if not args.prompt:
                logger.error("單次推理模式需要指定 --prompt 參數")
                return

            print(f"輸入: {args.prompt}")
            print("輸出: ", end="", flush=True)
            start_time = time.time()
            for output in inference.stream_generate(args.prompt, args.max_new_tokens):
                if output["finished"]:
                    end_time = time.time()
                    if not output.get("error", False):
                        print(f"\n\n生成完成！用時: {end_time - start_time:.2f}秒")
                        print(f"生成token數: {output.get('tokens_generated', 0)}")
                    break

                if output["delta"] and not output.get("error", False):
                    print(output["delta"], end="", flush=True)

    except Exception as e:
        logger.error(f"程序執行失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
