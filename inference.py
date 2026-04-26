import os
import sys
import json
import time
import logging
import argparse
import warnings
import random
import subprocess
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from transformers import PreTrainedTokenizerFast

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 導入自定義模型
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.nagato_sakura_model import NagatoSakuraForCausalLM, NSConfig, ConversationHistory
    from src.tokenizer import TokenizerManager
except ImportError as e:
    print(f"錯誤：無法導入自定義模型。錯誤詳情: {e}")
    sys.exit(1)

@dataclass
class InferenceConfig:
    """推理配置"""
    model_path: str
    tokenizer_path: str
    device: str = "auto"
    max_length: Optional[int] = None  # None=自動使用模型 max_position_embeddings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    do_sample: bool = True
    base_seed: int = -1
    silent_mode: bool = True  # 靜默模式，不顯示內存監控信息
    quantize_kv_cache: Optional[bool] = None
    kv_cache_bits: Optional[int] = None
    kv_quant_group_size: Optional[int] = None
    kv_residual_sign_correction: Optional[bool] = None
    num_key_value_heads: Optional[int] = None
    stateless_chat: bool = False  # 臨時聊天模式，每一輪都視為新對話

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

def setup_logging(log_level: str = "INFO"):
    """設置日誌系統"""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logging.getLogger(__name__)

class RandomSeedManager:
    """隨機種子管理器 - 每次生成使用不同的隨機種子"""
    def __init__(self, base_seed: int = -1):
        """
        初始化種子管理器
        Args:
            base_seed: 基礎種子值，-1表示使用隨機種子
        """
        if base_seed == -1:
            self.base_seed = int(time.time() * 1000) % (2**32)
        else:
            self.base_seed = base_seed
        self.counter = 0
        
    def get_new_seed(self) -> int:
        """獲取新的隨機種子"""
        new_seed = (self.base_seed + self.counter + random.randint(1, 10000)) % (2**32)
        self.counter += 1
        return new_seed

class NagatoSakuraInference:
    """長門櫻模型流式推理器"""
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

    def _log_tokenizer_health(self):
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
            if "�" in probe_decoded:
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
    
    def _load_model_and_tokenizer(self):
        """加載模型和分詞器"""
        try:
            # 使用TokenizerManager加載分詞器
            self.tokenizer_manager = TokenizerManager(Path(self.config.tokenizer_path))
            self.tokenizer_manager.load_tokenizer()
            self.tokenizer = self.tokenizer_manager.transformers_tokenizer
            self.logger.info(
                f"分詞器就緒: {self.config.tokenizer_path} (詞彙量 {len(self.tokenizer)})"
            )
            self._log_tokenizer_health()
            
            # 加載模型配置
            config_path = Path(self.config.model_path) / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                model_config = NSConfig.from_dict(config_dict)
            else:
                # 使用默認配置
                self.logger.warning("未找到配置文件，使用默認配置")
                model_config = NSConfig(vocab_size=len(self.tokenizer))
            
            # 確保配置與分詞器一致
            model_config.vocab_size = len(self.tokenizer)
            model_config.pad_token_id = self.tokenizer.pad_token_id
            model_config.bos_token_id = self.tokenizer.bos_token_id
            model_config.eos_token_id = self.tokenizer.eos_token_id
            model_config.unk_token_id = self.tokenizer.unk_token_id

            # 套用命令列覆寫（0.5.0）
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

            # 推理上下文上限：預設追隨模型配置，可由 --max_length 覆寫（但不超模型上限）
            model_max_length = int(getattr(model_config, "max_position_embeddings", 512))
            if self.config.max_length is None or int(self.config.max_length) <= 0:
                self.config.max_length = model_max_length
            else:
                self.config.max_length = min(int(self.config.max_length), model_max_length)
            self.logger.info(f"推理上下文上限: {self.config.max_length} (模型上限: {model_max_length})")
            
            # 創建模型
            self.logger.info(f"加載模型: {self.config.model_path}")
            self.model = NagatoSakuraForCausalLM(model_config)
            
            # 嘗試多種權重文件名
            weight_files = ["model.pt", "pytorch_model.bin", "model.safetensors"]
            model_loaded = False
            
            for weight_file in weight_files:
                model_path = Path(self.config.model_path) / weight_file
                if model_path.exists():
                    try:
                        state_dict = torch.load(model_path, map_location=self.device)
                        self.model.load_state_dict(state_dict)
                        self.logger.info(f"模型權重從 {weight_file} 加載完成")
                        model_loaded = True
                        break
                    except Exception as e:
                        self.logger.warning(f"從 {weight_file} 加載權重失敗: {e}")
                        continue
            
            if not model_loaded:
                self.logger.warning("未找到模型權重文件，使用隨機初始化的權重")
            
            # 移動到設備並設置評估模式
            self.model.to(self.device)
            self.model.eval()

            # 模型規模摘要（參數量不含 tokenizer 檔案）
            param_stats = self.model.get_parameter_stats()
            self.logger.info(
                f"模型參數量: {param_stats['total_params']/1e6:.2f}M"
            )
            self.logger.info(
                f"參數組成: Embedding {param_stats['embedding_params']/1e6:.2f}M "
                f"+ 非Embedding {param_stats['non_embedding_params']/1e6:.2f}M"
            )

            tokenizer_file = Path(self.config.tokenizer_path)
            if tokenizer_file.exists():
                tokenizer_size_bytes = tokenizer_file.stat().st_size
                tokenizer_size_mb = tokenizer_size_bytes / (1024 ** 2)
                tokenizer_ratio = (tokenizer_size_bytes / max(1, param_stats["parameter_memory_bytes"])) * 100.0
                self.logger.info(
                    f"Tokenizer檔案: {tokenizer_size_mb:.2f}MB "
                    f"(約 {tokenizer_ratio:.2f}% 參數記憶體)"
                )
            else:
                self.logger.info("Tokenizer檔案: 未找到（不影響模型參數統計）")

            # 詳細拆分放在 DEBUG，避免預設輸出過於冗長。
            self.logger.debug(f"可訓練參數: {param_stats['trainable_params']/1e6:.2f}M")
            self.logger.debug(
                f"Embedding矩陣: {param_stats['vocab_size']} x {param_stats['hidden_size']}"
            )
            if param_stats["lm_head_tied_with_embedding"]:
                self.logger.debug("LM Head參數: 0 (與Embedding共享權重)")
            else:
                self.logger.debug(f"LM Head參數: {param_stats['lm_head_params']/1e6:.2f}M")
            self.logger.debug(f"參數記憶體估算(目前dtype): {param_stats['parameter_memory_gb']:.2f}GB")
            
        except Exception as e:
            self.logger.error(f"加載模型或分詞器失敗: {e}")
            raise
    
    def _prepare_input(self, prompt: str) -> torch.Tensor:
        """準備輸入"""
        # 對齊訓練格式: <bos> + instruction(+input) + "\n"
        bos = self.tokenizer.bos_token or "<s>"
        normalized_prompt = str(prompt).strip()
        formatted_prompt = f"{bos}{normalized_prompt}\n" if normalized_prompt else bos
        
        # 編碼
        input_ids = self.tokenizer.encode(
            formatted_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        return input_ids.to(self.device)

    def stream_generate(self, prompt: str, max_new_tokens: int = 512, **kwargs):
        """
        流式生成響應 - 內存優化版本
        
        Args:
            prompt: 輸入提示
            max_new_tokens: 最大新生成token數
            **kwargs: 其他生成參數
        
        Yields:
            dict: 包含增量文本和狀態信息的字典
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("模型或分詞器未初始化")
        
        try:
            max_context_len = int(
                self.config.max_length
                or getattr(self.model.config, "max_position_embeddings", 512)
            )

            # 準備輸入
            input_ids = self._prepare_input(prompt)
            
            # 檢查輸入長度
            if input_ids.shape[1] >= max_context_len - 10:
                yield {"delta": "輸入太長，請縮短後重試。", "finished": True, "error": True}
                return
            
            # 獲取新的隨機種子
            current_seed = self.seed_manager.get_new_seed()
            
            # 設置生成參數 - 限制最大長度防止內存爆炸
            available_tokens = max_context_len - input_ids.shape[1] - 10
            if available_tokens <= 0:
                yield {"delta": "可用上下文長度不足，請縮短輸入或提高 max_length。", "finished": True, "error": True}
                return

            effective_max_tokens = min(max(1, int(max_new_tokens)), max(1, int(available_tokens)))
            
            generation_params = {
                'max_new_tokens': effective_max_tokens,
                'temperature': kwargs.get('temperature', self.config.temperature),
                'top_k': kwargs.get('top_k', self.config.top_k),
                'top_p': kwargs.get('top_p', self.config.top_p),
                'repetition_penalty': kwargs.get('repetition_penalty', self.config.repetition_penalty),
                'do_sample': kwargs.get('do_sample', self.config.do_sample),
                'generation_seed': current_seed
            }
            
            self.logger.debug(f"使用種子: {current_seed}, 最大token數: {effective_max_tokens}")
            
            full_response = ""
            last_memory_report = 0
            
            # 初始內存狀態
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                self.logger.debug(f"初始GPU內存: {initial_memory:.1f}MB")
            
            with torch.no_grad():
                for step, output in enumerate(self.model.stream_generate(
                    input_ids=input_ids,
                    **generation_params
                )):
                    # 靜默模式下不進行內存監控日誌輸出
                    if not self.config.silent_mode and torch.cuda.is_available() and step > 0 and step % 100 == 0:
                        current_memory = torch.cuda.memory_allocated() / 1024**2
                        if current_memory > last_memory_report + 100:  # 內存增長超過100MB時才記錄
                            self.logger.debug(f"步驟 {step}: GPU內存 {current_memory:.1f}MB (+{current_memory - initial_memory:.1f}MB)")
                            last_memory_report = current_memory
                            
                            # 只有在內存使用率過高時才發出警告
                            if torch.cuda.is_available():
                                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
                                usage_rate = current_memory / total_memory
                                if usage_rate > 0.85:  # 使用率超過85%時發出靜默警告（不打印）
                                    self.logger.error(f"GPU內存使用率過高: {usage_rate*100:.1f}%")
                                    # 強制清理
                                    torch.cuda.empty_cache()
                    
                    if output["finished"]:
                        # 最終清理
                        if torch.cuda.is_available():
                            final_memory = torch.cuda.memory_allocated() / 1024**2
                            self.logger.debug(f"最終GPU內存: {final_memory:.1f}MB")
                            torch.cuda.empty_cache()
                        
                        yield {
                            "delta": "",
                            "finished": True,
                            "tokens_generated": output.get('tokens_generated', 0),
                            "stop_reason": output.get('stop_reason', 'completed'),
                            "full_response": full_response,
                            "error": False
                        }
                        break
                    
                    # 處理流式輸出
                    if "generated_ids" in output and output["generated_ids"] is not None:
                        try:
                            # 獲取新生成的token（排除輸入部分）
                            new_tokens = output["generated_ids"][:, input_ids.shape[1]:]
                            if new_tokens.shape[1] > 0:
                                # 解碼當前完整響應
                                current_response = self.tokenizer.decode(
                                    new_tokens[0], 
                                    skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=True
                                )
                                
                                # 計算與之前響應的差異（增量）
                                if len(current_response) > len(full_response):
                                    delta = current_response[len(full_response):]
                                    full_response = current_response
                                    
                                    # 只輸出有內容的增量
                                    if delta:
                                        yield {
                                            "delta": delta,
                                            "finished": False,
                                            "full_response": full_response,
                                            "error": False
                                        }
                                        
                            # 清理臨時變量
                            del new_tokens
                            if step % 20 == 0:  # 每20步清理一次
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    
                        except Exception as e:
                            self.logger.warning(f"解碼token失敗: {e}")
                            continue
                            
        except Exception as e:
            self.logger.error(f"流式生成錯誤: {e}")
            # 發生錯誤時清理GPU內存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            yield {
                "delta": f"生成失敗: {str(e)}",
                "finished": True,
                "error": True
            }

    def interactive_chat(self):
        """交互式流式對話"""
        print("🌸 長門櫻流式推理程序 🌸")
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
                # 獲取用戶輸入
                user_input = input("\n👤 用戶: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', '退出', '結束']:
                    print("\n🌸 長門櫻: 謝謝您的陪伴，再見！")
                    break
                
                # 特殊命令處理
                if user_input.startswith('/'):
                    self._handle_command(user_input, conversation_history)
                    continue
                
                # 生成流式響應
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
                                print(f"\n⏱️  生成時間: {end_time - start_time:.2f}秒，"
                                      f"{output.get('tokens_generated', 0)} tokens")
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
        
        # 保存對話歷史
        self._save_conversation_history(conversation_history)
    
    def _handle_command(self, command: str, conversation_history: ConversationHistory):
        """處理特殊命令"""
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
- KV量化: {self.config.quantize_kv_cache if self.config.quantize_kv_cache is not None else '依模型配置'}
- KV位寬: {self.config.kv_cache_bits if self.config.kv_cache_bits is not None else '依模型配置'}
- KV分組: {self.config.kv_quant_group_size if self.config.kv_quant_group_size is not None else '依模型配置'}
- 殘差符號修正: {self.config.kv_residual_sign_correction if self.config.kv_residual_sign_correction is not None else '依模型配置'}
- num_key_value_heads: {self.config.num_key_value_heads if self.config.num_key_value_heads is not None else '依模型配置'}
- 臨時聊天模式: {'開啟' if self.config.stateless_chat else '關閉'}
- 靜默模式: {'開啟' if self.config.silent_mode else '關閉'}
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
                for i, turn in enumerate(conversation_history.turns[-10:], 1):  # 顯示最近10輪
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
                value = float(cmd_parts[2]) if '.' in cmd_parts[2] else int(cmd_parts[2])
                if param == "temperature":
                    self.config.temperature = max(0.1, min(2.0, value))
                elif param == "top_k":
                    self.config.top_k = max(1, min(100, int(value)))
                elif param == "top_p":
                    self.config.top_p = max(0.1, min(1.0, value))
                elif param == "repetition_penalty":
                    self.config.repetition_penalty = max(1.0, min(2.0, value))
                elif param == "max_length":
                    model_max = int(getattr(self.model.config, "max_position_embeddings", 8192)) if self.model else 8192
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

    def _show_memory_info(self):
        """顯示內存使用情況"""
        print("\n💾 內存使用情況:")
        
        if torch.cuda.is_available():
            # GPU內存
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            reserved = torch.cuda.memory_reserved() / 1024**2   # MB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            
            print(f"🔧 GPU內存:")
            print(f"   - 已分配: {allocated:.1f}MB")
            print(f"   - 已保留: {reserved:.1f}MB") 
            print(f"   - 總容量: {total:.1f}MB")
            print(f"   - 使用率: {allocated/total*100:.1f}%")
            
            # 檢查內存壓力
            if allocated / total > 0.8:
                print("⚠️  警告: GPU內存使用率過高，建議執行 /cleanup")
        else:
            print("❌ CUDA不可用")
        
        # 模型參數內存估算
        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            param_memory = total_params * 4 / 1024**2  # 假設float32，4字節/參數
            print(f"📊 模型參數: {total_params/1e6:.2f}M ({param_memory:.1f}MB)")
    
    def _cleanup_memory(self):
        """手動清理GPU內存"""
        print("🧹 正在清理GPU內存...")
        
        if torch.cuda.is_available():
            before = torch.cuda.memory_allocated() / 1024**2
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            after = torch.cuda.memory_allocated() / 1024**2
            
            print(f"✅ 內存清理完成: {before:.1f}MB → {after:.1f}MB (釋放 {before-after:.1f}MB)")
        else:
            print("❌ CUDA不可用，無法清理GPU內存")

    def _save_conversation_history(self, history: ConversationHistory):
        """保存對話歷史"""
        if not history.turns:
            return
            
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(history.to_dict(), f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"對話歷史已保存: {filename}")
        except Exception as e:
            self.logger.error(f"保存對話歷史失敗: {e}")

    def single_inference(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        """
        單次推理（非交互式）
        
        Args:
            prompt: 輸入提示
            max_new_tokens: 最大新生成token數
            **kwargs: 其他生成參數
        
        Returns:
            str: 生成的完整響應
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


def _append_common_web_args(command: list, args: argparse.Namespace):
    """附加通用 Web 參數到子程序命令列。"""
    def append_arg(name: str, value):
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


def launch_flask_web_demo(args: argparse.Namespace, logger: logging.Logger) -> int:
    """啟動 Flask Web Demo（tools/web_demo_flask.py）。"""
    web_demo_path = Path(__file__).resolve().parent / "tools" / "web_demo_flask.py"
    if not web_demo_path.exists():
        logger.error(f"找不到 Flask Web Demo 腳本: {web_demo_path}")
        return 1

    try:
        import flask  # noqa: F401
    except ImportError:
        logger.error("未安裝 flask。請先執行: pip install flask")
        return 1

    command = [
        sys.executable,
        str(web_demo_path),
        "--web_host",
        str(args.web_host),
        "--web_port",
        str(args.web_port),
    ]
    _append_common_web_args(command, args)

    logger.info(
        f"啟動 Flask Web Demo: http://{args.web_host}:{args.web_port} "
        f"(腳本: {web_demo_path})"
    )

    try:
        result = subprocess.run(command, check=False)
        return int(result.returncode)
    except Exception as e:
        logger.error(f"啟動 Flask Web Demo 失敗: {e}")
        return 1


def launch_web_demo(args: argparse.Namespace, logger: logging.Logger) -> int:
    """啟動 Flask Web UI。"""
    return launch_flask_web_demo(args, logger)

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="長門櫻模型流式推理程序")
    
    # 必需參數
    parser.add_argument("--model_path", type=str, default="NS-LLM-1.3/checkpoint-epoch-10", help="模型路徑")
    parser.add_argument("--tokenizer_path", type=str, help="分詞器路徑（如果未指定，將在模型路徑中查找）")
    
    # 推理模式
    parser.add_argument("--mode", type=str, default="web", 
                       choices=["interactive", "single", "web"],
                       help="推理模式: interactive=交互式對話, single=單次推理, web=啟動Web介面")
    
    # 單次推理參數
    parser.add_argument("--prompt", type=str, help="單次推理的輸入提示")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大新生成token數")
    parser.add_argument("--max_length", type=int, default=0, help="推理上下文上限（0=自動使用模型 max_position_embeddings）")
    
    # 生成參數
    parser.add_argument("--temperature", type=float, default=0.7, help="溫度參數")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k參數")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p參數")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="重複懲罰")
    parser.add_argument("--no_sample", action="store_true", help="禁用採樣（使用貪婪解碼）")

    # 0.5.0 量化/架構覆寫
    parser.add_argument("--quantize_kv_cache", action="store_true", help="啟用 KV cache 量化")
    parser.add_argument("--kv_cache_bits", type=int, choices=[3, 4, 8, 16, 32], help="KV cache 位寬")
    parser.add_argument("--kv_quant_group_size", type=int, help="KV 量化分組大小")
    parser.add_argument("--kv_residual_sign_correction", action="store_true", help="啟用 1-bit 殘差符號修正")
    parser.add_argument("--num_key_value_heads", type=int, help="覆寫 GQA key/value 頭數")
    
    # 種子控制
    parser.add_argument("--seed", type=int, default=-1, help="基礎種子（-1為隨機種子）")
    parser.add_argument("--history_rounds", type=int, default=3, help="Web 對話保留歷史輪次")
    
    # 調試選項
    parser.add_argument("--verbose", action="store_true", help="啟用詳細輸出（包括內存監控）")
    parser.add_argument("--stateless_chat", action="store_true", default=True, help="臨時聊天模式：每一輪都視為新對話")
    
    # 其他
    parser.add_argument("--device", type=str, default="auto", help="指定設備")
    parser.add_argument("--log_level", type=str, default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日誌級別")
    parser.add_argument("--web_host", type=str, default="127.0.0.1", help="Web UI 監聽位址（mode=web）")
    parser.add_argument("--web_port", type=int, default=8501, help="Web UI 埠號（mode=web）")
    
    args = parser.parse_args()
    
    # 設置日誌
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # 自動查找分詞器
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
        
        # 創建推理配置
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
            silent_mode=not args.verbose,  # 默認靜默模式，除非用戶指定verbose
            quantize_kv_cache=args.quantize_kv_cache if args.quantize_kv_cache else None,
            kv_cache_bits=args.kv_cache_bits,
            kv_quant_group_size=args.kv_quant_group_size,
            kv_residual_sign_correction=(True if args.kv_residual_sign_correction else None),
            num_key_value_heads=args.num_key_value_heads,
            stateless_chat=args.stateless_chat,
        )
        
        # 創建推理器
        inference = NagatoSakuraInference(config)
        
        # 運行指定模式
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
    print("🌸 長門櫻流式推理程序 🌸")
    print("=" * 50)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("CUDA: 不可用")
    print("=" * 50)
    main()
