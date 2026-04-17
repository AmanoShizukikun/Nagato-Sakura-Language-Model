import os
import sys
import json
import time
import logging
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import numpy as np

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
class TestConfig:
    """測試配置"""
    model_path: str
    tokenizer_path: str
    device: str = "auto"
    max_length: int = 1024
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    num_beams: int = 1
    do_sample: bool = True
    generation_seed: int = -1  # -1 表示隨機種子

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
    
    # 清除現有處理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 控制台處理器（彩色）
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

class DeterministicSeedManager:
    """確定性種子管理器，確保生成一致性"""
    
    def __init__(self, seed: int = -1):
        """
        初始化種子管理器
        Args:
            seed: 種子值，-1表示使用隨機種子
        """
        self.base_seed = seed if seed != -1 else int(time.time() * 1000) % (2**32)
        self.current_step = 0
        self.initial_state = None
        
    def set_global_seed(self):
        """設置全局種子狀態"""
        # 保存PyTorch隨機狀態
        torch.manual_seed(self.base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.base_seed)
        
        # 保存numpy隨機狀態
        np.random.seed(self.base_seed)
        
        # 重置步數
        self.current_step = 0
        
        # 保存初始狀態
        self.initial_state = {
            'torch_state': torch.get_rng_state(),
            'cuda_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy_state': np.random.get_state()
        }
    
    def restore_initial_state(self):
        """恢復到初始狀態"""
        if self.initial_state is not None:
            torch.set_rng_state(self.initial_state['torch_state'])
            if self.initial_state['cuda_state'] is not None:
                torch.cuda.set_rng_state_all(self.initial_state['cuda_state'])
            np.random.set_state(self.initial_state['numpy_state'])
            self.current_step = 0
    
    def get_deterministic_seed(self) -> int:
        """獲取當前步驟的確定性種子"""
        return (self.base_seed + self.current_step) % (2**32)
    
    def next_step(self):
        """前進到下一步"""
        self.current_step += 1

class NagatoSakuraTester:
    """長門櫻模型測試器 - 支援流式與傳統模式，確保生成一致性"""
    
    def __init__(self, test_config: TestConfig):
        self.config = test_config
        self.logger = setup_logging()
        
        # 種子管理器
        self.seed_manager = DeterministicSeedManager(test_config.generation_seed)
        
        # 設備設置
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        self.logger.info(f"使用設備: {self.device}")
        if self.config.generation_seed == -1:
            self.logger.info(f"使用隨機種子: {self.seed_manager.base_seed}")
        else:
            self.logger.info(f"使用固定種子: {self.seed_manager.base_seed}")
        
        # 模型和分詞器
        self.model: Optional[NagatoSakuraForCausalLM] = None
        self.tokenizer_manager: Optional[TokenizerManager] = None
        self.tokenizer: Optional[PreTrainedTokenizerFast] = None
        
        # 對話歷史
        self.conversation_history: Optional[ConversationHistory] = None
        
        # 加載模型和分詞器
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        """加載模型和分詞器"""
        try:
            # 使用TokenizerManager加載分詞器
            self.logger.info(f"加載分詞器: {self.config.tokenizer_path}")
            self.tokenizer_manager = TokenizerManager(Path(self.config.tokenizer_path))
            self.tokenizer_manager.load_tokenizer()
            self.tokenizer = self.tokenizer_manager.transformers_tokenizer
            self.logger.info(f"分詞器加載完成，詞彙量: {self.tokenizer.vocab_size}")
            
            # 加載模型配置
            config_path = Path(self.config.model_path) / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                model_config = NSConfig.from_dict(config_dict)
            else:
                # 使用默認配置
                self.logger.warning("未找到配置文件，使用默認配置")
                model_config = NSConfig(vocab_size=self.tokenizer.vocab_size)
            
            # 確保配置與分詞器一致
            model_config.vocab_size = self.tokenizer.vocab_size
            model_config.pad_token_id = self.tokenizer.pad_token_id
            model_config.bos_token_id = self.tokenizer.bos_token_id
            model_config.eos_token_id = self.tokenizer.eos_token_id
            
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
            
            # 計算參數數量
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"模型參數數量: {total_params/1e6:.2f}M")
            
        except Exception as e:
            self.logger.error(f"加載模型或分詞器失敗: {e}")
            raise
    
    def _prepare_input(self, prompt: str) -> torch.Tensor:
        """準備輸入"""
        # 添加BOS令牌
        bos = self.tokenizer.bos_token or "<s>"
        formatted_prompt = f"{bos}{prompt}"
        
        # 編碼
        input_ids = self.tokenizer.encode(
            formatted_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        return input_ids.to(self.device)

    def generate_response_unified(self, prompt: str, use_streaming: bool = False, **generation_kwargs) -> Union[str, List[Dict]]:
        """
        統一的生成函數，確保流式和非流式的完全一致性
        
        Args:
            prompt: 輸入提示
            use_streaming: 是否使用流式生成
            **generation_kwargs: 生成參數
        
        Returns:
            非流式: 返回字符串
            流式: 返回生成步驟列表
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("模型或分詞器未初始化")
        
        # 設置確定性種子
        self.seed_manager.set_global_seed()
        
        try:
            # 準備輸入
            input_ids = self._prepare_input(prompt)
            input_length = input_ids.shape[1]
            
            # 檢查輸入長度
            if input_length >= self.config.max_length:
                if use_streaming:
                    return [{"delta": "輸入太長，請縮短後重試。", "finished": True, "stop_reason": "max_length"}]
                else:
                    return "輸入太長，請縮短後重試。"
            
            # 設置生成參數
            gen_kwargs = {
                'max_new_tokens': min(self.config.max_length - input_length, 100),
                'temperature': self.config.temperature,
                'top_k': self.config.top_k,
                'top_p': self.config.top_p,
                'repetition_penalty': self.config.repetition_penalty,
                'do_sample': self.config.do_sample,
                'generation_seed': self.seed_manager.base_seed,
            }
            gen_kwargs.update(generation_kwargs)
            
            with torch.no_grad():
                if use_streaming:
                    return self._generate_streaming(input_ids, **gen_kwargs)
                else:
                    return self._generate_traditional(input_ids, **gen_kwargs)
                    
        except Exception as e:
            self.logger.error(f"生成過程中發生錯誤: {e}", exc_info=True)
            error_msg = f"生成失敗: {str(e)}"
            if use_streaming:
                return [{"delta": error_msg, "finished": True, "stop_reason": "error"}]
            else:
                return error_msg

    def _generate_traditional(self, input_ids: torch.Tensor, **gen_kwargs) -> str:
        """傳統生成（非流式）"""
        # 恢復初始種子狀態
        self.seed_manager.restore_initial_state()
        
        generated_ids = self.model.generate(
            input_ids=input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs
        )
        
        # 檢查生成結果
        if generated_ids is None or generated_ids.shape[1] <= input_ids.shape[1]:
            return "生成失敗，請重試。"
        
        # 解碼響應（只保留新生成的部分）
        response_ids = generated_ids[0, input_ids.shape[1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        return response.strip() if response.strip() else "生成了空響應，請重試。"

    def _generate_streaming(self, input_ids: torch.Tensor, **gen_kwargs) -> List[Dict]:
        """流式生成（修復版）"""
        # 恢復初始種子狀態
        self.seed_manager.restore_initial_state()
        
        streaming_results = []
        full_response = ""
        
        try:
            for output in self.model.stream_generate(
                input_ids=input_ids,
                **gen_kwargs
            ):
                if output["finished"]:
                    streaming_results.append({
                        "delta": "",
                        "finished": True,
                        "stop_reason": output.get("stop_reason", "completed"),
                        "tokens_generated": output.get("tokens_generated", 0),
                        "full_response": full_response
                    })
                    break
                
                # 解碼並累積新token - 修復邏輯
                if "generated_ids" in output and output["generated_ids"] is not None:
                    try:
                        new_tokens = output["generated_ids"][:, input_ids.shape[1]:]
                        if new_tokens.shape[1] > 0:
                            # 解碼完整序列以確保一致性
                            current_response = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
                            # 計算增量
                            if len(current_response) > len(full_response):
                                delta = current_response[len(full_response):]
                                full_response = current_response
                                streaming_results.append({
                                    "delta": delta,
                                    "finished": False,
                                    "full_response": full_response
                                })
                    except Exception as e:
                        self.logger.warning(f"解碼token失敗: {e}")
                        continue
        
        except Exception as e:
            streaming_results.append({
                "delta": f"流式生成錯誤: {e}",
                "finished": True,
                "stop_reason": "error"
            })
        
        return streaming_results

    def generate_response(self, prompt: str, **generation_kwargs) -> str:
        """傳統生成回應（非流式），保持向後兼容"""
        return self.generate_response_unified(prompt, use_streaming=False, **generation_kwargs)

    def traditional_interactive_chat(self):
        """傳統交互式對話（非流式）"""
        self.logger.info("🌸 歡迎與長門櫻對話！（傳統模式）輸入 'quit' 或 'exit' 退出")
        print("=" * 60)
        print("💡 提示: 這是傳統對話模式，會等待完整回應後一次性顯示")
        print("   - 如果遇到錯誤，可以嘗試:")
        print("   - 縮短輸入長度")
        print("   - 使用 /set temperature 0.3 調低溫度")
        print("   - 使用 /help 查看更多命令")
        print(f"   - 當前種子: {self.seed_manager.base_seed}")
        print("=" * 60)
        
        conversation_history = []
        
        while True:
            try:
                # 獲取用戶輸入
                user_input = input("\n👤 主人: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', '退出', '結束']:
                    print("\n🌸 長門櫻: 謝謝主人的陪伴，再見！")
                    break
                
                # 特殊命令處理
                if user_input.startswith('/'):
                    self._handle_traditional_command(user_input)
                    continue
                
                # 生成響應
                print("\n🌸 長門櫻正在思考中...")
                start_time = time.time()
                
                response = self.generate_response(user_input)
                
                generation_time = time.time() - start_time
                
                # 顯示響應
                print(f"\n🌸 長門櫻: {response}")
                print(f"\n⏱️  生成時間: {generation_time:.2f}秒")
                print(f"🎲 使用種子: {self.seed_manager.base_seed}")
                
                # 記錄對話歷史
                conversation_history.append({
                    "user": user_input,
                    "assistant": response,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "generation_time": generation_time,
                    "seed": self.seed_manager.base_seed
                })
                
            except KeyboardInterrupt:
                print("\n\n🌸 長門櫻: 檢測到中斷，再見主人！")
                break
            except Exception as e:
                print(f"\n❌ 錯誤: {e}")
                self.logger.error(f"交互過程中發生錯誤: {e}", exc_info=True)
        
        # 保存對話歷史
        if conversation_history:
            self._save_traditional_conversation_history(conversation_history)

    def _handle_traditional_command(self, command: str):
        """處理傳統模式的特殊命令"""
        cmd_parts = command[1:].split()
        cmd = cmd_parts[0].lower()
        
        if cmd == "help":
            print("""
🌸 可用命令（傳統模式）:
/help - 顯示此幫助
/config - 顯示當前配置
/set <param> <value> - 設置生成參數
/seed <seed> - 設置種子（-1為隨機）
/test_consistency - 測試生成一致性
/test - 運行快速測試
/benchmark - 運行性能測試
/debug - 運行調試模式測試
""")
        elif cmd == "config":
            print(f"""
🔧 當前配置:
- 最大長度: {self.config.max_length}
- 溫度: {self.config.temperature}
- Top-k: {self.config.top_k}
- Top-p: {self.config.top_p}
- 重複懲罰: {self.config.repetition_penalty}
- 採樣: {self.config.do_sample}
- 設備: {self.device}
- 當前種子: {self.seed_manager.base_seed}
""")
        elif cmd == "seed" and len(cmd_parts) >= 2:
            try:
                new_seed = int(cmd_parts[1])
                self.seed_manager = DeterministicSeedManager(new_seed)
                self.config.generation_seed = new_seed
                print(f"✅ 種子已設置為 {self.seed_manager.base_seed}")
            except ValueError:
                print(f"❌ 無效的種子值: {cmd_parts[1]}")
        elif cmd == "test_consistency":
            self.test_generation_consistency()
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
                    self.config.max_length = max(50, min(2048, int(value)))
                print(f"✅ {param} 已設置為 {value}")
            except ValueError:
                print(f"❌ 無效的值: {cmd_parts[2]}")
        elif cmd == "test":
            self.run_quick_test()
        elif cmd == "benchmark":
            self.run_benchmark()
        elif cmd == "debug":
            self._debug_test()
        else:
            print(f"❌ 未知命令: {command}")

    def test_generation_consistency(self):
        """測試生成一致性（修復版）"""
        print("\n🧪 測試生成一致性...")
        
        test_prompts = [
            "你好，長門櫻",
            "今天天氣如何？",
            "請說一句鼓勵的話"
        ]
        
        test_seed = 12345
        all_passed = True
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n測試 {i}/{len(test_prompts)}: {prompt}")
            
            # 設置測試種子
            original_seed = self.seed_manager.base_seed
            
            # 傳統生成
            self.seed_manager = DeterministicSeedManager(test_seed)
            traditional_response = self.generate_response_unified(prompt, use_streaming=False, max_new_tokens=50)
            
            # 重置種子進行流式生成
            self.seed_manager = DeterministicSeedManager(test_seed)
            streaming_results = self.generate_response_unified(prompt, use_streaming=True, max_new_tokens=50)
            
            # 從流式結果中提取完整響應
            streaming_response = ""
            for result in streaming_results:
                if result.get("delta"):
                    streaming_response += result["delta"]
            
            # 比較結果
            if traditional_response.strip() == streaming_response.strip():
                print(f"✅ 一致性測試通過")
                print(f"   響應: {traditional_response}")
            else:
                print(f"❌ 一致性測試失敗")
                print(f"   傳統模式: '{traditional_response}'")
                print(f"   流式模式: '{streaming_response}'")
                all_passed = False
            
            # 恢復原始種子
            self.seed_manager = DeterministicSeedManager(original_seed)
        
        return all_passed

    def interactive_streaming_chat(self):
        """交互式流式對話"""
        self.logger.info("🌸 歡迎與長門櫻進行流式對話！輸入 'quit' 或 'exit' 退出")
        print("=" * 60)
        print("💡 提示: 這是流式輸出模式，您將看到實時生成的回應")
        print("   - 輸入 /help 查看命令")
        print("   - 輸入 /clear 清空對話歷史")
        print("   - 輸入 /config 查看當前配置")
        print(f"   - 當前種子: {self.seed_manager.base_seed}")
        print("=" * 60)
        
        conversation_history = ConversationHistory()
        
        while True:
            try:
                # 獲取用戶輸入
                user_input = input("\n👤 主人: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', '退出', '結束']:
                    print("\n🌸 長門櫻: 謝謝主人的陪伴，再見！")
                    break
                
                # 特殊命令處理
                if user_input.startswith('/'):
                    self._handle_streaming_command(user_input, conversation_history)
                    continue
                
                # 生成流式響應
                print("\n🌸 長門櫻: ", end="", flush=True)
                start_time = time.time()
                full_response = ""
                
                try:
                    for output in self.model.chat_stream(
                        tokenizer=self.tokenizer,
                        query=user_input,
                        history=conversation_history,
                        max_new_tokens=1024,
                        temperature=self.config.temperature,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p,
                        generation_seed=self.seed_manager.base_seed
                    ):
                        if output["finished"]:
                            end_time = time.time()
                            print(f"\n⏱️  生成時間: {end_time - start_time:.2f}秒，"
                                  f"{output.get('tokens_generated', 0)} tokens")
                            print(f"🎲 使用種子: {self.seed_manager.base_seed}")
                            conversation_history = output.get("history", conversation_history)
                            break
                        
                        if output["delta"]:
                            print(output["delta"], end="", flush=True)
                            full_response += output["delta"]
                
                except Exception as e:
                    print(f"\n❌ 生成過程中出錯: {e}")
                    self.logger.error(f"流式生成錯誤: {e}", exc_info=True)
                
            except KeyboardInterrupt:
                print("\n\n🌸 長門櫻: 檢測到中斷，再見主人！")
                break
            except Exception as e:
                print(f"\n❌ 錯誤: {e}")
                self.logger.error(f"交互過程中發生錯誤: {e}", exc_info=True)
        
        # 保存對話歷史
        self._save_conversation_history(conversation_history)
    
    def _handle_streaming_command(self, command: str, conversation_history: ConversationHistory):
        """處理流式模式的特殊命令"""
        cmd_parts = command[1:].split()
        cmd = cmd_parts[0].lower()
        
        if cmd == "help":
            print("""
🌸 可用命令（流式模式）:
/help - 顯示此幫助
/config - 顯示當前配置
/clear - 清空對話歷史
/history - 顯示對話歷史
/set <param> <value> - 設置生成參數
/seed <seed> - 設置種子（-1為隨機）
/test_consistency - 測試生成一致性
/test - 運行快速測試
/benchmark - 運行性能測試
""")
        elif cmd == "config":
            print(f"""
🔧 當前配置:
- 最大長度: {self.config.max_length}
- 溫度: {self.config.temperature}
- Top-k: {self.config.top_k}
- Top-p: {self.config.top_p}
- 重複懲罰: {self.config.repetition_penalty}
- 採樣: {self.config.do_sample}
- 設備: {self.device}
- 當前種子: {self.seed_manager.base_seed}
""")
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
                self.seed_manager = DeterministicSeedManager(new_seed)
                self.config.generation_seed = new_seed
                print(f"✅ 種子已設置為 {self.seed_manager.base_seed}")
            except ValueError:
                print(f"❌ 無效的種子值: {cmd_parts[1]}")
        
        elif cmd == "test_consistency":
            self.test_generation_consistency()
        
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
                    self.config.max_length = max(50, min(2048, int(value)))
                print(f"✅ {param} 已設置為 {value}")
            except ValueError:
                print(f"❌ 無效的值: {cmd_parts[2]}")
        
        elif cmd == "test":
            self.run_quick_streaming_test()
        
        elif cmd == "benchmark":
            self.test_streaming_performance()
        
        else:
            print(f"❌ 未知命令: {command}")

    def _debug_test(self):
        """調試測試"""
        print("\n🔧 運行調試測試...")
        
        # 簡單測試
        simple_prompts = ["你好", "測試", "1+1=?"]
        
        for prompt in simple_prompts:
            print(f"\n測試提示: '{prompt}'")
            try:
                # 使用保守參數
                response = self.generate_response(
                    prompt,
                    max_new_tokens=50,
                    temperature=0.3,
                    do_sample=False
                )
                print(f"響應: {response}")
            except Exception as e:
                print(f"錯誤: {e}")
                self.logger.error(f"調試測試失敗: {e}", exc_info=True)

    def _save_traditional_conversation_history(self, history: List[Dict]):
        """保存傳統對話歷史"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"traditional_conversation_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"對話歷史已保存: {filename}")
        except Exception as e:
            self.logger.error(f"保存對話歷史失敗: {e}")

    def _save_conversation_history(self, history: ConversationHistory):
        """保存對話歷史"""
        if not history.turns:
            return
            
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"streaming_conversation_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(history.to_dict(), f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"對話歷史已保存: {filename}")
        except Exception as e:
            self.logger.error(f"保存對話歷史失敗: {e}")

    def run_quick_test(self):
        """運行快速測試"""
        print("\n🧪 運行快速測試...")
        
        test_prompts = [
            "你好",
            "你是誰？",
            "今天天氣如何？",
            "1+1等於多少？",
            "請說一句話"
        ]
        
        results = []
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n測試 {i}/{len(test_prompts)}: {prompt}")
            
            start_time = time.time()
            response = self.generate_response(prompt, max_new_tokens=100)
            generation_time = time.time() - start_time
            
            print(f"回應: {response}")
            print(f"時間: {generation_time:.2f}秒")
            
            results.append({
                "prompt": prompt,
                "response": response,
                "time": generation_time
            })
        
        # 統計
        avg_time = np.mean([r["time"] for r in results])
        print(f"\n📊 測試完成 - 平均生成時間: {avg_time:.2f}秒")
        
        return results

    def run_benchmark(self):
        """運行性能基準測試"""
        print("\n⚡ 運行性能基準測試...")
        
        # 測試不同長度的輸入
        test_cases = [
            ("短文本", "你好"),
            ("中等文本", "請幫我寫一個短故事"),
            ("長文本", "主人，請詳細介紹人工智慧的發展歷史")
        ]
        
        # 測試不同的生成參數
        param_sets = [
            {"name": "保守生成", "temperature": 0.3, "top_p": 0.8, "do_sample": True},
            {"name": "平衡生成", "temperature": 0.8, "top_p": 0.9, "do_sample": True},
            {"name": "貪婪解碼", "temperature": 1.0, "do_sample": False}
        ]
        
        results = []
        
        for case_name, prompt in test_cases:
            print(f"\n📝 測試用例: {case_name}")
            case_results = []
            
            for params in param_sets:
                print(f"  參數集: {params['name']}")
                
                # 多次運行取平均
                times = []
                responses = []
                
                for run in range(2):  # 減少運行次數以節省時間
                    try:
                        start_time = time.time()
                        response = self.generate_response(
                            prompt,
                            max_new_tokens=100,
                            temperature=params['temperature'],
                            top_p=params.get('top_p', 0.9),
                            do_sample=params['do_sample']
                        )
                        generation_time = time.time() - start_time
                        
                        times.append(generation_time)
                        responses.append(response)
                    except Exception as e:
                        print(f"    運行 {run+1} 失敗: {e}")
                        continue
                
                if times:
                    avg_time = np.mean(times)
                    std_time = np.std(times) if len(times) > 1 else 0
                    print(f"    平均時間: {avg_time:.2f}±{std_time:.2f}秒")
                    
                    case_results.append({
                        "params": params,
                        "avg_time": avg_time,
                        "std_time": std_time,
                        "responses": responses
                    })
            
            results.append({
                "case": case_name,
                "prompt": prompt,
                "results": case_results
            })
        
        # 總結報告
        print("\n📊 基準測試報告:")
        for result in results:
            print(f"\n{result['case']}:")
            for case_result in result['results']:
                params = case_result['params']
                print(f"  {params['name']}: {case_result['avg_time']:.2f}±{case_result['std_time']:.2f}秒")
        
        return results

    # 保持原有的流式測試方法...
    def test_basic_streaming(self):
        """測試基本流式生成（修復版）"""
        self.logger.info("🧪 測試基本流式生成功能...")
        
        test_prompt = "你好，長門櫻"
        print(f"\n輸入: {test_prompt}")
        print("流式輸出: ", end="", flush=True)
        
        # 編碼輸入
        input_ids = self._prepare_input(test_prompt)
        
        full_generated_text = ""
        start_time = time.time()
        
        try:
            for output in self.model.stream_generate(
                input_ids=input_ids,
                max_new_tokens=100,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                generation_seed=self.seed_manager.base_seed
            ):
                if output["finished"]:
                    end_time = time.time()
                    print(f"\n✅ 基本流式生成測試完成!")
                    print(f"   生成時間: {end_time - start_time:.2f}秒")
                    print(f"   生成token數: {output['tokens_generated']}")
                    print(f"   停止原因: {output['stop_reason']}")
                    if output['tokens_generated'] > 0:
                        print(f"   平均速度: {output['tokens_generated']/(end_time - start_time):.1f} tokens/秒")
                    break
                
                # 解碼並顯示新token
                if "generated_ids" in output and output["generated_ids"] is not None:
                    try:
                        new_tokens = output["generated_ids"][:, input_ids.shape[1]:]
                        if new_tokens.shape[1] > 0:
                            current_text = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
                            if len(current_text) > len(full_generated_text):
                                delta = current_text[len(full_generated_text):]
                                print(delta, end="", flush=True)
                                full_generated_text = current_text
                    except Exception as e:
                        self.logger.warning(f"解碼token失敗: {e}")
                        continue
        
        except Exception as e:
            print(f"\n❌ 基本流式生成測試失敗: {e}")
            return False
        
        return True

    def run_quick_streaming_test(self):
        """運行快速流式測試（修復版）"""
        print("\n🧪 運行快速流式測試...")
        
        test_prompts = [
            "你好",
            "你是誰？",
            "今天天氣如何？",
            "1+1等於多少？"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n測試 {i}/{len(test_prompts)}: {prompt}")
            print("回應: ", end="", flush=True)
            
            input_ids = self._prepare_input(prompt)
            start_time = time.time()
            full_text = ""
            
            try:
                for output in self.model.stream_generate(
                    input_ids=input_ids,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    generation_seed=self.seed_manager.base_seed
                ):
                    if output["finished"]:
                        end_time = time.time()
                        print(f"\n   [完成，{output['tokens_generated']} tokens，{end_time - start_time:.2f}秒]")
                        break
                    
                    if "generated_ids" in output and output["generated_ids"] is not None:
                        try:
                            new_tokens = output["generated_ids"][:, input_ids.shape[1]:]
                            if new_tokens.shape[1] > 0:
                                current_text = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
                                if len(current_text) > len(full_text):
                                    delta = current_text[len(full_text):]
                                    print(delta, end="", flush=True)
                                    full_text = current_text
                        except Exception as e:
                            continue
            
            except Exception as e:
                print(f"\n❌ 測試失敗: {e}")

    # 保持其他測試方法...
    def test_chat_streaming(self):
        """測試對話流式生成"""
        self.logger.info("🧪 測試對話流式生成功能...")
        
        test_queries = [
            "你是誰？",
            "你能做什麼？",
            "請說一句鼓勵的話",
            "今天天氣如何？",
            "1+1等於多少？"
        ]
        
        # 初始化對話歷史
        conversation_history = ConversationHistory()
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n對話 {i}:")
            print(f"👤 用戶: {query}")
            print("🌸 長門櫻: ", end="", flush=True)
            
            start_time = time.time()
            response = ""
            
            try:
                for output in self.model.chat_stream(
                    tokenizer=self.tokenizer,
                    query=query,
                    history=conversation_history,
                    max_new_tokens=100,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    generation_seed=self.seed_manager.base_seed
                ):
                    if output["finished"]:
                        end_time = time.time()
                        print(f"\n   [生成完成，{output.get('tokens_generated', 0)} tokens，"
                              f"{end_time - start_time:.2f}秒]")
                        conversation_history = output.get("history", conversation_history)
                        break
                    
                    if output["delta"]:
                        print(output["delta"], end="", flush=True)
                        response += output["delta"]
            
            except Exception as e:
                print(f"\n❌ 對話流式生成測試失敗: {e}")
                return False
        
        print("\n✅ 對話流式生成測試完成!")
        return True

    def test_streaming_performance(self):
        """測試流式輸出性能"""
        self.logger.info("⚡ 測試流式輸出性能...")
        
        test_cases = [
            ("短文本", "你好"),
            ("中文本", "請介紹一下人工智慧"),
            ("長文本", "請詳細說明深度學習的發展歷史和未來趨勢")
        ]
        
        results = []
        
        for case_name, prompt in test_cases:
            print(f"\n📝 測試 {case_name}: {prompt}")
            
            input_ids = self._prepare_input(prompt)
            
            # 測試多次取平均
            times = []
            token_counts = []
            
            for run in range(3):
                start_time = time.time()
                token_count = 0
                
                try:
                    for output in self.model.stream_generate(
                        input_ids=input_ids,
                        max_new_tokens=50,  # 限制長度以加速測試
                        temperature=0.7,
                        do_sample=True,
                        generation_seed=self.seed_manager.base_seed
                    ):
                        if output["finished"]:
                            end_time = time.time()
                            token_count = output.get("tokens_generated", 0)
                            break
                    
                    times.append(end_time - start_time)
                    token_counts.append(token_count)
                
                except Exception as e:
                    print(f"   運行 {run+1} 失敗: {e}")
                    continue
            
            if times and token_counts:
                avg_time = np.mean(times)
                avg_tokens = np.mean(token_counts)
                tokens_per_sec = avg_tokens / avg_time if avg_time > 0 else 0
                
                print(f"   平均時間: {avg_time:.2f}秒")
                print(f"   平均tokens: {avg_tokens:.1f}")
                print(f"   生成速度: {tokens_per_sec:.1f} tokens/秒")
                
                results.append({
                    "case": case_name,
                    "avg_time": avg_time,
                    "avg_tokens": avg_tokens,
                    "tokens_per_sec": tokens_per_sec
                })
        
        # 性能總結
        if results:
            print(f"\n📊 性能測試總結:")
            for result in results:
                print(f"   {result['case']}: {result['tokens_per_sec']:.1f} tokens/秒")
            
            overall_speed = np.mean([r['tokens_per_sec'] for r in results])
            print(f"   平均速度: {overall_speed:.1f} tokens/秒")
        
        return len(results) > 0

    def test_legacy_generation(self):
        """測試傳統生成方法（非流式）進行對比"""
        self.logger.info("🔄 測試傳統生成方法...")
        
        test_prompt = "你好，長門櫻"
        print(f"\n輸入: {test_prompt}")
        print("傳統生成: ", end="", flush=True)
        
        input_ids = self._prepare_input(test_prompt)
        
        try:
            start_time = time.time()
            
            # 使用傳統generate方法
            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + 50,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                generation_seed=self.seed_manager.base_seed
            )
            
            end_time = time.time()
            
            # 解碼結果
            response_ids = generated_ids[0, input_ids.shape[1]:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            print(response)
            print(f"\n✅ 傳統生成測試完成!")
            print(f"   生成時間: {end_time - start_time:.2f}秒")
            print(f"   生成token數: {len(response_ids)}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ 傳統生成測試失敗: {e}")
            return False

    def run_complete_streaming_test(self):
        """運行完整的流式輸出測試（修復版）"""
        print("🌸 長門櫻流式輸出測試程序 🌸")
        print("=" * 60)
        
        test_results = []
        
        # 1. 生成一致性測試
        try:
            print("🧪 測試生成一致性...")
            consistency_passed = self.test_generation_consistency()
            test_results.append(("生成一致性", consistency_passed))
        except Exception as e:
            print(f"❌ 生成一致性測試異常: {e}")
            test_results.append(("生成一致性", False))
        
        print("\n" + "-" * 60)
        
        # 2. 基本流式生成測試
        try:
            result = self.test_basic_streaming()
            test_results.append(("基本流式生成", result))
        except Exception as e:
            print(f"❌ 基本流式生成測試異常: {e}")
            test_results.append(("基本流式生成", False))
        
        print("\n" + "-" * 60)
        
        # 3. 對話流式生成測試
        try:
            result = self.test_chat_streaming()
            test_results.append(("對話流式生成", result))
        except Exception as e:
            print(f"❌ 對話流式生成測試異常: {e}")
            test_results.append(("對話流式生成", False))
        
        print("\n" + "-" * 60)
        
        # 4. 性能測試
        try:
            result = self.test_streaming_performance()
            test_results.append(("性能測試", result))
        except Exception as e:
            print(f"❌ 性能測試異常: {e}")
            test_results.append(("性能測試", False))
        
        print("\n" + "-" * 60)
        
        # 5. 傳統生成對比測試
        try:
            result = self.test_legacy_generation()
            test_results.append(("傳統生成對比", result))
        except Exception as e:
            print(f"❌ 傳統生成測試異常: {e}")
            test_results.append(("傳統生成對比", False))
        
        # 測試總結
        print("\n" + "=" * 60)
        print("🎊 流式輸出測試總結:")
        
        passed_tests = 0
        for test_name, result in test_results:
            status = "✅ 通過" if result else "❌ 失敗"
            print(f"   {test_name}: {status}")
            if result:
                passed_tests += 1
        
        print(f"\n總結: {passed_tests}/{len(test_results)} 項測試通過")
        print(f"🎲 測試種子: {self.seed_manager.base_seed}")
        
        if passed_tests == len(test_results):
            print("🎉 恭喜！您的模型完全支援流式輸出功能並且生成一致性良好！")
            print("💡 提示: 您可以使用 --mode streaming 開始流式對話")
        elif passed_tests > 0:
            print("⚠️  部分流式輸出功能正常，建議檢查失敗的測試")
        else:
            print("❌ 流式輸出功能存在問題，請檢查模型和配置")
        
        print("=" * 60)

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="長門櫻模型測試程序 - 支援流式與傳統模式，確保生成一致性")
    
    # 必需參數
    parser.add_argument("--model_path", type=str, default="NS-LLM/checkpoint-epoch-10", help="模型路徑")
    parser.add_argument("--tokenizer_path", type=str, help="分詞器路徑（如果未指定，將在模型路徑中查找）")
    
    # 測試模式
    parser.add_argument("--mode", type=str, default="complete", 
                       choices=["traditional", "streaming", "complete", "basic", "chat", "performance", "legacy", "test", "benchmark", "debug", "consistency"],
                       help="測試模式")
    
    # 生成參數
    parser.add_argument("--max_length", type=int, default=1024, help="最大生成長度")
    parser.add_argument("--temperature", type=float, default=0.7, help="溫度參數")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k參數")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top-p參數")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重複懲罰")
    parser.add_argument("--no_sample", action="store_true", help="禁用採樣（使用貪婪解碼）")
    
    # 種子控制
    parser.add_argument("--seed", type=int, default=-1, help="生成種子（-1為隨機種子）")
    
    # 其他
    parser.add_argument("--device", type=str, default="auto", help="指定設備")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日誌級別")
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
        
        # 創建測試配置
        test_config = TestConfig(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=not args.no_sample,
            generation_seed=args.seed
        )
        
        # 創建測試器
        tester = NagatoSakuraTester(test_config)
        
        # 運行指定模式
        if args.mode == "traditional":
            tester.traditional_interactive_chat()
        elif args.mode == "streaming":
            tester.interactive_streaming_chat()
        elif args.mode == "complete":
            tester.run_complete_streaming_test()
        elif args.mode == "basic":
            tester.test_basic_streaming()
        elif args.mode == "chat":
            tester.test_chat_streaming()
        elif args.mode == "performance":
            tester.test_streaming_performance()
        elif args.mode == "legacy":
            tester.test_legacy_generation()
        elif args.mode == "test":
            tester.run_quick_test()
        elif args.mode == "benchmark":
            tester.run_benchmark()
        elif args.mode == "debug":
            tester._debug_test()
        elif args.mode == "consistency":
            tester.test_generation_consistency()
        
    except Exception as e:
        logger.error(f"程序執行失敗: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    print("🌸 長門櫻模型測試程序 🌸")
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