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
    from src.nagato_sakura_model import NagatoSakuraForCausalLM, NSConfig
except ImportError as e:
    print(f"錯誤：無法導入自定義模型。錯誤詳情: {e}")
    sys.exit(1)

@dataclass
class TestConfig:
    """測試配置"""
    model_path: str
    tokenizer_path: str
    device: str = "auto"
    max_length: int = 512
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    num_beams: int = 1
    do_sample: bool = True

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

class NagatoSakuraTester:
    """長門櫻模型測試器"""
    
    def __init__(self, test_config: TestConfig):
        self.config = test_config
        self.logger = setup_logging()
        
        # 設備設置
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        self.logger.info(f"使用設備: {self.device}")
        
        # 模型和分詞器
        self.model: Optional[NagatoSakuraForCausalLM] = None
        self.tokenizer: Optional[PreTrainedTokenizerFast] = None
        
        # 加載模型和分詞器
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        """加載模型和分詞器"""
        try:
            # 加載分詞器
            self.logger.info(f"加載分詞器: {self.config.tokenizer_path}")
            from tokenizers import Tokenizer
            tokenizer_obj = Tokenizer.from_file(self.config.tokenizer_path)
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer_obj,
                unk_token="<unk>",
                pad_token="<pad>", 
                bos_token="<s>",
                eos_token="</s>",
                mask_token="<mask>"
            )
            self.logger.info(f"分詞器加載完成，詞彙量: {self.tokenizer.vocab_size}")
            
            # 加載模型配置
            config_path = Path(self.config.model_path) / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                model_config = NSConfig(**config_dict)
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
            
            # 加載權重
            model_path = Path(self.config.model_path) / "model.pt"
            if model_path.exists():
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.logger.info("模型權重加載完成")
            else:
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
    
    def generate_response(self, prompt: str, **generation_kwargs) -> str:
        """生成回應"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("模型或分詞器未初始化")
        
        try:
            # 準備輸入
            input_ids = self._prepare_input(prompt)
            input_length = input_ids.shape[1]
            
            # 檢查輸入長度
            if input_length >= self.config.max_length:
                return "輸入太長，請縮短後重試。"
            
            # 設置生成參數
            gen_kwargs = {
                'max_length': min(self.config.max_length, input_length + 100),  # 限制生成長度
                'temperature': self.config.temperature,
                'top_k': self.config.top_k,
                'top_p': self.config.top_p,
                'repetition_penalty': self.config.repetition_penalty,
                'do_sample': self.config.do_sample,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
            }
            gen_kwargs.update(generation_kwargs)
            
            # 生成
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    **gen_kwargs
                )
                
                # 檢查生成結果
                if generated_ids is None or generated_ids.shape[1] <= input_length:
                    return "生成失敗，請重試。"
                
                # 解碼響應（只保留新生成的部分）
                response_ids = generated_ids[0, input_length:]
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                
                return response.strip() if response.strip() else "生成了空響應，請重試。"
                
        except Exception as e:
            self.logger.error(f"生成過程中發生錯誤: {e}", exc_info=True)
            return f"生成失敗: {str(e)}"
    
    def interactive_chat(self):
        """交互式對話"""
        self.logger.info("🌸 歡迎與長門櫻對話！輸入 'quit' 或 'exit' 退出")
        print("=" * 60)
        print("💡 提示: 如果遇到錯誤，可以嘗試:")
        print("   - 縮短輸入長度")
        print("   - 使用 /set temperature 0.3 調低溫度")
        print("   - 使用 /help 查看更多命令")
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
                    self._handle_command(user_input)
                    continue
                
                # 生成響應
                print("\n🌸 長門櫻正在思考中...")
                start_time = time.time()
                
                response = self.generate_response(user_input)
                
                generation_time = time.time() - start_time
                
                # 顯示響應
                print(f"\n🌸 長門櫻: {response}")
                print(f"\n⏱️  生成時間: {generation_time:.2f}秒")
                
                # 記錄對話歷史
                conversation_history.append({
                    "user": user_input,
                    "assistant": response,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "generation_time": generation_time
                })
                
            except KeyboardInterrupt:
                print("\n\n🌸 長門櫻: 檢測到中斷，再見主人！")
                break
            except Exception as e:
                print(f"\n❌ 錯誤: {e}")
                self.logger.error(f"交互過程中發生錯誤: {e}", exc_info=True)
        
        # 保存對話歷史
        if conversation_history:
            self._save_conversation_history(conversation_history)
    
    def _handle_command(self, command: str):
        """處理特殊命令"""
        cmd_parts = command[1:].split()
        cmd = cmd_parts[0].lower()
        
        if cmd == "help":
            print("""
可用命令:
/help - 顯示此幫助
/config - 顯示當前配置
/set <param> <value> - 設置生成參數
/test - 運行快速測試
/benchmark - 運行性能測試
/debug - 運行調試模式測試
""")
        elif cmd == "config":
            print(f"""
當前配置:
- 最大長度: {self.config.max_length}
- 溫度: {self.config.temperature}
- Top-k: {self.config.top_k}
- Top-p: {self.config.top_p}
- 重複懲罰: {self.config.repetition_penalty}
- 採樣: {self.config.do_sample}
""")
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
                    max_length=50,
                    temperature=0.3,
                    do_sample=False
                )
                print(f"響應: {response}")
            except Exception as e:
                print(f"錯誤: {e}")
                self.logger.error(f"調試測試失敗: {e}", exc_info=True)
    
    def _save_conversation_history(self, history: List[Dict]):
        """保存對話歷史"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_history_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            
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
            response = self.generate_response(prompt, max_length=100)
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
                            max_length=100,
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

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="長門櫻模型測試程序")
    
    # 必需參數
    parser.add_argument("--model_path", type=str, default="NS-LLM-V1-1024/best_model", help="模型路徑")
    parser.add_argument("--tokenizer_path", type=str, help="分詞器路徑（如果未指定，將在模型路徑中查找）")
    
    # 測試模式
    parser.add_argument("--mode", type=str, default="interactive", 
                       choices=["interactive", "test", "benchmark", "debug"],
                       help="測試模式")
    
    # 生成參數
    parser.add_argument("--max_length", type=int, default=1024, help="最大生成長度")
    parser.add_argument("--temperature", type=float, default=0.7, help="溫度參數")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k參數")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p參數")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重複懲罰")
    parser.add_argument("--no_sample", action="store_true", help="禁用採樣（使用貪婪解碼）")
    
    # 其他
    parser.add_argument("--device", type=str, default="auto", help="指定設備")
    parser.add_argument("--log_level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日誌級別")
    
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
            do_sample=not args.no_sample
        )
        
        # 創建測試器
        tester = NagatoSakuraTester(test_config)
        
        # 運行指定模式
        if args.mode == "interactive":
            tester.interactive_chat()
        elif args.mode == "test":
            tester.run_quick_test()
        elif args.mode == "benchmark":
            tester.run_benchmark()
        elif args.mode == "debug":
            tester._debug_test()
        
    except Exception as e:
        logger.error(f"程序執行失敗: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    print("🌸 長門櫻模型測試程序 (修復版) 🌸")
    print("=" * 50)
    
    # 系統信息
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA: 不可用")
    
    print("=" * 50)
    
    main()