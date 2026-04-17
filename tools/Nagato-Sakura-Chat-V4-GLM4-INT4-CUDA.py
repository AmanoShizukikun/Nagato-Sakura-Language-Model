import os
import torch
import gc
from threading import Thread
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, AutoModelForCausalLM, BitsAndBytesConfig
from opencc import OpenCC

# 預設模型路徑 - 可以是本地路徑或 HuggingFace 模型名稱
MODEL_PATH = os.environ.get("GLM_MODEL_PATH", "THUDM/GLM-4-9B-0414")

# 如果你想直接使用本地模型，請將下面這行取消註解並修改路徑
MODEL_PATH = r"C:\Users\timac\Documents\model\GLM-4-9B-0414"

def check_model_path(model_path):
    """檢查模型路徑是否存在本地文件"""
    if os.path.exists(model_path):
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            return True
    return False

class StopOnTokens(StoppingCriteria):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.stop_ids = self.model.config.eos_token_id
        if not isinstance(self.stop_ids, (list, tuple)):
            self.stop_ids = [self.stop_ids]
        self.stop_ids_tensor = torch.tensor(self.stop_ids, device=self.model.device)
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_token = input_ids[0][-1]
        return last_token.item() in self.stop_ids

class Chat:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if check_model_path(MODEL_PATH):
            print("模型路徑驗證成功")
        else:
            print(f"將從 HuggingFace 下載模型: {MODEL_PATH}")
        print(f"正在載入模型: {MODEL_PATH}")
        is_local_model = check_model_path(MODEL_PATH)
        load_kwargs = {}
        if is_local_model:
            load_kwargs["local_files_only"] = True
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, **load_kwargs)
            print("分詞器載入成功")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, 
                bnb_4bit_quant_storage=torch.uint8
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, 
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True, 
                trust_remote_code=True, 
                quantization_config=quantization_config,
                **load_kwargs
            ).eval()
            print("模型載入成功")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            print(f"載入失敗: {e}")
            if is_local_model:
                print("請檢查本地模型路徑是否正確")
            else:
                print("請檢查網路連接或嘗試使用本地模型")
            raise e
        self.converter = OpenCC('s2t')
        self.history = []
        self.stop = StopOnTokens(self.model)
        
    def chat(self, user_input):
        self.history.append([user_input, ""])
        messages = self._build_messages()
        with torch.no_grad():
            model_inputs = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True, 
                return_dict=True, 
                return_tensors="pt"
            ).to(self.model.device, non_blocking=True)
        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer, 
            timeout=120, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        generate_kwargs = {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "streamer": streamer,
            "max_new_tokens": 8192,
            "do_sample": True,
            "top_p": 0.8,
            "temperature": 0.6,
            "stopping_criteria": StoppingCriteriaList([self.stop]),
            "repetition_penalty": 1.0,
            "eos_token_id": self.model.config.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }
        with torch.no_grad():
            t = Thread(target=self.model.generate, kwargs=generate_kwargs)
            t.start()
        response_traditional = ""
        token_count = 0
        print("模型: ", end="", flush=True)
        for new_token in streamer:
            if new_token:
                new_token_converted = self.converter.convert(new_token)
                response_traditional += new_token_converted
                print(new_token_converted, end="", flush=True)
                self.history[-1][1] += new_token
                token_count += 1
        print("") 
        self.history[-1][1] = self.history[-1][1].strip()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return response_traditional, token_count
    
    def _build_messages(self):
        """構建消息列表的輔助方法"""
        messages = []
        for idx, (user_msg, model_msg) in enumerate(self.history):
            if idx == len(self.history) - 1 and not model_msg:
                messages.append({"role": "user", "content": user_msg})
                break
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if model_msg:
                messages.append({"role": "assistant", "content": model_msg})
        return messages

    def clear_history(self):
        """清空對話歷史並釋放記憶體"""
        self.history = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    def get_memory_usage(self):
        """獲取當前記憶體使用情況"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3     # GB
            return f"GPU 記憶體使用: {allocated:.2f}GB 已分配, {cached:.2f}GB 已快取"
        return "CPU 模式"

if __name__ == "__main__":
    print("=== Nagato Sakura Chat V4 優化版本 ===")
    try:
        chat_bot = Chat()
        print(f"初始{chat_bot.get_memory_usage()}")
    except Exception as e:
        print(f"初始化失敗: {e}")
        exit(1)
    print("\n<測試開始>：")
    print("- 輸入內容即可進行對話")
    print("- 輸入 'clear' 清空對話歷史")
    print("- 輸入 'memory' 查看記憶體使用情況") 
    print("- 輸入 'stop' 或 'exit' 終止程序")
    print("-" * 50)
    while True:
        try:
            user_input = input("用戶: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["stop", "exit"]:
                print("已終止對話")
                break
            elif user_input.lower() == "clear":
                chat_bot.clear_history()
                print("✓ 已清除記憶和快取")
            elif user_input.lower() == "memory":
                print(f"📊 {chat_bot.get_memory_usage()}")
            else:
                import time
                start_time = time.time()
                response, token_count = chat_bot.chat(user_input)
                end_time = time.time()
                response_time = end_time - start_time
                tokens_per_second = token_count / response_time if response_time > 0 else 0
                print(f"\n⏱️  回應時間: {response_time:.2f}秒")
                print(f"🚀 輸出速度: {tokens_per_second:.2f} tokens/秒 ({token_count} tokens)\n")
        except KeyboardInterrupt:
            print("\n\n👋 已終止對話")
            break
        except Exception as e:
            print(f"❌ 輸入處理時發生錯誤: {e}")
            print("請嘗試重新輸入或使用 'clear' 清除歷史記錄")
    print("正在清理記憶體...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("清理完成，程序結束。")