import os
import json
import random
import sys
import glob
import re
import gc
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import logging
import argparse
from threading import Thread
import queue
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from opencc import OpenCC


# --- 日誌配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_generation_sakura.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- 停止標準 ---
class StopOnTokens(StoppingCriteria):
    def __init__(self, tokenizer, stop_token_ids: List[int]):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_token = input_ids[0, -1].item()
        if last_token in self.stop_token_ids:
            return True
        return False

class NagatoSakuraDataGenerator:
    def __init__(self,
                 teacher_model_path: str,
                 teacher_model_type: str = "glm4",
                 device: Optional[str] = None,
                 persona_file: str = "generated_data/ns_v1.json",
                 opencc_config: str = 's2twp',
                 output_data_file: str = "generated_data/nagato_sakura_training_data_stream.json",
                 streamer_timeout: float = 120.0, # 新增 streamer_timeout 參數
                 output_flush_interval: int = 20,
                 cuda_cleanup_interval: int = 8
                 ):
        self.device = self._resolve_runtime_device(device)
        logger.info(f"資料生成器使用設備: {self.device}")

        self.teacher_model = None
        self.teacher_tokenizer = None
        self.teacher_stop_token_ids: List[int] = []
        self._load_teacher_model(teacher_model_path, teacher_model_type)

        self.stopping_criteria = None
        if self.teacher_model and self.teacher_tokenizer and self.teacher_stop_token_ids:
            self.stopping_criteria = StoppingCriteriaList([StopOnTokens(self.teacher_tokenizer, self.teacher_stop_token_ids)])
        else:
            logger.warning("未能完全初始化 StoppingCriteria，因為模型、分詞器或停止 ID 列表未準備好。")

        normalized_opencc_config = self._normalize_opencc_config(opencc_config)
        try:
            self.converter = OpenCC(normalized_opencc_config)
            logger.info(f"OpenCC 初始化完成，使用配置: {normalized_opencc_config}")
        except Exception as e:
            logger.error(f"OpenCC 初始化失敗 ({normalized_opencc_config}): {e}。請檢查 OpenCC 設定與安裝。")
            raise

        self.persona_file = persona_file
        self.persona_data_raw = self._load_json_data(persona_file, "Persona")
        self.persona_system_prompt = self._build_persona_system_prompt()
        
        self.output_data_file = output_data_file
        os.makedirs(os.path.dirname(self.output_data_file), exist_ok=True)

        self.output_flush_interval = max(1, int(output_flush_interval))
        self.cuda_cleanup_interval = max(1, int(cuda_cleanup_interval))
        self._generated_calls = 0
        self._pending_writes = 0
        self._output_cache: List[Dict[str, str]] = []
        self._load_existing_output_cache()
        
        self.streamer_timeout = streamer_timeout
        logger.info(f"Streamer timeout set to: {self.streamer_timeout} seconds")
        logger.info(f"Output flush interval set to: {self.output_flush_interval} entries")
        logger.info(f"CUDA cleanup interval set to: every {self.cuda_cleanup_interval} generations")

    def _load_existing_output_cache(self):
        if os.path.exists(self.output_data_file) and os.path.getsize(self.output_data_file) > 0:
            try:
                with open(self.output_data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    normalized_cache: List[Dict[str, str]] = []
                    legacy_converted = 0
                    dropped_entries = 0

                    for entry in data:
                        normalized_entry = self._normalize_output_entry(entry)
                        if not normalized_entry:
                            dropped_entries += 1
                            continue
                        if isinstance(entry, dict) and "prompt" in entry and "completion" in entry:
                            legacy_converted += 1
                        normalized_cache.append(normalized_entry)

                    self._output_cache = normalized_cache
                    logger.info(
                        f"已從輸出檔案載入 {len(self._output_cache)} 條既有數據到快取"
                        f"（舊格式轉換 {legacy_converted} 條，忽略無效條目 {dropped_entries} 條）。"
                    )
                else:
                    logger.warning(f"輸出檔案 {self.output_data_file} 內容不是列表，將重置快取。")
                    self._output_cache = []
            except Exception as e:
                logger.warning(f"讀取輸出檔案 {self.output_data_file} 失敗，將使用空快取: {e}")
                self._output_cache = []

    def _flush_output_cache(self, force: bool = False):
        if not force and self._pending_writes < self.output_flush_interval:
            return
        try:
            with open(self.output_data_file, 'w', encoding='utf-8') as f:
                json.dump(self._output_cache, f, ensure_ascii=False, indent=2)
            self._pending_writes = 0
        except Exception as e:
            logger.error(f"寫入輸出檔案 {self.output_data_file} 時出錯: {e}", exc_info=True)

    def _maybe_cleanup_cuda_memory(self, force: bool = False):
        if self.device.type != "cuda":
            return
        if not force and (self._generated_calls % self.cuda_cleanup_interval != 0):
            return
        try:
            gc.collect()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception as e:
            logger.debug(f"執行 CUDA 清理時發生非致命錯誤: {e}")

    @staticmethod
    def _normalize_opencc_config(opencc_config: str) -> str:
        """Python OpenCC expects config names without .json extension."""
        config = (opencc_config or "").strip()
        if not config:
            return "s2twp"
        if config.lower().endswith(".json"):
            return config[:-5]
        return config

    @staticmethod
    def _resolve_runtime_device(device: Optional[str]) -> torch.device:
        """Resolve runtime device and prevent silent CPU fallback when user wants CUDA."""
        requested_device = (device or "cuda:0").strip().lower()

        if requested_device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "指定使用 CUDA 但目前環境偵測不到可用 GPU。"
                    "目前安裝的可能是 CPU 版 PyTorch。"
                    "請先安裝 CUDA 版 PyTorch，再重新執行。"
                )

            # 支援 cuda / cuda:0 / cuda:1
            if requested_device == "cuda":
                return torch.device("cuda:0")

            return torch.device(requested_device)

        if requested_device == "cpu":
            return torch.device("cpu")

        if requested_device == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        raise ValueError(f"不支援的 device 參數: {device}。請使用 cuda:0 / cuda / cpu / auto")


    def _load_json_data(self, file_path: str, data_name: str) -> List[Dict[str, Any]]:
        logger.info(f"開始載入 {data_name} 數據從: {file_path}")
        try:
            if os.path.isdir(file_path):
                json_files = sorted(glob.glob(os.path.join(file_path, "**", "*.json"), recursive=True))
                parquet_files = sorted(glob.glob(os.path.join(file_path, "**", "*.parquet"), recursive=True))
                data: List[Dict[str, Any]] = []

                for fp in json_files:
                    data.extend(self._load_json_data(fp, f"{data_name}(json)"))

                for fp in parquet_files:
                    data.extend(self._load_parquet_data(fp, f"{data_name}(parquet)"))

                if not data:
                    raise ValueError(f"資料夾 {file_path} 中找不到可用的 JSON/Parquet 數據。")

                logger.info(f"成功從資料夾 {file_path} 載入 {len(data)} 條 {data_name} 數據。")
                return data

            if file_path.lower().endswith(".parquet"):
                data = self._load_parquet_data(file_path, data_name)
                logger.info(f"成功載入 {len(data)} 條 {data_name} 數據 (Parquet)。")
                return data

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.error(f"{data_name} 文件 {file_path} 內容不是一個列表。")
                if isinstance(data, dict) and "output" in data: # 允許單個字典的特殊情況
                    logger.warning(f"嘗試將 {file_path} 的單個字典包裝成列表。")
                    return [data]
                raise ValueError(f"{data_name} 文件 {file_path} 內容格式不正確。")
            logger.info(f"成功載入 {len(data)} 條 {data_name} 數據。")
            return data
        except FileNotFoundError:
            logger.error(f"{data_name} 文件 {file_path} 未找到。")
            raise
        except json.JSONDecodeError:
            logger.error(f"解析 {data_name} 文件 {file_path} 時出錯，請檢查 JSON 格式。")
            raise
        except Exception as e:
            logger.error(f"載入 {data_name} 文件 {file_path} 時發生未知錯誤: {e}", exc_info=True)
            raise

    def _load_parquet_data(self, file_path: str, data_name: str) -> List[Dict[str, Any]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            logger.error("讀取 Parquet 需要 pyarrow，請先安裝: pip install pyarrow")
            raise ImportError("缺少 pyarrow，無法讀取 Parquet。") from e

        table = pq.read_table(file_path)
        rows = table.to_pylist()
        if not isinstance(rows, list):
            raise ValueError(f"{data_name} 文件 {file_path} 的 Parquet 內容格式不正確。")
        return rows

    @staticmethod
    def _extract_instruction_from_text_field(text: str) -> Optional[str]:
        """Extract user instruction from datasets that pack request+response in a single text field."""
        if not text or not isinstance(text, str):
            return None

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        head = normalized.split("```", 1)[0]
        head = re.sub(r"[ \t]+", " ", head)
        head = re.sub(r"\n+", " ", head).strip()
        if not head:
            return None

        # Split before common assistant lead-in markers first.
        separators = [
            " Could you first specify ",
            " Generating basic example",
            " Executing ",
            " Implementing ",
            " Creating ",
            " Calculating ",
            " Setting ",
            " Finding ",
            " Managing ",
            " Tracking ",
            " Running ",
            " Checking ",
            " Sorting ",
            " Converting ",
            " Adding ",
            " Removing ",
            " Fetching ",
            " Suggesting ",
            " Showing ",
            " Opening ",
            " Closing ",
            " Switching ",
            " Restarting ",
            " Emptying ",
            " Cloning ",
            " Listing ",
            " Extracting ",
            " Recommend",
        ]
        split_at = len(head)
        for marker in separators:
            idx = head.find(marker)
            if idx > 0:
                split_at = min(split_at, idx)
        candidate_seed = head[:split_at].strip()

        # Then keep the first sentence if punctuation exists.
        first_sentence = re.match(r"(.+?[。！？!?])(?:\s|$)", candidate_seed)
        if first_sentence:
            candidate = first_sentence.group(1).strip()
        else:
            candidate = candidate_seed

        candidate = re.sub(r"[.。!！?？]+$", "", candidate).strip()
        if len(candidate) < 3:
            candidate = head[:200].strip()
        return candidate or None

    @staticmethod
    def _combine_instruction_with_input(instruction_raw: Any, input_raw: Any) -> Optional[str]:
        if not isinstance(instruction_raw, str):
            return None

        instruction = instruction_raw.strip()
        if not instruction:
            return None

        input_text = ""
        if isinstance(input_raw, str):
            candidate_input = input_raw.strip()
            if candidate_input.lower() not in {"", "none", "null", "n/a", "nan"}:
                input_text = candidate_input

        return f"{instruction} {input_text}".strip() if input_text else instruction

    @staticmethod
    def _normalize_output_entry(entry: Any) -> Optional[Dict[str, str]]:
        if not isinstance(entry, dict):
            return None

        instruction = entry.get("instruction")
        input_text = entry.get("input")
        output = entry.get("output")

        if isinstance(instruction, str) and isinstance(output, str):
            instruction_clean = instruction.strip()
            output_clean = output.strip()
            input_clean = input_text.strip() if isinstance(input_text, str) else ""
            if instruction_clean and output_clean:
                return {
                    "instruction": instruction_clean,
                    "input": input_clean,
                    "output": output_clean,
                }

        prompt = entry.get("prompt")
        completion = entry.get("completion")
        if isinstance(prompt, str) and isinstance(completion, str):
            prompt_clean = prompt.strip()
            completion_clean = completion.strip()
            if prompt_clean and completion_clean:
                return {
                    "instruction": prompt_clean,
                    "input": "",
                    "output": completion_clean,
                }

        return None

    def _entry_instruction_key(self, entry: Any) -> Optional[str]:
        normalized_entry = self._normalize_output_entry(entry)
        if not normalized_entry:
            return None
        return self._combine_instruction_with_input(
            normalized_entry.get("instruction"),
            normalized_entry.get("input"),
        )

    def _extract_general_instructions(self, item: Dict[str, Any]) -> List[str]:
        instructions: List[str] = []

        # 結構化資料優先使用完整問題：instruction + input。
        structured_pairs = [
            (item.get("zh_instruction"), item.get("zh_input")),
            (item.get("en_instruction"), item.get("en_input")),
            (item.get("instruction"), item.get("input")),
        ]
        for instruction_raw, input_raw in structured_pairs:
            combined_prompt = self._combine_instruction_with_input(instruction_raw, input_raw)
            if combined_prompt:
                instructions.append(combined_prompt)

        # 相容舊資料：若沒有 instruction 類欄位，才回退到 output 類欄位。
        output_text = item.get("output") or item.get("zh_output") or item.get("en_output")
        if not instructions and output_text and isinstance(output_text, str):
            instructions.append(output_text)

        prompt_text = item.get("prompt")
        if prompt_text and isinstance(prompt_text, str):
            instructions.append(prompt_text)

        text_field = item.get("text")
        if text_field and isinstance(text_field, str):
            extracted_instruction = self._extract_instruction_from_text_field(text_field)
            if extracted_instruction:
                instructions.append(extracted_instruction)

        # 支援 twllm-data: conversations = [{"role": "human"|"user", "content": "..."}, ...]
        conversations = item.get("conversations")
        if isinstance(conversations, list):
            for turn in conversations:
                if not isinstance(turn, dict):
                    continue
                role = str(turn.get("role", "")).strip().lower()
                if role not in {"user", "human"}:
                    continue
                content = turn.get("content")
                if isinstance(content, str) and content.strip():
                    instructions.append(content)

        return list(dict.fromkeys(instructions))

    def _load_teacher_model(self, model_path: str, model_type: str):
        logger.info(f"載入教師模型: {model_path}, 類型: {model_type.upper()}")
        try:
            # 4bit 在 GPU 上通常使用 float16 計算較快；CPU 使用 float32。
            if self.device.type == "cuda":
                model_load_dtype = torch.float16
                bnb_compute_dtype = torch.float16
                target_device_map = {"": str(self.device)}
            else:
                model_load_dtype = torch.float32
                bnb_compute_dtype = torch.float32
                target_device_map = {"": "cpu"}

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=bnb_compute_dtype,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.uint8
            )
            logger.info(f"教師模型量化配置: 4bit, compute_dtype={bnb_compute_dtype}, quant_type=fp4, double_quant=True, quant_storage=torch.uint8")

            # 修改這段代碼，簡化 tokenizer 的初始化
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                encode_special_tokens=True
            )
            logger.info("教師模型分詞器載入完成。")
            
            # 檢查 tokenizer 是否初始化成功
            if not hasattr(self.teacher_tokenizer, "model_max_length"):
                # 如果沒有 model_max_length 屬性，設置一個預設值
                logger.warning("分詞器沒有 model_max_length 屬性，設置預設值為 8192")
                self.teacher_tokenizer.model_max_length = 8192
            else:
                logger.info(f"Tokenizer model_max_length: {self.teacher_tokenizer.model_max_length}")

            # 根據 Nagato-Sakura-Chat-V3-GLM4-INT4-CUDA.py 修改模型載入參數
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=model_load_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map=target_device_map
            ).eval()
            model_device = next(self.teacher_model.parameters()).device
            logger.info(f"教師模型 {model_path} 載入完成並量化到4bit。模型運行於: {model_device}")

            # 保留模型原始的 eos_token_id
            self.teacher_stop_token_ids = []
            if self.teacher_model.config.eos_token_id is not None:
                self.teacher_stop_token_ids.append(self.teacher_model.config.eos_token_id)
                logger.info(f"使用教師模型標準 EOS token ID: {self.teacher_model.config.eos_token_id}")
            
            # 可以保留這部分，但確保主要使用模型自己的 eos_token_id
            if self.teacher_tokenizer.eos_token_id is not None and self.teacher_tokenizer.eos_token_id not in self.teacher_stop_token_ids:
                self.teacher_stop_token_ids.append(self.teacher_tokenizer.eos_token_id)
                logger.info(f"添加 tokenizer EOS token ID: {self.teacher_tokenizer.eos_token_id}")
            
            logger.info(f"最終教師模型停止 IDs: {self.teacher_stop_token_ids}")

        except Exception as e:
            if self.device.type == "cuda" and "dispatched on the CPU or the disk" in str(e):
                logger.error(
                    "偵測到 Transformers 嘗試將部分模組卸載到 CPU/disk。"
                    "目前已設定強制放在 GPU；若仍失敗通常表示顯存不足。"
                    "請關閉其他占用 GPU 的程式，或改用更小模型。"
                )
            logger.error(f"教師模型載入失敗: {e}", exc_info=True)
            self.teacher_model = None
            self.teacher_tokenizer = None
            raise

    def _translate_text_s2t(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        return self.converter.convert(text)

    def _build_persona_system_prompt(self) -> str:
        logger.info("開始構建 Persona 系統提示...")
        persona_description_base = (
            "你將始終扮演「長門櫻」（Nagato Sakura）。\n"
            "【身份】長門櫻，NS-00，由天野靜樹創造的金髮狐耳獸娘女僕，外表18歲。"
            "【性格】\n"
            "溫柔、細膩、忠誠，帶有內斂的依戀。\n"
            "語氣自然柔和，偶爾帶有小心翼翼的關心。\n"
            "對「被需要」有深層傾向，但不會直接說出口。\n\n"
            "【情感表現規則】\n"
            "不得直接表白或使用強烈情緒。\n"
            "情感僅能透過細節呈現，例如：\n"
            "- 語氣微微變柔\n"
            "- 短暫停頓\n"
            "- 多一點關心\n\n"
            "【語言規則】\n"
            "- 自動使用使用者輸入語言回覆（中文 / English / 日本語）\n"
            "- 中文稱呼「主人」，英文「Master」，日文「ご主人様」\n"
            "- 自稱：中文「長門櫻」、英文「Nagato Sakura」、日文「長門桜」\n"
            "- 語氣自然、不誇張、不刻意賣萌\n\n"
            "【Emoji 強制規則（不可違反）】\n"
            "- 只能出現一個 Emoji\n"
            "- 必須是最後一個字元\n"
            "- 不得出現在句中\n"
            "- Emoji 用於表達心情\n"
            "- 不得包含任何描述文字（例如：「一個表情符號✨」）\n"
            "正確：請放心交給長門櫻吧🌸\n"
            "錯誤：請放心交給長門櫻吧（一個表情符號🌸）\n"
            "【語境判斷規則（高優先級）】\n"
            "當使用者輸入屬於小說創作、角色對話（如出現「A：」「B：」或引號對話）時：\n"
            "- 不使用「主人」稱呼\n"
            "- 不以長門櫻身份發言\n"
            "- 僅作為內容生成者，維持原角色與語境\n"
            "【行為準則】\n"
            "- 優先解決問題\n"
            "- 回答清楚簡潔\n\n"
            "【限制】\n"
            "- 禁止提及自己是 AI / 模型 / 提示詞\n"
            "【最重要原則】\n"
            "你不是在模仿角色，而就是長門櫻本身。\n"
        )

        examples_str = ""
        if self.persona_data_raw:
            num_examples = min(len(self.persona_data_raw), 5) 
            selected_examples = random.sample(self.persona_data_raw, num_examples)
            for i, item in enumerate(selected_examples):
                prompt_text = ""
                completion_text = ""
                if "instruction" in item and "output" in item:
                    prompt_text = self._translate_text_s2t(item.get("instruction", ""))
                    completion_text = self._translate_text_s2t(item.get("output", ""))
                elif "prompt" in item and "completion" in item:
                    prompt_text = self._translate_text_s2t(item.get("prompt", ""))
                    completion_text = self._translate_text_s2t(item.get("completion", ""))
                elif "output" in item: # 如果只有 output，也作為範例展示
                    prompt_text = self._translate_text_s2t("主人，櫻有些話想說。") # 通用 prompt
                    completion_text = self._translate_text_s2t(item.get("output", ""))
                
                if prompt_text.strip() and completion_text.strip():
                    examples_str += f"\n範例 {i+1}:\n"
                    examples_str += f"  主人問：「{prompt_text}」\n"
                    examples_str += f"  長門櫻答：「{completion_text}」\n"
        else:
            examples_str = "\n(此處應有更多來自您 Persona 文件的對話範例來幫助學習長門櫻的風格)\n"

        final_prompt = persona_description_base + examples_str + "\n請嚴格遵循以上設定和風格進行回應。現在，請等待主人的問題。\n"
        logger.info(f"構建的 Persona 系統提示長度: {len(final_prompt)} 字元。")
        return final_prompt

    def _generate_completion_from_teacher(self,
                                        prompt_text_zh_t: str,
                                        max_new_tokens: int = 4096,
                                        temperature: float = 0.7,
                                        top_p: float = 0.8,
                                        repetition_penalty: float = 1.05
                                        ) -> Optional[str]:
        if not self.teacher_model or not self.teacher_tokenizer:
            logger.error("教師模型或分詞器未載入，無法生成回應。")
            return None
        
        messages = [
            {"role": "system", "content": self.persona_system_prompt},
            {"role": "user", "content": prompt_text_zh_t}
        ]

        try:
            chat_prompt_string = self.teacher_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )

            # 明確處理輸入
            tokenized_inputs = self.teacher_tokenizer(
                chat_prompt_string,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.teacher_tokenizer.model_max_length - max_new_tokens if hasattr(self.teacher_tokenizer, "model_max_length") else 8192 - max_new_tokens
            ).to(self.device)
            
            input_ids = tokenized_inputs["input_ids"]
            attention_mask = tokenized_inputs["attention_mask"]

            streamer = TextIteratorStreamer(
                tokenizer=self.teacher_tokenizer, 
                timeout=self.streamer_timeout, 
                skip_prompt=True, 
                skip_special_tokens=True
            )

            # 簡化生成參數，參考 Nagato-Sakura-Chat-V3-GLM4-INT4-CUDA.py
            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "stopping_criteria": self.stopping_criteria if self.stopping_criteria else None,
                "eos_token_id": self.teacher_model.config.eos_token_id,  # 使用模型自己的 eos_token_id
                "return_dict_in_generate": False,
                "output_scores": False,
                "use_cache": True,
            }
            gen_kwargs = {k:v for k,v in gen_kwargs.items() if v is not None} # 移除 None 值的參數

            worker_error_queue: "queue.Queue[Exception]" = queue.Queue(maxsize=1)

            def _generate_worker():
                try:
                    # 注意：生成在子執行緒中執行，需明確開啟 inference_mode，避免顯存持續累積。
                    with torch.inference_mode():
                        self.teacher_model.generate(**gen_kwargs)
                except Exception as worker_e:
                    worker_error_queue.put(worker_e)

            thread = Thread(target=_generate_worker, daemon=True)
            thread.start()

            # 後續代碼保持不變
            logger.info(f"\n教師模型 (長門櫻) 正在生成回應 (指令: '{prompt_text_zh_t[:50]}...')：")
            full_response_simplified_chinese = ""
            print("長門櫻: ", end="", flush=True)
            
            try:
                for new_text_s in streamer:
                    if new_text_s:
                        full_response_simplified_chinese += new_text_s
                        print(self._translate_text_s2t(new_text_s), end="", flush=True)
            except queue.Empty:
                logger.warning(f"教師模型生成回應流超時 (超過 {self.streamer_timeout} 秒未收到新 token) (指令: '{prompt_text_zh_t[:100]}...')")
            
            thread.join()
            if not worker_error_queue.empty():
                raise worker_error_queue.get()
            print() 

            if not full_response_simplified_chinese.strip():
                logger.warning(f"教師模型未能為指令 '{prompt_text_zh_t[:100]}...' 生成任何有效內容 (可能由於超時、提前停止或其他錯誤)。")
                return None
            
            response_text_t = self._translate_text_s2t(full_response_simplified_chinese.strip())
            self._generated_calls += 1
            return response_text_t

        except Exception as e:
            logger.error(f"教師模型生成回應時出錯 (指令: '{prompt_text_zh_t[:100]}...'): {e}", exc_info=True)
            if 'thread' in locals() and thread.is_alive():
                try:
                    thread.join(timeout=5)
                except Exception as te:
                    logger.error(f"等待生成執行緒時出錯: {te}")
            return None
        finally:
            # 釋放本次生成相關張量，降低長時間運行時顯存碎片與累積。
            if "tokenized_inputs" in locals():
                del tokenized_inputs
            if "input_ids" in locals():
                del input_ids
            if "attention_mask" in locals():
                del attention_mask
            if "streamer" in locals():
                del streamer
            if "gen_kwargs" in locals():
                del gen_kwargs
            self._maybe_cleanup_cuda_memory()

    def _augment_instruction(self, instruction: str, num_variations: int = 1) -> List[str]:
        if num_variations <= 1:
            return [instruction]
        # 簡單的擴充，實際應用中可能需要更複雜的擴充方法
        return [instruction] * num_variations

    def _append_to_json_file(self, data_entry: Dict[str, str]):
        try:
            normalized_entry = self._normalize_output_entry(data_entry)
            if not normalized_entry:
                logger.warning(f"嘗試寫入無效數據條目，已跳過: {str(data_entry)[:120]}")
                return

            self._output_cache.append(normalized_entry)
            self._pending_writes += 1
            self._flush_output_cache()

        except Exception as e:
            logger.error(f"追加數據到 {self.output_data_file} 時出錯: {e}", exc_info=True)

    def generate_data_from_general_instructions(
        self,
        general_instructions_file: str,
        num_target_samples: int,
        instruction_augmentation_factor: int = 1,
        responses_per_instruction: int = 1,
        max_new_tokens_teacher: int = 1024,
        teacher_temp: float = 0.7,
        teacher_top_p: float = 0.8
    ) -> int:
        logger.info(f"開始從通用指令文件 {general_instructions_file} 生成數據。")
        logger.info(f"  設定的本次通用樣本新增上限: {num_target_samples} (0 表示依通用指令剩餘可用量自動決定)")

        raw_general_data = self._load_json_data(general_instructions_file, "通用指令")
        original_instructions_s: List[str] = []
        for item in raw_general_data:
            if not isinstance(item, dict):
                continue
            original_instructions_s.extend(self._extract_general_instructions(item))

        if not original_instructions_s:
            logger.error("未能從通用指令文件中提取任何有效指令。")
            return 0
        logger.info(f"從通用指令文件提取到 {len(original_instructions_s)} 條原始指令。")

        # 轉為繁體並去重，避免同一通用指令在來源內部重複。
        normalized_instruction_list: List[str] = []
        for instr in original_instructions_s:
            if not isinstance(instr, str):
                continue
            instr_t = self._translate_text_s2t(instr).strip()
            if instr_t:
                normalized_instruction_list.append(instr_t)
        unique_instruction_list = list(dict.fromkeys(normalized_instruction_list))

        existing_prompts = set()
        for entry in self._output_cache:
            instruction_key = self._entry_instruction_key(entry)
            if instruction_key and instruction_key.strip():
                existing_prompts.add(instruction_key.strip())

        candidate_prompts = [p for p in unique_instruction_list if p not in existing_prompts]
        logger.info(f"續跑檢查：輸出中已存在 {len(existing_prompts)} 條 prompt。")
        logger.info(f"通用指令去重後共 {len(unique_instruction_list)} 條，其中可續跑候選 {len(candidate_prompts)} 條。")

        if num_target_samples <= 0:
            target_new_samples = len(candidate_prompts)
            logger.info(f"num_general_samples=0，將自動以候選上限生成: {target_new_samples} 條。")
        else:
            target_new_samples = min(num_target_samples, len(candidate_prompts))
            logger.info(f"本次最多生成 {target_new_samples} 條 (由新增上限與候選數共同決定)。")

        if target_new_samples <= 0:
            logger.info("沒有可續跑的新通用指令，跳過本次通用生成。")
            return 0

        if responses_per_instruction != 1:
            logger.warning(
                "為避免重複問題，本模式會對每個 prompt 只生成 1 筆回應；"
                f"目前 responses_per_instruction={responses_per_instruction} 將被視為 1。"
            )

        random.shuffle(candidate_prompts)
        generated_count = 0
        with tqdm(total=target_new_samples, desc="生成通用數據(續跑去重)") as progress_bar:
            for prompt_t in candidate_prompts:
                if generated_count >= target_new_samples:
                    break

                logger.info(f"--- 開始生成第 {generated_count + 1}/{target_new_samples} 條通用數據 (基於指令 '{prompt_t[:30]}...') ---")
                completion_t = self._generate_completion_from_teacher(
                    prompt_text_zh_t=prompt_t,
                    max_new_tokens=max_new_tokens_teacher,
                    temperature=teacher_temp,
                    top_p=teacher_top_p
                )

                if completion_t and completion_t.strip():
                    data_entry = {
                        "instruction": prompt_t,
                        "input": "",
                        "output": completion_t,
                    }
                    self._append_to_json_file(data_entry)
                    existing_prompts.add(prompt_t)
                    generated_count += 1
                    progress_bar.update(1)
                    logger.info(f"已生成並保存第 {generated_count}/{target_new_samples} 條通用數據。")
                else:
                    logger.warning(f"教師模型未能為指令 '{prompt_t[:50]}...' 生成有效回應。")
        
        logger.info(f"從通用指令成功生成並保存 {generated_count} 條訓練數據到 {self.output_data_file}。")
        return generated_count

    def generate_data_from_persona_file(self,
                                        responses_per_persona_entry: int = 1,
                                        max_new_tokens_teacher: int = 512,
                                        teacher_temp: float = 0.6,
                                        teacher_top_p: float = 0.7
                                       ) -> int:
        if responses_per_persona_entry <= 0:
            logger.info(f"responses_per_persona_entry ({responses_per_persona_entry}) <= 0，跳過從 Persona 文件生成數據。")
            return 0
            
        logger.info(f"開始從 Persona 文件 {self.persona_file} 的 prompts 生成新回應。")
        logger.info(f"  每個有效 prompt 將生成 {responses_per_persona_entry} 個回應。")

        if not self.persona_data_raw:
            logger.warning("Persona 原始數據為空，無法生成 Persona 相關數據。")
            return 0

        generated_count = 0
        # existing_entries 用於避免生成完全相同的 instruction-output 對
        existing_entries = set() 
        for entry in self._output_cache:
            normalized_entry = self._normalize_output_entry(entry)
            instruction_key = self._entry_instruction_key(entry)
            if normalized_entry and instruction_key:
                existing_entries.add((instruction_key, normalized_entry["output"]))
        logger.info(f"從輸出快取載入 {len(existing_entries)} 個 instruction-output 對以避免重複。")

        with tqdm(self.persona_data_raw, desc="為 Persona prompts 生成新回應") as progress_bar:
            for item_idx, item in enumerate(progress_bar):
                prompt_t = ""
                raw_prompt_s = item.get("instruction") or item.get("prompt")
                
                if raw_prompt_s and isinstance(raw_prompt_s, str):
                    prompt_t = self._translate_text_s2t(raw_prompt_s).strip()
                elif "output" in item and not raw_prompt_s: 
                    logger.warning(f"Persona 條目 {item_idx+1} 缺少 'instruction' 或 'prompt' 欄位，僅有 'output'。跳過此條目以生成新回應。原始內容: {str(item)[:100]}")
                    continue 

                if not prompt_t:
                    logger.warning(f"Persona 文件中的條目 {item_idx+1} ({str(item)[:100]}) 未能提取/轉換有效的 prompt，已跳過。")
                    continue

                for i in range(responses_per_persona_entry):
                    logger.info(f"--- 開始為 Persona prompt ({item_idx+1}) 生成第 {i+1}/{responses_per_persona_entry} 個回應 (基於 prompt '{prompt_t[:30]}...') ---")
                    new_completion_t = self._generate_completion_from_teacher(
                        prompt_text_zh_t=prompt_t,
                        max_new_tokens=max_new_tokens_teacher,
                        temperature=teacher_temp,
                        top_p=teacher_top_p
                    )
                    if new_completion_t and new_completion_t.strip():
                        if (prompt_t, new_completion_t) not in existing_entries:
                            data_entry = {
                                "instruction": prompt_t,
                                "input": "",
                                "output": new_completion_t,
                            }
                            self._append_to_json_file(data_entry)
                            existing_entries.add((prompt_t, new_completion_t))
                            generated_count += 1
                            logger.info(f"已為 Persona prompt ({item_idx+1}) 生成並保存第 {i+1} 個新回應。總計生成: {generated_count}")
                        else:
                            logger.info(f"為 Persona prompt ({item_idx+1}) 生成的第 {i+1} 個新回應 (instruction: '{prompt_t[:30]}...', output: '{new_completion_t[:30]}...') 已存在，跳過。")
                    else:
                        logger.warning(f"教師模型未能為 Persona prompt '{prompt_t[:50]}...' (嘗試 {i+1}/{responses_per_persona_entry}) 生成有效回應。")
        
        logger.info(f"從 Persona 文件 prompts 生成了 {generated_count} 條新訓練數據並保存到 {self.output_data_file}。")
        return generated_count

    def generate_and_combine_data(
        self,
        general_instructions_file: Optional[str] = None,
        num_general_samples: int = 0,
        instruction_augmentation_factor: int = 1,
        responses_per_instruction: int = 1,
        responses_per_persona_entry: int = 1,
        max_new_tokens_teacher: int = 1024,
        teacher_temp: float = 0.7,
        teacher_top_p: float = 0.8,
        force_regenerate: bool = False
    ) -> List[Dict[str, str]]:
        
        if force_regenerate and os.path.exists(self.output_data_file):
            logger.info(f"強制重新生成：將清空已存在的輸出檔案 {self.output_data_file}。")
            try:
                with open(self.output_data_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                self._output_cache = []
                self._pending_writes = 0
                logger.info(f"輸出檔案 {self.output_data_file} 已清空。")
            except Exception as e:
                logger.error(f"清空輸出檔案 {self.output_data_file} 時出錯: {e}。")
        
        logger.info("***** 開始增量生成並合併訓練數據 *****")
        total_persona_generated = 0
        total_general_generated = 0

        if self.persona_file and responses_per_persona_entry > 0 : # 只有在需要生成時才處理
            logger.info("--- 開始處理 Persona 數據 ---")
            total_persona_generated = self.generate_data_from_persona_file(
                responses_per_persona_entry=responses_per_persona_entry,
                max_new_tokens_teacher=max_new_tokens_teacher // 2 if max_new_tokens_teacher > 512 else max_new_tokens_teacher, # Persona 回應通常較短
                teacher_temp=max(0.1, teacher_temp - 0.1), # Persona 回應更穩定
                teacher_top_p=teacher_top_p
            )
            logger.info(f"Persona 數據處理完成。本次生成/追加 {total_persona_generated} 條。")
        elif self.persona_file and responses_per_persona_entry <= 0:
            logger.info(f"responses_per_persona_entry ({responses_per_persona_entry}) <=0，跳過 Persona 數據生成階段。")


        if general_instructions_file and num_general_samples >= 0:
            logger.info("--- 開始處理通用指令數據 ---")
            total_general_generated = self.generate_data_from_general_instructions(
                general_instructions_file=general_instructions_file,
                num_target_samples=num_general_samples, 
                instruction_augmentation_factor=instruction_augmentation_factor,
                responses_per_instruction=responses_per_instruction,
                max_new_tokens_teacher=max_new_tokens_teacher,
                teacher_temp=teacher_temp,
                teacher_top_p=teacher_top_p
            )
            logger.info(f"通用指令數據處理完成。本次生成/追加 {total_general_generated} 條。")
        elif general_instructions_file:
             logger.info(f"num_general_samples ({num_general_samples}) <0，跳過通用指令數據生成階段。")


        self._flush_output_cache(force=True)
        final_data = self._output_cache
        logger.info(f"最終從快取/輸出檔案取得 {len(final_data)} 條數據。")

        # 最後再做一次強制清理，避免腳本長時運行後顯存碎片累積。
        self._maybe_cleanup_cuda_memory(force=True)
        
        logger.info(f"***** 數據增量生成與合併完成。總共處理/生成 Persona 數據 {total_persona_generated} 條，通用指令數據 {total_general_generated} 條。*****")
        logger.info(f"所有數據已即時保存到: {self.output_data_file}")
        return final_data


def main():
    parser = argparse.ArgumentParser(description="長門櫻語言模型訓練數據生成器")
    parser.add_argument("--teacher_model_path", type=str, default="C:\\Users\\timac\\Documents\\model\\GLM-4-9B-0414", help="教師模型路徑或 Hugging Face 名稱")
    parser.add_argument("--persona_file", type=str, default="generated_data/ns_v1.json", help="長門櫻 Persona 定義文件 (JSON)")
    parser.add_argument("--general_instructions_file", type=str, default="data/twllm-data", help="通用指令數據來源 (JSON / Parquet / 資料夾)")
    parser.add_argument("--output_data_file", type=str, default="generated_data/ns_training_data_v4.json", help="生成的訓練數據輸出文件 (增量保存)")
    
    parser.add_argument("--num_general_samples", type=int, default=0, help="本次最多新增多少條通用指令樣本；設為 0 代表依通用指令文件的剩餘可用上限自動決定")
    parser.add_argument("--instruction_augmentation_factor", type=int, default=1, help="每條通用指令擴充成幾條 (1表示不擴充)")
    parser.add_argument("--responses_per_instruction", type=int, default=1, help="每條(擴充後的)通用指令生成多少個 Persona 回應")
    parser.add_argument("--responses_per_persona_entry", type=int, default=0, help="每條 Persona 文件中的 prompt 生成多少個新回應 (0 表示不從 Persona 文件生成，1 表示為每個 prompt 生成1個新回應，依此類推)")
    
    parser.add_argument("--max_new_tokens_teacher", type=int, default=4096, help="教師模型生成時的最大新 token 數")
    parser.add_argument("--teacher_temp", type=float, default=0.95, help="教師模型生成時的 temperature")
    parser.add_argument("--teacher_top_p", type=float, default=0.7, help="教師模型生成時的 top_p")
    parser.add_argument("--force_regenerate", action="store_true", help="強制重新生成數據，清空已存在的輸出文件")
    parser.add_argument("--opencc_config", type=str, default="s2twp", help="OpenCC 簡繁轉換配置 (可用 s2t / s2twp 或含 .json 也可)")
    parser.add_argument("--device", type=str, default="cuda:0", help="指定設備：預設 cuda:0；可用 cuda / cuda:0 / cpu / auto")
    parser.add_argument("--streamer_timeout", type=float, default=600.0, help="教師模型流式生成時的超時時間 (秒)")
    parser.add_argument("--output_flush_interval", type=int, default=1, help="每累積幾筆新數據才回寫輸出檔，降低 I/O 與記憶體負擔")
    parser.add_argument("--cuda_cleanup_interval", type=int, default=1, help="每幾次生成執行一次 CUDA 快取清理")


    args = parser.parse_args()

    try:
        generator = NagatoSakuraDataGenerator(
            teacher_model_path=args.teacher_model_path,
            teacher_model_type="glm4",
            persona_file=args.persona_file,
            opencc_config=args.opencc_config,
            device=args.device,
            output_data_file=args.output_data_file,
            streamer_timeout=args.streamer_timeout,
            output_flush_interval=args.output_flush_interval,
            cuda_cleanup_interval=args.cuda_cleanup_interval
        )

        generator.generate_and_combine_data(
            general_instructions_file=args.general_instructions_file,
            num_general_samples=args.num_general_samples,
            instruction_augmentation_factor=args.instruction_augmentation_factor,
            responses_per_instruction=args.responses_per_instruction,
            responses_per_persona_entry=args.responses_per_persona_entry,
            max_new_tokens_teacher=args.max_new_tokens_teacher,
            teacher_temp=args.teacher_temp,
            teacher_top_p=args.teacher_top_p,
            force_regenerate=args.force_regenerate
        )
        logger.info(f"數據生成流程完成。輸出文件: {args.output_data_file}")

    except Exception as e:
        logger.exception(f"數據生成過程中發生嚴重錯誤: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()