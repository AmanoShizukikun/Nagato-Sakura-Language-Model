import torch
import logging
from typing import List, Dict, Any, Optional, Tuple, Iterator
from transformers import PreTrainedTokenizerFast
from dataclasses import dataclass
import json
import time

@dataclass
class ConversationSample:
    """對話樣本數據結構"""
    conversation_id: str
    turns: List[Dict[str, str]]  # [{"role": "user/assistant", "content": "..."}]
    metadata: Optional[Dict[str, Any]] = None

class ConversationDataProcessor:
    """對話數據處理器"""
    
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)
    
    def format_conversation(self, turns: List[Dict[str, str]], 
                          system_message: Optional[str] = None) -> str:
        """格式化對話為文本"""
        formatted_text = ""
        
        if system_message:
            formatted_text += f"系統：{system_message}\n"
        
        for turn in turns:
            role = turn.get("role", "")
            content = turn.get("content", "")
            
            if role == "user":
                formatted_text += f"用戶：{content}\n"
            elif role == "assistant":
                formatted_text += f"長門櫻：{content}\n"
        
        return formatted_text.strip()
    
    def prepare_training_sample(self, conversation: ConversationSample) -> Optional[Dict[str, str]]:
        """準備單個對話樣本用於訓練"""
        if len(conversation.turns) < 2:
            return None
        
        # 找到最後一對用戶-助手對話
        user_turns = [t for t in conversation.turns if t["role"] == "user"]
        assistant_turns = [t for t in conversation.turns if t["role"] == "assistant"]
        
        if not user_turns or not assistant_turns:
            return None
        
        # 構建prompt（包含歷史對話）
        prompt_turns = conversation.turns[:-1]  # 除了最後一個助手回應
        prompt = self.format_conversation(prompt_turns)
        prompt += "\n長門櫻："
        
        # 最後一個助手回應作為completion
        last_assistant_turn = conversation.turns[-1]
        if last_assistant_turn["role"] != "assistant":
            return None
        
        completion = last_assistant_turn["content"]
        
        return {
            "prompt": prompt,
            "completion": completion,
            "conversation_id": conversation.conversation_id
        }
    
    def process_conversation_dataset(self, conversations: List[ConversationSample]) -> List[Dict[str, str]]:
        """處理整個對話數據集"""
        training_samples = []
        
        for conv in conversations:
            sample = self.prepare_training_sample(conv)
            if sample:
                training_samples.append(sample)
        
        self.logger.info(f"從 {len(conversations)} 個對話中生成了 {len(training_samples)} 個訓練樣本")
        return training_samples

def smart_collate_fn(batch: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerFast, 
                    max_seq_length: int, pack_sequences: bool = True) -> Dict[str, Optional[torch.Tensor]]:
    """智能數據整理函數，支持序列打包和流式對話"""
    
    # 過濾無效數據
    valid_batch = []
    for item in batch:
        if (item and isinstance(item, dict) and 
            "prompt" in item and "completion" in item and
            item["prompt"] and item["completion"]):
            valid_batch.append(item)
    
    if not valid_batch:
        return {"input_ids": None, "attention_mask": None, "labels": None}
    
    bos = tokenizer.bos_token or "<s>"
    eos = tokenizer.eos_token or "</s>"
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    
    if pad_token_id is None:
        raise ValueError("分詞器缺少 pad_token_id")
    
    # 構建完整文本和計算prompt長度
    full_texts = []
    prompt_lengths = []
    
    for item in valid_batch:
        prompt = str(item["prompt"]).strip()
        completion = str(item["completion"]).strip()
        full_text = f"{bos}{prompt}{completion}{eos}"
        full_texts.append(full_text)
        
        # 計算prompt長度（包含BOS）
        prompt_with_bos = f"{bos}{prompt}"
        prompt_length = len(tokenizer.encode(prompt_with_bos, add_special_tokens=False))
        prompt_lengths.append(prompt_length)
    
    # 序列打包優化
    if pack_sequences and len(full_texts) > 1:
        # 按長度排序以更好地打包
        text_length_pairs = [(text, length) for text, length in zip(full_texts, prompt_lengths)]
        text_length_pairs.sort(key=lambda x: len(tokenizer.encode(x[0], add_special_tokens=False)))
        full_texts = [pair[0] for pair in text_length_pairs]
        prompt_lengths = [pair[1] for pair in text_length_pairs]
    
    try:
        # 編碼批次
        encoded_batch = tokenizer.batch_encode_plus(
            full_texts,
            padding="longest",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        input_ids = encoded_batch["input_ids"]
        attention_mask = encoded_batch["attention_mask"]
        labels = input_ids.clone()
        
        # 掩蔽prompt部分
        for i, prompt_len in enumerate(prompt_lengths):
            actual_prompt_len = min(prompt_len, labels.shape[1])
            labels[i, :actual_prompt_len] = -100
        
        # 掩蔽填充令牌
        labels[attention_mask == 0] = -100
        
        # 檢查是否有有效標籤
        if not (labels != -100).any():
            return {"input_ids": None, "attention_mask": None, "labels": None}
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
    except Exception as e:
        logging.getLogger(__name__).error(f"批次編碼錯誤: {e}")
        return {"input_ids": None, "attention_mask": None, "labels": None}

def stream_collate_fn(batch: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerFast,
                     max_seq_length: int) -> Dict[str, torch.Tensor]:
    """流式生成專用的collate函數"""
    # 簡化版本，主要用於推理
    texts = []
    for item in batch:
        if "text" in item:
            texts.append(item["text"])
        elif "prompt" in item:
            texts.append(item["prompt"])
    
    if not texts:
        return {"input_ids": torch.empty(0, 0, dtype=torch.long)}
    
    encoded = tokenizer.batch_encode_plus(
        texts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
        return_attention_mask=True
    )
    
    return encoded

class EarlyStoppingCallback:
    """早停回調"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.patience_counter = 0
        self.should_stop = False
    
    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        if self.mode == 'min':
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        else:  # mode == 'max'
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            self.should_stop = True
            return True
        return False

class StreamingDataLoader:
    """流式數據加載器，用於處理大型對話數據集"""
    
    def __init__(self, data_source: str, tokenizer: PreTrainedTokenizerFast,
                 batch_size: int = 1, max_length: int = 2048):
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.processor = ConversationDataProcessor(tokenizer, max_length)
    
    def load_conversations_from_file(self, file_path: str) -> Iterator[ConversationSample]:
        """從文件流式加載對話"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # 處理不同的數據格式
                        if "conversation" in item:
                            # 多輪對話格式
                            yield ConversationSample(
                                conversation_id=item.get("id", str(time.time())),
                                turns=item["conversation"],
                                metadata=item.get("metadata")
                            )
                        elif "turns" in item:
                            # 直接turns格式
                            yield ConversationSample(
                                conversation_id=item.get("id", str(time.time())),
                                turns=item["turns"],
                                metadata=item.get("metadata")
                            )
                        elif "prompt" in item and "completion" in item:
                            # 轉換為對話格式
                            turns = [
                                {"role": "user", "content": item["prompt"]},
                                {"role": "assistant", "content": item["completion"]}
                            ]
                            yield ConversationSample(
                                conversation_id=item.get("id", str(time.time())),
                                turns=turns,
                                metadata=item.get("metadata")
                            )
        except Exception as e:
            logging.getLogger(__name__).error(f"加載對話數據失敗: {e}")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """迭代器，返回批次數據"""
        current_batch = []
        
        for conversation in self.load_conversations_from_file(self.data_source):
            sample = self.processor.prepare_training_sample(conversation)
            if sample:
                current_batch.append(sample)
                
                if len(current_batch) >= self.batch_size:
                    # 處理批次
                    batch_data = smart_collate_fn(
                        current_batch, self.tokenizer, self.max_length, pack_sequences=True
                    )
                    if batch_data["input_ids"] is not None:
                        yield batch_data
                    current_batch = []
        
        # 處理剩餘數據
        if current_batch:
            batch_data = smart_collate_fn(
                current_batch, self.tokenizer, self.max_length, pack_sequences=True
            )
            if batch_data["input_ids"] is not None:
                yield batch_data

class ConversationMetrics:
    """對話評估指標"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置指標"""
        self.total_tokens = 0
        self.total_conversations = 0
        self.avg_turn_length = 0
        self.response_times = []
    
    def update(self, conversation_length: int, response_time: float):
        """更新指標"""
        self.total_conversations += 1
        self.total_tokens += conversation_length
        self.response_times.append(response_time)
        self.avg_turn_length = self.total_tokens / self.total_conversations
    
    def get_metrics(self) -> Dict[str, float]:
        """獲取當前指標"""
        if not self.response_times:
            return {}
        
        return {
            "total_conversations": self.total_conversations,
            "avg_turn_length": self.avg_turn_length,
            "avg_response_time": sum(self.response_times) / len(self.response_times),
            "tokens_per_second": self.total_tokens / sum(self.response_times) if sum(self.response_times) > 0 else 0
        }