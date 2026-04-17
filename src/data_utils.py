import torch
import logging
from typing import List, Dict, Any, Optional
from transformers import PreTrainedTokenizerFast


def smart_collate_fn(batch: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerFast, 
                    max_seq_length: int, pack_sequences: bool = True) -> Dict[str, Optional[torch.Tensor]]:
    """智能數據整理函數，支持序列打包"""
    
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