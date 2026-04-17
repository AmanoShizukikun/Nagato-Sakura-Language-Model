import json
import logging
from pathlib import Path
from typing import Iterator, Optional, Any

from transformers import PreTrainedTokenizerFast
from tqdm import tqdm


class TokenizerManager:
    """分詞器管理器"""
    
    def __init__(self, tokenizer_path: Path):
        self.tokenizer_path = tokenizer_path
        self.logger = logging.getLogger(__name__)
        self.tokenizer_object: Optional[Any] = None
        self.transformers_tokenizer: Optional[PreTrainedTokenizerFast] = None
    
    def load_tokenizer(self):
        """加載分詞器"""
        if self.tokenizer_path.exists():
            try:
                from tokenizers import Tokenizer
                self.tokenizer_object = Tokenizer.from_file(str(self.tokenizer_path))
                self.transformers_tokenizer = PreTrainedTokenizerFast(
                    tokenizer_object=self.tokenizer_object,
                    unk_token="<unk>",
                    pad_token="<pad>",
                    bos_token="<s>",
                    eos_token="</s>",
                    mask_token="<mask>"
                )
                self.logger.info(f"分詞器已加載，詞彙量: {self.transformers_tokenizer.vocab_size}")
            except Exception as e:
                self.logger.error(f"分詞器加載失敗: {e}")
                raise
        else:
            raise FileNotFoundError("分詞器文件不存在")
    
    def create_and_train_tokenizer(self, texts_iterator: Iterator[str], vocab_size: int):
        """創建並訓練分詞器"""
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers
            
            # 創建BPE分詞器
            tokenizer = Tokenizer(models.BPE())
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.decoder = decoders.ByteLevel()
            
            # 特殊令牌
            special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
            
            # 訓練器
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                min_frequency=2,
                show_progress=True
            )
            
            # 訓練
            self.logger.info(f"開始訓練分詞器，目標詞彙量: {vocab_size}")
            tokenizer.train_from_iterator(texts_iterator, trainer=trainer)
            
            # 保存
            tokenizer.save(str(self.tokenizer_path))
            self.logger.info(f"分詞器已保存到: {self.tokenizer_path}")
            
            # 包裝
            self.tokenizer_object = tokenizer
            self.transformers_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                unk_token="<unk>",
                pad_token="<pad>",
                bos_token="<s>",
                eos_token="</s>",
                mask_token="<mask>"
            )
            
            actual_vocab_size = self.transformers_tokenizer.vocab_size
            self.logger.info(f"分詞器訓練完成，實際詞彙量: {actual_vocab_size}")
            
        except Exception as e:
            self.logger.error(f"分詞器訓練失敗: {e}")
            raise
    
    def prepare_tokenizer(self, training_data: list, target_vocab_size: int, 
                         force_retrain: bool = False):
        """準備分詞器"""
        tokenizer_exists = self.tokenizer_path.exists()
        
        if not force_retrain and tokenizer_exists:
            self.logger.info("使用現有分詞器")
            try:
                self.load_tokenizer()
            except Exception as e:
                self.logger.warning(f"加載現有分詞器失敗: {e}，將重新訓練")
                force_retrain = True
        
        if force_retrain or not self.transformers_tokenizer:
            self.logger.info("訓練新分詞器...")
            
            # 收集文本
            all_texts = []
            for item in tqdm(training_data, desc="收集文本"):
                prompt = str(item.get("prompt", "")).strip()
                completion = str(item.get("completion", "")).strip()
                if prompt:
                    all_texts.append(prompt)
                if completion:
                    all_texts.append(completion)
            
            if not all_texts:
                raise ValueError("沒有收集到用於訓練分詞器的文本")
            
            # 訓練分詞器
            def text_iterator():
                for text in all_texts:
                    yield text
            
            self.create_and_train_tokenizer(text_iterator(), target_vocab_size)
        
        if self.transformers_tokenizer is None:
            raise RuntimeError("分詞器初始化失敗")