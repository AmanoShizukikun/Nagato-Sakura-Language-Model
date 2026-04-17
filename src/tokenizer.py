import json
import logging
import os
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
                self.logger.debug(f"分詞器已加載，詞彙量: {self.transformers_tokenizer.vocab_size}")
            except Exception as e:
                self.logger.error(f"分詞器加載失敗: {e}")
                raise
        else:
            raise FileNotFoundError("分詞器文件不存在")
    
    def create_and_train_tokenizer(
        self,
        texts_iterator: Iterator[str],
        vocab_size: int,
        min_frequency: int = 2,
        total_texts: Optional[int] = None,
        num_threads: int = 0,
    ):
        """創建並訓練分詞器"""
        previous_tokenizers_parallelism = os.environ.get("TOKENIZERS_PARALLELISM")
        previous_rayon_threads = os.environ.get("RAYON_NUM_THREADS")
        tokenizers_parallelism_override_applied = False
        rayon_override_applied = False
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers

            # 讓 Rust tokenizers 明確啟用平行化，並允許手動指定執行緒。
            if previous_tokenizers_parallelism is None:
                os.environ["TOKENIZERS_PARALLELISM"] = "true"
                tokenizers_parallelism_override_applied = True
            if num_threads and int(num_threads) > 0:
                os.environ["RAYON_NUM_THREADS"] = str(max(1, int(num_threads)))
                rayon_override_applied = True
            
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
                min_frequency=max(1, int(min_frequency)),
                show_progress=True
            )
            
            # 訓練
            self.logger.info(f"開始訓練分詞器，目標詞彙量: {vocab_size}")
            if total_texts is not None and total_texts > 0:
                self.logger.info(f"分詞器語料總文本數: {total_texts}")
                tokenizer.train_from_iterator(texts_iterator, trainer=trainer, length=int(total_texts))
            else:
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
        finally:
            if tokenizers_parallelism_override_applied:
                if previous_tokenizers_parallelism is None:
                    os.environ.pop("TOKENIZERS_PARALLELISM", None)
                else:
                    os.environ["TOKENIZERS_PARALLELISM"] = previous_tokenizers_parallelism
            if rayon_override_applied:
                if previous_rayon_threads is None:
                    os.environ.pop("RAYON_NUM_THREADS", None)
                else:
                    os.environ["RAYON_NUM_THREADS"] = previous_rayon_threads
    
    def prepare_tokenizer(
        self,
        training_data: list,
        target_vocab_size: int,
        force_retrain: bool = False,
        min_frequency: int = 2,
        max_training_samples: int = 0,
        num_threads: int = 0,
    ):
        """準備分詞器"""

        def _clean_optional_text(value: Any) -> str:
            if not isinstance(value, str):
                return ""
            text = value.strip()
            if text.lower() in {"", "none", "null", "n/a", "nan"}:
                return ""
            return text

        def _extract_supervised_fields(item: dict) -> tuple[str, str, str]:
            candidates = [
                (item.get("instruction"), item.get("input"), item.get("output")),
                (item.get("zh_instruction"), item.get("zh_input"), item.get("zh_output")),
                (item.get("en_instruction"), item.get("en_input"), item.get("en_output")),
                (item.get("prompt"), "", item.get("completion")),
            ]

            for instruction_raw, input_raw, output_raw in candidates:
                if not isinstance(instruction_raw, str) or not isinstance(output_raw, str):
                    continue

                instruction = _clean_optional_text(instruction_raw)
                output = output_raw.strip()
                input_text = _clean_optional_text(input_raw)
                if instruction and output:
                    return instruction, input_text, output

            return "", "", ""

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
            sample_cap = int(max_training_samples) if max_training_samples else 0
            if sample_cap > 0:
                training_subset = training_data[:sample_cap]
                self.logger.info(f"分詞器訓練樣本上限: {sample_cap}（實際使用 {len(training_subset)} 筆）")
            else:
                training_subset = training_data
                if len(training_subset) > 1_000_000:
                    self.logger.warning(
                        "分詞器訓練樣本超過 100 萬，可能耗時很長。"
                        "可使用 --tokenizer_train_max_samples 限制樣本數以提速。"
                    )

            valid_sample_count = 0
            for item in tqdm(training_subset, desc="統計分詞樣本"):
                if not isinstance(item, dict):
                    continue
                instruction, input_text, output = _extract_supervised_fields(item)
                if instruction and output:
                    valid_sample_count += 1

            if valid_sample_count <= 0:
                raise ValueError("沒有收集到用於訓練分詞器的文本")

            total_texts = valid_sample_count * 2

            def text_iterator():
                for item in training_subset:
                    if not isinstance(item, dict):
                        continue
                    instruction, input_text, output = _extract_supervised_fields(item)
                    if not instruction or not output:
                        continue
                    instruction_with_input = f"{instruction}\n{input_text}".strip() if input_text else instruction
                    yield instruction_with_input
                    yield output
            
            self.create_and_train_tokenizer(
                text_iterator(),
                target_vocab_size,
                min_frequency=min_frequency,
                total_texts=total_texts,
                num_threads=num_threads,
            )
        
        if self.transformers_tokenizer is None:
            raise RuntimeError("分詞器初始化失敗")