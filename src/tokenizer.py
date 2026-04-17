import logging
import os
from pathlib import Path
from typing import Any, Iterator, Optional

from transformers import PreTrainedTokenizerFast
from tqdm import tqdm


class TokenizerManager:
    """Manage tokenizer loading, training, and persistence."""

    def __init__(self, tokenizer_path: Path):
        self.tokenizer_path = tokenizer_path
        self.logger = logging.getLogger(__name__)
        self.tokenizer_object: Optional[Any] = None
        self.transformers_tokenizer: Optional[PreTrainedTokenizerFast] = None

    def load_tokenizer(self):
        """Load a previously saved tokenizer."""
        if not self.tokenizer_path.exists():
            raise FileNotFoundError("Tokenizer file not found")

        try:
            from tokenizers import Tokenizer

            self.tokenizer_object = Tokenizer.from_file(str(self.tokenizer_path))
            self.transformers_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=self.tokenizer_object,
                unk_token="<unk>",
                pad_token="<pad>",
                bos_token="<s>",
                eos_token="</s>",
                mask_token="<mask>",
            )
            self.logger.debug("Loaded tokenizer with vocab size %s", self.transformers_tokenizer.vocab_size)
        except Exception as exc:
            self.logger.error("Failed to load tokenizer: %s", exc)
            raise

    def create_and_train_tokenizer(
        self,
        texts_iterator: Iterator[str],
        vocab_size: int,
        min_frequency: int = 2,
        total_texts: Optional[int] = None,
        num_threads: int = 0,
    ):
        """Create and train a byte-level BPE tokenizer."""
        previous_tokenizers_parallelism = os.environ.get("TOKENIZERS_PARALLELISM")
        previous_rayon_threads = os.environ.get("RAYON_NUM_THREADS")
        tokenizers_parallelism_override_applied = False
        rayon_override_applied = False
        try:
            from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

            if previous_tokenizers_parallelism is None:
                os.environ["TOKENIZERS_PARALLELISM"] = "true"
                tokenizers_parallelism_override_applied = True
            if num_threads and int(num_threads) > 0:
                os.environ["RAYON_NUM_THREADS"] = str(max(1, int(num_threads)))
                rayon_override_applied = True

            tokenizer = Tokenizer(models.BPE())
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.decoder = decoders.ByteLevel()

            special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                min_frequency=max(1, int(min_frequency)),
                show_progress=True,
            )

            self.logger.info("Training tokenizer with target vocab size %s", vocab_size)
            if total_texts is not None and total_texts > 0:
                self.logger.info("Tokenizer corpus text count: %s", total_texts)
                tokenizer.train_from_iterator(texts_iterator, trainer=trainer, length=int(total_texts))
            else:
                tokenizer.train_from_iterator(texts_iterator, trainer=trainer)

            tokenizer.save(str(self.tokenizer_path))
            self.logger.info("Saved tokenizer to %s", self.tokenizer_path)

            self.tokenizer_object = tokenizer
            self.transformers_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                unk_token="<unk>",
                pad_token="<pad>",
                bos_token="<s>",
                eos_token="</s>",
                mask_token="<mask>",
            )
            self.logger.info("Actual tokenizer vocab size: %s", self.transformers_tokenizer.vocab_size)
        except Exception as exc:
            self.logger.error("Failed to train tokenizer: %s", exc)
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
        """Ensure a tokenizer is available, training one when required."""

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
            self.logger.info("Using existing tokenizer")
            try:
                self.load_tokenizer()
            except Exception as exc:
                self.logger.warning("Failed to load existing tokenizer (%s), retraining", exc)
                force_retrain = True

        if force_retrain or not self.transformers_tokenizer:
            self.logger.info("Training tokenizer from dataset")

            sample_cap = int(max_training_samples) if max_training_samples else 0
            if sample_cap > 0:
                training_subset = training_data[:sample_cap]
                self.logger.info(
                    "Tokenizer sample cap: %s (using %s records)",
                    sample_cap,
                    len(training_subset),
                )
            else:
                training_subset = training_data
                if len(training_subset) > 1_000_000:
                    self.logger.warning(
                        "Tokenizer training data exceeds 1M rows; consider --tokenizer_train_max_samples"
                    )

            valid_sample_count = 0
            for item in tqdm(training_subset, desc="Counting tokenizer samples"):
                if not isinstance(item, dict):
                    continue
                instruction, input_text, output = _extract_supervised_fields(item)
                if instruction and output:
                    valid_sample_count += 1

            if valid_sample_count <= 0:
                raise ValueError("No valid supervised samples were found for tokenizer training")

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
            raise RuntimeError("Tokenizer is still unavailable after preparation")
