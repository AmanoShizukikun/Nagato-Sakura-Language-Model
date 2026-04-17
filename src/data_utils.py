import json
import logging
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


def _clean_optional_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if text.lower() in {"", "none", "null", "n/a", "nan"}:
        return ""
    return text


def _normalize_supervised_item(item: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if not isinstance(item, dict):
        return None

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
            return {
                "instruction": instruction,
                "input": input_text,
                "output": output,
            }

    return None


def _compose_instruction_text(item: Dict[str, str]) -> str:
    instruction = _clean_optional_text(item.get("instruction"))
    input_text = _clean_optional_text(item.get("input"))
    if input_text:
        return f"{instruction}\n{input_text}".strip()
    return instruction


def _normalize_pretokenized_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None

    input_ids = item.get("input_ids")
    prompt_len = item.get("prompt_len")
    seq_len = item.get("seq_len")
    label_token_count = item.get("label_token_count")

    if not isinstance(input_ids, list) or not input_ids:
        return None
    if not all(isinstance(token_id, int) for token_id in input_ids):
        return None
    if not isinstance(prompt_len, int):
        return None

    actual_seq_len = len(input_ids)
    if isinstance(seq_len, int):
        actual_seq_len = max(0, min(seq_len, actual_seq_len))
    actual_prompt_len = max(0, min(prompt_len, actual_seq_len))

    actual_label_token_count = actual_seq_len - actual_prompt_len
    if isinstance(label_token_count, int):
        actual_label_token_count = max(0, min(label_token_count, actual_label_token_count))

    if actual_label_token_count <= 0:
        return None

    return {
        "input_ids": input_ids[:actual_seq_len],
        "prompt_len": actual_prompt_len,
        "seq_len": actual_seq_len,
        "label_token_count": actual_label_token_count,
    }


def _tokenize_batch_with_fallback(
    tokenizer: PreTrainedTokenizerFast,
    texts: List[str],
    max_seq_length: int,
    padding: Union[bool, str] = "longest",
) -> Dict[str, torch.Tensor]:
    try:
        return tokenizer(
            texts,
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
    except Exception:
        encoded_sequences = []
        for text in texts:
            token_ids = tokenizer.encode(str(text), add_special_tokens=False)
            if max_seq_length and len(token_ids) > max_seq_length:
                token_ids = token_ids[:max_seq_length]
            encoded_sequences.append(token_ids)

        if not encoded_sequences:
            return {
                "input_ids": torch.empty((0, 0), dtype=torch.long),
                "attention_mask": torch.empty((0, 0), dtype=torch.long),
            }

        if padding in (True, "longest"):
            target_len = max(len(ids) for ids in encoded_sequences)
        elif padding == "max_length":
            target_len = max_seq_length
        else:
            target_len = max(len(ids) for ids in encoded_sequences)

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer is missing both pad_token_id and eos_token_id")

        padded_ids = []
        attention_masks = []
        for ids in encoded_sequences:
            pad_len = max(0, target_len - len(ids))
            padded_ids.append(ids + [pad_token_id] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }


def build_pretokenized_supervised_item(
    item: Dict[str, Any],
    tokenizer: PreTrainedTokenizerFast,
    max_seq_length: int,
) -> Optional[Dict[str, Any]]:
    pretokenized = _normalize_pretokenized_item(item)
    if pretokenized is not None:
        return pretokenized

    normalized_item = _normalize_supervised_item(item)
    if normalized_item is None:
        return None

    bos = tokenizer.bos_token or "<s>"
    eos = tokenizer.eos_token or "</s>"
    prompt = _compose_instruction_text(normalized_item)
    completion = str(normalized_item["output"]).strip()
    if not prompt or not completion:
        return None

    prompt_with_bos = f"{bos}{prompt}\n"
    full_text = f"{prompt_with_bos}{completion}{eos}"

    input_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if max_seq_length and len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
    if not input_ids:
        return None

    prompt_len = len(tokenizer.encode(prompt_with_bos, add_special_tokens=False))
    prompt_len = min(prompt_len, len(input_ids))
    label_token_count = len(input_ids) - prompt_len
    if label_token_count <= 0:
        return None

    return {
        "input_ids": input_ids,
        "prompt_len": prompt_len,
        "seq_len": len(input_ids),
        "label_token_count": label_token_count,
    }


def _batch_rows_from_columnar_batch(batch: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not batch:
        return []

    keys = list(batch.keys())
    if not keys:
        return []

    batch_size = len(batch[keys[0]])
    rows: List[Dict[str, Any]] = []
    for row_idx in range(batch_size):
        rows.append({key: batch[key][row_idx] for key in keys})
    return rows


def _batched_tokenize_texts(
    tokenizer: PreTrainedTokenizerFast,
    texts: List[str],
    max_seq_length: int,
) -> List[List[int]]:
    if not texts:
        return []

    try:
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_attention_mask=False,
        )
        input_ids = encoded.get("input_ids", [])
        if isinstance(input_ids, list):
            return [list(map(int, ids)) for ids in input_ids]
    except Exception:
        pass

    encoded_sequences: List[List[int]] = []
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if max_seq_length and len(token_ids) > max_seq_length:
            token_ids = token_ids[:max_seq_length]
        encoded_sequences.append([int(token_id) for token_id in token_ids])
    return encoded_sequences


def _batched_pretokenize_records(
    batch: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizerFast,
    max_seq_length: int,
) -> Dict[str, List[Any]]:
    rows = _batch_rows_from_columnar_batch(batch)
    if not rows:
        return {
            "input_ids": [],
            "prompt_len": [],
            "seq_len": [],
            "label_token_count": [],
        }

    bos = tokenizer.bos_token or "<s>"
    eos = tokenizer.eos_token or "</s>"
    prepared_items: List[Dict[str, Any]] = []
    prompt_texts: List[str] = []
    full_texts: List[str] = []

    for row in rows:
        pretokenized = _normalize_pretokenized_item(row)
        if pretokenized is not None:
            prepared_items.append(pretokenized)
            continue

        normalized_item = _normalize_supervised_item(row)
        if normalized_item is None:
            continue

        prompt = _compose_instruction_text(normalized_item)
        completion = str(normalized_item["output"]).strip()
        if not prompt or not completion:
            continue

        prompt_with_bos = f"{bos}{prompt}\n"
        prepared_items.append({
            "_batched_prompt_index": len(prompt_texts),
        })
        prompt_texts.append(prompt_with_bos)
        full_texts.append(f"{prompt_with_bos}{completion}{eos}")

    encoded_prompt_ids = _batched_tokenize_texts(tokenizer, prompt_texts, max_seq_length=max_seq_length)
    encoded_full_ids = _batched_tokenize_texts(tokenizer, full_texts, max_seq_length=max_seq_length)

    output = {
        "input_ids": [],
        "prompt_len": [],
        "seq_len": [],
        "label_token_count": [],
    }

    for item in prepared_items:
        if "input_ids" in item:
            input_ids = item["input_ids"]
            prompt_len = item["prompt_len"]
            seq_len = item["seq_len"]
            label_token_count = item["label_token_count"]
        else:
            encoded_idx = int(item["_batched_prompt_index"])
            if encoded_idx >= len(encoded_prompt_ids) or encoded_idx >= len(encoded_full_ids):
                continue

            input_ids = encoded_full_ids[encoded_idx]
            if not input_ids:
                continue

            seq_len = len(input_ids)
            prompt_len = min(len(encoded_prompt_ids[encoded_idx]), seq_len)
            label_token_count = seq_len - prompt_len
            if label_token_count <= 0:
                continue

        output["input_ids"].append(input_ids[:seq_len])
        output["prompt_len"].append(int(prompt_len))
        output["seq_len"].append(int(seq_len))
        output["label_token_count"].append(int(label_token_count))

    return output


def _build_pretokenize_cache_fingerprint(
    data_list: List[Dict[str, Any]],
    max_seq_length: int,
    cache_namespace: str = "",
) -> str:
    digest = hashlib.sha256()
    digest.update(str(max_seq_length).encode("utf-8"))
    digest.update(cache_namespace.encode("utf-8"))
    digest.update(str(len(data_list)).encode("utf-8"))

    if not data_list:
        return digest.hexdigest()[:16]

    sample_budget = min(len(data_list), 4096)
    if sample_budget <= 0:
        return digest.hexdigest()[:16]

    if len(data_list) <= sample_budget:
        sample_indices = list(range(len(data_list)))
    else:
        sample_indices = sorted({
            0,
            len(data_list) - 1,
            len(data_list) // 2,
            *[
                min(len(data_list) - 1, (len(data_list) * step) // max(1, sample_budget - 1))
                for step in range(sample_budget)
            ],
        })

    for idx in sample_indices:
        normalized = _normalize_supervised_item(data_list[idx])
        if normalized is None:
            normalized = _normalize_pretokenized_item(data_list[idx]) or {}
        digest.update(json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8"))

    return digest.hexdigest()[:16]


def pretokenize_supervised_dataset(
    data_list: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerFast,
    max_seq_length: int,
    desc: str = "Pretokenizing",
    batch_size: int = 1024,
    num_proc: int = 1,
    cache_dir: Optional[Union[str, Path]] = None,
    cache_namespace: str = "",
    use_cache: bool = True,
    return_dataset: bool = False,
) -> Union[List[Dict[str, Any]], Dataset]:
    if not data_list:
        empty_dataset = Dataset.from_list([])
        return empty_dataset if return_dataset else []

    batch_size = max(1, int(batch_size))
    num_proc = max(1, int(num_proc))
    if os.name == "nt":
        num_proc = 1

    if num_proc > 1:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    else:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_fingerprint = _build_pretokenize_cache_fingerprint(
            data_list=data_list,
            max_seq_length=max_seq_length,
            cache_namespace=cache_namespace,
        )
        cache_path = cache_root / f"{desc.lower().replace(' ', '_')}_{cache_fingerprint}"

        if use_cache and cache_path.exists():
            cached_dataset = load_from_disk(str(cache_path))
            return cached_dataset if return_dataset else list(cached_dataset)

    raw_dataset = Dataset.from_list(data_list)
    map_kwargs: Dict[str, Any] = {
        "batched": True,
        "batch_size": batch_size,
        "remove_columns": raw_dataset.column_names,
        "fn_kwargs": {
            "tokenizer": tokenizer,
            "max_seq_length": max_seq_length,
        },
        "desc": desc,
        "load_from_cache_file": False,
    }
    if num_proc > 1:
        map_kwargs["num_proc"] = num_proc

    tokenized_dataset = raw_dataset.map(
        _batched_pretokenize_records,
        **map_kwargs,
    )

    if len(tokenized_dataset) <= 0:
        empty_dataset = Dataset.from_list([])
        return empty_dataset if return_dataset else []

    if cache_path is not None and use_cache:
        tokenized_dataset.save_to_disk(str(cache_path))

    return tokenized_dataset if return_dataset else list(tokenized_dataset)


@dataclass
class ConversationSample:
    conversation_id: str
    turns: List[Dict[str, str]]
    metadata: Optional[Dict[str, Any]] = None


class ConversationDataProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)

    def format_conversation(
        self,
        turns: List[Dict[str, str]],
        system_message: Optional[str] = None,
    ) -> str:
        formatted_text = ""

        if system_message:
            formatted_text += f"System: {system_message}\n"

        for turn in turns:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "user":
                formatted_text += f"User: {content}\n"
            elif role == "assistant":
                formatted_text += f"Assistant: {content}\n"

        return formatted_text.strip()

    def prepare_training_sample(self, conversation: ConversationSample) -> Optional[Dict[str, str]]:
        if len(conversation.turns) < 2:
            return None

        user_turns = [turn for turn in conversation.turns if turn["role"] == "user"]
        assistant_turns = [turn for turn in conversation.turns if turn["role"] == "assistant"]
        if not user_turns or not assistant_turns:
            return None

        prompt_turns = conversation.turns[:-1]
        prompt = self.format_conversation(prompt_turns)
        prompt += "\nAssistant:"

        last_assistant_turn = conversation.turns[-1]
        if last_assistant_turn["role"] != "assistant":
            return None

        return {
            "instruction": prompt,
            "input": "",
            "output": last_assistant_turn["content"],
            "conversation_id": conversation.conversation_id,
        }

    def process_conversation_dataset(self, conversations: List[ConversationSample]) -> List[Dict[str, str]]:
        training_samples = []
        for conversation in conversations:
            sample = self.prepare_training_sample(conversation)
            if sample:
                training_samples.append(sample)

        self.logger.info(
            "Converted %s conversation samples into %s supervised samples",
            len(conversations),
            len(training_samples),
        )
        return training_samples


def smart_collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerFast,
    max_seq_length: int,
    pack_sequences: bool = True,
) -> Dict[str, Optional[torch.Tensor]]:
    valid_batch = []
    for item in batch:
        prepared = build_pretokenized_supervised_item(item, tokenizer, max_seq_length)
        if prepared is not None:
            valid_batch.append(prepared)

    if not valid_batch:
        return {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
            "batch_token_count": 0,
        }

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer is missing pad_token_id")

    if pack_sequences and len(valid_batch) > 1:
        valid_batch.sort(key=lambda item: item["seq_len"])

    target_len = max(item["seq_len"] for item in valid_batch)
    batch_size = len(valid_batch)
    input_ids = torch.full((batch_size, target_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, target_len), dtype=torch.long)
    labels = torch.full((batch_size, target_len), -100, dtype=torch.long)
    batch_token_count = 0

    for row_idx, item in enumerate(valid_batch):
        seq_len = min(item["seq_len"], target_len)
        if seq_len <= 0:
            continue

        ids_tensor = torch.tensor(item["input_ids"][:seq_len], dtype=torch.long)
        input_ids[row_idx, :seq_len] = ids_tensor
        attention_mask[row_idx, :seq_len] = 1

        prompt_len = min(item["prompt_len"], seq_len)
        if prompt_len < seq_len:
            labels[row_idx, prompt_len:seq_len] = ids_tensor[prompt_len:seq_len]
            batch_token_count += int(item["label_token_count"])

    if batch_token_count <= 0:
        return {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
            "batch_token_count": 0,
        }

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "batch_token_count": batch_token_count,
    }


def stream_collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerFast,
    max_seq_length: int,
) -> Dict[str, torch.Tensor]:
    texts = []
    for item in batch:
        normalized_item = _normalize_supervised_item(item) if isinstance(item, dict) else None
        if normalized_item:
            composed_prompt = _compose_instruction_text(normalized_item)
            if composed_prompt:
                texts.append(composed_prompt)
                continue

        if isinstance(item, dict) and "text" in item:
            texts.append(item["text"])
        elif isinstance(item, dict) and "prompt" in item:
            texts.append(item["prompt"])

    if not texts:
        return {"input_ids": torch.empty(0, 0, dtype=torch.long)}

    return _tokenize_batch_with_fallback(
        tokenizer=tokenizer,
        texts=texts,
        max_seq_length=max_seq_length,
        padding=True,
    )


class EarlyStoppingCallback:
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.patience_counter = 0
        self.should_stop = False
        self.last_improved = False

    def __call__(self, current_score: float) -> bool:
        self.last_improved = False
        if self.best_score is None:
            self.best_score = current_score
            self.last_improved = True
            return False

        if self.mode == "min":
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.patience_counter = 0
                self.last_improved = True
            else:
                self.patience_counter += 1
        else:
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.patience_counter = 0
                self.last_improved = True
            else:
                self.patience_counter += 1

        if self.patience_counter >= self.patience:
            self.should_stop = True
            return True
        return False

    @property
    def patience_remaining(self) -> int:
        return max(0, self.patience - self.patience_counter)


class StreamingDataLoader:
    def __init__(
        self,
        data_source: str,
        tokenizer: PreTrainedTokenizerFast,
        batch_size: int = 1,
        max_length: int = 2048,
    ):
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.processor = ConversationDataProcessor(tokenizer, max_length)

    def load_conversations_from_file(self, file_path: str) -> Iterator[ConversationSample]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue

                    if "conversation" in item:
                        yield ConversationSample(
                            conversation_id=item.get("id", str(time.time())),
                            turns=item["conversation"],
                            metadata=item.get("metadata"),
                        )
                    elif "turns" in item:
                        yield ConversationSample(
                            conversation_id=item.get("id", str(time.time())),
                            turns=item["turns"],
                            metadata=item.get("metadata"),
                        )
                    elif "prompt" in item and "completion" in item:
                        yield ConversationSample(
                            conversation_id=item.get("id", str(time.time())),
                            turns=[
                                {"role": "user", "content": item["prompt"]},
                                {"role": "assistant", "content": item["completion"]},
                            ],
                            metadata=item.get("metadata"),
                        )
                    elif "instruction" in item and "output" in item:
                        instruction = _clean_optional_text(item.get("instruction"))
                        input_text = _clean_optional_text(item.get("input"))
                        prompt_text = f"{instruction}\n{input_text}".strip() if input_text else instruction
                        output_text = str(item.get("output", "")).strip()
                        if prompt_text and output_text:
                            yield ConversationSample(
                                conversation_id=item.get("id", str(time.time())),
                                turns=[
                                    {"role": "user", "content": prompt_text},
                                    {"role": "assistant", "content": output_text},
                                ],
                                metadata=item.get("metadata"),
                            )
        except Exception as e:
            logging.getLogger(__name__).error("Failed to load conversation dataset: %s", e)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        current_batch = []

        for conversation in self.load_conversations_from_file(self.data_source):
            sample = self.processor.prepare_training_sample(conversation)
            if sample:
                current_batch.append(sample)

                if len(current_batch) >= self.batch_size:
                    batch_data = smart_collate_fn(
                        current_batch,
                        self.tokenizer,
                        self.max_length,
                        pack_sequences=True,
                    )
                    if batch_data["input_ids"] is not None:
                        yield batch_data
                    current_batch = []

        if current_batch:
            batch_data = smart_collate_fn(
                current_batch,
                self.tokenizer,
                self.max_length,
                pack_sequences=True,
            )
            if batch_data["input_ids"] is not None:
                yield batch_data


class ConversationMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_tokens = 0
        self.total_conversations = 0
        self.avg_turn_length = 0.0
        self.response_times = []

    def update(self, conversation_length: int, response_time: float):
        self.total_conversations += 1
        self.total_tokens += conversation_length
        self.response_times.append(response_time)
        self.avg_turn_length = self.total_tokens / self.total_conversations

    def get_metrics(self) -> Dict[str, float]:
        if not self.response_times:
            return {}

        total_time = sum(self.response_times)
        return {
            "total_conversations": float(self.total_conversations),
            "avg_turn_length": float(self.avg_turn_length),
            "avg_response_time": total_time / len(self.response_times),
            "tokens_per_second": self.total_tokens / total_time if total_time > 0 else 0.0,
        }
