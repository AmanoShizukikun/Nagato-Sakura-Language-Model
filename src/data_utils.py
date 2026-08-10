import bisect
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from datasets import Dataset, load_from_disk
from datasets.utils.logging import disable_progress_bar as disable_datasets_progress_bar
from datasets.utils.logging import enable_progress_bar as enable_datasets_progress_bar
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


# ==============================================================================
# 資料正規化與預分詞 (Data Normalization & Pre-tokenization)
# ==============================================================================


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
        if not instruction or not output:
            continue

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


def build_pretokenized_supervised_item(
    item: Dict[str, Any], tokenizer: PreTrainedTokenizerFast, max_seq_length: int
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

    try:
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
            return_offsets_mapping=True,
        )
        input_ids = encoded.get("input_ids", [])
        offsets = encoded.get("offset_mapping", [])
    except Exception:
        input_ids = tokenizer.encode(full_text, add_special_tokens=False)
        if max_seq_length and len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
        offsets = []

    if not input_ids:
        return None

    seq_len = len(input_ids)
    prompt_char_len = len(prompt_with_bos)

    if offsets and len(offsets) == seq_len:
        starts = [s for s, e in offsets]
        prompt_len = bisect.bisect_left(starts, prompt_char_len)
    else:
        # 降級相容路徑：精確編碼 prompt 取得真實 Token 長度，確保 Loss Mask 邊界絕對正確
        prompt_len = min(seq_len, len(tokenizer.encode(prompt_with_bos, add_special_tokens=False)))

    label_token_count = seq_len - prompt_len
    if label_token_count <= 0:
        return None

    return {
        "input_ids": input_ids,
        "prompt_len": prompt_len,
        "seq_len": seq_len,
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


def _batched_tokenize_texts_with_offsets(
    tokenizer: PreTrainedTokenizerFast, texts: List[str], max_seq_length: int
) -> Tuple[List[List[int]], List[List[Tuple[int, int]]]]:
    if not texts:
        return [], []

    try:
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        input_ids = encoded.get("input_ids", [])
        offset_mapping = encoded.get("offset_mapping", [])
        if isinstance(input_ids, list) and isinstance(offset_mapping, list):
            return input_ids, offset_mapping
    except Exception:
        pass

    encoded_sequences: List[List[int]] = []
    offsets_sequences: List[List[Tuple[int, int]]] = []
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if max_seq_length and len(token_ids) > max_seq_length:
            token_ids = token_ids[:max_seq_length]
        encoded_sequences.append([int(token_id) for token_id in token_ids])
        offsets_sequences.append([])  # 空串列明確代表「無真實 offset」，觸發呼叫端精確計算降級路徑

    return encoded_sequences, offsets_sequences


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
            "_batched_prompt_index": len(full_texts),
            "_prompt_char_len": len(prompt_with_bos),
            "_prompt_text": prompt_with_bos,
        })
        full_texts.append(f"{prompt_with_bos}{completion}{eos}")

    # 單次高效分詞 (單次呼叫 Fast Tokenizer，降低 50% FFI 與 GIL 邊界開銷)
    encoded_full_ids, encoded_offsets = _batched_tokenize_texts_with_offsets(
        tokenizer, full_texts, max_seq_length=max_seq_length
    )

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
            if encoded_idx >= len(encoded_full_ids):
                continue

            input_ids = encoded_full_ids[encoded_idx]
            if not input_ids:
                continue

            seq_len = len(input_ids)
            prompt_char_len = int(item["_prompt_char_len"])
            offsets = encoded_offsets[encoded_idx] if encoded_idx < len(encoded_offsets) else []

            # 使用 Offset Mapping 與二分搜尋在 O(log N) 極速計算 prompt_len，無雙重分詞
            if offsets and len(offsets) == seq_len:
                starts = [s for s, e in offsets]
                prompt_len = bisect.bisect_left(starts, prompt_char_len)
            else:
                # 降級相容路徑：精確編碼 prompt 取得真實 Token 長度，確保 Loss Mask 邊界絕對正確
                p_text = str(item.get("_prompt_text", ""))
                prompt_len = min(seq_len, len(tokenizer.encode(p_text, add_special_tokens=False)))

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
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(max_seq_length).encode("utf-8"))
    digest.update(cache_namespace.encode("utf-8"))
    digest.update(str(len(data_list)).encode("utf-8"))
    if tokenizer is not None:
        tokenizer_signature: Dict[str, Any] = {
            "len": int(len(tokenizer)),
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
            "unk_token": tokenizer.unk_token,
            "pad_token": tokenizer.pad_token,
            "bos_token": tokenizer.bos_token,
            "eos_token": tokenizer.eos_token,
            "unk_token_id": tokenizer.unk_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        try:
            backend_payload = json.loads(tokenizer.backend_tokenizer.to_str())
            model_payload = (backend_payload.get("model", {}) if isinstance(backend_payload, dict) else {})
            if isinstance(model_payload, dict):
                tokenizer_signature["model_type"] = model_payload.get("type")
                tokenizer_signature["model_unk_token"] = model_payload.get("unk_token")
                tokenizer_signature["byte_fallback"] = model_payload.get("byte_fallback")
        except Exception:
            pass
        
        digest.update(json.dumps(tokenizer_signature, ensure_ascii=False, sort_keys=True).encode("utf-8"))

    if not data_list:
        return digest.hexdigest()[:16]

    sample_budget = min(len(data_list), 4096)
    if sample_budget <= 0:
        return digest.hexdigest()[:16]

    if len(data_list) <= sample_budget:
        sample_indices = list(range(len(data_list)))
    else:
        sample_indices = sorted(
            {
                0,
                len(data_list) - 1,
                len(data_list) // 2,
                *[
                    min(len(data_list) - 1, (len(data_list) * step) // max(1, sample_budget - 1))
                    for step in range(sample_budget)
                ],
            }
        )

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
    batch_size: int = 8192,
    num_proc: int = 1,
    cache_dir: Optional[Union[str, Path]] = None,
    cache_namespace: str = "",
    use_cache: bool = True,
    return_dataset: bool = False,
) -> Union[List[Dict[str, Any]], Dataset]:
    if not data_list:
        empty_dataset = Dataset.from_list([])
        return empty_dataset if return_dataset else []

    if num_proc is None or num_proc <= 0:
        num_workers = 1
    else:
        num_workers = max(1, int(num_proc))

    # 動態控制 Tokenizer 平行化：多 Worker 模式下禁用 Rust 內部平行處理以防核心爭搶與 Thrashing；單 Worker 模式下啟用 Rust 原生 Rayon 滿核加速
    if num_workers > 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    cache_fingerprint = _build_pretokenize_cache_fingerprint(
        data_list=data_list,
        max_seq_length=max_seq_length,
        cache_namespace=cache_namespace,
        tokenizer=tokenizer,
    )
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"{desc.lower().replace(' ', '_')}_{cache_fingerprint}"
        if use_cache and cache_path.exists():
            cached_dataset = load_from_disk(str(cache_path))
            return cached_dataset if return_dataset else list(cached_dataset)

    slices = [data_list[i : i + batch_size] for i in range(0, len(data_list), batch_size)]
    disable_datasets_progress_bar()
    pbar = tqdm(
        total=len(data_list),
        desc=desc,
        ncols=115,
        leave=False,
        bar_format="{l_bar}{bar}{r_bar}",
    )

    def _process_slice(data_slice: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        batch_dict: Dict[str, List[Any]] = {}
        if data_slice:
            keys = data_slice[0].keys()
            for k in keys:
                batch_dict[k] = [d.get(k) for d in data_slice]
                
        return _batched_pretokenize_records(batch_dict, tokenizer, max_seq_length)

    all_input_ids: List[List[int]] = []
    all_prompt_len: List[int] = []
    all_seq_len: List[int] = []
    all_label_token_count: List[int] = []
    try:
        if num_workers <= 1 or len(slices) <= 1:
            for sl in slices:
                res = _process_slice(sl)
                all_input_ids.extend(res["input_ids"])
                all_prompt_len.extend(res["prompt_len"])
                all_seq_len.extend(res["seq_len"])
                all_label_token_count.extend(res["label_token_count"])
                pbar.update(len(sl))
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_process_slice, sl) for sl in slices]
                for sl, future in zip(slices, futures):
                    res = future.result()
                    all_input_ids.extend(res["input_ids"])
                    all_prompt_len.extend(res["prompt_len"])
                    all_seq_len.extend(res["seq_len"])
                    all_label_token_count.extend(res["label_token_count"])
                    pbar.update(len(sl))
                    
    finally:
        pbar.close()

    if not all_input_ids:
        empty_dataset = Dataset.from_list([])
        return empty_dataset if return_dataset else []

    tokenized_dict = {
        "input_ids": all_input_ids,
        "prompt_len": all_prompt_len,
        "seq_len": all_seq_len,
        "label_token_count": all_label_token_count,
    }
    tokenized_dataset = Dataset.from_dict(tokenized_dict)
    if cache_path is not None and use_cache:
        tokenized_dataset.save_to_disk(str(cache_path))
        # 重新以磁碟記憶體映射方式載入，避免 DataLoader 多進程（尤其 Windows spawn）
        # 需要將整個 in-memory Arrow table 序列化進 pipe，導致大資料集下
        # OSError: [Errno 22] Invalid argument
        tokenized_dataset = load_from_disk(str(cache_path))

    return tokenized_dataset if return_dataset else list(tokenized_dataset)


def _tokenize_batch_with_fallback(
    tokenizer: PreTrainedTokenizerFast,
    texts: List[str],
    max_seq_length: int,
    padding: Union[bool, str] = "longest",
    padding_side: Optional[str] = "left",
) -> Dict[str, torch.Tensor]:
    """批次編碼相容層：優先使用 tokenizer(...)，失敗時退回手動編碼。支援 left-padding 供批次生成。"""
    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    if padding_side is not None and orig_padding_side != padding_side:
        tokenizer.padding_side = padding_side

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
            raise ValueError("分詞器缺少 pad_token_id 與 eos_token_id")

        padded_ids = []
        attention_masks = []
        for ids in encoded_sequences:
            pad_len = max(0, target_len - len(ids))
            if padding_side == "left":
                padded_ids.append([pad_token_id] * pad_len + ids)
                attention_masks.append([0] * pad_len + [1] * len(ids))
            else:
                padded_ids.append(ids + [pad_token_id] * pad_len)
                attention_masks.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }
    finally:
        if padding_side is not None and orig_padding_side != padding_side:
            tokenizer.padding_side = orig_padding_side


# ==============================================================================
# 對話處理器與 Dataset (Conversation Data Processor & Dataset)
# ==============================================================================


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

    def format_conversation(self, turns: List[Dict[str, str]], system_message: Optional[str] = None) -> str:
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
        prompt_turns = conversation.turns[:-1]
        prompt = self.format_conversation(prompt_turns)
        prompt += "\n長門櫻："
        last_assistant_turn = conversation.turns[-1]
        if last_assistant_turn["role"] != "assistant":
            return None

        completion = last_assistant_turn["content"]

        return {
            "instruction": prompt,
            "input": "",
            "output": completion,
            "conversation_id": conversation.conversation_id,
        }

    def process_conversation_dataset(
        self, conversations: List[ConversationSample]
    ) -> List[Dict[str, str]]:
        """處理整個對話數據集"""
        training_samples = []

        for conv in conversations:
            sample = self.prepare_training_sample(conv)
            if sample:
                training_samples.append(sample)

        self.logger.info(
            f"從 {len(conversations)} 個對話中生成了 {len(training_samples)} 個訓練樣本"
        )
        return training_samples


# ==============================================================================
# Collate 函數與 DataLoader (Collate Functions & Streaming DataLoader)
# ==============================================================================


def _round_to_bucket(seq_len: int, max_seq_length: int, min_bucket: int = 128) -> int:
    """將序列長度向上對齊至最近的桶 (Bucket)，平衡動態 Batch 長度算力利用率與 torch.compile 的圖編譯穩定度"""
    if seq_len >= max_seq_length:
        return max_seq_length
    buckets = []
    curr = min_bucket
    while curr < max_seq_length:
        buckets.append(curr)
        if curr < 1024:
            curr += 128
        elif curr < 4096:
            curr += 512
        else:
            curr += 1024
    buckets.append(max_seq_length)
    buckets = sorted(list(set(buckets)))

    for b in buckets:
        if b >= seq_len:
            return b
    return max_seq_length


def smart_collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerFast,
    max_seq_length: int,
    sort_by_length: bool = True,
    pack_sequences: bool = True,
    pad_to_max_length: bool = True,
) -> Dict[str, Optional[torch.Tensor]]:
    """
    智能數據整理函數，支援：
    1. 真實 Sequence Packing (pack_sequences=True)：將多個樣本拼接到 Bucket 區塊，0 算力浪費。
    2. Bucket 離散 Padding (pad_to_max_length=True)：對齊至最近長度桶 (128, 256, 512...)，大幅省顯存與 FLOPs，維護 torch.compile 穩定。
    """
    valid_batch = []
    untokenized_items = []
    for item in batch:
        if item and isinstance(item, dict):
            if "input_ids" in item:
                valid_batch.append(item)
            else:
                untokenized_items.append(item)

    if untokenized_items:
        columnar_batch: Dict[str, List[Any]] = {}
        keys = list(untokenized_items[0].keys())
        for k in keys:
            columnar_batch[k] = [d.get(k) for d in untokenized_items]
        pretokenized_res = _batched_pretokenize_records(columnar_batch, tokenizer, max_seq_length)
        if pretokenized_res and "input_ids" in pretokenized_res:
            for idx in range(len(pretokenized_res["input_ids"])):
                valid_batch.append({
                    "input_ids": pretokenized_res["input_ids"][idx],
                    "prompt_len": pretokenized_res["prompt_len"][idx],
                    "seq_len": pretokenized_res["seq_len"][idx],
                    "label_token_count": pretokenized_res["label_token_count"][idx],
                })

    if not valid_batch:
        return {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
            "batch_token_count": 0,
            "valid_label_count": 0,
            "valid_label_ratio": 0.0,
        }

    pad_token_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )

    if pad_token_id is None:
        raise ValueError("分詞器缺少 pad_token_id")

    if pack_sequences:
        # === 1. First-Fit-Decreasing (FFD) 降序 Bin Packing 算法 ===
        # 先將本批樣本按序列長度從大到小排序，再依次放入能容納它的第一個 Bin (最佳化密度與最低 Padding 浪費)
        valid_batch.sort(key=lambda item: item["seq_len"], reverse=True)

        bins_ids: List[List[int]] = []
        bins_mask: List[List[int]] = []
        bins_labels: List[List[int]] = []
        bins_positions: List[List[int]] = []
        bins_segments: List[List[int]] = []
        bins_segment_counters: List[int] = []

        for item in valid_batch:
            seq_ids = item["input_ids"]
            prompt_len = item["prompt_len"]
            item_labels = [-100] * prompt_len + seq_ids[prompt_len:]
            item_mask = [1] * len(seq_ids)
            item_positions = list(range(len(seq_ids)))

            placed = False
            for b_idx in range(len(bins_ids)):
                if len(bins_ids[b_idx]) + len(seq_ids) <= max_seq_length:
                    seg_id = bins_segment_counters[b_idx]
                    bins_ids[b_idx].extend(seq_ids)
                    bins_mask[b_idx].extend(item_mask)
                    bins_labels[b_idx].extend(item_labels)
                    bins_positions[b_idx].extend(item_positions)
                    bins_segments[b_idx].extend([seg_id] * len(seq_ids))
                    bins_segment_counters[b_idx] += 1
                    placed = True
                    break

            if not placed:
                bins_ids.append(list(seq_ids[:max_seq_length]))
                bins_mask.append(list(item_mask[:max_seq_length]))
                bins_labels.append(list(item_labels[:max_seq_length]))
                bins_positions.append(list(item_positions[:max_seq_length]))
                bins_segments.append([1] * min(len(seq_ids), max_seq_length))
                bins_segment_counters.append(2)

        packed_input_ids = []
        packed_attention_masks = []
        packed_labels = []
        packed_position_ids = []
        packed_segment_ids = []

        max_in_batch = max(len(b) for b in bins_ids) if bins_ids else max_seq_length
        S = _round_to_bucket(max_in_batch, max_seq_length) if pad_to_max_length else max_in_batch
        for b_idx in range(len(bins_ids)):
            curr_ids = bins_ids[b_idx]
            curr_mask = bins_mask[b_idx]
            curr_labels = bins_labels[b_idx]
            curr_positions = bins_positions[b_idx]
            curr_segments = bins_segments[b_idx]

            pad_len = S - len(curr_ids)
            if pad_len > 0:
                curr_ids.extend([pad_token_id] * pad_len)
                curr_mask.extend([0] * pad_len)
                curr_labels.extend([-100] * pad_len)
                curr_positions.extend([0] * pad_len)
                curr_segments.extend([0] * pad_len)

            packed_input_ids.append(curr_ids[:S])
            packed_attention_masks.append(curr_mask[:S])
            packed_labels.append(curr_labels[:S])
            packed_position_ids.append(curr_positions[:S])
            packed_segment_ids.append(curr_segments[:S])

        input_ids = torch.tensor(packed_input_ids, dtype=torch.long)
        attention_mask = torch.tensor(packed_attention_masks, dtype=torch.long)
        labels = torch.tensor(packed_labels, dtype=torch.long)
        position_ids = torch.tensor(packed_position_ids, dtype=torch.long)
        segment_ids = torch.tensor(packed_segment_ids, dtype=torch.long)

    else:
        # === 傳統 Padding 模式 ===
        if sort_by_length and len(valid_batch) > 1:
            valid_batch.sort(key=lambda item: item["seq_len"])

        max_in_batch = max(item["seq_len"] for item in valid_batch) if valid_batch else max_seq_length
        target_len = _round_to_bucket(max_in_batch, max_seq_length) if pad_to_max_length else max_in_batch
        batch_size = len(valid_batch)
        input_ids = torch.full((batch_size, target_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, target_len), dtype=torch.long)
        labels = torch.full((batch_size, target_len), -100, dtype=torch.long)
        position_ids = torch.zeros((batch_size, target_len), dtype=torch.long)
        segment_ids = torch.ones((batch_size, target_len), dtype=torch.long)

        for row_idx, item in enumerate(valid_batch):
            seq_len = min(item["seq_len"], target_len)
            if seq_len <= 0:
                continue

            ids_tensor = torch.tensor(item["input_ids"][:seq_len], dtype=torch.long)
            input_ids[row_idx, :seq_len] = ids_tensor
            attention_mask[row_idx, :seq_len] = 1
            position_ids[row_idx, :seq_len] = torch.arange(seq_len, dtype=torch.long)

            prompt_len = min(item["prompt_len"], seq_len)
            if prompt_len < seq_len:
                labels[row_idx, prompt_len:seq_len] = ids_tensor[prompt_len:seq_len]

    valid_label_count = int((labels != -100).sum().item())
    total_label_count = int(labels.numel())
    valid_label_ratio = float(valid_label_count / max(1, total_label_count))

    if valid_label_count <= 0:
        return {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
            "position_ids": None,
            "segment_ids": None,
            "batch_token_count": 0,
            "valid_label_count": 0,
            "valid_label_ratio": 0.0,
        }

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "position_ids": position_ids,
        "segment_ids": segment_ids,
        "batch_token_count": valid_label_count,
        "valid_label_count": valid_label_count,
        "valid_label_ratio": valid_label_ratio,
    }


def stream_collate_fn(
    batch: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerFast, max_seq_length: int
) -> Dict[str, torch.Tensor]:
    """流式生成專用的collate函數 (預設使用 left-padding 以支援正確的批次生成)"""
    
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

    encoded = _tokenize_batch_with_fallback(
        tokenizer=tokenizer,
        texts=texts,
        max_seq_length=max_seq_length,
        padding=True,
        padding_side="left",
    )

    return encoded


# ==============================================================================
# 訓練回調與指標 (Training Callbacks & Metrics)
# ==============================================================================


class EarlyStoppingCallback:
    """早停回調"""

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
        else:  # mode == 'max'
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
    """流式數據加載器，用於處理大型對話數據集"""

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
        """
        從文件加載對話。
        - 對於 .jsonl 格式：支援真正逐行流式讀取，具備超低 RAM 佔用 ($O(1)$ 常數記憶體)。
        - 對於標準 .json 格式：將整檔載入記憶體 (超大型數據集建議轉為 .jsonl 格式以獲得最佳記憶體效能)。
        """
        file_path_obj = Path(file_path)
        try:
            if file_path_obj.suffix.lower() == ".jsonl":
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            sample = self._parse_item_dict(item)
                            if sample:
                                yield sample
                        except json.JSONDecodeError:
                            continue
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for item in data:
                        sample = self._parse_item_dict(item)
                        if sample:
                            yield sample
        except Exception as e:
            logging.getLogger(__name__).error(f"加載對話數據失敗: {e}")

    def _parse_item_dict(self, item: Dict[str, Any]) -> Optional[ConversationSample]:
        """解析單筆條目為 ConversationSample 結構"""
        if not isinstance(item, dict):
            return None
        if "conversation" in item:
            return ConversationSample(
                conversation_id=item.get("id", str(time.time())),
                turns=item["conversation"],
                metadata=item.get("metadata"),
            )
        elif "turns" in item:
            return ConversationSample(
                conversation_id=item.get("id", str(time.time())),
                turns=item["turns"],
                metadata=item.get("metadata"),
            )
        elif "prompt" in item and "completion" in item:
            turns = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["completion"]},
            ]
            return ConversationSample(
                conversation_id=item.get("id", str(time.time())),
                turns=turns,
                metadata=item.get("metadata"),
            )
        elif "instruction" in item and "output" in item:
            instruction = _clean_optional_text(item.get("instruction"))
            input_text = _clean_optional_text(item.get("input"))
            prompt_text = (
                f"{instruction}\n{input_text}".strip()
                if input_text
                else instruction
            )
            output_text = str(item.get("output", "")).strip()
            if prompt_text and output_text:
                turns = [
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": output_text},
                ]
                return ConversationSample(
                    conversation_id=item.get("id", str(time.time())),
                    turns=turns,
                    metadata=item.get("metadata"),
                )
        return None

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """迭代器，返回批次數據"""
        
        current_batch = []
        for conversation in self.load_conversations_from_file(self.data_source):
            sample = self.processor.prepare_training_sample(conversation)
            if sample:
                current_batch.append(sample)
                if len(current_batch) >= self.batch_size:
                    batch_data = smart_collate_fn(current_batch, self.tokenizer, self.max_length, pack_sequences=True)
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
            "tokens_per_second": self.total_tokens / sum(self.response_times)
            if sum(self.response_times) > 0
            else 0,
        }
