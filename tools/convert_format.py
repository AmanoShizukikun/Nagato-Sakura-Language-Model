#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON 格式轉換工具

功能：
1) 將 alpaca-chinese-52k-v3.json 轉為兩份標準資料：en.json / zh.json
   - English: en_instruction / en_input / en_output -> instruction / input / output
   - Chinese: zh_instruction / zh_input / zh_output -> instruction / input / output
     並透過 OpenCC 將簡體中文轉為繁體中文。
2) 保留舊功能：prompt/completion -> instruction/input/output
3) 支援 CodeFeedback-Python105K：query/response -> instruction/input/output
4) 支援 python_code_instructions_18k_alpaca：instruction/input/output（含 prompt 回填）
5) 支援 Stable-Code-Python-SFT：instruction/output(jsonl.gz) -> instruction/input/output
"""

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from opencc import OpenCC


def _normalize_opencc_config(opencc_config: str) -> str:
    """Python OpenCC expects config names without .json extension."""
    config = (opencc_config or "").strip()
    if not config:
        return "s2twp"
    if config.lower().endswith(".json"):
        return config[:-5]
    return config


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _write_json(file_path: Path, data: List[Dict[str, str]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _load_parquet_rows(file_path: Path) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("缺少 pyarrow，請先安裝：pip install pyarrow") from exc

    table = pq.read_table(file_path)
    rows = table.to_pylist()
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _load_json_rows(file_path: Path) -> List[Dict[str, Any]]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        # Some datasets (e.g., ChatMed) are JSONL but use .json extension.
        pass

    return _load_jsonl_rows(file_path)


def _load_jsonl_rows(file_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _load_jsonl_gz_rows(file_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with gzip.open(file_path, "rt", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _collect_twllm_source_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files: List[Path] = []
        files.extend(sorted(input_path.rglob("*.parquet")))
        files.extend(sorted(input_path.rglob("*.jsonl.gz")))
        files.extend(sorted(input_path.rglob("*.json")))
        files.extend(sorted(input_path.rglob("*.jsonl")))
        return files
    return []


def _load_rows_from_file(file_path: Path) -> List[Dict[str, Any]]:
    file_name = file_path.name.lower()
    suffix = file_path.suffix.lower()

    if suffix == ".parquet":
        return _load_parquet_rows(file_path)
    if file_name.endswith(".jsonl.gz"):
        return _load_jsonl_gz_rows(file_path)
    if suffix == ".json":
        return _load_json_rows(file_path)
    if suffix == ".jsonl":
        return _load_jsonl_rows(file_path)

    return []


def _extract_twllm_pairs(
    conversations: Iterable[Any],
    converter: Optional[OpenCC] = None,
) -> List[Dict[str, str]]:
    converted: List[Dict[str, str]] = []
    pending_instruction = ""

    for turn in conversations:
        if not isinstance(turn, dict):
            continue

        role = _safe_text(turn.get("role")).strip().lower()
        content = _safe_text(turn.get("content")).strip()
        if not content:
            continue

        if converter is not None:
            content = converter.convert(content)

        if role in {"human", "user"}:
            pending_instruction = content
            continue

        if role in {"gpt", "assistant", "bot"} and pending_instruction:
            converted.append(
                {
                    "instruction": pending_instruction,
                    "input": "",
                    "output": content,
                }
            )
            pending_instruction = ""

    return converted


def _extract_dialog_pairs(
    turns: Iterable[Any],
    role_key: str,
    content_key: str,
    human_roles: set[str],
    assistant_roles: set[str],
    converter: Optional[OpenCC] = None,
) -> List[Dict[str, str]]:
    converted: List[Dict[str, str]] = []
    pending_instruction = ""

    for turn in turns:
        if not isinstance(turn, dict):
            continue

        role = _safe_text(turn.get(role_key)).strip().lower()
        content = _safe_text(turn.get(content_key)).strip()
        if not content:
            continue

        if converter is not None:
            content = converter.convert(content)

        if role in human_roles:
            pending_instruction = content
            continue

        if role in assistant_roles and pending_instruction:
            converted.append(
                {
                    "instruction": pending_instruction,
                    "input": "",
                    "output": content,
                }
            )
            pending_instruction = ""

    return converted


def _extract_taiwanchat_pairs(item: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = item.get("messages")
    if isinstance(messages, list) and messages:
        return _extract_dialog_pairs(
            turns=messages,
            role_key="role",
            content_key="content",
            human_roles={"user"},
            assistant_roles={"assistant"},
        )

    conversations = item.get("conversations")
    if isinstance(conversations, list) and conversations:
        return _extract_dialog_pairs(
            turns=conversations,
            role_key="from",
            content_key="value",
            human_roles={"human"},
            assistant_roles={"gpt"},
        )

    return []


def convert_legacy_prompt_completion(
    input_file_path: Path,
    output_file_path: Optional[Path] = None,
) -> bool:
    """舊格式轉換：prompt/completion -> instruction/input/output。"""
    if not input_file_path.exists():
        print(f"錯誤：找不到輸入文件 {input_file_path}")
        return False

    if output_file_path is None:
        output_file_path = input_file_path.parent / f"{input_file_path.stem}_converted{input_file_path.suffix}"

    try:
        print(f"正在讀取文件：{input_file_path}")
        with open(input_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            print("❌ 輸入 JSON 不是列表格式")
            return False

        converted_data: List[Dict[str, str]] = []
        for item in data:
            if not isinstance(item, dict):
                continue

            prompt = _safe_text(item.get("prompt")).strip()
            completion = _safe_text(item.get("completion")).strip()
            if not prompt or not completion:
                continue

            converted_data.append(
                {
                    "instruction": prompt,
                    "input": "",
                    "output": completion,
                }
            )

        _write_json(output_file_path, converted_data)

        print("✅ 轉換完成！")
        print(f"原始項目數量：{len(data)}")
        print(f"轉換項目數量：{len(converted_data)}")
        print(f"輸出文件：{output_file_path}")
        return True
    except json.JSONDecodeError as exc:
        print(f"❌ JSON 解析錯誤：{exc}")
        return False
    except Exception as exc:
        print(f"❌ 轉換過程中發生錯誤：{exc}")
        return False


def _extract_lang_record(
    item: Dict[str, Any],
    prefix: str,
    converter: Optional[OpenCC] = None,
) -> Optional[Dict[str, str]]:
    instruction = _safe_text(item.get(f"{prefix}_instruction"))
    input_text = _safe_text(item.get(f"{prefix}_input"))
    output = _safe_text(item.get(f"{prefix}_output"))

    if not instruction.strip() or not output.strip():
        return None

    if converter is not None:
        instruction = converter.convert(instruction)
        input_text = converter.convert(input_text)
        output = converter.convert(output)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }


def convert_alpaca_bilingual(
    input_file_path: Path,
    en_output_path: Optional[Path] = None,
    zh_output_path: Optional[Path] = None,
    opencc_config: str = "s2twp",
) -> Tuple[bool, int, int]:
    """
    將 Alpaca 雙語格式拆成兩份標準格式資料。

    Returns:
        (success, en_count, zh_count)
    """
    if not input_file_path.exists():
        print(f"錯誤：找不到輸入文件 {input_file_path}")
        return False, 0, 0

    if en_output_path is None:
        en_output_path = input_file_path.parent / "en.json"
    if zh_output_path is None:
        zh_output_path = input_file_path.parent / "zh.json"

    normalized_config = _normalize_opencc_config(opencc_config)

    try:
        converter = OpenCC(normalized_config)
    except Exception as exc:
        print(f"❌ OpenCC 初始化失敗 ({normalized_config})：{exc}")
        return False, 0, 0

    try:
        print(f"正在讀取文件：{input_file_path}")
        with open(input_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            print("❌ 輸入 JSON 不是列表格式")
            return False, 0, 0

        en_data: List[Dict[str, str]] = []
        zh_data: List[Dict[str, str]] = []

        for item in data:
            if not isinstance(item, dict):
                continue

            en_record = _extract_lang_record(item, "en")
            if en_record:
                en_data.append(en_record)

            zh_record = _extract_lang_record(item, "zh", converter=converter)
            if zh_record:
                zh_data.append(zh_record)

        _write_json(en_output_path, en_data)
        _write_json(zh_output_path, zh_data)

        print("✅ 轉換完成！")
        print(f"原始項目數量：{len(data)}")
        print(f"英文輸出數量：{len(en_data)} -> {en_output_path}")
        print(f"中文輸出數量：{len(zh_data)} -> {zh_output_path}")
        return True, len(en_data), len(zh_data)
    except json.JSONDecodeError as exc:
        print(f"❌ JSON 解析錯誤：{exc}")
        return False, 0, 0
    except Exception as exc:
        print(f"❌ 轉換過程中發生錯誤：{exc}")
        return False, 0, 0


def convert_twllm_data(
    input_path: Path,
    output_file_path: Optional[Path] = None,
    opencc_config: str = "s2twp",
) -> Tuple[bool, int, int]:
    """
    將 twllm-data conversations 轉為 instruction/input/output。

    支援輸入：
    - parquet
    - json
    - jsonl
    - 資料夾（會遞迴讀取上述格式）

    Returns:
        (success, conversation_count, converted_count)
    """
    if not input_path.exists():
        print(f"錯誤：找不到輸入路徑 {input_path}")
        return False, 0, 0

    if output_file_path is None:
        if input_path.is_dir():
            output_file_path = input_path / "twllm_converted.json"
        else:
            output_file_path = input_path.parent / f"{input_path.stem}_converted.json"

    source_files = _collect_twllm_source_files(input_path)
    if output_file_path is not None:
        output_resolved = output_file_path.resolve()
        source_files = [fp for fp in source_files if fp.resolve() != output_resolved]
    if not source_files:
        print("❌ 找不到可轉換的來源檔案（parquet/json/jsonl）")
        return False, 0, 0

    normalized_config = _normalize_opencc_config(opencc_config)
    try:
        converter = OpenCC(normalized_config)
    except Exception as exc:
        print(f"❌ OpenCC 初始化失敗 ({normalized_config})：{exc}")
        return False, 0, 0

    try:
        all_rows: List[Dict[str, Any]] = []
        for file_path in source_files:
            all_rows.extend(_load_rows_from_file(file_path))

        converted_data: List[Dict[str, str]] = []
        for item in all_rows:
            conversations = item.get("conversations") if isinstance(item, dict) else None
            if not isinstance(conversations, list):
                continue
            converted_data.extend(_extract_twllm_pairs(conversations, converter=converter))

        _write_json(output_file_path, converted_data)

        print("✅ 轉換完成！")
        print(f"來源檔案數量：{len(source_files)}")
        print(f"原始對話樣本數量：{len(all_rows)}")
        print(f"轉換後樣本數量：{len(converted_data)}")
        print(f"輸出文件：{output_file_path}")
        return True, len(all_rows), len(converted_data)
    except Exception as exc:
        print(f"❌ 轉換過程中發生錯誤：{exc}")
        return False, 0, 0


def convert_codefeedback_python105k(
    input_path: Path,
    output_file_path: Optional[Path] = None,
    opencc_config: str = "s2twp",
) -> Tuple[bool, int, int]:
    """
    將 CodeFeedback-Python105K（query/response）轉為 instruction/input/output。

    也支援 question/answer 欄位，並可透過 OpenCC 轉為繁體。

    支援輸入：
    - parquet
    - json
    - jsonl
    - 資料夾（會遞迴讀取上述格式）

    Returns:
        (success, source_count, converted_count)
    """
    if not input_path.exists():
        print(f"錯誤：找不到輸入路徑 {input_path}")
        return False, 0, 0

    if output_file_path is None:
        if input_path.is_dir():
            output_file_path = input_path / "codefeedback_converted.json"
        else:
            output_file_path = input_path.parent / f"{input_path.stem}_converted.json"

    source_files = _collect_twllm_source_files(input_path)
    if output_file_path is not None:
        output_resolved = output_file_path.resolve()
        source_files = [fp for fp in source_files if fp.resolve() != output_resolved]
    if not source_files:
        print("❌ 找不到可轉換的來源檔案（parquet/json/jsonl）")
        return False, 0, 0

    normalized_config = _normalize_opencc_config(opencc_config)
    try:
        converter = OpenCC(normalized_config)
    except Exception as exc:
        print(f"❌ OpenCC 初始化失敗 ({normalized_config})：{exc}")
        return False, 0, 0

    try:
        all_rows: List[Dict[str, Any]] = []
        for file_path in source_files:
            all_rows.extend(_load_rows_from_file(file_path))

        converted_data: List[Dict[str, str]] = []
        for item in all_rows:
            if not isinstance(item, dict):
                continue

            instruction = _safe_text(
                item.get("query")
                or item.get("question")
                or item.get("instruction")
                or item.get("prompt")
            ).strip()
            input_text = _safe_text(item.get("input") or item.get("context") or item.get("related_diseases")).strip()
            output = _safe_text(item.get("response") or item.get("answer") or item.get("output") or item.get("completion")).strip()

            if not instruction or not output:
                continue

            instruction = converter.convert(instruction)
            input_text = converter.convert(input_text)
            output = converter.convert(output)

            converted_data.append(
                {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output,
                }
            )

        _write_json(output_file_path, converted_data)

        print("✅ 轉換完成！")
        print(f"來源檔案數量：{len(source_files)}")
        print(f"原始樣本數量：{len(all_rows)}")
        print(f"轉換後樣本數量：{len(converted_data)}")
        print(f"輸出文件：{output_file_path}")
        return True, len(all_rows), len(converted_data)
    except Exception as exc:
        print(f"❌ 轉換過程中發生錯誤：{exc}")
        return False, 0, 0


def convert_stable_code_python_sft(
    input_path: Path,
    output_file_path: Optional[Path] = None,
) -> Tuple[bool, int, int]:
    """
    將 Stable-Code-Python-SFT 轉為 instruction/input/output。

    主要來源欄位：instruction, output（jsonl/jsonl.gz）

    Returns:
        (success, source_count, converted_count)
    """
    if not input_path.exists():
        print(f"錯誤：找不到輸入路徑 {input_path}")
        return False, 0, 0

    if output_file_path is None:
        if input_path.is_dir():
            output_file_path = input_path / "stablecode_converted.json"
        else:
            output_file_path = input_path.parent / f"{input_path.stem}_converted.json"

    source_files = _collect_twllm_source_files(input_path)
    if output_file_path is not None:
        output_resolved = output_file_path.resolve()
        source_files = [fp for fp in source_files if fp.resolve() != output_resolved]
    if not source_files:
        print("❌ 找不到可轉換的來源檔案（parquet/json/jsonl/jsonl.gz）")
        return False, 0, 0

    try:
        all_rows: List[Dict[str, Any]] = []
        for file_path in source_files:
            all_rows.extend(_load_rows_from_file(file_path))

        converted_data: List[Dict[str, str]] = []
        for item in all_rows:
            if not isinstance(item, dict):
                continue

            instruction = _safe_text(item.get("instruction") or item.get("query") or item.get("prompt")).strip()
            output = _safe_text(item.get("output") or item.get("response") or item.get("completion")).strip()

            if not instruction or not output:
                continue

            converted_data.append(
                {
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                }
            )

        _write_json(output_file_path, converted_data)

        print("✅ 轉換完成！")
        print(f"來源檔案數量：{len(source_files)}")
        print(f"原始樣本數量：{len(all_rows)}")
        print(f"轉換後樣本數量：{len(converted_data)}")
        print(f"輸出文件：{output_file_path}")
        return True, len(all_rows), len(converted_data)
    except Exception as exc:
        print(f"❌ 轉換過程中發生錯誤：{exc}")
        return False, 0, 0


def convert_chatmed_consult_dataset(
    input_path: Path,
    output_file_path: Optional[Path] = None,
    opencc_config: str = "s2twp",
) -> Tuple[bool, int, int]:
    """
    將 ChatMed_Consult_Dataset（query/response）轉為 instruction/input/output（繁體）。

    Returns:
        (success, source_count, converted_count)
    """
    if not input_path.exists():
        print(f"錯誤：找不到輸入路徑 {input_path}")
        return False, 0, 0

    if output_file_path is None:
        if input_path.is_dir():
            output_file_path = input_path / "chatmed_converted.json"
        else:
            output_file_path = input_path.parent / f"{input_path.stem}_converted.json"

    source_files = _collect_twllm_source_files(input_path)
    if output_file_path is not None:
        output_resolved = output_file_path.resolve()
        source_files = [fp for fp in source_files if fp.resolve() != output_resolved]
    if not source_files:
        print("❌ 找不到可轉換的來源檔案（parquet/json/jsonl/jsonl.gz）")
        return False, 0, 0

    normalized_config = _normalize_opencc_config(opencc_config)
    try:
        converter = OpenCC(normalized_config)
    except Exception as exc:
        print(f"❌ OpenCC 初始化失敗 ({normalized_config})：{exc}")
        return False, 0, 0

    try:
        all_rows: List[Dict[str, Any]] = []
        for file_path in source_files:
            all_rows.extend(_load_rows_from_file(file_path))

        converted_data: List[Dict[str, str]] = []
        for item in all_rows:
            if not isinstance(item, dict):
                continue

            instruction = _safe_text(item.get("query") or item.get("instruction") or item.get("prompt")).strip()
            output = _safe_text(item.get("response") or item.get("output") or item.get("completion")).strip()

            if not instruction or not output:
                continue

            converted_data.append(
                {
                    "instruction": converter.convert(instruction),
                    "input": "",
                    "output": converter.convert(output),
                }
            )

        _write_json(output_file_path, converted_data)

        print("✅ 轉換完成！")
        print(f"來源檔案數量：{len(source_files)}")
        print(f"原始樣本數量：{len(all_rows)}")
        print(f"轉換後樣本數量：{len(converted_data)}")
        print(f"輸出文件：{output_file_path}")
        return True, len(all_rows), len(converted_data)
    except Exception as exc:
        print(f"❌ 轉換過程中發生錯誤：{exc}")
        return False, 0, 0


def convert_taiwanchat(
    input_path: Path,
    output_file_path: Optional[Path] = None,
) -> Tuple[bool, int, int]:
    """
    將 TaiwanChat 轉為 instruction/input/output。

    支援來源欄位：
    - messages: [{role, content}] (user/assistant)
    - conversations: [{from, value}] (human/gpt)

    Returns:
        (success, source_count, converted_count)
    """
    if not input_path.exists():
        print(f"錯誤：找不到輸入路徑 {input_path}")
        return False, 0, 0

    if output_file_path is None:
        if input_path.is_dir():
            output_file_path = input_path / "taiwanchat_converted.json"
        else:
            output_file_path = input_path.parent / f"{input_path.stem}_converted.json"

    source_files = _collect_twllm_source_files(input_path)
    if output_file_path is not None:
        output_resolved = output_file_path.resolve()
        source_files = [fp for fp in source_files if fp.resolve() != output_resolved]
    if not source_files:
        print("❌ 找不到可轉換的來源檔案（parquet/json/jsonl/jsonl.gz）")
        return False, 0, 0

    try:
        all_rows: List[Dict[str, Any]] = []
        for file_path in source_files:
            all_rows.extend(_load_rows_from_file(file_path))

        converted_data: List[Dict[str, str]] = []
        for item in all_rows:
            if not isinstance(item, dict):
                continue
            converted_data.extend(_extract_taiwanchat_pairs(item))

        _write_json(output_file_path, converted_data)

        print("✅ 轉換完成！")
        print(f"來源檔案數量：{len(source_files)}")
        print(f"原始樣本數量：{len(all_rows)}")
        print(f"轉換後樣本數量：{len(converted_data)}")
        print(f"輸出文件：{output_file_path}")
        return True, len(all_rows), len(converted_data)
    except Exception as exc:
        print(f"❌ 轉換過程中發生錯誤：{exc}")
        return False, 0, 0


def _extract_from_alpaca_prompt(prompt_text: str) -> Tuple[str, str, str]:
    if not prompt_text or not isinstance(prompt_text, str):
        return "", "", ""

    normalized = prompt_text.replace("\r\n", "\n").replace("\r", "\n")

    def _grab(section_name: str, next_sections: List[str]) -> str:
        if next_sections:
            next_pattern = "|".join([rf"\n###\s*{re.escape(name)}\s*:" for name in next_sections])
            pattern = rf"###\s*{re.escape(section_name)}\s*:\s*(.*?)(?:{next_pattern}|\Z)"
        else:
            pattern = rf"###\s*{re.escape(section_name)}\s*:\s*(.*)\Z"

        match = re.search(pattern, normalized, flags=re.DOTALL)
        return match.group(1).strip() if match else ""

    instruction = _grab("Instruction", ["Input", "Output"])
    input_text = _grab("Input", ["Output"])
    output = _grab("Output", [])

    return instruction, input_text, output


def convert_python_code_instructions_18k_alpaca(
    input_path: Path,
    output_file_path: Optional[Path] = None,
) -> Tuple[bool, int, int]:
    """
    將 python_code_instructions_18k_alpaca 轉為 instruction/input/output。

    主要來源欄位：instruction, input, output
    回填來源欄位：prompt（Alpaca 模板）

    Returns:
        (success, source_count, converted_count)
    """
    if not input_path.exists():
        print(f"錯誤：找不到輸入路徑 {input_path}")
        return False, 0, 0

    if output_file_path is None:
        if input_path.is_dir():
            output_file_path = input_path / "python18k_converted.json"
        else:
            output_file_path = input_path.parent / f"{input_path.stem}_converted.json"

    source_files = _collect_twllm_source_files(input_path)
    if output_file_path is not None:
        output_resolved = output_file_path.resolve()
        source_files = [fp for fp in source_files if fp.resolve() != output_resolved]
    if not source_files:
        print("❌ 找不到可轉換的來源檔案（parquet/json/jsonl）")
        return False, 0, 0

    try:
        all_rows: List[Dict[str, Any]] = []
        for file_path in source_files:
            all_rows.extend(_load_rows_from_file(file_path))

        converted_data: List[Dict[str, str]] = []
        for item in all_rows:
            if not isinstance(item, dict):
                continue

            instruction = _safe_text(item.get("instruction")).strip()
            input_text = _safe_text(item.get("input")).strip()
            output = _safe_text(item.get("output")).strip()

            if (not instruction or not output) and item.get("prompt"):
                prompt_instruction, prompt_input, prompt_output = _extract_from_alpaca_prompt(
                    _safe_text(item.get("prompt"))
                )
                if not instruction:
                    instruction = prompt_instruction
                if not input_text:
                    input_text = prompt_input
                if not output:
                    output = prompt_output

            if not instruction or not output:
                continue

            converted_data.append(
                {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output,
                }
            )

        _write_json(output_file_path, converted_data)

        print("✅ 轉換完成！")
        print(f"來源檔案數量：{len(source_files)}")
        print(f"原始樣本數量：{len(all_rows)}")
        print(f"轉換後樣本數量：{len(converted_data)}")
        print(f"輸出文件：{output_file_path}")
        return True, len(all_rows), len(converted_data)
    except Exception as exc:
        print(f"❌ 轉換過程中發生錯誤：{exc}")
        return False, 0, 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NS-LLM JSON 格式轉換工具")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/alpaca-chinese-52k-v3.json"),
        help="輸入 JSON 檔案路徑",
    )
    parser.add_argument(
        "--mode",
        choices=["alpaca", "legacy", "twllm", "codefeedback", "python18k", "stablecode", "chatmed", "taiwanchat"],
        default="alpaca",
        help="alpaca: 產生 en.json/zh.json；legacy: 轉換 prompt/completion；twllm: conversations 轉 instruction/input/output；codefeedback: query/response 轉 instruction/input/output；python18k: python_code_instructions_18k_alpaca 轉 instruction/input/output；stablecode: Stable-Code-Python-SFT 轉 instruction/input/output；chatmed: ChatMed_Consult_Dataset 轉 instruction/input/output；taiwanchat: TaiwanChat 轉 instruction/input/output",
    )
    parser.add_argument(
        "--en-output",
        type=Path,
        default=None,
        help="alpaca 模式英文輸出路徑，預設為輸入檔同資料夾下 en.json",
    )
    parser.add_argument(
        "--zh-output",
        type=Path,
        default=None,
        help="alpaca 模式中文輸出路徑，預設為輸入檔同資料夾下 zh.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="legacy 模式輸出路徑，預設為 <input>_converted.json",
    )
    parser.add_argument(
        "--opencc-config",
        type=str,
        default="s2twp",
        help="OpenCC 簡繁轉換配置 (可用 s2t / s2twp 或含 .json 也可)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=== NS-LLM JSON 格式轉換工具 ===")
    print(f"輸入文件：{args.input}")
    print(f"模式：{args.mode}")
    print("-" * 50)

    if args.mode == "alpaca":
        success, en_count, zh_count = convert_alpaca_bilingual(
            input_file_path=args.input,
            en_output_path=args.en_output,
            zh_output_path=args.zh_output,
            opencc_config=args.opencc_config,
        )
        if success:
            print("\n🎉 轉換成功完成！")
            print(f"英文資料筆數：{en_count}")
            print(f"中文資料筆數：{zh_count}")
        else:
            print("\n❌ 轉換失敗！")
        return

    if args.mode == "twllm":
        success, src_count, converted_count = convert_twllm_data(
            input_path=args.input,
            output_file_path=args.output,
            opencc_config=args.opencc_config,
        )
        if success:
            print("\n🎉 轉換成功完成！")
            print(f"原始對話樣本數量：{src_count}")
            print(f"轉換後樣本數量：{converted_count}")
        else:
            print("\n❌ 轉換失敗！")
        return

    if args.mode == "codefeedback":
        success, src_count, converted_count = convert_codefeedback_python105k(
            input_path=args.input,
            output_file_path=args.output,
            opencc_config=args.opencc_config,
        )
        if success:
            print("\n🎉 轉換成功完成！")
            print(f"原始樣本數量：{src_count}")
            print(f"轉換後樣本數量：{converted_count}")
        else:
            print("\n❌ 轉換失敗！")
        return

    if args.mode == "python18k":
        success, src_count, converted_count = convert_python_code_instructions_18k_alpaca(
            input_path=args.input,
            output_file_path=args.output,
        )
        if success:
            print("\n🎉 轉換成功完成！")
            print(f"原始樣本數量：{src_count}")
            print(f"轉換後樣本數量：{converted_count}")
        else:
            print("\n❌ 轉換失敗！")
        return

    if args.mode == "stablecode":
        success, src_count, converted_count = convert_stable_code_python_sft(
            input_path=args.input,
            output_file_path=args.output,
        )
        if success:
            print("\n🎉 轉換成功完成！")
            print(f"原始樣本數量：{src_count}")
            print(f"轉換後樣本數量：{converted_count}")
        else:
            print("\n❌ 轉換失敗！")
        return

    if args.mode == "chatmed":
        success, src_count, converted_count = convert_chatmed_consult_dataset(
            input_path=args.input,
            output_file_path=args.output,
            opencc_config=args.opencc_config,
        )
        if success:
            print("\n🎉 轉換成功完成！")
            print(f"原始樣本數量：{src_count}")
            print(f"轉換後樣本數量：{converted_count}")
        else:
            print("\n❌ 轉換失敗！")
        return

    if args.mode == "taiwanchat":
        success, src_count, converted_count = convert_taiwanchat(
            input_path=args.input,
            output_file_path=args.output,
        )
        if success:
            print("\n🎉 轉換成功完成！")
            print(f"原始樣本數量：{src_count}")
            print(f"轉換後樣本數量：{converted_count}")
        else:
            print("\n❌ 轉換失敗！")
        return

    success = convert_legacy_prompt_completion(
        input_file_path=args.input,
        output_file_path=args.output,
    )
    if success:
        print("\n🎉 轉換成功完成！")
    else:
        print("\n❌ 轉換失敗！")


if __name__ == "__main__":
    main()
