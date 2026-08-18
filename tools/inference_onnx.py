#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NagatoSakura LLM 輕量級 ONNX 推理引擎 (Lightweight ONNX Inference Engine)

專為資源受限之邊緣裝置或輕量系統設計，基於 ONNX Runtime 與 NumPy 進行高效推導。
支援 KV-Cache 極速 O(1) 逐字生成，大幅降低推導延遲。
支援模式：
1. interactive : 互動式流式 CLI 對話
2. single      : 單次提示詞推理

使用範例：
    python tools/inference_onnx.py --model_path NS-LM-1.6-micro/export --mode interactive
    python tools/inference_onnx.py --model_path NS-LM-1.6-micro/export --mode single --prompt "你好！"
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import onnxruntime as ort
except ImportError:
    print("❌ 錯誤：未找到 onnxruntime 套件。請先執行: pip install onnxruntime")
    sys.exit(1)

# 加入 repo root 至 sys.path 以便導入 TokenizerManager
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.tokenizer import TokenizerManager
except ImportError:
    TokenizerManager = None

try:
    from transformers import PreTrainedTokenizerFast
except ImportError:
    PreTrainedTokenizerFast = None

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ONNXInference")


def fix_utf8_artifacts(text: str) -> str:
    """修復單字節解碼產生的替換符號"""
    if not text:
        return text
    return text.replace("\ufffd", "·")


@dataclass
class ONNXInferenceConfig:
    model_path: str = "NS-LM-1.6-micro/export"
    tokenizer_path: Optional[str] = None
    mode: str = "interactive"
    prompt: Optional[str] = None
    max_length: int = 512
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    do_sample: bool = True
    threads: int = 0
    stateless_chat: bool = True
    device: str = "auto"


@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: float


class ConversationHistory:
    def __init__(self, max_turns: int = 50):
        self.turns: List[ConversationTurn] = []
        self.max_turns = max_turns

    def add_turn(self, role: str, content: str):
        self.turns.append(ConversationTurn(role=role, content=content, timestamp=time.time()))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def clear(self):
        self.turns.clear()


class ONNXSampler:
    """純 NumPy 實現的 Sampling 採樣器 (Temperature, Top-K, Top-P, Repetition Penalty)"""

    @staticmethod
    def sample(
        logits: np.ndarray,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        generated_tokens: Optional[List[int]] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> int:
        logits = logits.copy().astype(np.float32)

        # 1. 重複懲罰 (Repetition Penalty)
        if repetition_penalty != 1.0 and generated_tokens:
            for token_id in set(generated_tokens):
                if token_id < len(logits):
                    if logits[token_id] > 0:
                        logits[token_id] /= repetition_penalty
                    else:
                        logits[token_id] *= repetition_penalty

        # 2. 貪婪解碼 (Greedy Decoding)
        if not do_sample or temperature <= 1e-4:
            return int(np.argmax(logits))

        # 3. 溫度調節 (Temperature Scaling)
        logits = logits / max(temperature, 1e-4)

        # 4. Top-K 篩選
        if top_k > 0 and top_k < len(logits):
            indices_to_remove = np.argpartition(logits, -top_k)[:-top_k]
            logits[indices_to_remove] = -np.inf

        # 5. Softmax 轉換為機率分佈
        max_logit = np.max(logits[np.isfinite(logits)]) if np.any(np.isfinite(logits)) else 0.0
        exp_logits = np.exp(logits - max_logit)
        exp_logits[~np.isfinite(exp_logits)] = 0.0
        sum_exp = np.sum(exp_logits)
        if sum_exp <= 0 or not np.isfinite(sum_exp):
            return int(np.argmax(logits))

        probs = exp_logits / sum_exp

        # 6. Top-P (Nucleus) 篩選
        if top_p < 1.0 and top_p > 0.0:
            sorted_indices = np.argsort(-probs)
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)

            # 移除累積機率超過 top_p 的 tokens (保留至少第一個 token)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0.0
            sum_probs = np.sum(probs)
            if sum_probs > 0:
                probs = probs / sum_probs
            else:
                return int(sorted_indices[0])

        # 7. 機率抽樣
        return int(np.random.choice(len(probs), p=probs))


class NagatoSakuraONNXEngine:
    """NagatoSakura ONNX 輕量推理引擎 (支援 KV-Cache 自動偵測)"""

    def __init__(self, config: ONNXInferenceConfig):
        self.config = config
        self.session: Optional[ort.InferenceSession] = None
        self.tokenizer = None
        self.model_config: Dict[str, Any] = {}
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.has_kv_cache: bool = False
        self.num_layers: int = 0
        self.num_kv_heads: int = 0
        self.head_dim: int = 0
        self._init_session()
        self._init_tokenizer()

    def _find_model_file(self, model_path_str: str) -> Path:
        model_path = Path(model_path_str)
        if model_path.is_file() and model_path.suffix == ".onnx":
            return model_path

        candidates = ["model.onnx", "nagato_sakura.onnx"]
        for cand in candidates:
            cand_path = model_path / cand
            if cand_path.exists():
                return cand_path

        for p in model_path.glob("*.onnx"):
            return p

        raise FileNotFoundError(f"在 {model_path_str} 中未找到任何 .onnx 模型檔案！")

    def _init_session(self) -> None:
        onnx_file = self._find_model_file(self.config.model_path)
        logger.info(f"正在載入 ONNX 模型: {onnx_file}...")

        opts = ort.SessionOptions()
        if self.config.threads > 0:
            opts.intra_op_num_threads = self.config.threads
            opts.inter_op_num_threads = max(1, self.config.threads // 2)

        # 選擇 Provider (優先支援 CUDA 或 CPU)
        if self.config.device in ("auto", "cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(onnx_file), sess_options=opts, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.has_kv_cache = "past_key_0" in self.input_names

        # 讀取 config.json (若存在)
        model_dir = onnx_file.parent
        config_file = model_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    self.model_config = json.load(f)
                max_pos = self.model_config.get("max_position_embeddings", 512)
                if self.config.max_length <= 0 or self.config.max_length > max_pos:
                    self.config.max_length = max_pos
            except Exception as e:
                logger.warning(f"讀取 config.json 失敗: {e}")

        # 解析 KV-Cache 維度
        if self.has_kv_cache:
            self.num_layers = len([k for k in self.input_names if k.startswith("past_key_")])
            num_heads = self.model_config.get("num_attention_heads", 32)
            self.num_kv_heads = self.model_config.get("num_key_value_heads") or num_heads
            hidden_size = self.model_config.get("hidden_size", 4096)
            self.head_dim = hidden_size // num_heads

        logger.info(
            f"ONNX Session 就緒 | 運算設備: {self.session.get_providers()[0]} (設定: {self.config.device}) | "
            f"KV-Cache: {'已啟用 (極速 O(1) 解碼)' if self.has_kv_cache else '未啟用 (基礎 O(N^2) 模式)'}"
        )

    def _init_tokenizer(self) -> None:
        tok_path = Path(self.config.tokenizer_path) if self.config.tokenizer_path else Path(self.config.model_path)
        if not (tok_path / "tokenizer.json").exists() and (tok_path.parent / "tokenizer.json").exists():
            tok_path = tok_path.parent

        tokenizer_json_file = tok_path / "tokenizer.json"
        if not tokenizer_json_file.exists():
            # 向上搜尋
            for parent_dir in [tok_path.parent, tok_path.parent.parent]:
                if (parent_dir / "tokenizer.json").exists():
                    tok_path = parent_dir
                    tokenizer_json_file = tok_path / "tokenizer.json"
                    break

        if not tokenizer_json_file.exists():
            raise FileNotFoundError(f"找不到 tokenizer.json，搜尋路徑: {tok_path}")

        logger.info(f"正在載入分詞器: {tokenizer_json_file}...")
        if TokenizerManager is not None:
            manager = TokenizerManager(tokenizer_json_file)
            manager.load_tokenizer()
            self.tokenizer = manager.transformers_tokenizer
        elif PreTrainedTokenizerFast is not None:
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json_file))
        else:
            raise ImportError("缺少 transformers / TokenizerManager，請先安裝: pip install transformers tokenizers")

        logger.info(f"分詞器載入完成 (詞彙量: {len(self.tokenizer)})")

    def _prepare_input_ids(self, prompt: str) -> List[int]:
        bos = getattr(self.tokenizer, "bos_token", None) or "<s>"
        normalized = str(prompt).strip()
        formatted = f"{bos}{normalized}\n" if normalized else bos
        input_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        return input_ids

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """流式生成器 (自動支援 KV-Cache 極速流水線)"""
        if max_new_tokens is None or max_new_tokens <= 0:
            max_new_tokens = self.config.max_new_tokens
        if temperature is None:
            temperature = self.config.temperature
        if top_k is None:
            top_k = self.config.top_k
        if top_p is None:
            top_p = self.config.top_p
        if repetition_penalty is None:
            repetition_penalty = self.config.repetition_penalty
        if do_sample is None:
            do_sample = self.config.do_sample

        input_ids = self._prepare_input_ids(prompt)
        max_ctx = self.config.max_length

        if len(input_ids) >= max_ctx - 4:
            yield {"delta": "輸入文本過長，超出模型上下文限制。", "finished": True, "error": True}
            return

        effective_max_tokens = min(max_new_tokens, max_ctx - len(input_ids))
        generated_tokens: List[int] = list(input_ids)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        sampler = ONNXSampler()

        full_response = ""
        tokens_generated_count = 0

        # ======================================================================
        # 1. 啟用 KV-Cache 模式 (極速 O(1) 逐字生成)
        # ======================================================================
        if self.has_kv_cache:
            # 階段 1: Prefill (首字生成)
            prompt_len = len(input_ids)
            curr_input = np.array([input_ids], dtype=np.int64)
            prefill_inputs = {
                "input_ids": curr_input,
                "attention_mask": np.ones((1, prompt_len), dtype=np.int64),
                "position_ids": np.arange(0, prompt_len, dtype=np.int64).reshape(1, -1),
            }
            for i in range(self.num_layers):
                prefill_inputs[f"past_key_{i}"] = np.zeros((1, self.num_kv_heads, 0, self.head_dim), dtype=np.float32)
                prefill_inputs[f"past_value_{i}"] = np.zeros((1, self.num_kv_heads, 0, self.head_dim), dtype=np.float32)

            ort_outputs = self.session.run(self.output_names, prefill_inputs)
            logits_3d = ort_outputs[0]
            present_kvs = ort_outputs[1:]
            next_token_logits = logits_3d[0, -1, :]

            next_token_id = sampler.sample(
                logits=next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=generated_tokens[len(input_ids) :],
                do_sample=do_sample,
                eos_token_id=eos_token_id,
            )

            generated_tokens.append(next_token_id)
            tokens_generated_count += 1

            if eos_token_id is not None and next_token_id == eos_token_id:
                yield {
                    "delta": "",
                    "finished": True,
                    "tokens_generated": tokens_generated_count,
                    "full_response": "",
                    "error": False,
                }
                return

            try:
                newly_decoded = self.tokenizer.decode([next_token_id], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                newly_decoded = fix_utf8_artifacts(newly_decoded)
                full_response = newly_decoded
                if newly_decoded:
                    yield {"delta": newly_decoded, "finished": False, "full_response": full_response, "error": False}
            except Exception:
                pass

            # 階段 2: Decode (逐字 O(1) 快取遞增)
            for _ in range(1, effective_max_tokens):
                total_seq_len = len(generated_tokens)
                decode_inputs = {
                    "input_ids": np.array([[next_token_id]], dtype=np.int64),
                    "attention_mask": np.ones((1, total_seq_len), dtype=np.int64),
                    "position_ids": np.array([[total_seq_len - 1]], dtype=np.int64),
                }
                for i in range(self.num_layers):
                    decode_inputs[f"past_key_{i}"] = present_kvs[2 * i]
                    decode_inputs[f"past_value_{i}"] = present_kvs[2 * i + 1]

                ort_outputs = self.session.run(self.output_names, decode_inputs)
                dec_logits = ort_outputs[0]
                present_kvs = ort_outputs[1:]
                next_token_logits = dec_logits[0, -1, :]

                next_token_id = sampler.sample(
                    logits=next_token_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    generated_tokens=generated_tokens[len(input_ids) :],
                    do_sample=do_sample,
                    eos_token_id=eos_token_id,
                )

                generated_tokens.append(next_token_id)
                tokens_generated_count += 1

                if eos_token_id is not None and next_token_id == eos_token_id:
                    break

                try:
                    newly_decoded = self.tokenizer.decode(
                        generated_tokens[len(input_ids) :],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                    newly_decoded = fix_utf8_artifacts(newly_decoded)
                    _REPL = "\ufffd"
                    stable_decoded = newly_decoded.rstrip(_REPL) if _REPL in newly_decoded else newly_decoded

                    if len(stable_decoded) > len(full_response):
                        delta = stable_decoded[len(full_response) :]
                        full_response = stable_decoded
                        if delta:
                            yield {
                                "delta": delta,
                                "finished": False,
                                "full_response": full_response,
                                "error": False,
                            }
                except Exception:
                    pass

        # ======================================================================
        # 2. 基礎無 KV-Cache 模式 (回退全序列重算)
        # ======================================================================
        else:
            for _ in range(effective_max_tokens):
                curr_input = np.array([generated_tokens], dtype=np.int64)
                ort_inputs = {"input_ids": curr_input}

                if "attention_mask" in self.input_names:
                    ort_inputs["attention_mask"] = np.ones_like(curr_input, dtype=np.int64)

                ort_outputs = self.session.run(self.output_names, ort_inputs)
                logits_3d = ort_outputs[0]
                next_token_logits = logits_3d[0, -1, :]

                next_token_id = sampler.sample(
                    logits=next_token_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    generated_tokens=generated_tokens[len(input_ids) :],
                    do_sample=do_sample,
                    eos_token_id=eos_token_id,
                )

                generated_tokens.append(next_token_id)
                tokens_generated_count += 1

                if eos_token_id is not None and next_token_id == eos_token_id:
                    break

                try:
                    newly_decoded = self.tokenizer.decode(
                        generated_tokens[len(input_ids) :],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                    newly_decoded = fix_utf8_artifacts(newly_decoded)
                    _REPL = "\ufffd"
                    stable_decoded = newly_decoded.rstrip(_REPL) if _REPL in newly_decoded else newly_decoded

                    if len(stable_decoded) > len(full_response):
                        delta = stable_decoded[len(full_response) :]
                        full_response = stable_decoded
                        if delta:
                            yield {
                                "delta": delta,
                                "finished": False,
                                "full_response": full_response,
                                "error": False,
                            }
                except Exception:
                    pass

        # 最終完整結果
        final_decoded = self.tokenizer.decode(
            generated_tokens[len(input_ids) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        final_decoded = fix_utf8_artifacts(final_decoded)

        yield {
            "delta": "",
            "finished": True,
            "tokens_generated": tokens_generated_count,
            "full_response": final_decoded,
            "error": False,
        }

    def single_inference(self, prompt: str, **kwargs) -> str:
        """單次推理"""
        full_text = ""
        for out in self.stream_generate(prompt, **kwargs):
            if out.get("error"):
                return f"❌ 錯誤: {out.get('delta')}"
            if out["delta"]:
                full_text += out["delta"]
            if out["finished"]:
                return out.get("full_response", full_text)
        return full_text

    def interactive_chat(self) -> None:
        """互動式對話"""
        print("\n" + "=" * 60)
        print("🌸 NagatoSakura ONNX 輕量推理終端 (Interactive Chat)")
        print("=" * 60)
        print("💡 提示:")
        print("   - 輸入 'quit' 或 'exit' 退出")
        print("   - 輸入 '/help' 查看可用指令")
        print("   - 輸入 '/temp' 切換記憶模式 (預設為不保留記憶的單輪獨立對話)")
        print("   - 輸入 '/clear' 清空歷史紀錄")
        print("   - 輸入 '/config' 查看目前推導超參數")
        print(f"   - 上下文長度上限: {self.config.max_length}")
        print(f"   - 對話記憶模式: {'保留多輪歷史' if not self.config.stateless_chat else '單輪獨立 (預設不保留記憶)'}")
        print(f"   - KV-Cache 加速: {'啟用 (O(1) 極速流水線)' if self.has_kv_cache else '未啟用'}")
        print("=" * 60)

        history = ConversationHistory()

        while True:
            try:
                user_input = input("\n👤 用戶: ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ("quit", "exit", "退出", "結束", "q"):
                    print("\n🌸 長門櫻: 謝謝您的使用，再見！")
                    break

                if user_input.startswith("/"):
                    self._handle_cmd(user_input, history)
                    continue

                # 準備 prompt (若為多輪模式)
                if not self.config.stateless_chat and history.turns:
                    history_context = "\n".join(
                        f"{'用戶：' if t.role == 'user' else '長門櫻：'}{t.content}" for t in history.turns
                    )
                    prompt = f"{history_context}\n用戶：{user_input}\n長門櫻："
                else:
                    prompt = user_input

                print("\n🌸 長門櫻: ", end="", flush=True)
                start_time = time.time()
                full_resp = ""

                for out in self.stream_generate(prompt):
                    if out.get("error"):
                        print(f"\n❌ 生成出錯: {out.get('delta')}")
                        break

                    if out["delta"]:
                        print(out["delta"], end="", flush=True)
                        full_resp += out["delta"]

                    if out["finished"]:
                        end_time = time.time()
                        total_time = end_time - start_time
                        tok_count = out.get("tokens_generated", 0)
                        speed = tok_count / max(total_time, 1e-4)
                        print(f"\n\n⏱️  耗時: {total_time:.2f}s | {tok_count} tokens | 速度: {speed:.1f} tok/s")

                        if not self.config.stateless_chat:
                            history.add_turn("user", user_input)
                            history.add_turn("assistant", out.get("full_response", full_resp))
                        break

            except KeyboardInterrupt:
                print("\n\n🌸 長門櫻: 偵測到中斷，再見！")
                break
            except Exception as e:
                print(f"\n❌ 錯誤: {e}")

    def _handle_cmd(self, command: str, history: ConversationHistory) -> None:
        parts = command[1:].strip().split()
        cmd = parts[0].lower() if parts else ""

        if cmd in ("help", "h"):
            print("""
🌸 可用指令:
/help                 - 顯示此說明
/config               - 顯示當前 ONNX 推理參數
/clear                - 清空對話歷史
/history              - 查看對話歷史紀錄
/temp [on|off]        - 切換臨時聊天模式 (不保留對話歷史)
/set <param> <value>  - 設定超參數 (temperature, top_k, top_p, repetition_penalty, max_new_tokens)
""")
        elif cmd == "config":
            print(f"""
🔧 當前 ONNX 推理配置:
- 最大上下文: {self.config.max_length}
- 單輪最大生成: {self.config.max_new_tokens}
- 溫度 (temperature): {self.config.temperature}
- Top-k: {self.config.top_k}
- Top-p: {self.config.top_p}
- 重複懲罰: {self.config.repetition_penalty}
- 採樣模式: {'啟用 (do_sample)' if self.config.do_sample else '貪婪 (greedy)'}
- 臨時模式 (stateless): {'開啟 (預設不保留記憶)' if self.config.stateless_chat else '關閉 (保留多輪記憶)'}
- 執行設備: {self.config.device}
- KV-Cache 狀態: {'已啟用 (O(1) 極速解碼)' if self.has_kv_cache else '未啟用'}
""")
        elif cmd == "clear":
            history.clear()
            print("✅ 對話歷史已清空。")
        elif cmd == "history":
            if not history.turns:
                print("📜 尚無對話歷史。")
            else:
                print("\n📜 對話歷史:")
                for i, turn in enumerate(history.turns, 1):
                    role_icon = "👤" if turn.role == "user" else "🌸"
                    print(f"  {i}. {role_icon} {turn.role}: {turn.content}")
        elif cmd == "temp":
            if len(parts) >= 2:
                self.config.stateless_chat = parts[1].lower() in ("on", "true", "1")
            else:
                self.config.stateless_chat = not self.config.stateless_chat
            status = "開啟 (單輪獨立，不保留記憶)" if self.config.stateless_chat else "關閉 (保留多輪記憶)"
            print(f"✅ 臨時聊天模式已{status}。")
        elif cmd == "set" and len(parts) >= 3:
            param, val = parts[1].lower(), parts[2]
            try:
                if param in ("temperature", "temp"):
                    self.config.temperature = float(val)
                elif param == "top_k":
                    self.config.top_k = int(val)
                elif param == "top_p":
                    self.config.top_p = float(val)
                elif param in ("repetition_penalty", "rep"):
                    self.config.repetition_penalty = float(val)
                elif param in ("max_new_tokens", "max_tokens"):
                    self.config.max_new_tokens = int(val)
                else:
                    print(f"❌ 不支援的參數: {param}")
                    return
                print(f"✅ 已將 {param} 設定為: {val}")
            except ValueError:
                print(f"❌ 數值格式錯誤: {val}")
        else:
            print(f"❌ 未知指令: /{cmd}，請輸入 /help 查看說明。")


def main() -> None:
    parser = argparse.ArgumentParser(description="NagatoSakura ONNX 輕量推理引擎 (Lightweight ONNX Inference Engine)")
    parser.add_argument("--model_path", type=str, default="NS-LM-1.6-micro/export", help="ONNX 模型檔案或包含 model.onnx 的資料夾路徑")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer 路徑 (未指定時自動在 model_path 中搜尋)")
    parser.add_argument("--mode", type=str, choices=["interactive", "single"], default="interactive", help="推理模式: interactive (互動對話), single (單次推導)")
    parser.add_argument("--prompt", type=str, default=None, help="單次推理時的輸入提示文字 (single 模式專用)")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="單輪最大新生成 token 數 (預設: 2048)")
    parser.add_argument("--max_length", type=int, default=0, help="推理上下文長度上限 (0=自動使用模型 config.json 設定)")
    parser.add_argument("--temperature", type=float, default=0.7, help="採樣溫度 (預設: 0.7)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K 採樣參數 (預設: 50)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P 核採樣參數 (預設: 0.9)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="重複懲罰係數 (預設: 1.0)")
    parser.add_argument("--no_sample", action="store_true", help="停用隨機採樣，改用貪婪解碼 (Greedy)")
    parser.add_argument("--threads", type=int, default=0, help="ONNX Runtime CPU 執行緒數 (0=自動)")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="運算設備: auto, cpu, cuda (預設: auto)")
    parser.add_argument("--keep_history", action="store_true", help="啟用多輪對話記憶（預設為不保留記憶的臨時聊天模式）")

    args = parser.parse_args()

    config = ONNXInferenceConfig(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        mode=args.mode,
        prompt=args.prompt,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=not args.no_sample,
        threads=args.threads,
        stateless_chat=not args.keep_history,
        device=args.device,
    )

    engine = NagatoSakuraONNXEngine(config)

    if args.mode == "single":
        if not args.prompt:
            print("❌ 錯誤: single 模式必須提供 --prompt 參數！")
            sys.exit(1)
        response = engine.single_inference(args.prompt)
        print(f"\n🌸 長門櫻: {response}")
    else:
        engine.interactive_chat()


if __name__ == "__main__":
    main()
