import argparse
import json
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List
import tempfile
import os

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("警告: Whisper 套件未安裝，語音轉文字功能將被禁用")

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference import InferenceConfig, NagatoSakuraInference


INFERENCE_INSTANCE: NagatoSakuraInference | None = None
INFERENCE_INIT_LOCK = Lock()
INFERENCE_RUN_LOCK = Lock()
RUNTIME_ARGS: argparse.Namespace | None = None
WHISPER_MODEL = None
WHISPER_INIT_LOCK = Lock()

WEBUI_DIR = Path(__file__).resolve().parent / "webui"
app = Flask(
    __name__,
    template_folder=str(WEBUI_DIR / "templates"),
    static_folder=str(WEBUI_DIR / "static"),
)
app.config["JSON_AS_ASCII"] = False


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path

    root_candidate = (ROOT_DIR / path).resolve()
    if root_candidate.exists():
        return root_candidate

    return (Path.cwd() / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nagato Sakura Flask Web Demo")
    parser.add_argument("--model_path", type=str, default="NS-LLM-0.1/checkpoint-epoch-17")
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_length", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--history_rounds", type=int, default=3)
    parser.add_argument("--no_sample", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--stateless_chat", action="store_true")
    parser.add_argument("--quantize_kv_cache", action="store_true")
    parser.add_argument("--kv_cache_bits", type=int, choices=[3, 4, 8, 16, 32])
    parser.add_argument("--kv_quant_group_size", type=int)
    parser.add_argument("--kv_residual_sign_correction", action="store_true")
    parser.add_argument("--num_key_value_heads", type=int)
    parser.add_argument("--web_host", type=str, default="127.0.0.1")
    parser.add_argument("--web_port", type=int, default=8501)

    args, _ = parser.parse_known_args()

    model_path = _resolve_path(args.model_path)
    args.model_path = str(model_path)

    if args.tokenizer_path:
        args.tokenizer_path = str(_resolve_path(args.tokenizer_path))
    else:
        args.tokenizer_path = str(model_path / "tokenizer.json")

    return args


def _build_model_prompt(messages: List[Dict[str, str]], history_rounds: int, stateless_chat: bool) -> str:
    if not messages:
        return ""

    if stateless_chat:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    valid_messages = [m for m in messages if m.get("role") in {"user", "assistant"}]
    if not valid_messages:
        return ""

    if history_rounds <= 0:
        selected_messages = valid_messages[-1:]
    else:
        selected_messages = valid_messages[-(history_rounds * 2):]

    lines: List[str] = []
    for message in selected_messages:
        role = message.get("role")
        content = str(message.get("content", ""))
        if role == "user":
            lines.append(f"用戶：{content}")
        else:
            lines.append(f"長門櫻：{content}")

    if selected_messages[-1].get("role") == "user":
        lines.append("長門櫻：")

    return "\n".join(lines)


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _normalize_history(history_raw) -> List[Dict[str, str]]:
    if not isinstance(history_raw, list):
        return []

    messages: List[Dict[str, str]] = []
    for item in history_raw[-80:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content", ""))
        if not content.strip():
            continue
        messages.append({"role": role, "content": content})

    return messages


def _build_inference(args: argparse.Namespace) -> NagatoSakuraInference:
    config = InferenceConfig(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        max_length=(None if int(args.max_length) <= 0 else int(args.max_length)),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=not args.no_sample,
        base_seed=args.seed,
        silent_mode=not args.verbose,
        quantize_kv_cache=(True if args.quantize_kv_cache else None),
        kv_cache_bits=args.kv_cache_bits,
        kv_quant_group_size=args.kv_quant_group_size,
        kv_residual_sign_correction=(True if args.kv_residual_sign_correction else None),
        num_key_value_heads=args.num_key_value_heads,
        stateless_chat=args.stateless_chat,
    )
    return NagatoSakuraInference(config)


def get_inference() -> NagatoSakuraInference:
    global INFERENCE_INSTANCE
    if INFERENCE_INSTANCE is not None:
        return INFERENCE_INSTANCE

    with INFERENCE_INIT_LOCK:
        if INFERENCE_INSTANCE is None:
            if RUNTIME_ARGS is None:
                raise RuntimeError("Flask runtime args 尚未初始化")
            INFERENCE_INSTANCE = _build_inference(RUNTIME_ARGS)
    return INFERENCE_INSTANCE


def _jsonl_line(payload: Dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


@app.get("/")
def index():
    if RUNTIME_ARGS is None:
        return "Runtime args not initialized", 500

    bootstrap = {
        "modelPath": RUNTIME_ARGS.model_path,
        "tokenizerPath": RUNTIME_ARGS.tokenizer_path,
        "modelName": Path(RUNTIME_ARGS.model_path).name,
        "defaults": {
            "historyRounds": int(RUNTIME_ARGS.history_rounds),
            "maxLength": int(RUNTIME_ARGS.max_length if RUNTIME_ARGS.max_length > 0 else 4096),
            "maxNewTokens": int(RUNTIME_ARGS.max_new_tokens),
            "temperature": float(RUNTIME_ARGS.temperature),
            "topK": int(RUNTIME_ARGS.top_k),
            "topP": float(RUNTIME_ARGS.top_p),
            "repetitionPenalty": float(RUNTIME_ARGS.repetition_penalty),
            "doSample": bool(not RUNTIME_ARGS.no_sample),
        },
        "uiDefaults": {
            "activeSection": "audioVideo",
            "themePreset": "cyber",
            "displayFontScale": 1.0,
            "chatUiMode": "bubbleOnly",
            "displayDensity": "normal",
            "displayAnimations": True,
            "metaShowTokensPerSec": True,
            "metaShowTokens": True,
            "metaShowElapsed": True,
            "metaShowStopReason": True,
            "statusIdleSeconds": 60,
            "shortcutEnterToSend": True,
            "shortcutEscClosePanels": True,
            "shortcutSlashFocusInput": True,
            "language": "zh-Hant",
            "timeFormat": "24h",
            "timezone": "local",
        },
        "deviceDefaults": {
            "echoCancellation": True,
            "noiseSuppression": True,
            "autoGainControl": True,
            "micGain": 1.0,
        },
        "featureDefaults": {
            "voiceInput": True,
            "conversationHistory": True,
            "messageActions": True,
            "healthMonitor": True,
        },
        "capabilities": {
            "voiceTranscription": bool(WHISPER_AVAILABLE),
            "deviceManagement": True,
            "speakerSelectionBrowserDependent": True,
            "cameraPreview": True,
        },
    }
    return render_template("chat.html", bootstrap=bootstrap)


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "backend": "flask", "model_loaded": INFERENCE_INSTANCE is not None})


def get_whisper_model():
    global WHISPER_MODEL
    if not WHISPER_AVAILABLE:
        return None
    if WHISPER_MODEL is not None:
        return WHISPER_MODEL
    with WHISPER_INIT_LOCK:
        if WHISPER_MODEL is None:
            try:
                # 使用 base 模型，兼顧速度與準確度
                WHISPER_MODEL = whisper.load_model("base")
            except Exception as e:
                print(f"Whisper model load error: {e}")
                return None
    return WHISPER_MODEL


def is_fp16_supported() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name().lower()
        if "1650" in device_name or "1660" in device_name or "gtx 16" in device_name:
            return False
        major, _ = torch.cuda.get_device_capability()
        if major < 7:
            return False
        return True
    except Exception:
        return False


@app.post("/api/transcribe")
def api_transcribe():
    if not WHISPER_AVAILABLE:
        return jsonify({"error": "Whisper disabled"}), 500
    
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"error": "Empty file"}), 400

    model = get_whisper_model()
    if not model:
        return jsonify({"error": "Whisper load fail"}), 500

    try:
        # 將音訊暫存
        fd, temp_path = tempfile.mkstemp(suffix=".webm")
        os.close(fd)
        audio_file.save(temp_path)

        result = model.transcribe(temp_path, fp16=is_fp16_supported())
        text = result.get("text", "").strip()

        os.remove(temp_path)
        return jsonify({"text": text})
    except Exception as e:
        print(f"Transcribe error: {e}")
        return jsonify({"error": str(e)}), 500


@app.post("/api/chat")
def api_chat():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400

    params = payload.get("params", {})
    if not isinstance(params, dict):
        params = {}

    message = str(payload.get("message", "")).strip()
    messages = _normalize_history(payload.get("history", []))

    if message and (not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != message):
        messages.append({"role": "user", "content": message})

    if not messages:
        return jsonify({"error": "Empty message"}), 400

    try:
        inference = get_inference()
    except Exception as e:
        return jsonify({"error": f"模型初始化失敗: {e}"}), 500

    model_limit = int(getattr(inference.model.config, "max_position_embeddings", 4096))

    history_rounds = _clamp_int(params.get("history_rounds", 3), 0, 40)
    stateless_chat = bool(params.get("stateless_chat", inference.config.stateless_chat))
    do_sample = bool(params.get("do_sample", inference.config.do_sample))

    max_length = _clamp_int(params.get("max_length", inference.config.max_length or model_limit), 128, model_limit)
    max_new_limit = max(16, max_length - 8)
    max_new_tokens = _clamp_int(params.get("max_new_tokens", inference.config.max_new_tokens), 16, max_new_limit)

    temperature = _clamp_float(params.get("temperature", inference.config.temperature), 0.1, 1.8)
    top_k = _clamp_int(params.get("top_k", inference.config.top_k), 1, 200)
    top_p = _clamp_float(params.get("top_p", inference.config.top_p), 0.1, 1.0)
    repetition_penalty = _clamp_float(params.get("repetition_penalty", inference.config.repetition_penalty), 1.0, 2.0)

    prompt = _build_model_prompt(messages, history_rounds=history_rounds, stateless_chat=stateless_chat)
    if not prompt:
        return jsonify({"error": "Prompt 為空，請重試"}), 400

    def generate_stream():
        with INFERENCE_RUN_LOCK:
            inference.config.max_length = max_length
            inference.config.max_new_tokens = max_new_tokens
            inference.config.temperature = temperature
            inference.config.top_k = top_k
            inference.config.top_p = top_p
            inference.config.repetition_penalty = repetition_penalty
            inference.config.do_sample = do_sample
            inference.config.stateless_chat = stateless_chat

            started_at = time.time()
            final_text = ""

            try:
                for output in inference.stream_generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                ):
                    if output.get("finished", False):
                        elapsed = time.time() - started_at
                        if output.get("error", False):
                            message_text = str(output.get("delta", "生成失敗"))
                            yield _jsonl_line({"type": "error", "message": message_text, "done": True})
                        else:
                            final_text = str(output.get("full_response", final_text))
                            yield _jsonl_line(
                                {
                                    "type": "done",
                                    "done": True,
                                    "text": final_text,
                                    "elapsed": elapsed,
                                    "tokens": int(output.get("tokens_generated", 0)),
                                    "stop_reason": str(output.get("stop_reason", "completed")),
                                }
                            )
                        return

                    delta = str(output.get("delta", ""))
                    if delta:
                        final_text += delta
                        yield _jsonl_line({"type": "delta", "delta": delta})
            except Exception as e:
                yield _jsonl_line({"type": "error", "message": f"生成例外: {e}", "done": True})

    return Response(
        stream_with_context(generate_stream()),
        mimetype="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def main() -> None:
    global RUNTIME_ARGS
    RUNTIME_ARGS = parse_args()

    model_path = Path(RUNTIME_ARGS.model_path)
    tokenizer_path = Path(RUNTIME_ARGS.tokenizer_path)

    if not model_path.exists():
        raise SystemExit(f"模型路徑不存在: {model_path}")
    if not tokenizer_path.exists():
        raise SystemExit(f"找不到 tokenizer.json: {tokenizer_path}")

    print("🌸 Nagato Sakura Flask Web Demo")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"URL: http://{RUNTIME_ARGS.web_host}:{RUNTIME_ARGS.web_port}")
    print("=" * 50)

    app.run(
        host=RUNTIME_ARGS.web_host,
        port=RUNTIME_ARGS.web_port,
        debug=False,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
