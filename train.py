import argparse
import multiprocessing
import os
import socket
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.nagato_sakura_model import NSConfig
    from src.trainer import AdvancedNagatoSakuraTrainer
except ImportError as exc:
    print(f"Failed to import training modules: {exc}")
    sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Nagato Sakura language model (single-node multi-GPU supported; multi-node unsupported)"
    )

    parser.add_argument("--training_data_file", type=str, help="Training data file or directory")
    parser.add_argument("--output_dir", type=str, default="NS-LLM-0.8", help="Output directory")
    parser.add_argument("--force_retrain_tokenizer", action="store_true", help="Force retraining the tokenizer")
    parser.add_argument("--eval_split_ratio", type=float, default=0.0, help="Eval split ratio when no eval file is supplied")
    parser.add_argument("--eval_data_file", type=str, help="Fixed evaluation data file or directory")

    parser.add_argument("--vocab_size", type=int, default=65536, help="Target tokenizer vocab size")
    parser.add_argument("--tokenizer_min_frequency", type=int, default=5, help="Tokenizer minimum token frequency")
    parser.add_argument("--tokenizer_train_max_samples", type=int, default=0, help="Limit tokenizer training samples, 0 disables the cap")
    parser.add_argument("--tokenizer_num_threads", type=int, default=0, help="Tokenizer training thread count, 0 uses library defaults")
    parser.add_argument("--hidden_size", type=int, default=512, help="Model hidden size")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_key_value_heads", type=int, default=4, help="Number of grouped-query KV heads")
    parser.add_argument("--intermediate_size", type=int, default=1536, help="MLP intermediate size")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--memory_tokens", type=int, default=32, help="Memory token count")
    parser.add_argument("--quantize_kv_cache", action="store_true", default=True, help="Enable KV cache quantization")
    parser.add_argument("--no_quantize_kv_cache", action="store_false", dest="quantize_kv_cache", help="Disable KV cache quantization")
    parser.add_argument("--kv_cache_bits", type=int, default=4, choices=[3, 4, 8, 16, 32], help="KV cache bit width")
    parser.add_argument("--kv_quant_group_size", type=int, default=64, help="KV quantization group size")
    parser.add_argument("--kv_residual_sign_correction", action="store_true", default=True, help="Enable KV residual sign correction")
    parser.add_argument("--no_kv_residual_sign_correction", action="store_false", dest="kv_residual_sign_correction", help="Disable KV residual sign correction")

    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--learning_rate", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "onecycle"], help="Learning rate scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing")
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing", help="Disable gradient checkpointing")
    parser.add_argument("--scheduler_target_epochs", type=int, default=150, help="Epoch span used to derive scheduler total steps")
    parser.add_argument("--pretokenize_batch_size", type=int, default=1024, help="Pretokenize batch size")
    parser.add_argument("--pretokenize_num_proc", type=int, default=None, help="Pretokenize worker process count")
    parser.add_argument("--pretokenize_cache", action="store_true", default=True, help="Enable pretokenize cache")
    parser.add_argument("--no_pretokenize_cache", action="store_false", dest="pretokenize_cache", help="Disable pretokenize cache")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader worker count")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="DataLoader prefetch factor")
    parser.add_argument("--persistent_workers", action="store_true", default=True, help="Enable persistent DataLoader workers")
    parser.add_argument("--no_persistent_workers", action="store_false", dest="persistent_workers", help="Disable persistent DataLoader workers")
    parser.add_argument("--fused_adamw", action="store_true", default=True, help="Enable fused AdamW when CUDA supports it")
    parser.add_argument("--no_fused_adamw", action="store_false", dest="fused_adamw", help="Disable fused AdamW")
    parser.add_argument("--tf32", action="store_true", default=True, help="Enable TF32 on Ampere+ GPUs")
    parser.add_argument("--no_tf32", action="store_false", dest="tf32", help="Disable TF32")

    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="Training precision mode")
    parser.add_argument("--multi_gpu_mode", type=str, default="auto", choices=["auto", "off"], help="Automatic single-node Linux multi-GPU DDP mode")
    parser.add_argument("--ddp_backend", type=str, default="nccl", choices=["nccl", "gloo"], help="Single-node DDP backend")
    parser.add_argument("--ddp_master_port", type=int, default=29500, help="Preferred master port for single-node DDP")

    parser.add_argument("--log_interval", type=int, default=1, help="Epoch logging interval")
    parser.add_argument("--save_interval_epochs", type=int, default=5, help="Checkpoint save interval in epochs")
    parser.add_argument("--early_stopping_patience", type=int, default=120, help="Early stopping patience")
    parser.add_argument("--early_stopping_monitor", type=str, default="train_loss", choices=["train_loss", "eval_loss"], help="Metric used by early stopping")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0005, help="Minimum early stopping improvement")
    parser.add_argument("--early_stopping_warmup_epochs", type=int, default=12, help="Warmup epochs before early stopping")
    parser.add_argument("--eval_interval_epochs", type=int, default=5, help="Eval interval in epochs")
    parser.add_argument("--eval_short_max_tokens", type=int, default=64, help="Short length bucket upper bound")
    parser.add_argument("--eval_medium_max_tokens", type=int, default=256, help="Medium length bucket upper bound")
    parser.add_argument("--metrics_log_interval_steps", type=int, default=200, help="CSV metrics logging interval in optimizer steps")
    parser.add_argument("--save_best_k", type=int, default=3, help="Keep this many best checkpoints")
    parser.add_argument("--save_latest_k", type=int, default=2, help="Keep this many latest checkpoints")
    parser.add_argument("--save_on_improve_delta", type=float, default=0.001, help="Improvement threshold for checkpoint saves")
    parser.add_argument("--checkpoint_cleanup", action="store_true", help="Enable checkpoint cleanup")
    parser.add_argument("--no_resume", action="store_true", help="Disable automatic checkpoint resume")
    parser.add_argument("--disable_tqdm_postfix", action="store_true", help="Disable tqdm postfix updates")

    parser.add_argument("--device", type=str, default=None, help="Manual device override")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="nagato-sakura", help="Weights & Biases project name")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Application log level")

    return parser


def _find_available_port(preferred_port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", int(preferred_port)))
            return int(preferred_port)
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


def _path_has_supervised_inputs(path_str: str | None) -> bool:
    if not path_str or not str(path_str).strip():
        return False

    candidate = Path(path_str)
    if not candidate.exists():
        return False
    if candidate.is_file():
        return candidate.suffix.lower() in {".json", ".jsonl"}

    for pattern in ("*.json", "*.jsonl"):
        if any(candidate.rglob(pattern)):
            return True
    return False


def _resolve_eval_data_source(
    args: argparse.Namespace,
    default_eval_path: str | None = None,
) -> str | None:
    if args.eval_data_file and str(args.eval_data_file).strip():
        return args.eval_data_file

    candidate = default_eval_path or os.path.join("data", "eval")
    if _path_has_supervised_inputs(candidate):
        return candidate
    return None


def _bootstrap_marker_paths(output_dir: str, bootstrap_id: str) -> tuple[Path, Path]:
    runtime_dir = Path(output_dir) / ".runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    success_path = runtime_dir / f"bootstrap_{bootstrap_id}.ok"
    error_path = runtime_dir / f"bootstrap_{bootstrap_id}.err"
    return success_path, error_path


def _new_bootstrap_id(seed: int | None = None) -> str:
    prefix = "single_node"
    if seed is None:
        return f"{prefix}_{os.getpid()}_{time.time_ns()}"
    return f"{prefix}_{int(seed)}_{os.getpid()}_{time.time_ns()}"


def _bootstrap_id_for_environment() -> str:
    explicit_id = str(os.environ.get("NSLLM_BOOTSTRAP_ID", "")).strip()
    if explicit_id:
        return explicit_id

    run_id = str(os.environ.get("TORCHELASTIC_RUN_ID", "")).strip()
    if run_id:
        master_port = str(os.environ.get("MASTER_PORT", "")).strip() or "unknown"
        return f"launcher_{master_port}_{run_id}"

    master_port = str(os.environ.get("MASTER_PORT", "")).strip() or "unknown"
    world_size = str(os.environ.get("WORLD_SIZE", "")).strip() or "1"
    return f"launcher_{master_port}_{world_size}"


def _has_launcher_distributed_env() -> bool:
    required_keys = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    return all(str(os.environ.get(key, "")).strip() for key in required_keys)


def _get_explicit_device_type(device: str | None) -> str | None:
    if device is None or not str(device).strip():
        return None
    return torch.device(device).type


def _should_bind_cuda_device(device: str | None) -> bool:
    if not torch.cuda.is_available():
        return False
    explicit_device_type = _get_explicit_device_type(device)
    if explicit_device_type is None:
        return True
    return explicit_device_type == "cuda"


def _is_multi_node_launcher_env() -> bool:
    if not _has_launcher_distributed_env():
        return False

    try:
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        return False

    local_world_size_raw = os.environ.get("LOCAL_WORLD_SIZE")
    if local_world_size_raw:
        try:
            local_world_size = max(1, int(local_world_size_raw))
            if world_size > local_world_size:
                return True
        except ValueError:
            pass

    for key in ("NODE_RANK", "GROUP_RANK"):
        raw_value = os.environ.get(key)
        if not raw_value or not str(raw_value).strip():
            continue
        try:
            if int(raw_value) > 0:
                return True
        except ValueError:
            continue

    return False


def _resolve_distributed_backend(requested_backend: str, device: str | None = None) -> str:
    backend = str(requested_backend).strip().lower()
    if backend == "nccl":
        if sys.platform == "win32":
            return "gloo"
        if _get_explicit_device_type(device) not in (None, "cuda"):
            return "gloo"
        if not torch.cuda.is_available():
            return "gloo"
        if hasattr(dist, "is_nccl_available") and not dist.is_nccl_available():
            return "gloo"
    return backend


def _should_use_auto_ddp(args: argparse.Namespace) -> bool:
    if _has_launcher_distributed_env():
        return False
    if args.multi_gpu_mode != "auto":
        return False
    if args.device is not None and str(args.device).strip():
        return False
    if sys.platform == "win32":
        return False
    if not torch.cuda.is_available():
        return False
    return torch.cuda.device_count() > 1


def _build_model_config(trainer: AdvancedNagatoSakuraTrainer, args: argparse.Namespace) -> NSConfig:
    tokenizer = trainer.tokenizer_manager.transformers_tokenizer
    return NSConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.max_seq_length,
        memory_tokens=args.memory_tokens,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        hidden_dropout=0.05,
        attention_dropout=0.05,
        gradient_checkpointing=args.gradient_checkpointing,
        quantize_kv_cache=args.quantize_kv_cache,
        kv_cache_bits=args.kv_cache_bits,
        kv_quant_group_size=args.kv_quant_group_size,
        kv_residual_sign_correction=args.kv_residual_sign_correction,
    )


def _run_single_node_data_bootstrap(
    args: argparse.Namespace,
    bootstrap_id: str,
):
    training_data_source = args.training_data_file or os.path.join("data", "train")
    eval_data_source = _resolve_eval_data_source(args)
    success_path, error_path = _bootstrap_marker_paths(args.output_dir, bootstrap_id)

    success_path.unlink(missing_ok=True)
    error_path.unlink(missing_ok=True)

    trainer = None
    training_data = None
    fixed_eval_data = None
    train_dataset = None
    eval_dataset = None

    try:
        trainer = AdvancedNagatoSakuraTrainer(
            model_config=None,
            output_dir=args.output_dir,
            device="cpu",
            use_wandb=False,
            project_name=args.wandb_project,
            precision="fp32",
            enable_tf32=False,
            is_distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
        )
        training_data, fixed_eval_data = _prepare_data(
            trainer,
            training_data_source,
            eval_data_source,
            args,
            rank=0,
            is_distributed=False,
        )
        trainer.model_config = _build_model_config(trainer, args)
        train_dataset, eval_dataset = _create_datasets(
            trainer,
            training_data,
            fixed_eval_data,
            args,
            rank=0,
            is_distributed=False,
        )
        success_path.write_text("ok", encoding="utf-8")
    except Exception as exc:
        error_path.write_text(str(exc), encoding="utf-8")
        raise
    finally:
        del train_dataset
        del eval_dataset
        del training_data
        del fixed_eval_data
        del trainer


def _wait_for_single_node_data_bootstrap(
    output_dir: str,
    bootstrap_id: str,
    poll_interval_seconds: float = 1.0,
):
    success_path, error_path = _bootstrap_marker_paths(output_dir, bootstrap_id)
    while True:
        if error_path.exists():
            message = error_path.read_text(encoding="utf-8").strip() or "bootstrap failed"
            raise RuntimeError(f"single-node data bootstrap failed: {message}")
        if success_path.exists():
            return
        time.sleep(max(0.1, float(poll_interval_seconds)))


def _bootstrap_single_node_training_artifacts(
    args: argparse.Namespace,
    bootstrap_id: str,
    local_rank: int,
):
    if local_rank == 0:
        print("Preparing tokenizer and pretokenized cache before initializing DDP...")
        _run_single_node_data_bootstrap(args, bootstrap_id)
    else:
        _wait_for_single_node_data_bootstrap(args.output_dir, bootstrap_id)


def _broadcast_rank0_stage_error(rank: int, stage_name: str, rank0_error: str | None):
    payload = [rank0_error if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError(f"{stage_name} failed on rank 0: {payload[0]}")


def _raise_if_any_rank_failed(stage_name: str, local_error: str | None, world_size: int):
    errors = [None] * world_size
    dist.all_gather_object(errors, local_error)
    failures = [f"rank {idx}: {error}" for idx, error in enumerate(errors) if error]
    if failures:
        raise RuntimeError(f"{stage_name} failed on {'; '.join(failures)}")


def _prepare_data(
    trainer: AdvancedNagatoSakuraTrainer,
    training_data_source: str,
    eval_data_source: str,
    args: argparse.Namespace,
    rank: int,
    is_distributed: bool,
):
    if not is_distributed:
        return trainer.prepare_data_and_tokenizer(
            training_data_source,
            args.vocab_size,
            args.force_retrain_tokenizer,
            tokenizer_min_frequency=args.tokenizer_min_frequency,
            eval_data_file=eval_data_source,
            tokenizer_train_max_samples=args.tokenizer_train_max_samples,
            tokenizer_num_threads=args.tokenizer_num_threads,
        )

    result = None
    rank0_error = None
    if rank == 0:
        try:
            result = trainer.prepare_data_and_tokenizer(
                training_data_source,
                args.vocab_size,
                args.force_retrain_tokenizer,
                tokenizer_min_frequency=args.tokenizer_min_frequency,
                eval_data_file=eval_data_source,
                tokenizer_train_max_samples=args.tokenizer_train_max_samples,
                tokenizer_num_threads=args.tokenizer_num_threads,
            )
        except Exception as exc:
            rank0_error = str(exc)

    _broadcast_rank0_stage_error(rank, "prepare_data_and_tokenizer", rank0_error)

    local_error = None
    if rank != 0:
        try:
            training_data = trainer._load_supervised_data_file(training_data_source, "training data")
            fixed_eval_data = None
            if eval_data_source:
                fixed_eval_data = trainer._load_supervised_data_file(eval_data_source, "evaluation data")
            trainer.tokenizer_manager.load_tokenizer()
            result = (training_data, fixed_eval_data)
        except Exception as exc:
            local_error = str(exc)

    _raise_if_any_rank_failed("distributed data preparation", local_error, world_size=trainer.world_size)
    return result


def _create_datasets(
    trainer: AdvancedNagatoSakuraTrainer,
    training_data,
    fixed_eval_data,
    args: argparse.Namespace,
    rank: int,
    is_distributed: bool,
):
    if not is_distributed:
        return trainer.create_datasets(
            training_data,
            args.eval_split_ratio,
            fixed_eval_data=fixed_eval_data,
            pretokenize_batch_size=args.pretokenize_batch_size,
            pretokenize_num_proc=args.pretokenize_num_proc,
            use_pretokenize_cache=args.pretokenize_cache,
        )

    datasets = None
    rank0_error = None
    if rank == 0:
        try:
            datasets = trainer.create_datasets(
                training_data,
                args.eval_split_ratio,
                fixed_eval_data=fixed_eval_data,
                pretokenize_batch_size=args.pretokenize_batch_size,
                pretokenize_num_proc=args.pretokenize_num_proc,
                use_pretokenize_cache=args.pretokenize_cache,
            )
        except Exception as exc:
            rank0_error = str(exc)

    _broadcast_rank0_stage_error(rank, "create_datasets", rank0_error)

    local_error = None
    if rank != 0:
        try:
            datasets = trainer.create_datasets(
                training_data,
                args.eval_split_ratio,
                fixed_eval_data=fixed_eval_data,
                pretokenize_batch_size=args.pretokenize_batch_size,
                pretokenize_num_proc=args.pretokenize_num_proc,
                use_pretokenize_cache=args.pretokenize_cache,
            )
        except Exception as exc:
            local_error = str(exc)

    _raise_if_any_rank_failed("distributed dataset creation", local_error, world_size=trainer.world_size)
    return datasets


def _distributed_worker(local_rank: int, world_size: int, args: argparse.Namespace, master_port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    bootstrap_id = _bootstrap_id_for_environment()

    _bootstrap_single_node_training_artifacts(args, bootstrap_id, local_rank)

    if _should_bind_cuda_device(args.device):
        torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=_resolve_distributed_backend(args.ddp_backend, device=args.device),
        rank=local_rank,
        world_size=world_size,
    )

    try:
        _run_training(
            args,
            rank=local_rank,
            local_rank=local_rank,
            world_size=world_size,
            is_distributed=True,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_distributed_from_environment(args: argparse.Namespace):
    if _is_multi_node_launcher_env():
        raise RuntimeError(
            "This runtime currently supports single-node multi-GPU launches only; "
            "multi-node training is intentionally rejected because output_dir artifacts "
            "and checkpoints are assumed to be shared across ranks."
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    is_distributed = world_size > 1

    if is_distributed:
        bootstrap_id = _bootstrap_id_for_environment()
        _bootstrap_single_node_training_artifacts(args, bootstrap_id, local_rank)
        if _should_bind_cuda_device(args.device):
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=_resolve_distributed_backend(args.ddp_backend, device=args.device),
            rank=rank,
            world_size=world_size,
        )

    try:
        _run_training(
            args,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            is_distributed=is_distributed,
        )
    finally:
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


def _run_training(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    is_distributed: bool,
    local_rank: int | None = None,
):
    training_data_source = args.training_data_file or os.path.join("data", "train")
    eval_data_source = _resolve_eval_data_source(args)
    local_rank = int(rank if local_rank is None else local_rank)

    try:
        device_arg = args.device
        trainer = AdvancedNagatoSakuraTrainer(
            model_config=None,
            output_dir=args.output_dir,
            device=device_arg,
            use_wandb=args.use_wandb,
            project_name=args.wandb_project,
            precision=args.precision,
            enable_tf32=args.tf32,
            is_distributed=is_distributed,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
        )

        training_data, fixed_eval_data = _prepare_data(
            trainer,
            training_data_source,
            eval_data_source,
            args,
            rank,
            is_distributed,
        )

        trainer.model_config = _build_model_config(trainer, args)
        trainer.initialize_model()

        train_dataset, eval_dataset = _create_datasets(
            trainer,
            training_data,
            fixed_eval_data,
            args,
            rank,
            is_distributed,
        )

        trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            max_grad_norm=args.max_grad_norm,
            log_interval=args.log_interval,
            save_interval_epochs=args.save_interval_epochs,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_monitor=args.early_stopping_monitor,
            early_stopping_min_delta=args.early_stopping_min_delta,
            early_stopping_warmup_epochs=args.early_stopping_warmup_epochs,
            resume_from_checkpoint=not args.no_resume,
            eval_interval_epochs=args.eval_interval_epochs,
            eval_short_max_tokens=args.eval_short_max_tokens,
            eval_medium_max_tokens=args.eval_medium_max_tokens,
            metrics_log_interval_steps=args.metrics_log_interval_steps,
            save_best_k=args.save_best_k,
            save_latest_k=args.save_latest_k,
            save_on_improve_delta=args.save_on_improve_delta,
            cleanup_old_checkpoints=args.checkpoint_cleanup,
            scheduler_target_epochs=args.scheduler_target_epochs,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            fused_adamw=args.fused_adamw,
            disable_tqdm_postfix=args.disable_tqdm_postfix,
        )
    except Exception as exc:
        import logging

        logger = logging.getLogger(__name__)
        if (not is_distributed) or rank == 0:
            logger.error(f"Training failed: {exc}", exc_info=True)
        if is_distributed:
            raise
        sys.exit(1)


def _print_system_environment():
    print("***** Nagato Sakura Training *****")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"Visible GPUs: {torch.cuda.device_count()}")
        for gpu_idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(gpu_idx)
            total_gb = props.total_memory / 1024**3
            print(f"GPU[{gpu_idx}]: {torch.cuda.get_device_name(gpu_idx)} ({total_gb:.1f} GB)")
        bf16_supported = bool(
            hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        )
        print(f"BF16 supported: {bf16_supported}")
    else:
        print("CUDA: unavailable")
    print("*******************************")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if _has_launcher_distributed_env():
        _run_distributed_from_environment(args)
    elif _should_use_auto_ddp(args):
        world_size = torch.cuda.device_count()
        master_port = _find_available_port(args.ddp_master_port)
        previous_bootstrap_id = os.environ.get("NSLLM_BOOTSTRAP_ID")
        os.environ["NSLLM_BOOTSTRAP_ID"] = _new_bootstrap_id(master_port)
        print(
            f"Launching single-node DDP training: world_size={world_size}, "
            f"backend={args.ddp_backend}, port={master_port}"
        )
        try:
            mp.spawn(_distributed_worker, args=(world_size, args, master_port), nprocs=world_size, join=True)
        finally:
            if previous_bootstrap_id is None:
                os.environ.pop("NSLLM_BOOTSTRAP_ID", None)
            else:
                os.environ["NSLLM_BOOTSTRAP_ID"] = previous_bootstrap_id
    else:
        _run_training(args, rank=0, world_size=1, is_distributed=False)


if __name__ == "__main__":
    _print_system_environment()
    if sys.platform == "win32":
        multiprocessing.freeze_support()
    main()
