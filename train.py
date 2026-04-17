import argparse
import multiprocessing
import os
import socket
import sys
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 自定義模型導入
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.trainer import AdvancedNagatoSakuraTrainer
    from src.nagato_sakura_model import NSConfig
except ImportError as e:
    print(f"錯誤：無法導入自定義模組。錯誤詳情: {e}")
    sys.exit(1)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="長門櫻語言模型訓練器")

    # 數據相關
    parser.add_argument("--training_data_file", type=str, help="訓練數據來源（檔案或資料夾）；未指定時使用 data/train")
    parser.add_argument("--output_dir", type=str, default="NS-LLM-0.8", help="輸出目錄")
    parser.add_argument("--force_retrain_tokenizer", action="store_true", help="強制重新訓練分詞器")
    parser.add_argument("--eval_split_ratio", type=float, default=0.0, help="評估集分割比例（使用固定評估集時請設為0）")
    parser.add_argument("--eval_data_file", type=str, help="固定評估集來源（檔案或資料夾）；未指定時使用 data/eval")

    # 模型配置
    parser.add_argument("--vocab_size", type=int, default=65536, help="詞彙表大小")
    parser.add_argument("--tokenizer_min_frequency", type=int, default=5, help="分詞器最小詞頻")
    parser.add_argument("--tokenizer_train_max_samples", type=int, default=0, help="分詞器最多使用多少訓練樣本（0=不限制）")
    parser.add_argument("--tokenizer_num_threads", type=int, default=0, help="分詞器執行緒數（0=自動）")
    parser.add_argument("--hidden_size", type=int, default=512, help="隱藏層大小")
    parser.add_argument("--num_layers", type=int, default=8, help="層數")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力頭數")
    parser.add_argument("--num_key_value_heads", type=int, default=4, help="GQA 的 key/value 頭數")
    parser.add_argument("--intermediate_size", type=int, default=1536, help="MLP中間層大小")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="最大序列長度")
    parser.add_argument("--memory_tokens", type=int, default=32, help="記憶令牌數量")
    parser.add_argument("--quantize_kv_cache", action="store_true", default=True, help="啟用 KV cache 量化")
    parser.add_argument("--no_quantize_kv_cache", action="store_false", dest="quantize_kv_cache", help="禁用 KV cache 量化")
    parser.add_argument("--kv_cache_bits", type=int, default=4, choices=[3, 4, 8, 16, 32], help="KV cache 位寬")
    parser.add_argument("--kv_quant_group_size", type=int, default=64, help="KV 量化分組大小")
    parser.add_argument("--kv_residual_sign_correction", action="store_true", default=True, help="啟用 1-bit 殘差符號修正")
    parser.add_argument("--no_kv_residual_sign_correction", action="store_false", dest="kv_residual_sign_correction", help="禁用 1-bit 殘差符號修正")

    # 訓練參數
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="梯度累積步數")
    parser.add_argument("--num_epochs", type=int, default=300, help="訓練輪數")
    parser.add_argument("--learning_rate", type=float, default=1.5e-4, help="學習率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="權重衰減")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "onecycle"], help="學習率調度器類型")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="預熱比例")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪閾值")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="啟用梯度檢查點")
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing", help="禁用梯度檢查點")
    parser.add_argument("--scheduler_target_epochs", type=int, default=150, help="學習率調度目標epoch（可低於實際訓練輪數）")
    parser.add_argument("--pretokenize_batch_size", type=int, default=1024, help="預分詞批次大小")
    parser.add_argument("--pretokenize_num_proc", type=int, default=None, help="預分詞進程數（Windows 將固定為 1）")
    parser.add_argument("--pretokenize_cache", action="store_true", default=True, help="啟用預分詞快取")
    parser.add_argument("--no_pretokenize_cache", action="store_false", dest="pretokenize_cache", help="禁用預分詞快取")

    # 精度與多卡
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="訓練精度模式")
    parser.add_argument("--multi_gpu_mode", type=str, default="auto", choices=["auto", "off"], help="多卡模式（auto 在 Linux + 多GPU 時自動啟動 DDP）")
    parser.add_argument("--ddp_backend", type=str, default="nccl", choices=["nccl", "gloo"], help="DDP 通訊後端")
    parser.add_argument("--ddp_master_port", type=int, default=29500, help="DDP Master Port")

    # DataLoader
    parser.add_argument("--dataloader_num_workers", type=int, default=-1, help="DataLoader workers；-1代表自動")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2, help="DataLoader prefetch factor")
    parser.add_argument("--dataloader_persistent_workers", action="store_true", default=True, help="啟用 persistent workers")
    parser.add_argument("--no_dataloader_persistent_workers", action="store_false", dest="dataloader_persistent_workers", help="禁用 persistent workers")
    parser.add_argument("--dataloader_drop_last", action="store_true", default=True, help="訓練 DataLoader 啟用 drop_last")
    parser.add_argument("--no_dataloader_drop_last", action="store_false", dest="dataloader_drop_last", help="訓練 DataLoader 禁用 drop_last")

    # 日誌和檢查點
    parser.add_argument("--log_interval", type=int, default=1, help="日誌記錄間隔（按epoch）")
    parser.add_argument("--save_interval_epochs", type=int, default=5, help="按epoch保存間隔")
    parser.add_argument("--early_stopping_patience", type=int, default=120, help="早停耐心值（按epoch）")
    parser.add_argument("--early_stopping_monitor", type=str, default="train_loss", choices=["train_loss", "eval_loss"], help="早停監控指標")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0005, help="早停最小改善幅度")
    parser.add_argument("--early_stopping_warmup_epochs", type=int, default=12, help="早停啟用前的預熱epoch")
    parser.add_argument("--eval_interval_epochs", type=int, default=1, help="評估間隔（按epoch）")
    parser.add_argument("--eval_short_max_tokens", type=int, default=64, help="短樣本分桶上限token數")
    parser.add_argument("--eval_medium_max_tokens", type=int, default=256, help="中樣本分桶上限token數")
    parser.add_argument("--metrics_log_interval_steps", type=int, default=50, help="CSV指標記錄間隔（按optimizer step）")
    parser.add_argument("--save_best_k", type=int, default=3, help="保留最佳checkpoint數量")
    parser.add_argument("--save_latest_k", type=int, default=2, help="保留最新checkpoint數量")
    parser.add_argument("--save_on_improve_delta", type=float, default=0.001, help="達到此改善幅度時觸發保存")
    parser.add_argument("--checkpoint_cleanup", action="store_true", help="啟用舊checkpoint自動清理")
    parser.add_argument("--no_resume", action="store_true", help="不從檢查點恢復")
    parser.add_argument("--resume_checkpoint", type=str, help="指定恢復的checkpoint路徑（可為相對或絕對路徑）")

    # 其他
    parser.add_argument("--device", type=str, default=None, help="指定設備")
    parser.add_argument("--use_wandb", action="store_true", help="使用WandB記錄")
    parser.add_argument("--wandb_project", type=str, default="nagato-sakura", help="WandB項目名稱")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日誌級別")

    args = parser.parse_args()

    if args.no_resume and args.resume_checkpoint:
        print("警告: 已設定 --no_resume，將忽略 --resume_checkpoint")

    if _should_use_auto_ddp(args):
        world_size = torch.cuda.device_count()
        master_port = _find_available_port(args.ddp_master_port)
        print(f"自動多卡啟動: 啟用 DDP，world_size={world_size}, backend={args.ddp_backend}, port={master_port}")
        mp.spawn(_distributed_worker, args=(world_size, args, master_port), nprocs=world_size, join=True)
    else:
        _run_training(args, rank=0, world_size=1, is_distributed=False)


def _find_available_port(preferred_port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", int(preferred_port)))
            return int(preferred_port)
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


def _should_use_auto_ddp(args: argparse.Namespace) -> bool:
    if args.multi_gpu_mode != "auto":
        return False
    if sys.platform == "win32":
        return False
    if not torch.cuda.is_available():
        return False
    return torch.cuda.device_count() > 1


def _distributed_worker(local_rank: int, world_size: int, args: argparse.Namespace, master_port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=args.ddp_backend,
        rank=local_rank,
        world_size=world_size,
    )

    try:
        _run_training(args, rank=local_rank, world_size=world_size, is_distributed=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_training(args: argparse.Namespace, rank: int, world_size: int, is_distributed: bool):
    training_data_source = args.training_data_file or os.path.join("data", "train")
    eval_data_source = args.eval_data_file or os.path.join("data", "eval")
    dataloader_num_workers = None if args.dataloader_num_workers < 0 else int(args.dataloader_num_workers)

    try:
        device_arg = f"cuda:{rank}" if is_distributed else args.device

        trainer = AdvancedNagatoSakuraTrainer(
            model_config=None,
            output_dir=args.output_dir,
            device=device_arg,
            use_wandb=args.use_wandb,
            project_name=args.wandb_project,
            precision=args.precision,
            is_distributed=is_distributed,
            rank=rank,
            local_rank=rank,
            world_size=world_size,
        )

        if is_distributed:
            if rank == 0:
                training_data, fixed_eval_data = trainer.prepare_data_and_tokenizer(
                    training_data_source,
                    args.vocab_size,
                    args.force_retrain_tokenizer,
                    tokenizer_min_frequency=args.tokenizer_min_frequency,
                    eval_data_file=eval_data_source,
                    tokenizer_train_max_samples=args.tokenizer_train_max_samples,
                    tokenizer_num_threads=args.tokenizer_num_threads,
                )
                dist.barrier()
            else:
                dist.barrier()
                training_data = trainer._load_supervised_data_file(training_data_source, "訓練數據")
                fixed_eval_data = None
                if eval_data_source:
                    fixed_eval_data = trainer._load_supervised_data_file(eval_data_source, "固定評估集")
                trainer.tokenizer_manager.load_tokenizer()
        else:
            training_data, fixed_eval_data = trainer.prepare_data_and_tokenizer(
                training_data_source,
                args.vocab_size,
                args.force_retrain_tokenizer,
                tokenizer_min_frequency=args.tokenizer_min_frequency,
                eval_data_file=eval_data_source,
                tokenizer_train_max_samples=args.tokenizer_train_max_samples,
                tokenizer_num_threads=args.tokenizer_num_threads,
            )

        model_config = NSConfig(
            vocab_size=trainer.tokenizer_manager.transformers_tokenizer.vocab_size,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            num_key_value_heads=args.num_key_value_heads,
            max_position_embeddings=args.max_seq_length,
            memory_tokens=args.memory_tokens,
            pad_token_id=trainer.tokenizer_manager.transformers_tokenizer.pad_token_id,
            bos_token_id=trainer.tokenizer_manager.transformers_tokenizer.bos_token_id,
            eos_token_id=trainer.tokenizer_manager.transformers_tokenizer.eos_token_id,
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

        trainer.model_config = model_config
        trainer.initialize_model()

        train_dataset, eval_dataset = trainer.create_datasets(
            training_data,
            args.eval_split_ratio,
            fixed_eval_data=fixed_eval_data,
            pretokenize_batch_size=args.pretokenize_batch_size,
            pretokenize_num_proc=args.pretokenize_num_proc,
            use_pretokenize_cache=args.pretokenize_cache,
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
            resume_checkpoint=args.resume_checkpoint,
            eval_interval_epochs=args.eval_interval_epochs,
            eval_short_max_tokens=args.eval_short_max_tokens,
            eval_medium_max_tokens=args.eval_medium_max_tokens,
            metrics_log_interval_steps=args.metrics_log_interval_steps,
            save_best_k=args.save_best_k,
            save_latest_k=args.save_latest_k,
            save_on_improve_delta=args.save_on_improve_delta,
            cleanup_old_checkpoints=args.checkpoint_cleanup,
            scheduler_target_epochs=args.scheduler_target_epochs,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_prefetch_factor=args.dataloader_prefetch_factor,
            dataloader_persistent_workers=args.dataloader_persistent_workers,
            dataloader_drop_last=args.dataloader_drop_last,
        )

    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        if (not is_distributed) or rank == 0:
            logger.error(f"程序執行失敗: {e}", exc_info=True)
        if is_distributed:
            raise
        sys.exit(1)


def _print_system_environment():
    print("***** 系統環境檢查 *****")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"可見GPU數量: {torch.cuda.device_count()}")
        for gpu_idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(gpu_idx)
            print(f"GPU[{gpu_idx}]: {torch.cuda.get_device_name(gpu_idx)} ({props.total_memory / 1024**3:.1f}GB)")
        bf16_supported = bool(
            hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        )
        print(f"BF16支援: {'可用' if bf16_supported else '不可用'}")
    else:
        print("CUDA: 不可用")
    try:
        from torch.cuda.amp import GradScaler
        _ = GradScaler
        print("混合精度: 可用")
    except ImportError:
        print("混合精度: 不可用")
    print("************************")


if __name__ == "__main__":
    _print_system_environment()
    if sys.platform == "win32":
        multiprocessing.freeze_support()
    main()