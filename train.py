import argparse
import logging
import multiprocessing
import os
import socket
import sys
import warnings
from typing import Optional
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# 忽略非必要警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 自定義模型與訓練器導入
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.nagato_sakura_model import NSConfig
    from src.trainer import AdvancedNagatoSakuraTrainer
except ImportError as e:
    print(f"錯誤：無法導入自定義模組。錯誤詳情: {e}")
    sys.exit(1)


# ==============================================================================
# 輔助與網絡功能函數
# ==============================================================================


def _find_available_port(preferred_port: int) -> int:
    """
    尋找可用網路埠號以供 DDP Master 使用。

    Args:
        preferred_port (int): 優先嘗試綁定的埠號。

    Returns:
        int: 成功綁定的可用埠號。
    """
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", int(preferred_port)))
            return int(preferred_port)
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


def _should_use_auto_ddp(args: argparse.Namespace) -> bool:
    """
    判斷當前環境是否符合自動啟用多卡 DDP 模式的條件。

    Args:
        args (argparse.Namespace): 命令列解析參數。

    Returns:
        bool: 若應啟用 DDP 則傳回 True，否則傳回 False。
    """
    
    if args.multi_gpu_mode != "auto":
        return False
    
    if sys.platform == "win32":
        return False
    
    if not torch.cuda.is_available():
        return False
    
    return torch.cuda.device_count() > 1


def _distributed_worker(local_rank: int, world_size: int, args: argparse.Namespace, master_port: int) -> None:
    """
    DDP 分散式訓練的 Worker 子進程執行函數。

    Args:
        local_rank (int): 當前 Worker 的 GPU 卡號 (Rank)。
        world_size (int): 總卡數/進程數。
        args (argparse.Namespace): 訓練配置參數。
        master_port (int): DDP 主機通訊埠號。
    """
    
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.ddp_backend, rank=local_rank, world_size=world_size)
    try:
        _run_training(args, rank=local_rank, world_size=world_size, is_distributed=True)
        
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_training(args: argparse.Namespace, rank: int, world_size: int, is_distributed: bool) -> None:
    """
    執行訓練的核心邏輯流程（包含資料集準備、模型建構、訓練器啟動與例外處理）。

    Args:
        args (argparse.Namespace): 命令列與組態參數。
        rank (int): 當前進程 Rank 號。
        world_size (int): 總進程數。
        is_distributed (bool): 是否為分散式訓練。
    """
    
    logger = logging.getLogger(__name__)
    training_data_source = args.training_data_file or os.path.join("data", "train")
    eval_data_source = args.eval_data_file or os.path.join("data", "eval")
    dataloader_num_workers: Optional[int] = None if args.dataloader_num_workers < 0 else int(args.dataloader_num_workers)
    
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
                    tokenizer_enable_universal_charset=args.tokenizer_universal_charset,
                    tokenizer_extra_chars_files=args.tokenizer_extra_chars_file,
                    tokenizer_random_sampling=args.tokenizer_random_sampling,
                    tokenizer_sample_ratio=args.tokenizer_sample_ratio,
                )
                dist.barrier()
            else:
                dist.barrier()
                training_data = trainer._load_supervised_data_file(training_data_source, "訓練數據")
                fixed_eval_data = None
                if eval_data_source:
                    fixed_eval_data = trainer._load_supervised_data_file(
                        eval_data_source, "固定評估集"
                    )
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
                tokenizer_enable_universal_charset=args.tokenizer_universal_charset,
                tokenizer_extra_chars_files=args.tokenizer_extra_chars_file,
                tokenizer_random_sampling=args.tokenizer_random_sampling,
                tokenizer_sample_ratio=args.tokenizer_sample_ratio,
                tokenizer_enable_zh_common=args.tokenizer_enable_zh_common,
                tokenizer_enable_zh_yi=args.tokenizer_enable_zh_yi,
                tokenizer_enable_zh_bing=args.tokenizer_enable_zh_bing,
                tokenizer_enable_ja=args.tokenizer_enable_ja,
                tokenizer_enable_emoji=args.tokenizer_enable_emoji,
                tokenizer_enable_symbols=args.tokenizer_enable_symbols,
                tokenizer_enable_programming=args.tokenizer_enable_programming,
            )

        model_config = NSConfig(
            vocab_size=len(trainer.tokenizer_manager.transformers_tokenizer),
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
            unk_token_id=trainer.tokenizer_manager.transformers_tokenizer.unk_token_id,
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
            use_compile=args.use_compile,
            compile_mode=args.compile_mode,
            debug_nan_inf=args.debug_nan_inf,
        )

        trainer.model_config = model_config
        trainer.initialize_model()
        train_dataset, eval_dataset = trainer.create_datasets(
            training_data,
            args.eval_split_ratio,
            fixed_eval_data=fixed_eval_data,
            enable_pretokenize=not args.disable_pretokenize,
            pretokenize_batch_size=args.pretokenize_batch_size,
            pretokenize_num_proc=args.pretokenize_num_proc,
            use_pretokenize_cache=args.pretokenize_cache,
            fail_on_unk_tokens=args.fail_on_unk_tokens,
            unk_audit_max_samples=args.unk_audit_max_samples,
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
            resume_model_only=args.resume_model_only,
            resume_lr_scale=args.resume_lr_scale,
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
            pack_sequences=args.pack_sequences,
        )

    except Exception as e:
        if (not is_distributed) or rank == 0:
            logger.error(f"程序執行失敗: {e}", exc_info=True)
            
        if is_distributed:
            raise
        sys.exit(1)


# ==============================================================================
# 主程式入口
# ==============================================================================


def main() -> None:
    """主函數：建構 CLI 解析器並啟動單卡或多卡訓練"""
    
    parser = argparse.ArgumentParser(description="長門櫻語言模型訓練器 (NagatoSakura LLM Trainer)")

    # 數據設定
    data_group = parser.add_argument_group("數據設定 (Data Configurations)")
    data_group.add_argument("--training_data_file", type=str, help="訓練數據來源（檔案或資料夾）；未指定時使用 data/train")
    data_group.add_argument("--output_dir", type=str, default="NS-LM-1.6-pico", help="輸出目錄")
    data_group.add_argument("--force_retrain_tokenizer", action="store_true", help="強制重新訓練分詞器")
    data_group.add_argument("--eval_split_ratio", type=float, default=0.0, help="評估集分割比例（使用固定評估集時請設為0）")
    data_group.add_argument("--eval_data_file", type=str, help="固定評估集來源（檔案或資料夾）；未指定時使用 data/eval")

    # 模型與分詞器配置
    model_group = parser.add_argument_group("模型與分詞器配置 (Model & Tokenizer Configurations)")
    model_group.add_argument("--vocab_size", type=int, default=32768, help="詞彙表大小")
    model_group.add_argument("--tokenizer_min_frequency", type=int, default=5, help="分詞器最小詞頻")
    model_group.add_argument("--tokenizer_train_max_samples", type=int, default=0,help="分詞器最多使用多少訓練樣本（0=不限制）")
    model_group.add_argument("--tokenizer_num_threads", type=int, default=0, help="分詞器執行緒數（0=自動）")
    model_group.add_argument("--tokenizer_universal_charset", action="store_true", default=True, help="啟用萬能分詞器分層保底")
    model_group.add_argument("--no_tokenizer_universal_charset", action="store_false", dest="tokenizer_universal_charset", help="禁用萬能分詞器保底字元")
    model_group.add_argument("--tokenizer_extra_chars_file", action="append", default=[], help="額外保底字元/詞彙檔案（例如甲乙丙字表），可重複指定")
    model_group.add_argument("--tokenizer_random_sampling", action="store_true", default=True, help="對大規模訓練數據進行隨機採樣（>10萬自動啟用）")
    model_group.add_argument("--no_tokenizer_random_sampling", action="store_false", dest="tokenizer_random_sampling", help="禁用分詞器隨機採樣")
    model_group.add_argument("--tokenizer_sample_ratio", type=float, default=0.1, help="分詞器採樣比率（當數據>10萬時自動應用，值為0.0-1.0）")
    model_group.add_argument("--tokenizer_enable_zh_common", action="store_true", default=True, help="啟用教育部甲表常用字 4808 字保底")
    model_group.add_argument("--no_tokenizer_enable_zh_common", action="store_false", dest="tokenizer_enable_zh_common", help="禁用甲表常用字保底")
    model_group.add_argument("--tokenizer_enable_zh_yi", action="store_true", default=False, help="啟用教育部乙表次常用字 6329 字保底")
    model_group.add_argument("--no_tokenizer_enable_zh_yi", action="store_false", dest="tokenizer_enable_zh_yi", help="禁用乙表次常用字保底")
    model_group.add_argument("--tokenizer_enable_zh_bing", action="store_true", default=False, help="啟用教育部丙表罕用字 6548 字保底")
    model_group.add_argument("--no_tokenizer_enable_zh_bing", action="store_false", dest="tokenizer_enable_zh_bing", help="禁用丙表罕用字保底")
    model_group.add_argument("--tokenizer_enable_ja", action="store_true", default=True, help="啟用日文常用漢字 2136 字保底")
    model_group.add_argument("--no_tokenizer_enable_ja", action="store_false", dest="tokenizer_enable_ja", help="禁用日文常用漢字保底")
    model_group.add_argument("--tokenizer_enable_emoji", action="store_true", default=True, help="啟用 Unicode Emoji 保底")
    model_group.add_argument("--no_tokenizer_enable_emoji", action="store_false", dest="tokenizer_enable_emoji", help="禁用 Emoji 保底")
    model_group.add_argument("--tokenizer_enable_symbols", action="store_true", default=True, help="啟用通用標點符號保底")
    model_group.add_argument("--no_tokenizer_enable_symbols", action="store_false", dest="tokenizer_enable_symbols", help="禁用通用標點符號保底")
    model_group.add_argument("--tokenizer_enable_programming", action="store_true", default=False, help="啟用程式關鍵字保底")
    model_group.add_argument("--no_tokenizer_enable_programming", action="store_false", dest="tokenizer_enable_programming", help="禁用程式關鍵字保底")
    model_group.add_argument("--hidden_size", type=int, default=128, help="隱藏層大小")
    model_group.add_argument("--num_layers", type=int, default=2, help="層數")
    model_group.add_argument("--num_heads", type=int, default=2, help="注意力頭數")
    model_group.add_argument("--num_key_value_heads", type=int, default=1, help="GQA 的 key/value 頭數")
    model_group.add_argument("--intermediate_size", type=int, default=256, help="MLP中間層大小")
    model_group.add_argument("--max_seq_length", type=int, default=4096, help="最大序列長度")
    model_group.add_argument("--memory_tokens", type=int, default=0, help="記憶令牌數量")
    model_group.add_argument("--quantize_kv_cache", action="store_true", default=False, help="啟用 KV cache 量化")
    model_group.add_argument("--no_quantize_kv_cache", action="store_false", dest="quantize_kv_cache", help="禁用 KV cache 量化")
    model_group.add_argument("--kv_cache_bits", type=int, default=4, choices=[3, 4, 8, 16, 32], help="KV cache 位寬")
    model_group.add_argument("--kv_quant_group_size", type=int, default=64, help="KV 量化分組大小")
    model_group.add_argument("--kv_residual_sign_correction", action="store_true", default=True, help="啟用 1-bit 殘差符號修正")
    model_group.add_argument("--no_kv_residual_sign_correction", action="store_false", dest="kv_residual_sign_correction", help="禁用 1-bit 殘差符號修正")
    model_group.add_argument("--use_compile", action="store_true", default=False, help="啟用 torch.compile 加速模型前向傳播 (Try-Except 容錯保護)")
    model_group.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"], help="torch.compile 模式")
    model_group.add_argument("--debug_nan_inf", action="store_true", default=False, help="啟用逐層 NaN/Inf 除錯掃描 (僅除錯用，常態訓練關閉以避免 GPU 同步開銷)")

    # 訓練參數
    train_group = parser.add_argument_group("訓練參數 (Training Hyperparameters)")
    train_group.add_argument("--batch_size", type=int, default=4, help="批次大小")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累積步數")
    train_group.add_argument("--num_epochs", type=int, default=30, help="訓練輪數")
    train_group.add_argument("--learning_rate", type=float, default=5e-4, help="學習率")
    train_group.add_argument("--weight_decay", type=float, default=0.005, help="權重衰減")
    train_group.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "onecycle"], help="學習率調度器類型")
    train_group.add_argument("--warmup_ratio", type=float, default=0.03, help="預熱比例")
    train_group.add_argument("--max_grad_norm", type=float, default=0.7, help="梯度裁剪閾值")
    train_group.add_argument("--gradient_checkpointing", action="store_true", default=False, help="啟用梯度檢查點 (節省顯存，但增加額外計算量；小模型建議預設關閉以最大化 GPU 吞吐量)")
    train_group.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing", help="禁用梯度檢查點")
    train_group.add_argument("--scheduler_target_epochs", type=int, default=150, help="學習率調度目標epoch（可低於實際訓練輪數）")
    train_group.add_argument("--pretokenize_batch_size", type=int, default=1024, help="預分詞批次大小")
    train_group.add_argument("--pretokenize_num_proc", type=int, default=1, help="預分詞進程數（預設 1，充分利用 Rust 原生 Rayon 滿核並行加速）")
    train_group.add_argument("--pretokenize_cache", action="store_true", default=True, help="啟用預分詞快取")
    train_group.add_argument("--no_pretokenize_cache", action="store_false", dest="pretokenize_cache", help="禁用預分詞快取")
    train_group.add_argument("--disable_pretokenize", action="store_true", default=False, help="停用啟動前預分詞（改為訓練時即時分詞）")
    train_group.add_argument("--unk_audit_max_samples", type=int, default=4096, help="資料集 <unk> 抽樣掃描上限（每個 split）")
    train_group.add_argument("--fail_on_unk_tokens", action="store_true", help="若資料集掃描發現 <unk> token，立即中止訓練")
    train_group.add_argument("--pack_sequences", action="store_true", default=True, help="啟用序列打包 (First-Fit-Decreasing Bin Packing)，提升有效 token/s 訓練吞吐量")
    train_group.add_argument("--no_pack_sequences", action="store_false", dest="pack_sequences", help="禁用序列打包（改用標準 Padding 模式）")

    # 精度與多卡
    precision_group = parser.add_argument_group("精度與分散式訓練 (Precision & Multi-GPU)")
    precision_group.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="訓練精度模式")
    precision_group.add_argument("--multi_gpu_mode", type=str, default="auto", choices=["auto", "off"], help="多卡模式（auto 在 Linux + 多GPU 時自動啟動 DDP）")
    precision_group.add_argument("--ddp_backend", type=str, default="nccl", choices=["nccl", "gloo"], help="DDP 通訊後端")
    precision_group.add_argument("--ddp_master_port", type=int, default=29500, help="DDP Master Port")

    # DataLoader
    dataloader_group = parser.add_argument_group("DataLoader 設定")
    dataloader_group.add_argument("--dataloader_num_workers", type=int, default=-1, help="DataLoader workers；-1 代表自動 (Windows 自動設為 0 以避免 spawn 啟動延遲，Linux 為 4/8)")
    dataloader_group.add_argument("--dataloader_prefetch_factor", type=int, default=2, help="DataLoader prefetch factor")
    dataloader_group.add_argument("--dataloader_persistent_workers", action="store_true", default=True, help="啟用 persistent workers")
    dataloader_group.add_argument("--no_dataloader_persistent_workers", action="store_false", dest="dataloader_persistent_workers", help="禁用 persistent workers")
    dataloader_group.add_argument("--dataloader_drop_last", action="store_true", default=True, help="訓練 DataLoader 啟用 drop_last")
    dataloader_group.add_argument("--no_dataloader_drop_last", action="store_false", dest="dataloader_drop_last", help="訓練 DataLoader 禁用 drop_last")

    # 日誌和檢查點
    checkpoint_group = parser.add_argument_group("日誌與 Checkpoint 設定 (Logging & Checkpoints)")
    checkpoint_group.add_argument("--log_interval", type=int, default=1, help="日誌記錄間隔（按epoch）")
    checkpoint_group.add_argument("--save_interval_epochs", type=int, default=5, help="按epoch保存間隔")
    checkpoint_group.add_argument("--early_stopping_patience", type=int, default=30, help="早停耐心值（按epoch）")
    checkpoint_group.add_argument("--early_stopping_monitor",type=str,default="train_loss",choices=["train_loss", "eval_loss"],help="早停監控指標")
    checkpoint_group.add_argument("--early_stopping_min_delta", type=float, default=0.0001, help="早停最小改善幅度")
    checkpoint_group.add_argument("--early_stopping_warmup_epochs", type=int, default=8, help="早停啟用前的預熱epoch")
    checkpoint_group.add_argument("--eval_interval_epochs", type=int, default=1, help="評估間隔（按epoch）")
    checkpoint_group.add_argument("--eval_short_max_tokens", type=int, default=64, help="短樣本分桶上限token數")
    checkpoint_group.add_argument("--eval_medium_max_tokens", type=int, default=256, help="中樣本分桶上限token數")
    checkpoint_group.add_argument("--metrics_log_interval_steps", type=int, default=50, help="CSV指標記錄間隔（按optimizer step）")
    checkpoint_group.add_argument("--save_best_k", type=int, default=1, help="保留最佳checkpoint數量")
    checkpoint_group.add_argument("--save_latest_k", type=int, default=1, help="保留最新checkpoint數量")
    checkpoint_group.add_argument("--save_on_improve_delta", type=float, default=0.0001, help="達到此改善幅度時觸發保存")
    checkpoint_group.add_argument("--checkpoint_cleanup", action="store_true", help="啟用舊checkpoint自動清理")
    checkpoint_group.add_argument("--no_resume", action="store_true", help="不從檢查點恢復")
    checkpoint_group.add_argument("--resume_checkpoint", type=str, help="指定恢復的checkpoint路徑（可為相對或絕對路徑）")
    checkpoint_group.add_argument("--resume_model_only", action="store_true", help="僅恢復模型權重，不恢復 optimizer/scheduler/scaler 與 step/epoch")
    checkpoint_group.add_argument("--resume_lr_scale", type=float, default=0.0, help="續訓後套用學習率縮放（<=0 代表自動對齊到 --learning_rate；例如 0.5 代表將當前LR減半）")

    # 其他系統參數
    other_group = parser.add_argument_group("其他系統參數 (Other Systems)")
    other_group.add_argument("--device", type=str, default=None, help="指定設備")
    other_group.add_argument("--use_wandb", action="store_true", help="使用WandB記錄")
    other_group.add_argument("--wandb_project", type=str, default="nagato-sakura", help="WandB項目名稱")
    other_group.add_argument("--log_level", type=str,default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日誌級別")

    args = parser.parse_args()

    if args.no_resume and args.resume_checkpoint:
        print("警告: 已設定 --no_resume，將忽略 --resume_checkpoint")
        
    if args.force_retrain_tokenizer and not args.no_resume:
        parser.error("啟用 --force_retrain_tokenizer 時必須同時指定 --no_resume，以避免新詞表與舊 checkpoint 不相容")
        
    if _should_use_auto_ddp(args):
        world_size = torch.cuda.device_count()
        master_port = _find_available_port(args.ddp_master_port)
        print(f"自動多卡啟動: 啟用 DDP，world_size={world_size}, backend={args.ddp_backend}, port={master_port}")
        mp.spawn(_distributed_worker, args=(world_size, args, master_port), nprocs=world_size, join=True)
    else:
        _run_training(args, rank=0, world_size=1, is_distributed=False)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        
    if sys.platform == "win32":
        multiprocessing.freeze_support()
    main()
