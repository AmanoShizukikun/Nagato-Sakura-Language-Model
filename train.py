import os
import sys
import multiprocessing
import warnings
import argparse

import torch
from datasets import Dataset

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
    parser.add_argument("--training_data_file", type=str, default="generated_data/all.json", help="訓練數據文件路徑")
    parser.add_argument("--output_dir", type=str, default="NS-LLM", help="輸出目錄")
    parser.add_argument("--force_retrain_tokenizer", action="store_true", help="強制重新訓練分詞器")
    parser.add_argument("--eval_split_ratio", type=float, default=0.01, help="評估集分割比例")
    
    # 模型配置
    parser.add_argument("--vocab_size", type=int, default=32768, help="詞彙表大小")
    parser.add_argument("--hidden_size", type=int, default=1024, help="隱藏層大小")
    parser.add_argument("--num_layers", type=int, default=12, help="層數")
    parser.add_argument("--num_heads", type=int, default=16, help="注意力頭數")
    parser.add_argument("--intermediate_size", type=int, default=4096, help="MLP中間層大小")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="最大序列長度")
    parser.add_argument("--memory_tokens", type=int, default=0, help="記憶令牌數量")
    
    # 訓練參數
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="梯度累積步數")
    parser.add_argument("--num_epochs", type=int, default=1000, help="訓練輪數")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="學習率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="權重衰減")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "onecycle"], help="學習率調度器類型")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="預熱比例")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪閾值")
    
    # 日誌和檢查點
    parser.add_argument("--log_interval", type=int, default=50, help="日誌記錄間隔")
    parser.add_argument("--save_interval_steps", type=int, default=500, help="按步數保存間隔")
    parser.add_argument("--save_interval_epochs", type=int, default=5, help="按epoch保存間隔")
    parser.add_argument("--eval_interval_steps", type=int, default=500, help="評估間隔")
    parser.add_argument("--early_stopping_patience", type=int, default=1000, help="早停耐心值")
    parser.add_argument("--no_resume", action="store_true", help="不從檢查點恢復")
    
    # 其他
    parser.add_argument("--device", type=str, default=None, help="指定設備")
    parser.add_argument("--use_wandb", action="store_true", help="使用WandB記錄")
    parser.add_argument("--wandb_project", type=str, default="nagato-sakura", help="WandB項目名稱")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日誌級別")
    
    args = parser.parse_args()
    
    try:
        # 創建訓練器
        trainer = AdvancedNagatoSakuraTrainer(
            model_config=None,
            output_dir=args.output_dir,
            device=args.device,
            use_wandb=args.use_wandb,
            project_name=args.wandb_project
        )
        
        # 準備數據和分詞器
        training_data = trainer.prepare_data_and_tokenizer(
            args.training_data_file,
            args.vocab_size,
            args.force_retrain_tokenizer
        )
        
        # 創建模型配置
        model_config = NSConfig(
            vocab_size=trainer.tokenizer_manager.transformers_tokenizer.vocab_size,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            max_position_embeddings=args.max_seq_length,
            memory_tokens=args.memory_tokens,
            pad_token_id=trainer.tokenizer_manager.transformers_tokenizer.pad_token_id,
            bos_token_id=trainer.tokenizer_manager.transformers_tokenizer.bos_token_id,
            eos_token_id=trainer.tokenizer_manager.transformers_tokenizer.eos_token_id,
            tie_word_embeddings=True,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            hidden_dropout=0.05,
            attention_dropout=0.05
        )
        
        trainer.model_config = model_config
        trainer.initialize_model()
        
        # 創建數據集
        train_dataset, eval_dataset = trainer.create_datasets(
            training_data,
            args.eval_split_ratio
        )
        
        # 開始訓練
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
            save_interval_steps=args.save_interval_steps,
            save_interval_epochs=args.save_interval_epochs,
            eval_interval_steps=args.eval_interval_steps,
            early_stopping_patience=args.early_stopping_patience,
            resume_from_checkpoint=not args.no_resume
        )
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"程序執行失敗: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # 系統檢查
    print("***** 系統環境檢查 *****")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("CUDA: 不可用")
    
    try:
        from torch.cuda.amp import GradScaler
        print("混合精度: 可用")
    except ImportError:
        print("混合精度: 不可用")
    
    print("************************")
    
    # Windows兼容性
    if sys.platform == "win32":
        multiprocessing.freeze_support()
    
    main()