import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from functools import partial
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from datasets import Dataset
import numpy as np
import wandb
from tqdm import tqdm

from .nagato_sakura_model import NagatoSakuraForCausalLM, NSConfig
from .logger import setup_logging, SystemMonitor
from .tokenizer import TokenizerManager
from .data_utils import smart_collate_fn, EarlyStoppingCallback

# 混合精度處理
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    class GradScaler:
        def __init__(self, *args, **kwargs): pass
        def scale(self, loss): return loss
        def step(self, optimizer): optimizer.step()
        def update(self): pass
        def unscale_(self, optimizer): pass
        def is_enabled(self): return False
    def autocast(enabled=False):
        import contextlib
        return contextlib.nullcontext()


class AdvancedNagatoSakuraTrainer:
    """高級長門櫻訓練器，參考ChatGLM訓練標準"""
    
    def __init__(self, model_config: Optional[NSConfig], output_dir: str = "nagato_sakura_output",
                 device: Optional[str] = None, use_wandb: bool = False, project_name: str = "nagato-sakura"):
        
        # 基礎設置
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # 初始化日誌
        self.logger = setup_logging(str(self.output_dir))
        self.logger.info(f"使用設備: {self.device}")
        
        # 系統監控
        self.system_monitor = SystemMonitor()
        self.system_monitor.log_system_status(self.logger)
        
        # 模型相關
        self.model_config = model_config
        self.model: Optional[NagatoSakuraForCausalLM] = None
        self.tokenizer_manager = TokenizerManager(self.output_dir / "tokenizer.json")
        
        # 訓練狀態
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        self.training_history = defaultdict(list)
        
        # WandB集成
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                wandb.init(project=project_name, dir=str(self.output_dir))
                self.logger.info("WandB 初始化成功")
            except Exception as e:
                self.logger.warning(f"WandB 初始化失敗: {e}")
                self.use_wandb = False
        
        # 其他組件
        self.early_stopping = None
        
        # 嘗試加載現有分詞器
        try:
            self.tokenizer_manager.load_tokenizer()
        except Exception as e:
            self.logger.info(f"未找到現有分詞器或加載失敗: {e}")

    def initialize_model(self):
        """初始化模型"""
        if self.model_config is None:
            raise ValueError("模型配置未設置")
        if self.tokenizer_manager.transformers_tokenizer is None:
            raise ValueError("分詞器未初始化")
        
        # 更新配置
        self.model_config.vocab_size = self.tokenizer_manager.transformers_tokenizer.vocab_size
        self.model_config.pad_token_id = self.tokenizer_manager.transformers_tokenizer.pad_token_id
        self.model_config.bos_token_id = self.tokenizer_manager.transformers_tokenizer.bos_token_id
        self.model_config.eos_token_id = self.tokenizer_manager.transformers_tokenizer.eos_token_id
        
        # 創建模型
        try:
            self.model = NagatoSakuraForCausalLM(self.model_config).to(self.device)
            
            # 計算參數數量
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            self.logger.info(f"模型初始化完成:")
            self.logger.info(f"  總參數量: {total_params/1e6:.2f}M")
            self.logger.info(f"  可訓練參數: {trainable_params/1e6:.2f}M")
            self.logger.info(f"  模型大小: {total_params * 4 / 1024**3:.2f}GB (FP32)")
            
            # 綁定權重
            self.model.tie_weights()
            
            if self.use_wandb:
                wandb.config.update({
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "model_size_gb": total_params * 4 / 1024**3
                })
                
        except Exception as e:
            self.logger.error(f"模型初始化失敗: {e}")
            raise

    def prepare_data_and_tokenizer(self, training_data_file: str, target_vocab_size: int,
                                 force_retrain_tokenizer: bool = False) -> List[Dict[str, str]]:
        """準備數據和分詞器"""
        
        # 加載數據
        self.logger.info(f"從 {training_data_file} 加載訓練數據...")
        try:
            with open(training_data_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            if not isinstance(loaded_data, list):
                raise ValueError("數據格式錯誤")
            
            # 驗證數據格式
            valid_data = []
            for i, item in enumerate(loaded_data):
                if isinstance(item, dict) and "prompt" in item and "completion" in item:
                    if item["prompt"] and item["completion"]:
                        valid_data.append(item)
                    else:
                        self.logger.warning(f"跳過空數據項 {i}")
                else:
                    self.logger.warning(f"跳過格式錯誤的數據項 {i}")
            
            self.logger.info(f"成功加載 {len(valid_data)} 條有效數據")
            
        except Exception as e:
            self.logger.error(f"數據加載失敗: {e}")
            raise
        
        # 處理分詞器
        self.tokenizer_manager.prepare_tokenizer(
            valid_data, target_vocab_size, force_retrain_tokenizer
        )
        
        return valid_data

    def create_datasets(self, data_list: List[Dict[str, str]], 
                       eval_split_ratio: float = 0.01) -> Tuple[Dataset, Optional[Dataset]]:
        """創建訓練和評估數據集"""
        
        self.logger.info(f"創建數據集，總數據量: {len(data_list)}")
        
        if not data_list:
            raise ValueError("數據列表為空")
        
        # 創建完整數據集
        dataset = Dataset.from_list(data_list)
        
        # 分割數據集
        if eval_split_ratio > 0 and eval_split_ratio < 1.0 and len(dataset) > 1:
            split_datasets = dataset.train_test_split(
                test_size=eval_split_ratio, 
                shuffle=True, 
                seed=42
            )
            train_dataset = split_datasets["train"]
            eval_dataset = split_datasets["test"]
            
            self.logger.info(f"數據集分割完成 - 訓練集: {len(train_dataset)}, 評估集: {len(eval_dataset)}")
            return train_dataset, eval_dataset
        else:
            self.logger.info(f"使用完整數據集進行訓練: {len(dataset)}")
            return dataset, None

    def setup_training_components(self, train_dataset: Dataset, batch_size: int,
                                num_epochs: int, learning_rate: float,
                                weight_decay: float, gradient_accumulation_steps: int,
                                warmup_ratio: float, lr_scheduler_type: str):
        """設置訓練組件"""
        
        # 數據加載器
        collate_fn = partial(
            smart_collate_fn,
            tokenizer=self.tokenizer_manager.transformers_tokenizer,
            max_seq_length=self.model_config.max_position_embeddings,
            pack_sequences=True
        )
        
        num_workers = min(4, os.cpu_count() // 2) if os.cpu_count() else 0
        if sys.platform == "win32":
            num_workers = 0  # Windows兼容性
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda',
            drop_last=True  # 確保批次大小一致
        )
        
        # 優化器（使用更先進的配置）
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(0.9, 0.95),  # ChatGLM風格的betas
            eps=1e-8,
            weight_decay=weight_decay
        )
        
        # 學習率調度器
        total_steps = len(train_dataloader) // gradient_accumulation_steps * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        if lr_scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif lr_scheduler_type == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                total_steps=total_steps,
                pct_start=warmup_ratio,
                anneal_strategy='cos'
            )
        else:  # linear
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        
        # 混合精度
        scaler = GradScaler(enabled=AMP_AVAILABLE and self.device.type == 'cuda')
        
        self.logger.info(f"訓練組件設置完成:")
        self.logger.info(f"  批次大小: {batch_size}")
        self.logger.info(f"  梯度累積步數: {gradient_accumulation_steps}")
        self.logger.info(f"  有效批次大小: {batch_size * gradient_accumulation_steps}")
        self.logger.info(f"  總訓練步數: {total_steps}")
        self.logger.info(f"  預熱步數: {warmup_steps}")
        self.logger.info(f"  學習率調度器: {lr_scheduler_type}")
        self.logger.info(f"  混合精度: {'啟用' if scaler.is_enabled() else '禁用'}")
        
        return train_dataloader, optimizer, scheduler, scaler

    def train_epoch(self, train_dataloader: DataLoader, optimizer, scheduler, 
                scaler: GradScaler, gradient_accumulation_steps: int,
                max_grad_norm: float, log_interval: int) -> float:
        """訓練一個epoch"""
        
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        accumulated_steps = 0
        
        progress_bar = tqdm(
            train_dataloader, 
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False
        )
        
        for step, batch in enumerate(progress_bar):
            # 跳過無效批次
            if batch["input_ids"] is None:
                continue
            
            # 移動到設備
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            
            # 前向傳播
            with autocast(enabled=scaler.is_enabled()):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(f"檢測到無效損失，跳過此批次: {loss}")
                accumulated_steps = 0
                optimizer.zero_grad()
                continue
            
            # 反向傳播
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
            
            epoch_loss += loss.item()
            num_batches += 1
            accumulated_steps += 1
            
            # 梯度更新
            if accumulated_steps >= gradient_accumulation_steps:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_grad_norm
                )
                
                # 檢查梯度
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    self.logger.warning(f"檢測到無效梯度範數: {grad_norm}")
                    accumulated_steps = 0
                    optimizer.zero_grad()
                    scaler.update()
                    continue
                
                # 優化器步驟
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                accumulated_steps = 0
                self.global_step += 1
                
                # 更新進度條
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    "Loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                    "LR": f"{current_lr:.2e}",
                    "Step": self.global_step
                })
                
                # 記錄日誌
                if self.global_step % log_interval == 0:
                    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                    self.logger.info(
                        f"Step {self.global_step}: Loss={avg_loss:.4f}, "
                        f"LR={current_lr:.2e}, GradNorm={grad_norm:.4f}"
                    )
                    
                    # WandB記錄
                    if self.use_wandb:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": current_lr,
                            "train/grad_norm": grad_norm,
                            "global_step": self.global_step
                        })
                
                # 系統監控
                if self.global_step % (log_interval * 5) == 0:
                    self.system_monitor.log_system_status(self.logger)
        
        # 處理剩餘的累積梯度
        if accumulated_steps > 0:
            try:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_grad_norm
                )
                
                if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    self.global_step += 1
                else:
                    scaler.update()
                
                optimizer.zero_grad()
            except RuntimeError as e:
                self.logger.warning(f"處理剩餘梯度時出錯: {e}")
                optimizer.zero_grad()
                scaler.update()
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        return avg_epoch_loss

    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """評估模型"""
        if eval_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="評估中", leave=False):
                if batch["input_ids"] is None:
                    continue
                
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                
                with autocast(enabled=AMP_AVAILABLE and self.device.type == 'cuda'):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                
                if loss is not None and not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    valid_tokens = (labels != -100).sum().item()
                    total_tokens += valid_tokens
                    num_batches += 1
        
        if num_batches == 0:
            return {"eval_loss": float('inf'), "perplexity": float('inf')}
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            "eval_loss": avg_loss,
            "perplexity": perplexity,
            "total_tokens": total_tokens
        }

    def save_checkpoint(self, checkpoint_name: str, optimizer, scheduler, 
                       scaler: Optional[GradScaler] = None, is_best: bool = False):
        """保存檢查點"""
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 模型狀態
            torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")
            
            # 配置
            with open(checkpoint_dir / "config.json", 'w', encoding='utf-8') as f:
                json.dump(vars(self.model_config), f, indent=2, ensure_ascii=False)
            
            # 分詞器
            if self.tokenizer_manager.tokenizer_object:
                self.tokenizer_manager.tokenizer_object.save(str(checkpoint_dir / "tokenizer.json"))
            
            # 訓練狀態
            training_state = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'best_eval_loss': self.best_eval_loss,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'training_history': dict(self.training_history)
            }
            
            if scaler and scaler.is_enabled():
                training_state['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(training_state, checkpoint_dir / "training_state.pt")
            
            # 如果是最佳模型，額外保存
            if is_best:
                best_model_dir = self.output_dir / "best_model"
                if best_model_dir.exists():
                    import shutil
                    shutil.rmtree(best_model_dir)
                import shutil
                shutil.copytree(checkpoint_dir, best_model_dir)
            
            self.logger.info(f"檢查點已保存: {checkpoint_dir}")
            
        except Exception as e:
            self.logger.error(f"保存檢查點失敗: {e}")

    def load_checkpoint(self, checkpoint_dir: str, optimizer, scheduler, 
                       scaler: Optional[GradScaler] = None) -> bool:
        """加載檢查點"""
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            self.logger.warning(f"檢查點目錄不存在: {checkpoint_dir}")
            return False
        
        try:
            # 加載訓練狀態
            state_path = checkpoint_path / "training_state.pt"
            if state_path.exists():
                state = torch.load(state_path, map_location=self.device)
                self.current_epoch = state.get('epoch', 0)
                self.global_step = state.get('global_step', 0)
                self.best_eval_loss = state.get('best_eval_loss', float('inf'))
                self.training_history = defaultdict(list, state.get('training_history', {}))
                
                optimizer.load_state_dict(state['optimizer_state_dict'])
                scheduler.load_state_dict(state['scheduler_state_dict'])
                
                if scaler and 'scaler_state_dict' in state:
                    scaler.load_state_dict(state['scaler_state_dict'])
            
            # 加載模型
            model_path = checkpoint_path / "model.pt"
            if model_path.exists():
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            self.logger.info(f"檢查點加載成功: {checkpoint_dir}")
            self.logger.info(f"恢復至 Epoch {self.current_epoch}, Step {self.global_step}")
            return True
            
        except Exception as e:
            self.logger.error(f"加載檢查點失敗: {e}")
            return False

    def find_latest_checkpoint(self) -> Optional[str]:
        """查找最新檢查點"""
        checkpoint_pattern = "checkpoint-*"
        checkpoints = list(self.output_dir.glob(checkpoint_pattern))
        
        if not checkpoints:
            return None
        
        # 按修改時間排序
        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        return str(latest_checkpoint)

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None,
              batch_size: int = 4, num_epochs: int = 3, learning_rate: float = 5e-5,
              gradient_accumulation_steps: int = 1, weight_decay: float = 0.01,
              lr_scheduler_type: str = "cosine", warmup_ratio: float = 0.03,
              max_grad_norm: float = 1.0, log_interval: int = 10,
              save_interval_steps: int = 500, eval_interval_steps: int = 500,
              save_interval_epochs: int = 2,
              early_stopping_patience: int = 5, resume_from_checkpoint: bool = True):
        """主訓練函數"""
        
        self.logger.info("***** 開始訓練 *****")
        
        # 設置早停
        if early_stopping_patience > 0:
            self.early_stopping = EarlyStoppingCallback(
                patience=early_stopping_patience,
                min_delta=0.001,
                mode='min'
            )
        
        # 設置訓練組件
        train_dataloader, optimizer, scheduler, scaler = self.setup_training_components(
            train_dataset, batch_size, num_epochs, learning_rate,
            weight_decay, gradient_accumulation_steps, warmup_ratio, lr_scheduler_type
        )
        
        # 設置評估數據加載器
        eval_dataloader = None
        if eval_dataset:
            collate_fn = partial(
                smart_collate_fn,
                tokenizer=self.tokenizer_manager.transformers_tokenizer,
                max_seq_length=self.model_config.max_position_embeddings,
                pack_sequences=False
            )
            
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=batch_size * 2,
                collate_fn=collate_fn,
                shuffle=False,
                num_workers=0,
                pin_memory=self.device.type == 'cuda'
            )
        
        # 恢復檢查點
        if resume_from_checkpoint:
            latest_checkpoint = self.find_latest_checkpoint()
            if latest_checkpoint:
                if self.load_checkpoint(latest_checkpoint, optimizer, scheduler, scaler):
                    self.logger.info(f"從檢查點恢復訓練: {latest_checkpoint}")
        
        # 訓練循環
        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                self.logger.info(f"=== Epoch {epoch + 1}/{num_epochs} ===")
                
                # 訓練一個epoch
                epoch_loss = self.train_epoch(
                    train_dataloader, optimizer, scheduler, scaler,
                    gradient_accumulation_steps, max_grad_norm, log_interval
                )
                
                # 記錄epoch結果
                self.training_history['train_loss'].append(epoch_loss)
                self.logger.info(f"Epoch {epoch + 1} 完成 - 平均損失: {epoch_loss:.4f}")
                
                # 評估
                eval_loss = None
                should_save_checkpoint = False
                is_best = False
                
                if eval_dataloader and (epoch + 1) % 1 == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    eval_loss = eval_metrics.get('eval_loss', float('inf'))
                    perplexity = eval_metrics.get('perplexity', float('inf'))
                    
                    self.training_history['eval_loss'].append(eval_loss)
                    self.training_history['perplexity'].append(perplexity)
                    
                    self.logger.info(f"評估結果 - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")
                    
                    # WandB記錄
                    if self.use_wandb:
                        wandb.log({
                            "epoch": epoch + 1,
                            "train/epoch_loss": epoch_loss,
                            "eval/loss": eval_loss,
                            "eval/perplexity": perplexity
                        })
                    
                    # 檢查是否是最佳模型
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        is_best = True
                        should_save_checkpoint = True
                        self.logger.info(f"🎉 新的最佳模型! 評估損失: {eval_loss:.4f}")
                    
                    # 早停檢查
                    if self.early_stopping and self.early_stopping(eval_loss):
                        self.logger.info(f"早停觸發! 最佳評估損失: {self.best_eval_loss:.4f}")
                        self.save_checkpoint(
                            f"checkpoint-early-stop-epoch-{epoch + 1}",
                            optimizer, scheduler, scaler, False
                        )
                        break
                
                # 檢查是否需要保存checkpoint
                if (epoch + 1) % save_interval_epochs == 0:
                    should_save_checkpoint = True
                
                if epoch == num_epochs - 1:
                    should_save_checkpoint = True
                
                # 保存檢查點
                if should_save_checkpoint:
                    self.save_checkpoint(
                        f"checkpoint-epoch-{epoch + 1}",
                        optimizer, scheduler, scaler, is_best
                    )
            
            # 訓練完成
            self.logger.info("***** 訓練完成 *****")
            self.save_checkpoint("final_model", optimizer, scheduler, scaler)
            
            # 最終評估
            if eval_dataloader:
                final_metrics = self.evaluate(eval_dataloader)
                self.logger.info(f"最終評估結果: {final_metrics}")
            
        except KeyboardInterrupt:
            self.logger.info("訓練被用戶中斷")
            self.save_checkpoint(
                f"checkpoint-interrupted-step-{self.global_step}",
                optimizer, scheduler, scaler
            )
        except Exception as e:
            self.logger.error(f"訓練過程中發生錯誤: {e}", exc_info=True)
            raise
        finally:
            if self.use_wandb:
                wandb.finish()