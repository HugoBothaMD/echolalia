"""
Training infrastructure for SSL pretraining and downstream fine-tuning.

Provides Trainer classes with logging, checkpointing, and evaluation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import numpy as np


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Basic training
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Logging
    log_every_n_steps: int = 100
    eval_every_n_epochs: int = 1
    
    # Checkpointing
    save_every_n_epochs: int = 10
    save_best: bool = True
    checkpoint_dir: str = "checkpoints"
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"  # "min" or "max"
    
    # Mixed precision
    use_amp: bool = False
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MetricTracker:
    """Tracks and aggregates metrics during training."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics: Dict[str, float], count: int = 1):
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value * count
            self.counts[key] += count
    
    def compute(self) -> Dict[str, float]:
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics
            if self.counts[key] > 0
        }


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(
        self,
        patience: int,
        metric: str = "val_loss",
        mode: str = "min",
    ):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, metrics: Dict[str, float]) -> bool:
        if self.metric not in metrics:
            return False
        
        score = metrics[self.metric]
        
        if self.best_score is None:
            self.best_score = score
            return False
        
        improved = (
            (self.mode == "min" and score < self.best_score) or
            (self.mode == "max" and score > self.best_score)
        )
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class SSLTrainer:
    """
    Trainer for SSL pretraining.
    
    Handles training loop, logging, checkpointing, and evaluation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional[_LRScheduler] = None,
        config: Optional[TrainingConfig] = None,
        logger: Optional[Any] = None,
    ):
        """
        Args:
            model: SSL model to train
            optimizer: Optimizer
            train_loader: Training data loader
            val_loader: Optional validation loader
            scheduler: Optional learning rate scheduler
            config: Training configuration
            logger: Optional logger (e.g., TensorBoard writer)
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = config or TrainingConfig()
        self.logger = logger
        
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler() if self.config.use_amp else None
        
        # Tracking
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = None
        
        # Early stopping
        self.early_stopping = None
        if self.config.early_stopping_patience:
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                metric=self.config.early_stopping_metric,
                mode=self.config.early_stopping_mode,
            )
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metric_tracker = MetricTracker()
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            leave=False,
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            waveforms = batch["waveforms"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            gammas = [g.to(self.device) for g in batch["gammas"]]

            # Optional GOP targets
            gop_targets = None
            if "gop_targets" in batch:
                gop_targets = [
                    g.to(self.device) if g is not None else None
                    for g in batch["gop_targets"]
                ]

            # Forward pass
            with torch.amp.autocast(device_type='cuda', enabled=self.config.use_amp):
                losses = self.model(
                    waveforms, gammas, lengths, gop_targets=gop_targets,
                )
                loss = losses["total_loss"] / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update metrics
            metric_tracker.update(
                {k: v.item() for k, v in losses.items()},
                count=waveforms.shape[0],
            )
            
            # Update progress bar
            progress_bar.set_postfix(loss=losses["total_loss"].item())
            
            # Logging
            if self.global_step % self.config.log_every_n_steps == 0:
                if self.logger:
                    for key, value in losses.items():
                        self.logger.add_scalar(
                            f"train/{key}",
                            value.item(),
                            self.global_step,
                        )
        
        return metric_tracker.compute()
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        metric_tracker = MetricTracker()
        
        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            waveforms = batch["waveforms"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            gammas = [g.to(self.device) for g in batch["gammas"]]

            gop_targets = None
            if "gop_targets" in batch:
                gop_targets = [
                    g.to(self.device) if g is not None else None
                    for g in batch["gop_targets"]
                ]

            with torch.amp.autocast(device_type='cuda', enabled=self.config.use_amp):
                losses = self.model(
                    waveforms, gammas, lengths, gop_targets=gop_targets,
                )
            
            metric_tracker.update(
                {k: v.item() for k, v in losses.items()},
                count=waveforms.shape[0],
            )
        
        metrics = metric_tracker.compute()
        
        # Add val_ prefix
        return {f"val_{k}": v for k, v in metrics.items()}
    
    def save_checkpoint(self, filename: str, metrics: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "config": self.config.__dict__,
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if metrics:
            checkpoint["metrics"] = metrics
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, path: Union[str, Path]):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    def train(self) -> Dict[str, List[float]]:
        """
        Run full training loop.
        
        Returns:
            Dictionary of metric histories
        """
        history = {"train": [], "val": []}
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            history["train"].append(train_metrics)
            
            # Evaluate
            val_metrics = {}
            if (epoch + 1) % self.config.eval_every_n_epochs == 0:
                val_metrics = self.evaluate()
                history["val"].append(val_metrics)
                
                # Log
                if self.logger:
                    for key, value in val_metrics.items():
                        self.logger.add_scalar(key, value, epoch)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Print progress
            all_metrics = {**train_metrics, **val_metrics}
            metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in all_metrics.items())
            print(f"Epoch {epoch}: {metric_str}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt", all_metrics)
            
            # Save best model
            if self.config.save_best and val_metrics:
                metric_key = self.config.early_stopping_metric
                if metric_key in val_metrics:
                    current = val_metrics[metric_key]
                    is_best = (
                        self.best_metric is None or
                        (self.config.early_stopping_mode == "min" and current < self.best_metric) or
                        (self.config.early_stopping_mode == "max" and current > self.best_metric)
                    )
                    if is_best:
                        self.best_metric = current
                        self.save_checkpoint("best_model.pt", all_metrics)
            
            # Early stopping
            if self.early_stopping and self.early_stopping(all_metrics):
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Save final model
        self.save_checkpoint("final_model.pt", all_metrics)
        
        return history


class DownstreamTrainer:
    """
    Trainer for downstream fine-tuning.
    
    Similar to SSLTrainer but handles classification/regression tasks.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        scheduler: Optional[_LRScheduler] = None,
        config: Optional[TrainingConfig] = None,
        logger: Optional[Any] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        """
        Args:
            model: Downstream model
            optimizer: Optimizer
            train_loader: Training loader
            val_loader: Validation loader
            test_loader: Test loader (for final evaluation)
            scheduler: LR scheduler
            config: Training config
            logger: Logger
            compute_metrics: Optional function to compute additional metrics
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.config = config or TrainingConfig()
        self.logger = logger
        self.compute_metrics = compute_metrics
        
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        
        self.scaler = torch.amp.GradScaler() if self.config.use_amp else None
        
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = None
        
        self.early_stopping = None
        if self.config.early_stopping_patience:
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                metric=self.config.early_stopping_metric,
                mode=self.config.early_stopping_mode,
            )
        
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metric_tracker = MetricTracker()
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            leave=False,
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            waveforms = batch["waveforms"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            
            # Handle different label formats
            if "clinical_labels" in batch:
                labels = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch["clinical_labels"].items()
                }
            elif "labels" in batch:
                labels = batch["labels"].to(self.device)
            else:
                raise ValueError("No labels found in batch")
            
            with torch.amp.autocast(device_type='cuda', enabled=self.config.use_amp):
                if isinstance(labels, dict):
                    losses = self.model.compute_loss(waveforms, labels, lengths)
                else:
                    losses = self.model.compute_loss(waveforms, labels, lengths)
                
                loss = losses["loss"] / self.config.gradient_accumulation_steps
            
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            metric_tracker.update(
                {k: v.item() if isinstance(v, torch.Tensor) else v 
                 for k, v in losses.items()},
                count=waveforms.shape[0],
            )
            
            progress_bar.set_postfix(loss=losses["loss"].item())
        
        return metric_tracker.compute()
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, prefix: str = "val") -> Dict[str, float]:
        """Evaluate on a data loader."""
        self.model.eval()
        metric_tracker = MetricTracker()
        
        all_predictions = []
        all_labels = []
        
        for batch in tqdm(loader, desc=f"Evaluating ({prefix})", leave=False):
            waveforms = batch["waveforms"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            
            if "clinical_labels" in batch:
                labels = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch["clinical_labels"].items()
                }
            elif "labels" in batch:
                labels = batch["labels"].to(self.device)
            
            with torch.amp.autocast(device_type='cuda', enabled=self.config.use_amp):
                if isinstance(labels, dict):
                    predictions = self.model(waveforms, lengths)
                    losses = self.model.head.compute_loss(predictions, labels)
                else:
                    predictions = self.model(waveforms, lengths)
                    losses = self.model.classifier.compute_loss(predictions, labels)
            
            metric_tracker.update(
                {k: v.item() if isinstance(v, torch.Tensor) else v 
                 for k, v in losses.items()},
                count=waveforms.shape[0],
            )
            
            # Store for additional metrics
            if isinstance(predictions, dict):
                all_predictions.append({k: v.cpu() for k, v in predictions.items()})
                all_labels.append({k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                  for k, v in labels.items()})
            else:
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu() if isinstance(labels, torch.Tensor) else labels)
        
        metrics = metric_tracker.compute()
        
        # Compute additional metrics if provided
        if self.compute_metrics:
            extra_metrics = self.compute_metrics(all_predictions, all_labels)
            metrics.update(extra_metrics)
        
        return {f"{prefix}_{k}": v for k, v in metrics.items()}
    
    def save_checkpoint(self, filename: str, metrics: Optional[Dict] = None):
        """Save checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if metrics:
            checkpoint["metrics"] = metrics
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, path: Union[str, Path]):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    def train(self) -> Dict[str, Any]:
        """Run training loop."""
        history = {"train": [], "val": [], "test": None}
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            history["train"].append(train_metrics)
            
            val_metrics = {}
            if self.val_loader and (epoch + 1) % self.config.eval_every_n_epochs == 0:
                val_metrics = self.evaluate(self.val_loader, "val")
                history["val"].append(val_metrics)
            
            if self.scheduler:
                self.scheduler.step()
            
            all_metrics = {**train_metrics, **val_metrics}
            metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in all_metrics.items())
            print(f"Epoch {epoch}: {metric_str}")
            
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt", all_metrics)
            
            if self.config.save_best and val_metrics:
                metric_key = self.config.early_stopping_metric
                if metric_key in val_metrics:
                    current = val_metrics[metric_key]
                    is_best = (
                        self.best_metric is None or
                        (self.config.early_stopping_mode == "min" and current < self.best_metric) or
                        (self.config.early_stopping_mode == "max" and current > self.best_metric)
                    )
                    if is_best:
                        self.best_metric = current
                        self.save_checkpoint("best_model.pt", all_metrics)
            
            if self.early_stopping and self.early_stopping(all_metrics):
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Final test evaluation
        if self.test_loader:
            # Load best model for testing
            best_path = self.checkpoint_dir / "best_model.pt"
            if best_path.exists():
                self.load_checkpoint(best_path)
            
            test_metrics = self.evaluate(self.test_loader, "test")
            history["test"] = test_metrics
            print(f"Test metrics: {test_metrics}")
        
        self.save_checkpoint("final_model.pt")
        
        return history
