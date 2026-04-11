"""
Multi-task head for combined classification and regression.

Supports training on multiple clinical endpoints simultaneously,
which is common in clinical speech analysis (e.g., predicting
both diagnosis and severity).
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TaskConfig:
    """Configuration for a single task."""
    name: str
    task_type: Literal["binary", "multiclass", "multilabel", "regression"]
    num_outputs: int  # num_classes for classification, num_values for regression
    loss_weight: float = 1.0
    output_activation: Optional[str] = None  # For regression
    output_range: Optional[tuple] = None  # For regression


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head.
    
    Supports multiple classification and regression tasks with
    shared or task-specific hidden layers.
    
    Example usage:
        tasks = [
            TaskConfig("diagnosis", "multiclass", 4),
            TaskConfig("severity", "regression", 1),
            TaskConfig("intelligibility", "regression", 1),
        ]
        head = MultiTaskHead(embed_dim=256, tasks=tasks)
    """
    
    def __init__(
        self,
        embed_dim: int,
        tasks: List[TaskConfig],
        shared_hidden_dims: Optional[List[int]] = None,
        task_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        pooling: Literal["mean", "attention", "first", "last"] = "mean",
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            tasks: List of task configurations
            shared_hidden_dims: Shared hidden layers before task-specific heads
            task_hidden_dims: Task-specific hidden layers
            dropout: Dropout probability
            pooling: How to pool frame representations
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.tasks = {t.name: t for t in tasks}
        self.task_names = [t.name for t in tasks]
        self.pooling = pooling
        
        # Pooling layer
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.Tanh(),
                nn.Linear(embed_dim // 4, 1),
            )
        
        # Shared layers
        current_dim = embed_dim
        if shared_hidden_dims:
            layers = []
            for hidden_dim in shared_hidden_dims:
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                current_dim = hidden_dim
            self.shared_layers = nn.Sequential(*layers)
        else:
            self.shared_layers = nn.Identity()
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        
        for task in tasks:
            head_input_dim = current_dim
            
            if task_hidden_dims:
                layers = []
                dim = head_input_dim
                for hidden_dim in task_hidden_dims:
                    layers.extend([
                        nn.Linear(dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ])
                    dim = hidden_dim
                layers.append(nn.Linear(dim, task.num_outputs))
                self.task_heads[task.name] = nn.Sequential(*layers)
            else:
                self.task_heads[task.name] = nn.Linear(head_input_dim, task.num_outputs)
    
    def pool(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool sequence features."""
        if self.pooling == "mean":
            if lengths is not None:
                mask = torch.arange(features.shape[1], device=features.device)
                mask = mask.unsqueeze(0) < lengths.unsqueeze(1)
                features = features * mask.unsqueeze(-1)
                pooled = features.sum(dim=1) / lengths.unsqueeze(-1).float().clamp(min=1)
            else:
                pooled = features.mean(dim=1)
        
        elif self.pooling == "attention":
            attn_weights = self.attention(features)
            
            if lengths is not None:
                mask = torch.arange(features.shape[1], device=features.device)
                mask = mask.unsqueeze(0) < lengths.unsqueeze(1)
                attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            
            attn_weights = F.softmax(attn_weights, dim=1)
            pooled = (features * attn_weights).sum(dim=1)
        
        elif self.pooling == "first":
            pooled = features[:, 0, :]
        
        elif self.pooling == "last":
            if lengths is not None:
                batch_indices = torch.arange(features.shape[0], device=features.device)
                pooled = features[batch_indices, lengths - 1]
            else:
                pooled = features[:, -1, :]
        
        return pooled
    
    def forward(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        task_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict outputs for all or specified tasks.
        
        Args:
            features: Encoder output [B, T, D]
            lengths: Sequence lengths [B]
            task_names: Optional subset of tasks to compute
            
        Returns:
            Dictionary mapping task name to predictions
        """
        # Pool features
        pooled = self.pool(features, lengths)
        
        # Shared transformation
        shared = self.shared_layers(pooled)
        
        # Task-specific predictions
        task_names = task_names or self.task_names
        outputs = {}
        
        for name in task_names:
            task = self.tasks[name]
            logits = self.task_heads[name](shared)
            
            # Apply output activation for regression tasks
            if task.task_type == "regression":
                if task.output_activation == "sigmoid":
                    logits = torch.sigmoid(logits)
                elif task.output_activation == "tanh":
                    logits = torch.tanh(logits)
                
                if task.output_range is not None:
                    min_val, max_val = task.output_range
                    logits = logits * (max_val - min_val) + min_val
            
            outputs[name] = logits
        
        return outputs
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Task predictions from forward()
            targets: Ground truth for each task
            class_weights: Optional class weights per task
            
        Returns:
            Dictionary with total loss, per-task losses, and metrics
        """
        losses = {}
        metrics = {}
        total_loss = 0.0
        
        for name, pred in predictions.items():
            task = self.tasks[name]
            target = targets[name]
            weight = class_weights.get(name) if class_weights else None
            
            if task.task_type == "binary":
                if pred.shape[-1] == 1:
                    pred = pred.squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(
                    pred, target.float(), pos_weight=weight
                )
                
                with torch.no_grad():
                    preds = (torch.sigmoid(pred) > 0.5).long()
                    acc = (preds == target).float().mean()
                    metrics[f"{name}_accuracy"] = acc
            
            elif task.task_type == "multiclass":
                loss = F.cross_entropy(pred, target, weight=weight)
                
                with torch.no_grad():
                    preds = pred.argmax(dim=-1)
                    acc = (preds == target).float().mean()
                    metrics[f"{name}_accuracy"] = acc
            
            elif task.task_type == "multilabel":
                loss = F.binary_cross_entropy_with_logits(
                    pred, target.float(), pos_weight=weight
                )
                
                with torch.no_grad():
                    preds = (torch.sigmoid(pred) > 0.5).long()
                    acc = (preds == target).float().mean()
                    metrics[f"{name}_accuracy"] = acc
            
            elif task.task_type == "regression":
                if target.dim() == 1:
                    target = target.unsqueeze(-1)
                loss = F.mse_loss(pred, target)
                
                with torch.no_grad():
                    mae = F.l1_loss(pred, target)
                    metrics[f"{name}_mae"] = mae
            
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            losses[f"{name}_loss"] = loss
            total_loss = total_loss + task.loss_weight * loss
        
        losses["total_loss"] = total_loss
        losses.update(metrics)
        
        return losses


class MultiTaskModel(nn.Module):
    """Complete multi-task model with encoder."""
    
    def __init__(
        self,
        encoder: nn.Module,
        multi_task_head: MultiTaskHead,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.head = multi_task_head
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        task_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        if hasattr(self.encoder, 'encode'):
            features = self.encoder.encode(waveform, lengths)
        else:
            features = self.encoder(waveform, lengths)
        
        feature_lengths = None
        if lengths is not None and hasattr(self.encoder, 'frontend'):
            if hasattr(self.encoder.frontend, 'get_output_length'):
                feature_lengths = torch.tensor([
                    self.encoder.frontend.get_output_length(l.item())
                    for l in lengths
                ], device=lengths.device)
        
        return self.head(features, feature_lengths, task_names)
    
    def compute_loss(
        self,
        waveform: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss."""
        predictions = self.forward(waveform, lengths)
        return self.head.compute_loss(predictions, targets, **kwargs)
