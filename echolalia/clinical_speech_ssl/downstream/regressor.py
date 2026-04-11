"""
Downstream regressor for fine-tuning on continuous clinical scores.

Supports single-output and multi-output regression.
"""

from typing import Dict, List, Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DownstreamRegressor(nn.Module):
    """
    Regression head for downstream fine-tuning on continuous targets.
    
    Can predict single or multiple continuous values (e.g., severity scores,
    intelligibility ratings, acoustic measurements).
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_outputs: int = 1,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        pooling: Literal["mean", "attention", "first", "last"] = "mean",
        output_activation: Optional[str] = None,
        output_range: Optional[tuple] = None,
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            num_outputs: Number of output values to predict
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            pooling: How to pool frame representations
            output_activation: Optional activation ("sigmoid", "tanh", None)
            output_range: Optional (min, max) to scale outputs
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_outputs = num_outputs
        self.pooling = pooling
        self.output_activation = output_activation
        self.output_range = output_range
        
        # Pooling layer
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.Tanh(),
                nn.Linear(embed_dim // 4, 1),
            )
        
        # Regression head
        if hidden_dims is None or len(hidden_dims) == 0:
            self.regressor = nn.Linear(embed_dim, num_outputs)
        else:
            layers = []
            current_dim = embed_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                current_dim = hidden_dim
            
            layers.append(nn.Linear(current_dim, num_outputs))
            self.regressor = nn.Sequential(*layers)
    
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
        
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        return pooled
    
    def forward(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict continuous values.
        
        Args:
            features: Encoder output [B, T, D]
            lengths: Sequence lengths [B]
            
        Returns:
            predictions: Regression predictions [B, num_outputs]
        """
        pooled = self.pool(features, lengths)
        predictions = self.regressor(pooled)
        
        # Apply output activation if specified
        if self.output_activation == "sigmoid":
            predictions = torch.sigmoid(predictions)
        elif self.output_activation == "tanh":
            predictions = torch.tanh(predictions)
        
        # Scale to output range if specified
        if self.output_range is not None:
            min_val, max_val = self.output_range
            predictions = predictions * (max_val - min_val) + min_val
        
        return predictions
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_type: Literal["mse", "mae", "huber"] = "mse",
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute regression loss.
        
        Args:
            predictions: Model predictions [B, num_outputs]
            targets: Ground truth values [B, num_outputs] or [B]
            loss_type: Type of loss function
            sample_weights: Optional per-sample weights [B]
            
        Returns:
            Dictionary with 'loss' and metrics
        """
        # Ensure consistent shapes
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(-1)
        
        # Compute loss
        if loss_type == "mse":
            loss = F.mse_loss(predictions, targets, reduction='none')
        elif loss_type == "mae":
            loss = F.l1_loss(predictions, targets, reduction='none')
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(predictions, targets, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Apply sample weights if provided
        if sample_weights is not None:
            loss = loss * sample_weights.unsqueeze(-1)
        
        loss = loss.mean()
        
        # Compute metrics
        with torch.no_grad():
            mae = F.l1_loss(predictions, targets)
            rmse = torch.sqrt(F.mse_loss(predictions, targets))
            
            # Pearson correlation (per output, then averaged)
            correlations = []
            for i in range(predictions.shape[-1]):
                pred_i = predictions[:, i]
                targ_i = targets[:, i]
                
                pred_centered = pred_i - pred_i.mean()
                targ_centered = targ_i - targ_i.mean()
                
                correlation = (pred_centered * targ_centered).sum() / (
                    pred_centered.norm() * targ_centered.norm() + 1e-8
                )
                correlations.append(correlation)
            
            avg_correlation = torch.stack(correlations).mean()
        
        return {
            'loss': loss,
            'mae': mae,
            'rmse': rmse,
            'correlation': avg_correlation,
        }


class DownstreamRegressorModel(nn.Module):
    """
    Complete downstream regression model.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        regressor: DownstreamRegressor,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.regressor = regressor
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        
        predictions = self.regressor(features, feature_lengths)
        return predictions
    
    def compute_loss(
        self,
        waveform: torch.Tensor,
        targets: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss."""
        predictions = self.forward(waveform, lengths)
        return self.regressor.compute_loss(predictions, targets, **kwargs)
