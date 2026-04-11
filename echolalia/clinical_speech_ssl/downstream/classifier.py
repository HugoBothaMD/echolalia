"""
Downstream classifier for fine-tuning on clinical labels.

Supports binary classification, multi-class classification,
and multi-label classification.
"""

from typing import Dict, List, Literal, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DownstreamClassifier(nn.Module):
    """
    Classification head for downstream fine-tuning.
    
    Can be used with frozen or fine-tuned encoder.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        pooling: Literal["mean", "attention", "first", "last"] = "mean",
        task_type: Literal["binary", "multiclass", "multilabel"] = "multiclass",
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            num_classes: Number of output classes
            hidden_dims: Hidden layer dimensions (if None, linear classifier)
            dropout: Dropout probability
            pooling: How to pool frame representations
            task_type: Type of classification task
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.pooling = pooling
        self.task_type = task_type
        
        # Pooling layer
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.Tanh(),
                nn.Linear(embed_dim // 4, 1),
            )
        
        # Classification head
        if hidden_dims is None or len(hidden_dims) == 0:
            self.classifier = nn.Linear(embed_dim, num_classes)
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
            
            layers.append(nn.Linear(current_dim, num_classes))
            self.classifier = nn.Sequential(*layers)
    
    def pool(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool sequence features.
        
        Args:
            features: [B, T, D]
            lengths: [B]
            
        Returns:
            pooled: [B, D]
        """
        if self.pooling == "mean":
            if lengths is not None:
                mask = torch.arange(features.shape[1], device=features.device)
                mask = mask.unsqueeze(0) < lengths.unsqueeze(1)
                features = features * mask.unsqueeze(-1)
                pooled = features.sum(dim=1) / lengths.unsqueeze(-1).float().clamp(min=1)
            else:
                pooled = features.mean(dim=1)
        
        elif self.pooling == "attention":
            attn_weights = self.attention(features)  # [B, T, 1]
            
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
        Classify input features.
        
        Args:
            features: Encoder output [B, T, D]
            lengths: Sequence lengths [B]
            
        Returns:
            logits: Classification logits [B, num_classes]
        """
        pooled = self.pool(features, lengths)
        logits = self.classifier(pooled)
        return logits
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute classification loss.
        
        Args:
            logits: Model logits [B, num_classes]
            labels: Ground truth labels [B] or [B, num_classes] for multilabel
            class_weights: Optional class weights for imbalanced data
            
        Returns:
            Dictionary with 'loss' and metrics
        """
        if self.task_type == "binary":
            if logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels.float(),
                pos_weight=class_weights,
            )
            
            # Compute accuracy
            with torch.no_grad():
                predictions = (torch.sigmoid(logits) > 0.5).long()
                accuracy = (predictions == labels).float().mean()
        
        elif self.task_type == "multiclass":
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                accuracy = (predictions == labels).float().mean()
        
        elif self.task_type == "multilabel":
            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels.float(),
                pos_weight=class_weights,
            )
            
            with torch.no_grad():
                predictions = (torch.sigmoid(logits) > 0.5).long()
                accuracy = (predictions == labels).float().mean()
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        
        return {
            'loss': loss,
            'accuracy': accuracy,
        }


class DownstreamModel(nn.Module):
    """
    Complete downstream model combining encoder and classifier.
    
    Supports freezing the encoder for linear probing or
    fine-tuning end-to-end.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        classifier: DownstreamClassifier,
        freeze_encoder: bool = False,
        freeze_encoder_layers: Optional[int] = None,
    ):
        """
        Args:
            encoder: Pretrained encoder (ClinicalSpeechSSL or similar)
            classifier: Classification head
            freeze_encoder: Whether to freeze all encoder parameters
            freeze_encoder_layers: Freeze first N encoder layers (if not freezing all)
        """
        super().__init__()
        
        self.encoder = encoder
        self.classifier = classifier
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif freeze_encoder_layers is not None:
            self._freeze_layers(freeze_encoder_layers)
    
    def _freeze_layers(self, num_layers: int):
        """Freeze first N encoder layers."""
        # Freeze frontend
        for param in self.encoder.frontend.parameters():
            param.requires_grad = False
        
        # Freeze first N encoder layers
        if hasattr(self.encoder.encoder, 'layers'):
            for i, layer in enumerate(self.encoder.encoder.layers):
                if i < num_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
    
    def forward(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through encoder and classifier.
        
        Args:
            waveform: Input audio [B, T]
            lengths: Audio lengths [B]
            
        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Encode
        if hasattr(self.encoder, 'encode'):
            features = self.encoder.encode(waveform, lengths)
        else:
            features = self.encoder(waveform, lengths)
        
        # Get feature lengths (approximate)
        feature_lengths = None
        if lengths is not None and hasattr(self.encoder, 'frontend'):
            if hasattr(self.encoder.frontend, 'get_output_length'):
                feature_lengths = torch.tensor([
                    self.encoder.frontend.get_output_length(l.item())
                    for l in lengths
                ], device=lengths.device)
        
        # Classify
        logits = self.classifier(features, feature_lengths)
        
        return logits
    
    def compute_loss(
        self,
        waveform: torch.Tensor,
        labels: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute forward pass and loss.
        
        Args:
            waveform: Input audio [B, T]
            labels: Ground truth labels
            lengths: Audio lengths [B]
            class_weights: Optional class weights
            
        Returns:
            Dictionary with 'loss' and metrics
        """
        logits = self.forward(waveform, lengths)
        return self.classifier.compute_loss(logits, labels, class_weights)
