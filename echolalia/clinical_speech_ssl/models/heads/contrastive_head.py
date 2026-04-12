"""
Contrastive learning head for SSL.

Implements contrastive objective with safe augmentations
to learn recording-condition invariant representations.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    
    Projects encoder outputs to a lower-dimensional space
    where contrastive loss is computed.
    """
    
    def __init__(
        self,
        embed_dim: int,
        projection_dim: int = 256,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        use_batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            projection_dim: Output projection dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of projection layers
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        hidden_dim = hidden_dim or embed_dim
        
        layers = []
        current_dim = embed_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            else:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(nn.ReLU(inplace=True))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            current_dim = hidden_dim
        
        # Final projection (no activation)
        layers.append(nn.Linear(current_dim, projection_dim))
        
        self.projection = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features.
        
        Args:
            x: Input features [B, D] or [B, T, D]
            
        Returns:
            projections: [B, projection_dim] or [B, T, projection_dim]
        """
        return self.projection(x)


class ContrastiveHead(nn.Module):
    """
    Contrastive learning head.
    
    Computes embeddings for contrastive loss where:
    - Positive pairs: same utterance with different safe augmentations
    - Negative pairs: different utterances
    """
    
    def __init__(
        self,
        embed_dim: int,
        projection_dim: int = 256,
        hidden_dim: Optional[int] = None,
        temperature: float = 0.07,
        use_pooling: str = "mean",
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            projection_dim: Projection dimension for contrastive loss
            hidden_dim: Hidden dimension in projection head
            temperature: Temperature for InfoNCE loss
            use_pooling: Pooling type ("mean", "cls", "attention")
        """
        super().__init__()
        
        self.temperature = temperature
        self.use_pooling = use_pooling
        
        self.projection = ProjectionHead(
            embed_dim=embed_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
        )
        
        if use_pooling == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(embed_dim, 1),
                nn.Softmax(dim=1),
            )
    
    def pool(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool sequence features to single vector.
        
        Args:
            features: [B, T, D]
            lengths: [B] actual lengths
            
        Returns:
            pooled: [B, D]
        """
        if self.use_pooling == "mean":
            if lengths is not None:
                # Masked mean pooling
                mask = torch.arange(
                    features.shape[1], device=features.device
                ).unsqueeze(0) < lengths.unsqueeze(1)
                
                features = features * mask.unsqueeze(-1)
                pooled = features.sum(dim=1) / lengths.unsqueeze(-1).float()
            else:
                pooled = features.mean(dim=1)
        
        elif self.use_pooling == "cls":
            # Use first token as CLS
            pooled = features[:, 0, :]
        
        elif self.use_pooling == "attention":
            # Attention-weighted pooling
            attn_weights = self.attention_pool(features)  # [B, T, 1]
            pooled = (features * attn_weights).sum(dim=1)  # [B, D]
        
        else:
            raise ValueError(f"Unknown pooling type: {self.use_pooling}")
        
        return pooled
    
    def forward(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive embeddings.
        
        Args:
            features: Encoder output [B, T, D]
            lengths: Sequence lengths [B]
            
        Returns:
            embeddings: L2-normalized projections [B, projection_dim]
        """
        # Pool to utterance level
        pooled = self.pool(features, lengths)
        
        # Project
        projected = self.projection(pooled)
        
        # L2 normalize
        embeddings = F.normalize(projected, dim=-1)
        
        return embeddings


class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss.

    Supports both standard and supervised contrastive loss.
    """

    def __init__(
        self,
        temperature: float = 0.07,
    ):
        """
        Args:
            temperature: Temperature scaling
        """
        super().__init__()

        self.temperature = temperature
    
    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss.
        
        Args:
            embeddings_a: First view embeddings [B, D]
            embeddings_b: Second view embeddings [B, D]
            labels: Optional labels for supervised contrastive [B]
            
        Returns:
            Dictionary with 'loss', 'accuracy'
        """
        B = embeddings_a.shape[0]
        device = embeddings_a.device
        
        # Concatenate embeddings
        embeddings = torch.cat([embeddings_a, embeddings_b], dim=0)  # [2B, D]
        
        # Compute similarity matrix
        similarity = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Create positive pair mask
        # Positive pairs: (i, i+B) and (i+B, i)
        pos_mask = torch.zeros(2 * B, 2 * B, dtype=torch.bool, device=device)
        pos_mask[torch.arange(B), torch.arange(B) + B] = True
        pos_mask[torch.arange(B) + B, torch.arange(B)] = True
        
        # Create negative mask (exclude self)
        neg_mask = ~torch.eye(2 * B, dtype=torch.bool, device=device)
        
        # If labels provided, also exclude same-class samples from negatives
        if labels is not None:
            labels_cat = torch.cat([labels, labels])
            same_class = labels_cat.unsqueeze(0) == labels_cat.unsqueeze(1)
            neg_mask = neg_mask & ~same_class
        
        # Compute InfoNCE loss
        # For each sample, positive is the augmented view, negatives are all others
        
        # Get positive similarities
        pos_sim = similarity[pos_mask].view(2 * B, 1)
        
        # Get negative similarities
        neg_sim = similarity.masked_fill(~neg_mask, float('-inf'))
        
        # Logits: positive vs all negatives
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        
        # Labels: positive is always index 0
        target = torch.zeros(2 * B, dtype=torch.long, device=device)
        
        loss = F.cross_entropy(logits, target)
        
        # Compute accuracy
        with torch.no_grad():
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == target).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
        }


class SimCLRLoss(nn.Module):
    """
    NT-Xent loss from SimCLR.
    
    Symmetric contrastive loss treating both views equally.
    """
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute NT-Xent loss.
        
        Args:
            embeddings_a: First view [B, D]
            embeddings_b: Second view [B, D]
            
        Returns:
            Dictionary with 'loss'
        """
        B = embeddings_a.shape[0]
        device = embeddings_a.device
        
        # Concatenate
        embeddings = torch.cat([embeddings_a, embeddings_b], dim=0)
        
        # Similarity matrix
        similarity = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Mask out self-similarity
        mask = torch.eye(2 * B, dtype=torch.bool, device=device)
        similarity = similarity.masked_fill(mask, float('-inf'))
        
        # Positive indices: (i, i+B) for first half, (i, i-B) for second half
        pos_indices = torch.cat([
            torch.arange(B, 2 * B, device=device),
            torch.arange(0, B, device=device),
        ])
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity, pos_indices)
        
        return {'loss': loss}


class ContrastiveModule(nn.Module):
    """
    Complete contrastive learning module.
    
    Handles augmentation, embedding, and loss computation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        projection_dim: int = 256,
        temperature: float = 0.07,
        loss_type: str = "infonce",
    ):
        """
        Args:
            embed_dim: Encoder output dimension
            projection_dim: Projection dimension
            temperature: Temperature for loss
            loss_type: "infonce" or "simclr"
        """
        super().__init__()
        
        self.head = ContrastiveHead(
            embed_dim=embed_dim,
            projection_dim=projection_dim,
            temperature=temperature,
        )
        
        if loss_type == "infonce":
            self.loss_fn = ContrastiveLoss(temperature=temperature)
        elif loss_type == "simclr":
            self.loss_fn = SimCLRLoss(temperature=temperature)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        lengths_a: Optional[torch.Tensor] = None,
        lengths_b: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss between two views.
        
        Args:
            features_a: First view encoder output [B, T, D]
            features_b: Second view encoder output [B, T, D]
            lengths_a: Lengths for first view [B]
            lengths_b: Lengths for second view [B]
            
        Returns:
            Dictionary with loss and metrics
        """
        # Get embeddings
        embeddings_a = self.head(features_a, lengths_a)
        embeddings_b = self.head(features_b, lengths_b)
        
        # Compute loss
        result = self.loss_fn(embeddings_a, embeddings_b)
        
        return result
