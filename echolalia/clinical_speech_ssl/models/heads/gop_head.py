"""
GOP prediction head for SSL.

Predicts per-phoneme Goodness of Pronunciation scores from
encoder features using gamma-weighted pooling.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GOPPredictionHead(nn.Module):
    """Predicts per-phoneme GOP scores from encoder output.

    Uses the gamma matrix to soft-pool encoder features to phoneme
    regions, then applies a shared MLP to predict each phoneme's GOP.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim // 2

        self.projector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        gammas: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Predict per-phoneme GOP scores.

        Args:
            features: Encoder output [B, T, D]
            gammas: List of [T_i, P_i] gamma matrices (one per sample)

        Returns:
            List of [P_i] predicted GOP score tensors (one per sample)
        """
        batch_predictions = []
        for b in range(features.shape[0]):
            gamma = gammas[b]  # [T_gamma, P]
            feat = features[b]  # [T_feat, D]

            # Align gamma to feature length (may differ due to frontend downsampling)
            T_feat = feat.shape[0]
            T_gamma = gamma.shape[0]
            if T_gamma != T_feat:
                gamma = F.interpolate(
                    gamma.unsqueeze(0).transpose(1, 2),  # [1, P, T_gamma]
                    size=T_feat,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2).squeeze(0)  # [T_feat, P]

            # Soft-pool features to phoneme regions
            # Normalize gamma per-phoneme (sum over time for each phoneme)
            gamma_norm = gamma / gamma.sum(dim=0, keepdim=True).clamp(min=1e-10)
            # [T, P]^T x [T, D] -> [P, D]
            phoneme_features = torch.einsum('tp,td->pd', gamma_norm, feat)

            # Predict GOP per phoneme
            predictions = self.projector(phoneme_features).squeeze(-1)  # [P]
            batch_predictions.append(predictions)

        return batch_predictions


class GOPPredictionLoss(nn.Module):
    """Loss for GOP prediction objective."""

    def __init__(self, loss_type: str = "smoothl1"):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute GOP prediction loss.

        Args:
            predictions: List of [P_i] predicted scores
            targets: List of [P_i] target GOP scores

        Returns:
            Dict with 'loss' key
        """
        losses = []
        for pred, tgt in zip(predictions, targets):
            if tgt is None:
                continue
            # Align lengths (predictions may cover CTC blanks; targets are phoneme-only)
            min_len = min(pred.shape[0], tgt.shape[0])
            p = pred[:min_len]
            t = tgt[:min_len].to(p.device)

            if self.loss_type == "smoothl1":
                losses.append(F.smooth_l1_loss(p, t))
            else:
                losses.append(F.mse_loss(p, t))

        if not losses:
            device = predictions[0].device if predictions else torch.device("cpu")
            return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

        return {"loss": torch.stack(losses).mean()}
