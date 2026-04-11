"""
Utility functions for clinical speech SSL.

Includes gamma matrix utilities, metric computation, and helper functions.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================================
# Gamma Matrix Utilities
# ============================================================================

def compute_phoneme_boundaries(
    gamma: torch.Tensor,
    threshold: float = 0.5,
) -> List[Tuple[int, int, int]]:
    """
    Compute phoneme boundaries from gamma matrix.
    
    Args:
        gamma: CTC posteriors [T, num_phonemes]
        threshold: Threshold for dominant phoneme detection
        
    Returns:
        List of (start_frame, end_frame, phoneme_idx) tuples
    """
    # Get dominant phoneme at each frame
    dominant = gamma.argmax(dim=-1)
    confidence = gamma.max(dim=-1).values
    
    # Find boundaries where dominant phoneme changes
    boundaries = []
    current_phoneme = dominant[0].item()
    start_frame = 0
    
    for t in range(1, len(dominant)):
        if dominant[t].item() != current_phoneme:
            if confidence[start_frame:t].mean() >= threshold:
                boundaries.append((start_frame, t, current_phoneme))
            current_phoneme = dominant[t].item()
            start_frame = t
    
    # Add final segment
    if confidence[start_frame:].mean() >= threshold:
        boundaries.append((start_frame, len(dominant), current_phoneme))
    
    return boundaries


def compute_transition_mask(
    gamma: torch.Tensor,
    entropy_threshold: float = 0.3,
) -> torch.Tensor:
    """
    Create a mask indicating transition frames.
    
    Args:
        gamma: CTC posteriors [T, num_phonemes]
        entropy_threshold: Normalized entropy threshold
        
    Returns:
        Boolean mask [T] where True = transition
    """
    # Compute entropy at each frame
    gamma_safe = gamma.clamp(min=1e-10)
    entropy = -(gamma_safe * gamma_safe.log()).sum(dim=-1)
    max_entropy = np.log(gamma.shape[-1])
    normalized_entropy = entropy / max_entropy
    
    return normalized_entropy > entropy_threshold


def pool_to_phonemes(
    features: torch.Tensor,
    gamma: torch.Tensor,
    reduction: str = "weighted_mean",
) -> torch.Tensor:
    """
    Pool frame features to phoneme-level representations.
    
    Args:
        features: Frame features [T, D]
        gamma: CTC posteriors [T, num_phonemes]
        reduction: "weighted_mean", "max", or "attention"
        
    Returns:
        phoneme_features: [num_phonemes, D]
    """
    if reduction == "weighted_mean":
        # Normalize gamma per phoneme
        gamma_sum = gamma.sum(dim=0, keepdim=True).clamp(min=1e-10)
        gamma_norm = gamma / gamma_sum
        
        # Weighted sum
        phoneme_features = torch.mm(gamma_norm.t(), features)
    
    elif reduction == "max":
        # Max pooling weighted by gamma
        expanded_gamma = gamma.unsqueeze(-1)  # [T, P, 1]
        expanded_features = features.unsqueeze(1)  # [T, 1, D]
        weighted = expanded_gamma * expanded_features  # [T, P, D]
        phoneme_features = weighted.max(dim=0).values
    
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    
    return phoneme_features


def align_features_to_waveform(
    features: torch.Tensor,
    feature_length: int,
    waveform_length: int,
    method: str = "nearest",
) -> torch.Tensor:
    """
    Upsample features to waveform resolution.
    
    Args:
        features: [T', D]
        feature_length: T' (may differ from features.shape[0] if padded)
        waveform_length: Target length
        method: "nearest" or "linear"
        
    Returns:
        aligned_features: [waveform_length, D]
    """
    # Use only valid portion
    features = features[:feature_length]
    
    # Transpose for interpolation [1, D, T']
    features = features.t().unsqueeze(0)
    
    # Interpolate
    aligned = F.interpolate(
        features,
        size=waveform_length,
        mode=method,
    )
    
    # Transpose back [waveform_length, D]
    return aligned.squeeze(0).t()


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    task_type: str = "multiclass",
    average: str = "macro",
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Model predictions (probabilities or logits)
        labels: Ground truth labels
        task_type: "binary", "multiclass", or "multilabel"
        average: Averaging method for multi-class
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    if task_type == "binary":
        if predictions.ndim == 1:
            pred_probs = 1 / (1 + np.exp(-predictions))  # Sigmoid
        else:
            pred_probs = predictions[:, 1] if predictions.shape[1] == 2 else predictions[:, 0]
        
        pred_labels = (pred_probs > 0.5).astype(int)
        
        metrics["accuracy"] = accuracy_score(labels, pred_labels)
        metrics["auc"] = roc_auc_score(labels, pred_probs)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_labels, average="binary"
        )
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
    
    elif task_type == "multiclass":
        pred_probs = np.exp(predictions) / np.exp(predictions).sum(axis=1, keepdims=True)
        pred_labels = predictions.argmax(axis=1)
        
        metrics["accuracy"] = accuracy_score(labels, pred_labels)
        
        try:
            metrics["auc"] = roc_auc_score(labels, pred_probs, multi_class="ovr", average=average)
        except ValueError:
            metrics["auc"] = 0.0  # Not enough classes in batch
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_labels, average=average
        )
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
    
    elif task_type == "multilabel":
        pred_probs = 1 / (1 + np.exp(-predictions))
        pred_labels = (pred_probs > 0.5).astype(int)
        
        metrics["accuracy"] = (pred_labels == labels).mean()
        
        try:
            metrics["auc"] = roc_auc_score(labels, pred_probs, average=average)
        except ValueError:
            metrics["auc"] = 0.0
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_labels, average=average
        )
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
    
    return metrics


def compute_regression_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        predictions: Model predictions
        labels: Ground truth values
        
    Returns:
        Dictionary of metrics
    """
    # Ensure same shape
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)
    
    metrics = {
        "mae": mean_absolute_error(labels, predictions),
        "rmse": np.sqrt(mean_squared_error(labels, predictions)),
        "r2": r2_score(labels, predictions),
    }
    
    # Pearson correlation
    if len(predictions) > 1:
        correlation = np.corrcoef(predictions, labels)[0, 1]
        metrics["correlation"] = correlation if not np.isnan(correlation) else 0.0
    
    return metrics


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    normalize: str = "true",
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        normalize: "true", "pred", "all", or None
        
    Returns:
        Confusion matrix array
    """
    return confusion_matrix(labels, predictions, normalize=normalize)


# ============================================================================
# Model Utilities
# ============================================================================

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Returns:
        Dictionary with total, trainable, and frozen counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def get_layer_wise_lr_groups(
    model: torch.nn.Module,
    base_lr: float,
    lr_decay: float = 0.9,
) -> List[Dict]:
    """
    Create parameter groups with layer-wise learning rate decay.
    
    Useful for fine-tuning pretrained models where lower layers
    should have smaller learning rates.
    
    Args:
        model: The model
        base_lr: Base learning rate for top layer
        lr_decay: Decay factor per layer
        
    Returns:
        List of parameter group dicts for optimizer
    """
    groups = []
    
    # Get all named parameters
    all_params = list(model.named_parameters())
    
    # Group by layer depth (heuristic: count '.' in name)
    depth_to_params = {}
    max_depth = 0
    
    for name, param in all_params:
        depth = name.count('.')
        max_depth = max(max_depth, depth)
        
        if depth not in depth_to_params:
            depth_to_params[depth] = []
        depth_to_params[depth].append(param)
    
    # Create groups with decaying LR
    for depth in sorted(depth_to_params.keys(), reverse=True):
        layers_from_top = max_depth - depth
        lr = base_lr * (lr_decay ** layers_from_top)
        
        groups.append({
            "params": depth_to_params[depth],
            "lr": lr,
        })
    
    return groups


def freeze_layers(
    model: torch.nn.Module,
    layers_to_freeze: List[str],
):
    """
    Freeze specific layers by name pattern.
    
    Args:
        model: The model
        layers_to_freeze: List of layer name patterns to freeze
    """
    for name, param in model.named_parameters():
        for pattern in layers_to_freeze:
            if pattern in name:
                param.requires_grad = False
                break


def unfreeze_layers(
    model: torch.nn.Module,
    layers_to_unfreeze: List[str],
):
    """
    Unfreeze specific layers by name pattern.
    
    Args:
        model: The model
        layers_to_unfreeze: List of layer name patterns to unfreeze
    """
    for name, param in model.named_parameters():
        for pattern in layers_to_unfreeze:
            if pattern in name:
                param.requires_grad = True
                break


# ============================================================================
# Audio Utilities
# ============================================================================

def compute_audio_stats(
    waveform: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute basic audio statistics.
    
    Args:
        waveform: Audio tensor [T] or [C, T]
        
    Returns:
        Dictionary of statistics
    """
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)
    
    return {
        "duration_samples": waveform.shape[0],
        "mean": waveform.mean().item(),
        "std": waveform.std().item(),
        "max": waveform.abs().max().item(),
        "rms": torch.sqrt((waveform ** 2).mean()).item(),
    }


def normalize_waveform(
    waveform: torch.Tensor,
    method: str = "peak",
) -> torch.Tensor:
    """
    Normalize audio waveform.
    
    Args:
        waveform: Audio tensor
        method: "peak", "rms", or "standard"
        
    Returns:
        Normalized waveform
    """
    if method == "peak":
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
    
    elif method == "rms":
        rms = torch.sqrt((waveform ** 2).mean())
        if rms > 0:
            waveform = waveform / rms
    
    elif method == "standard":
        mean = waveform.mean()
        std = waveform.std()
        if std > 0:
            waveform = (waveform - mean) / std
    
    return waveform
