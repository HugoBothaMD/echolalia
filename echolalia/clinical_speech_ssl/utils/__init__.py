"""Utility functions."""

from clinical_speech_ssl.utils.utils import (
    # Gamma utilities
    compute_phoneme_boundaries,
    compute_transition_mask,
    pool_to_phonemes,
    align_features_to_waveform,
    # Metrics
    compute_classification_metrics,
    compute_regression_metrics,
    compute_confusion_matrix,
    # Model utilities
    count_parameters,
    get_layer_wise_lr_groups,
    freeze_layers,
    unfreeze_layers,
    # Audio utilities
    compute_audio_stats,
    normalize_waveform,
)

__all__ = [
    "compute_phoneme_boundaries",
    "compute_transition_mask",
    "pool_to_phonemes",
    "align_features_to_waveform",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "compute_confusion_matrix",
    "count_parameters",
    "get_layer_wise_lr_groups",
    "freeze_layers",
    "unfreeze_layers",
    "compute_audio_stats",
    "normalize_waveform",
]
