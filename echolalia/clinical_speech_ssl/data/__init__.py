"""Data loading and augmentation modules."""

from clinical_speech_ssl.data.dataset import (
    SpeechSample,
    ClinicalSpeechDataset,
    SSLCollator,
    create_dataloaders,
)
from clinical_speech_ssl.data.augmentations import (
    AugmentationType,
    AugmentationConfig,
    AugmentationResult,
    RegionAugmentationLabel,
    AudioAugmentor,
    RegionAugmentor,
    SafeAugmentor,
    CLINICAL_AUGMENTATIONS,
    CLINICAL_AUGMENTATION_INDEX,
    SAFE_AUGMENTATIONS,
)

__all__ = [
    "SpeechSample",
    "ClinicalSpeechDataset",
    "SSLCollator",
    "create_dataloaders",
    "AugmentationType",
    "AugmentationConfig",
    "AugmentationResult",
    "RegionAugmentationLabel",
    "AudioAugmentor",
    "RegionAugmentor",
    "SafeAugmentor",
    "CLINICAL_AUGMENTATIONS",
    "CLINICAL_AUGMENTATION_INDEX",
    "SAFE_AUGMENTATIONS",
]
