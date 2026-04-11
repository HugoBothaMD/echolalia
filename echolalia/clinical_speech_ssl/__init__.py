"""
Clinical Speech SSL: Transition-aware multi-objective self-supervised learning for clinical speech.

This package provides a modular framework for:
- Self-supervised pretraining on clinical speech data
- Multiple input modalities (waveform, spectrogram patches)
- Multiple SSL objectives (augmentation prediction, masked reconstruction, contrastive)
- Transition-focused learning with CTC-based soft alignments
- Downstream fine-tuning for classification and regression tasks
"""

__version__ = "0.1.0"

from clinical_speech_ssl.models.ssl_model import ClinicalSpeechSSL
from clinical_speech_ssl.downstream.classifier import DownstreamClassifier
from clinical_speech_ssl.downstream.regressor import DownstreamRegressor
from clinical_speech_ssl.downstream.multi_task import MultiTaskHead
from clinical_speech_ssl.data.dataset import ClinicalSpeechDataset
from clinical_speech_ssl.training.trainer import SSLTrainer, DownstreamTrainer

__all__ = [
    "ClinicalSpeechSSL",
    "DownstreamClassifier",
    "DownstreamRegressor",
    "MultiTaskHead",
    "ClinicalSpeechDataset",
    "SSLTrainer",
    "DownstreamTrainer",
]
