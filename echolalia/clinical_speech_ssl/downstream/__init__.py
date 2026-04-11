"""Downstream fine-tuning modules."""

from clinical_speech_ssl.downstream.classifier import (
    DownstreamClassifier,
    DownstreamModel,
)
from clinical_speech_ssl.downstream.regressor import (
    DownstreamRegressor,
    DownstreamRegressorModel,
)
from clinical_speech_ssl.downstream.multi_task import (
    TaskConfig,
    MultiTaskHead,
    MultiTaskModel,
)

__all__ = [
    "DownstreamClassifier",
    "DownstreamModel",
    "DownstreamRegressor",
    "DownstreamRegressorModel",
    "TaskConfig",
    "MultiTaskHead",
    "MultiTaskModel",
]
