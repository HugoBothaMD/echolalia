"""SoftAlign measure extraction for downstream auxiliary targets."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from soft_align import (
    AudioBackbone,
    MeasureExtractor,
    MeasureBundle,
    AlignmentResult,
)
from soft_align.measures import (
    SoftDuration,
    TransitionSharpness,
    GammaEntropy,
    CoarticulationIndex,
    AlignmentConfidence,
    PairwiseVariabilityIndex,
)

# Default measures for SSL auxiliary targets
DEFAULT_MEASURES = [
    SoftDuration(),
    TransitionSharpness(),
    GammaEntropy(),
    CoarticulationIndex(),
    AlignmentConfidence(),
    PairwiseVariabilityIndex(scope="vocalic"),
]


def compute_measure_vector(
    audio: torch.Tensor,
    transcript: str,
    backbone: AudioBackbone,
    sample_rate: int = 16000,
    measures: Optional[List] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute a flat feature vector of SoftAlign measures.

    Args:
        audio: Waveform tensor
        transcript: Target transcript
        backbone: soft_align AudioBackbone
        sample_rate: Audio sample rate
        measures: List of SoftAlignMeasure instances (uses defaults if None)

    Returns:
        Tuple of (feature_vector [D], measure_keys [D])
        where measure_keys describes what each dimension represents.
    """
    if measures is None:
        measures = DEFAULT_MEASURES

    extractor = MeasureExtractor(backbone, measures=measures)
    bundle: MeasureBundle = extractor.extract(audio, transcript, sample_rate=sample_rate)

    feature_vector = bundle.to_feature_vector()
    # Build key names from sorted measure results
    keys = []
    for name in sorted(bundle.measure_results.keys()):
        result = bundle.measure_results[name]
        if result.utterance_value is not None:
            keys.append(name)

    return feature_vector, keys
