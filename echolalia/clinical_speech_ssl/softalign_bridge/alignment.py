"""Gamma matrix computation using soft_align backbones."""

from typing import Optional, Tuple

import torch

from soft_align import (
    CTCAligner,
    AlignmentResult,
    PhonemeCTCBackbone,
    PhonemeClassifierBackbone,
    AudioBackbone,
)


def get_alignment_backbone(
    backbone_type: str = "phoneme_classifier",
    classifier_checkpoint: Optional[str] = None,
    model_name: Optional[str] = None,
) -> AudioBackbone:
    """Load a soft_align backbone for gamma computation.

    Args:
        backbone_type: "phoneme_classifier" (recommended) or "phoneme_ctc"
        classifier_checkpoint: Path to trained classifier checkpoint
            (required for phoneme_classifier)
        model_name: HuggingFace model name for CTC backbone

    Returns:
        An AudioBackbone instance
    """
    if backbone_type == "phoneme_classifier":
        if classifier_checkpoint is None:
            print(
                "Warning: No classifier checkpoint provided, falling back "
                "to PhonemeCTCBackbone. Gamma will be peaky (CTC artifact). "
                "Train a PhonemeClassifierBackbone for smoother posteriors."
            )
            return PhonemeCTCBackbone(model_name=model_name)
        return PhonemeClassifierBackbone(
            classifier_checkpoint=classifier_checkpoint,
        )
    elif backbone_type == "phoneme_ctc":
        return PhonemeCTCBackbone(model_name=model_name)
    else:
        raise ValueError(
            f"Unknown backbone_type: {backbone_type}. "
            f"Choose 'phoneme_classifier' or 'phoneme_ctc'."
        )


def compute_gamma(
    audio: torch.Tensor,
    transcript: str,
    backbone: AudioBackbone,
    sample_rate: int = 16000,
) -> Tuple[torch.Tensor, AlignmentResult]:
    """Compute gamma matrix for an audio/transcript pair.

    Args:
        audio: Waveform tensor (samples,) or (1, samples)
        transcript: Target transcript
        backbone: soft_align AudioBackbone
        sample_rate: Audio sample rate

    Returns:
        Tuple of (gamma [T, P], full AlignmentResult)
    """
    output = backbone.process_audio(audio, sample_rate)
    target_ids = backbone.tokenize_target(transcript)
    aligner = CTCAligner(backbone.get_blank_index())
    alignment = aligner.compute_posterior(output.log_probs, target_ids)
    return alignment.gamma, alignment
