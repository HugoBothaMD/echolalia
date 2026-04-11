"""Encoder architectures for sequence modeling."""

from clinical_speech_ssl.models.encoders.transformer import (
    TransformerEncoder,
    TransformerEncoderSmall,
    TransformerEncoderBase,
    TransformerEncoderLarge,
    TransformerEncoderLayer,
    MultiHeadAttention,
    FeedForward,
)
from clinical_speech_ssl.models.encoders.conformer import (
    ConformerEncoder,
    ConformerEncoderSmall,
    ConformerEncoderMedium,
    ConformerEncoderLarge,
    ConformerBlock,
    ConvolutionModule,
)

__all__ = [
    "TransformerEncoder",
    "TransformerEncoderSmall",
    "TransformerEncoderBase",
    "TransformerEncoderLarge",
    "TransformerEncoderLayer",
    "MultiHeadAttention",
    "FeedForward",
    "ConformerEncoder",
    "ConformerEncoderSmall",
    "ConformerEncoderMedium",
    "ConformerEncoderLarge",
    "ConformerBlock",
    "ConvolutionModule",
]
