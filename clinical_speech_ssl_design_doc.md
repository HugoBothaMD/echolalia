# Clinical Speech SSL: Design Document

**Transition-Aware Multi-Objective Self-Supervised Learning for Clinical Speech Analysis**

*A framework for learning speech representations that are sensitive to clinically-relevant acoustic variations while being robust to recording conditions.*

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [The Core Insight](#2-the-core-insight)
3. [Multi-Objective SSL Framework](#3-multi-objective-ssl-framework)
4. [Implementation Overview](#4-implementation-overview)
5. [Package Architecture](#5-package-architecture)
6. [Integration with SoftAlign](#6-integration-with-softalign)
7. [Future Directions](#7-future-directions)
8. [Appendix: Configuration Reference](#appendix-configuration-reference)

---

## 1. Background and Motivation

### 1.1 The Problem with Standard Speech SSL

Self-supervised learning models for speech (wav2vec 2.0, HuBERT, WavLM) have achieved remarkable success by learning representations that are invariant to acoustic perturbations. The typical SSL recipe:

1. Apply augmentations (time stretching, pitch shifting, spectral warping, noise)
2. Train the model to produce similar representations for augmented and original audio
3. Learn to ignore these "nuisance" variations

**This is exactly wrong for clinical speech.**

In motor speech disorders, the diagnostic signal *is* the acoustic perturbation:

| Clinical Feature | Acoustic Correlate | Standard SSL Treatment |
|------------------|-------------------|----------------------|
| Bradykinesia (Parkinson's) | Slowed articulation, reduced rate | Time-stretch invariance removes this |
| Monotone speech | Reduced pitch range | Pitch-shift invariance removes this |
| Imprecise articulation | Formant undershoot, blurred transitions | Spectral warping invariance removes this |
| Hypernasality | Altered formant structure | Spectral perturbation invariance removes this |

When we fine-tune WavLM on clinical speech classification, we're fighting against representations that were explicitly trained to ignore the very features we need.

### 1.2 The Fixed-Utterance Advantage

The Speech AI Program uses a standardized utterance: *"My physician wrote out a prescription."* Every patient says the same phrase. This creates a unique opportunity:

- **Known phoneme sequence** without runtime alignment ambiguity
- **Natural correspondence points** across patients (the /p/ in "physician" from patient A can be compared to the /p/ from patient B)
- **Clean prototype construction** from healthy speakers
- **Localized learning objectives** targeting specific phoneme regions

This fixed-utterance paradigm enables learning approaches that would be impossible with variable transcripts.

### 1.3 Soft Alignment as Foundation

Hard phoneme boundaries (from tools like MFA) fail on disordered speech. The **SoftAlign** package provides CTC-based soft alignments—a gamma matrix of posterior probabilities over phoneme positions at each time frame. This:

- Degrades gracefully when articulation is imprecise
- Provides uncertainty-aware phoneme masks
- Identifies transitions (high-entropy frames) naturally
- Enables weighted feature pooling without hard cutpoints

Our SSL framework assumes access to these gamma matrices as given.

---

## 2. The Core Insight

### 2.1 Inverting the Augmentation Paradigm

**Standard SSL:** Learn representations invariant TO augmentations  
**Clinical Speech SSL:** Learn representations that can DETECT augmentations

If a model can detect that a segment has been time-stretched, it must have learned a rich representation of normal temporal dynamics. If it can detect pitch shifting, it has learned pitch structure. The augmentation prediction task forces the model to be *sensitive* to exactly the acoustic properties that clinical assessment requires.

### 2.2 Augmentation Taxonomy

We categorize augmentations by their clinical relevance:

**Clinical Augmentations** (mimic pathology—use as prediction targets):

| Augmentation | What It Mimics | Clinical Correlate |
|--------------|----------------|-------------------|
| Time stretch | Bradykinesia, scanning speech | Rate disorders |
| Pitch shift | Monotone, pitch breaks | Prosodic disorders |
| Formant shift | Vowel distortion, undershoot | Articulatory imprecision |
| Amplitude modulation | Tremor, flutter | Laryngeal instability |

**Safe Augmentations** (affect recording, not speech—use for contrastive invariance):

| Augmentation | What It Simulates |
|--------------|-------------------|
| Additive noise | Background noise, mic quality |
| Reverb | Room acoustics |
| Gain | Recording level variation |
| Filtering | Telephone bandwidth, equipment |

### 2.3 Transition-Focused Learning

Phoneme transitions carry disproportionate diagnostic information:
- Coarticulation patterns reveal motor planning
- Transition sharpness indicates articulatory precision
- VOT (voice onset time) is a key dysarthria marker
- Stop releases are often the first casualty of motor impairment

We bias all learning objectives toward phoneme transitions by:
- Augmenting transitions more frequently
- Masking transitions at higher probability
- Weighting transition-region losses higher

Transitions are identified as high-entropy frames in the gamma matrix—frames where the model is uncertain which phoneme is active.

---

## 3. Multi-Objective SSL Framework

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ClinicalSpeechSSL                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────┐                                               │
│   │  Raw Waveform   │                                               │
│   └────────┬────────┘                                               │
│            │                                                        │
│            ▼                                                        │
│   ┌─────────────────┐     ┌─────────────────┐                       │
│   │    Frontend     │     │   Alternatives  │                       │
│   │  ┌───────────┐  │     │  ┌───────────┐  │                       │
│   │  │ Waveform  │  │ OR  │  │ Spectro-  │  │                       │
│   │  │    CNN    │  │     │  │  gram     │  │                       │
│   │  └───────────┘  │     │  │ Patches   │  │                       │
│   └────────┬────────┘     └───────┬───────┘                         │
│            │                      │                                 │
│            └──────────┬───────────┘                                 │
│                       ▼                                             │
│            ┌─────────────────┐                                      │
│            │     Encoder     │                                      │
│            │  ┌───────────┐  │     ┌─────────────┐                  │
│            │  │Transformer│  │ OR  │  Conformer  │                  │
│            │  └───────────┘  │     └─────────────┘                  │
│            └────────┬────────┘                                      │
│                     │                                               │
│                     ▼                                               │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                      SSL Objectives                          │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │   │
│   │  │ Augmentation │  │   Masked     │  │   Contrastive    │   │   │
│   │  │  Prediction  │  │Reconstruction│  │    Learning      │   │   │
│   │  │              │  │              │  │                  │   │   │
│   │  │ Detect type  │  │ Reconstruct  │  │ Same utterance,  │   │   │
│   │  │ + magnitude  │  │ masked frames│  │ different safe   │   │   │
│   │  │ per region   │  │ (transition  │  │ augmentations    │   │   │
│   │  │              │  │  bias)       │  │ = positive pair  │   │   │
│   │  └──────────────┘  └──────────────┘  └──────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Objective 1: Augmentation Prediction

**Goal:** Force the encoder to learn fine-grained acoustic sensitivity.

**Mechanism:**
1. Apply clinical augmentations to random phoneme regions (with transition bias)
2. Predict: (a) which augmentation types were applied, (b) their magnitude
3. Use gamma matrix to pool encoder outputs to phoneme regions
4. Multi-label classification + regression per region

**What the model learns:**
- Precise temporal dynamics (to detect time stretch)
- Pitch structure (to detect pitch shift)
- Formant patterns (to detect formant shift)
- Amplitude envelope (to detect tremor-like modulation)

**Loss:**
```
L_aug = λ_presence * BCE(pred_presence, true_presence) 
      + λ_magnitude * MSE(pred_magnitude, true_magnitude)
```

### 3.3 Objective 2: Masked Reconstruction

**Goal:** Learn temporal dynamics and coarticulation patterns.

**Mechanism:**
1. Generate mask with bias toward transition frames (high gamma entropy)
2. Replace masked frames with learnable mask token
3. Predict original frame features from context
4. Use cosine similarity or MSE loss

**Transition bias implementation:**
```python
# Identify transitions via gamma entropy
entropy = -(gamma * log(gamma)).sum(dim=-1)
is_transition = entropy > threshold

# Bias masking probability
mask_prob = base_prob * (1 + transition_bias * is_transition)
```

**What the model learns:**
- How phonemes flow into each other
- Expected coarticulation patterns
- Temporal dependencies across the utterance

### 3.4 Objective 3: Contrastive Learning (Safe Augmentations Only)

**Goal:** Learn representations invariant to recording conditions.

**Mechanism:**
1. Create two views of each utterance using only safe augmentations
2. Same utterance + different noise/reverb/gain = positive pair
3. Different utterances = negative pairs
4. InfoNCE or NT-Xent loss

**What the model learns:**
- Speaker-independent, recording-independent representations
- Focus on speech content, not channel effects

**Crucial constraint:** Never use clinical augmentations (pitch shift, time stretch) for contrastive pairs—this would teach the model to ignore exactly what we need.

### 3.5 Combined Training

All three objectives train simultaneously on the shared encoder:

```
L_total = λ_aug * L_augmentation 
        + λ_mask * L_reconstruction 
        + λ_contrast * L_contrastive
```

Default weights: `λ_aug = 1.0`, `λ_mask = 1.0`, `λ_contrast = 0.5`

The contrastive weight is lower because its role is regularization (don't overfit to recording conditions) rather than primary representation learning.

---

## 4. Implementation Overview

### 4.1 What Was Built

A complete, modular PyTorch package (`clinical_speech_ssl`) with ~8,200 lines of Python:

| Component | Lines | Description |
|-----------|-------|-------------|
| Data & Augmentations | ~1,400 | Dataset loading, clinical/safe augmentation taxonomy, region-aware augmentation |
| Frontends | ~800 | Waveform CNN (wav2vec-style), spectrogram patchers (tall-narrow, ViT) |
| Encoders | ~1,200 | Transformer, Conformer with relative position encodings |
| SSL Heads | ~1,000 | Augmentation prediction, masked reconstruction, contrastive learning |
| Main SSL Model | ~600 | Unified model combining all components |
| Downstream | ~900 | Classification, regression, multi-task heads |
| Training | ~800 | SSL trainer, downstream trainer, checkpointing, early stopping |
| Tests | ~1,500 | Comprehensive test suite for all components |

### 4.2 Key Design Decisions

**Modularity:** Every component is swappable:
```python
config = ClinicalSpeechSSLConfig(
    input_type="waveform",           # or "spectrogram"
    frontend_type="cnn_base",        # or "patch_tall_narrow", "patch_vit"
    encoder_type="conformer_medium", # or "transformer_*"
    use_augmentation_prediction=True,
    use_masked_reconstruction=True,
    use_contrastive=True,
)
```

**Transition bias is configurable:**
```python
config = ClinicalSpeechSSLConfig(
    transition_bias=2.0,  # 2x more likely to augment/mask transitions
    mask_prob=0.15,
    mask_span_length=10,
)
```

**Gamma matrix integration:**
- Dataset yields gamma matrices alongside audio
- Augmentation prediction uses gamma to pool features to phonemes
- Masked reconstruction uses gamma entropy to identify transitions
- All measures can be computed per-phoneme using soft alignment

### 4.3 Frontend Options

**Waveform CNN (Default):**
- 7-layer convolutional feature extractor (wav2vec 2.0 style)
- 320x downsampling (20ms frames at 16kHz)
- Group normalization, GELU activation

**Spectrogram Patches:**

*Tall-Narrow Patches:*
- Full frequency range × few time frames
- Respects that frequency is not translation-invariant
- Better for formant patterns, spectral envelopes

*ViT-Style Patches:*
- Square/rectangular grid patches
- More general, image-like treatment
- May miss frequency-specific structure

### 4.4 Encoder Options

| Encoder | Layers | Heads | Dim | Parameters | Use Case |
|---------|--------|-------|-----|------------|----------|
| `transformer_small` | 4 | 4 | 256 | ~5M | Testing |
| `transformer_base` | 12 | 12 | 768 | ~85M | Research baseline |
| `conformer_small` | 4 | 4 | 256 | ~8M | Fast experiments |
| `conformer_medium` | 12 | 4 | 256 | ~25M | **Recommended** |
| `conformer_large` | 17 | 8 | 512 | ~100M | Maximum capacity |

Conformer is preferred because the convolution module captures local patterns (important for coarticulation) while attention captures global context.

---

## 5. Package Architecture

```
clinical_speech_ssl/
├── clinical_speech_ssl/
│   ├── __init__.py                 # Public API
│   │
│   ├── data/
│   │   ├── augmentations.py        # AudioAugmentor, RegionAugmentor, SafeAugmentor
│   │   │                           # AugmentationType enum (CLINICAL vs SAFE)
│   │   └── dataset.py              # ClinicalSpeechDataset, SSLCollator
│   │
│   ├── models/
│   │   ├── frontends/
│   │   │   ├── waveform_cnn.py     # ConvBlock, WaveformCNNFrontend
│   │   │   └── spectrogram_patcher.py  # TallNarrowPatcher, ViTStylePatcher
│   │   │
│   │   ├── encoders/
│   │   │   ├── transformer.py      # TransformerEncoder, MultiHeadAttention
│   │   │   └── conformer.py        # ConformerEncoder, ConvolutionModule
│   │   │
│   │   ├── heads/
│   │   │   ├── augmentation_head.py    # AugmentationPredictionHead
│   │   │   ├── reconstruction_head.py  # MaskGenerator, ReconstructionHead
│   │   │   └── contrastive_head.py     # ContrastiveHead, ProjectionHead
│   │   │
│   │   └── ssl_model.py            # ClinicalSpeechSSL, ClinicalSpeechSSLConfig
│   │
│   ├── downstream/
│   │   ├── classifier.py           # DownstreamClassifier, DownstreamModel
│   │   ├── regressor.py            # DownstreamRegressor
│   │   └── multi_task.py           # TaskConfig, MultiTaskHead
│   │
│   ├── training/
│   │   └── trainer.py              # SSLTrainer, DownstreamTrainer
│   │                               # TrainingConfig, CheckpointManager
│   │
│   └── utils/
│       └── utils.py                # Helpers
│
├── tests/
│   ├── test_ssl_model.py           # Main model tests
│   ├── test_frontends.py           # Frontend tests
│   ├── test_downstream.py          # Downstream tests
│   └── test_data.py                # Augmentation and dataset tests
│
├── configs/
│   ├── ssl_pretraining.yaml        # SSL pretraining config
│   ├── downstream_classification.yaml  # Fine-tuning config
│   └── ablations.yaml              # Systematic ablation study config
│
├── examples/
│   └── train_example.py            # Working example with synthetic data
│
├── README.md                       # Full documentation
└── pyproject.toml                  # Package configuration
```

---

## 6. Integration with SoftAlign

The `clinical_speech_ssl` package and `soft_align` package are natural complements. SoftAlign provides the alignment foundation; Clinical Speech SSL learns representations on top of it.

### 6.1 SoftAlign Overview (from documentation)

SoftAlign replaces hard phoneme boundaries with a probability distribution over all possible alignments, computed via the CTC forward-backward algorithm. The result is a **gamma matrix** — a (time × phoneme) posterior probability surface.

Key SoftAlign capabilities relevant to our SSL framework:

| Component | Description | Relevance to SSL |
|-----------|-------------|------------------|
| `PhonemeCTCBackbone` | Wav2Vec2/XLSR/WavLM + CTC | Produces gamma matrix |
| `PhonemeClassifierBackbone` | Frame-level MLP on frozen WavLM | Smoother, better-calibrated posteriors |
| `CTCAligner` | Vectorized forward-backward | Computes gamma from log-probs |
| `MeasureExtractor` | Extracts clinical measures | TransitionSharpness, GammaEntropy, etc. |
| `UtteranceAnalysisPipeline` | Handles false starts, multiple attempts | Preprocessing for messy recordings |
| `TrajectoryExtractor` + OT | Alignment-free comparison | Fallback for severe pathology |

### 6.2 Current Integration Points

**Gamma Matrix as Input:**
```python
# SoftAlign produces gamma matrices
from soft_align import PhonemeCTCBackbone, CTCAligner

backbone = PhonemeCTCBackbone()
aligner = CTCAligner(backbone.get_blank_index())

output = backbone.process_audio(audio, sr)
target_ids = backbone.tokenize_target("my physician wrote out a prescription")
alignment = aligner.compute_posterior(output.log_probs, target_ids)

gamma = alignment.gamma  # (T, S) posterior matrix

# Clinical Speech SSL consumes gamma matrices
from clinical_speech_ssl import ClinicalSpeechSSL, create_ssl_model

ssl_model = create_ssl_model("base")
losses = ssl_model(waveform, gammas=[gamma], lengths=lengths)
```

**Transition Detection via Gamma Entropy:**

Both packages identify transitions the same way—high-entropy frames in the gamma matrix:

```python
# SoftAlign's GammaEntropy measure
from soft_align.measures import GammaEntropy
ent = GammaEntropy()
result = ent.compute(alignment, phoneme_labels=labels)

# clinical_speech_ssl's transition detection
# Uses same entropy calculation
entropy = -(gamma * gamma.log()).sum(dim=-1)
is_transition = entropy > threshold
```

### 6.3 Recommended Enhancements

#### 6.3.1 Use SoftAlign's PhonemeClassifierBackbone (HIGH PRIORITY)

**The Problem:** CTC models produce **peaky posteriors**—probability mass concentrates on a few frames per phoneme, with blank dominating elsewhere. This is an artifact of CTC training, not genuine phonetic structure.

From SoftAlign docs:
> *"CTC models produce **peaky posteriors**: the training objective rewards concentrating probability mass on a few frames per phoneme, with blank dominating elsewhere. This creates problems:*
> - *Transition sharpness is artificially high (reflects CTC dynamics, not phonetic reality)*
> - *Coarticulation index is underestimated (peaks don't overlap)*
> - *Soft duration concentrates on spike frames instead of the full phoneme extent*
> - *Gamma entropy is falsely low (overconfident)"*

**The Solution:** SoftAlign's `PhonemeClassifierBackbone` trains a lightweight MLP on frozen WavLM features to directly classify each frame's phoneme. The posteriors are naturally smooth and well-calibrated.

```python
# Peaky CTC posteriors
from soft_align import PhonemeCTCBackbone
ctc_backbone = PhonemeCTCBackbone()

# Smooth classifier posteriors
from soft_align import PhonemeClassifierBackbone
classifier_backbone = PhonemeClassifierBackbone(
    classifier_checkpoint="models/classifier/best_classifier.pt"
)
```

**Why this matters for Clinical Speech SSL:**
- **TransitionSharpness** reflects genuine coarticulation, not CTC dynamics
- **Transition identification** is more accurate—we augment/mask real transitions
- **GammaEntropy** is honestly calibrated, not falsely low from peakiness
- **Soft pooling** spreads across the full phoneme extent

**Integration approach:**

```python
# Modify ClinicalSpeechDataset to optionally use classifier backend
class ClinicalSpeechDataset:
    def __init__(self, ..., gamma_backend="classifier", classifier_checkpoint=None):
        if gamma_backend == "classifier":
            self.backbone = PhonemeClassifierBackbone(
                classifier_checkpoint=classifier_checkpoint
            )
        else:
            self.backbone = PhonemeCTCBackbone()
        self.aligner = CTCAligner(self.backbone.get_blank_index())
    
    def _compute_gamma(self, audio, transcript):
        output = self.backbone.process_audio(audio, self.sample_rate)
        target_ids = self.backbone.tokenize_target(transcript)
        alignment = self.aligner.compute_posterior(output.log_probs, target_ids)
        return alignment.gamma
```

**Training the classifier** (from SoftAlign CLI):
```bash
# Generate frame-level labels from MFA
soft-align prepare-labels \
    --textgrid-dir ./mfa_output \
    --audio-dir ./audio \
    --output ./labels

# Train classifier
soft-align train-classifier \
    --audio-dir ./audio \
    --label-dir ./labels/labels \
    --output ./models/classifier \
    --epochs 10
```

#### 6.3.2 SoftAlign Measures as Downstream Targets (MEDIUM PRIORITY)

**Idea:** Use SoftAlign's clinical measures as auxiliary regression targets during downstream fine-tuning. This teaches the SSL encoder to capture information relevant to established clinical metrics.

```python
from soft_align import MeasureExtractor
from soft_align.measures import (
    SoftDuration, TransitionSharpness, GammaEntropy,
    CoarticulationIndex, AlignmentConfidence, PairwiseVariabilityIndex,
)
from clinical_speech_ssl.downstream import TaskConfig, MultiTaskHead

# Extract SoftAlign measures as targets
extractor = MeasureExtractor(backbone, measures=[
    SoftDuration(),
    TransitionSharpness(),
    PairwiseVariabilityIndex(scope="vocalic"),
])

# Define multi-task head with SoftAlign measure targets
tasks = [
    TaskConfig("msd_diagnosis", "multiclass", 5, loss_weight=1.0),
    TaskConfig("transition_sharpness", "regression", 1, loss_weight=0.3),
    TaskConfig("npvi_vocalic", "regression", 1, loss_weight=0.3),
]

head = MultiTaskHead(embed_dim=256, tasks=tasks)
```

This joint training encourages the SSL encoder to learn features predictive of both the primary diagnosis task AND established clinical measures.

#### 6.3.3 UtteranceAnalysisPipeline Preprocessing (MEDIUM PRIORITY)

**The Problem:** Clinical recordings are messy—false starts, multiple attempts, off-target responses.

From SoftAlign docs:
> *"Patients may:*
> - *Start over: "ple- please call stella"*
> - *Try multiple times: "please call stella... please call stella"*
> - *Produce the wrong word*
> - *Abandon mid-word*
> - *Add filler: "um... please call stella"*
> 
> *If you run standard GOP analysis on such recordings without preprocessing, the false start or filler material corrupts the alignment."*

**Solution:** Use SoftAlign's `UtteranceAnalysisPipeline` to:
1. Decode what was actually said (CTC beam search)
2. Align decoded sequence to target (Smith-Waterman)
3. Classify utterance structure (clean, false start, multiple attempts)
4. Isolate valid target region
5. Assess quality and flag unreliable measurements

```python
from soft_align import UtteranceAnalysisPipeline, UtteranceType, QualityTier

class RobustSSLDataset:
    """Preprocess with SoftAlign before SSL feature extraction."""
    
    def __init__(self, ssl_model, soft_align_backbone):
        self.ssl_model = ssl_model
        self.pipeline = UtteranceAnalysisPipeline(soft_align_backbone)
    
    def process(self, audio, transcript, sample_rate):
        # Preprocess with SoftAlign pipeline
        result = self.pipeline.analyze(audio, transcript, sample_rate=sample_rate)
        
        if not result.usable_for_analysis:
            return None  # Skip this sample
        
        # Extract only the valid target region
        if result.target_attempt:
            start_sample = int(result.target_attempt.start_frame * 320)
            end_sample = int(result.target_attempt.end_frame * 320)
            valid_audio = audio[:, start_sample:end_sample]
        else:
            valid_audio = audio
        
        return {
            "audio": valid_audio,
            "utterance_type": result.utterance_type,
            "quality_tier": result.quality_tier,
            "had_false_start": result.utterance_type == UtteranceType.FALSE_START,
        }
```

**Quality tiers** (from SoftAlign):

| Tier | Match Score | PER | Use |
|------|-------------|-----|-----|
| **HIGH** | > 0.85 | < 0.15 | Full analysis |
| **MEDIUM** | 0.5 - 0.85 | < 0.4 | Use with caution |
| **LOW** | 0.2 - 0.5 | any | Flag for review |
| **FAILED** | < 0.2 | any | Skip |

#### 6.3.4 Trajectory Analysis Fallback (MEDIUM PRIORITY)

**The Problem:** For severely disordered speech (groping, repetitions, false starts), even soft alignment can fail because CTC can't find a plausible path through the target phoneme sequence.

**Solution:** SoftAlign's alignment-free trajectory analysis using optimal transport.

From SoftAlign docs:
> *"The **alignment-free trajectory analysis** module sidesteps this entirely: it treats each utterance as a distribution of frames in feature space and compares them using optimal transport (OT)."*

```python
from soft_align.trajectory import (
    TrajectoryExtractor,
    OptimalTransportAnalyzer,
    TrajectoryNormDatabase,
)

class HybridSSLPipeline:
    """Use alignment-based SSL for mild-moderate pathology;
    fall back to trajectory analysis for severe cases."""
    
    def __init__(self, ssl_model, backbone, control_trajectory):
        self.ssl_model = ssl_model
        self.backbone = backbone
        self.aligner = CTCAligner(backbone.get_blank_index())
        self.trajectory_extractor = TrajectoryExtractor(backbone)
        self.ot_analyzer = OptimalTransportAnalyzer(
            regularization=0.05, 
            silence_handling="trim"
        )
        self.control_trajectory = control_trajectory
    
    def analyze(self, audio, transcript, sample_rate):
        # Try alignment-based approach first
        output = self.backbone.process_audio(audio, sample_rate)
        target_ids = self.backbone.tokenize_target(transcript)
        alignment = self.aligner.compute_posterior(output.log_probs, target_ids)
        
        # Check alignment confidence
        confidence = alignment.gamma.max(dim=-1).values.mean().item()
        
        if confidence > 0.5:
            # Good alignment: use SSL features + gamma
            ssl_features = self.ssl_model.encode(audio)
            return {
                "method": "alignment_based",
                "ssl_features": ssl_features,
                "gamma": alignment.gamma,
                "confidence": confidence,
            }
        else:
            # Poor alignment: fall back to trajectory analysis
            patient_traj = self.trajectory_extractor.extract(audio, sample_rate)
            ot_result = self.ot_analyzer.compute(
                patient_traj, 
                self.control_trajectory
            )
            
            return {
                "method": "trajectory",
                "ot_distance": ot_result.distance,
                "monotonicity": ot_result.monotonicity,
                "anomaly_regions": ot_result.anomaly_regions,
                "per_frame_cost": ot_result.per_frame_cost,
            }
```

**OT metrics** (from SoftAlign):
- `distance`: Wasserstein distance between patient and reference
- `monotonicity`: How well temporal ordering is preserved (1.0 = perfect)
- `per_frame_cost`: Cost assigned to each patient frame
- `anomaly_regions`: Detected anomalous segments

#### 6.3.5 GOP Prediction as SSL Objective (MEDIUM PRIORITY)

**Idea:** Add GOP (Goodness of Pronunciation) prediction as a fourth SSL objective.

```python
from soft_align import compute_gop, GOPMode

class GOPPredictionHead(nn.Module):
    """Predict per-phoneme GOP scores from SSL encoder output."""
    
    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, features, gamma):
        """
        Args:
            features: [B, T, D] encoder output
            gamma: List of [T, S] gamma matrices
        Returns:
            gop_predictions: List of [N_phonemes] tensors
        """
        batch_predictions = []
        for b in range(features.shape[0]):
            num_phonemes = (gamma[b].shape[1] - 1) // 2
            phoneme_features = []
            
            for i in range(num_phonemes):
                # Soft mask for phoneme i (odd indices in CTC sequence)
                mask = gamma[b][:, 2*i + 1]
                weights = mask / (mask.sum() + 1e-8)
                phoneme_feat = (weights.unsqueeze(-1) * features[b]).sum(dim=0)
                phoneme_features.append(phoneme_feat)
            
            phoneme_features = torch.stack(phoneme_features)
            predictions = self.projector(phoneme_features).squeeze(-1)
            batch_predictions.append(predictions)
        
        return batch_predictions
```

**Integration with SoftAlign's GOP modes:**

```python
# SoftAlign provides multiple GOP modes
for mode in [GOPMode.SOFT, GOPMode.ALIGNMENT_FREE, GOPMode.SELF_ALIGNED]:
    result = compute_gop(
        output.log_probs, target_ids,
        mode=mode,
        blank_idx=backbone.get_blank_index(),
    )
```

| Mode | Description | Use Case |
|------|-------------|----------|
| `SOFT` | Gamma-weighted log-posteriors | Default; robust |
| `ALIGNMENT_FREE` | Full CTC likelihood ratio | Most robust; no alignment needed |
| `SELF_ALIGNED` | Viterbi-based boundaries | Peaky posteriors |

#### 6.3.6 Multi-Space Dissociation Analysis (LOW PRIORITY)

**Idea:** Compare SSL representations across different encoder layers to identify dissociations characteristic of specific pathology types.

From SoftAlign docs on multi-space OT:
> *"Compare patient in multiple feature spaces for dissociation scoring"*

```python
from soft_align.trajectory import MultiSpaceOTAnalyzer

class SSLMultiSpaceAnalyzer:
    """Analyze dissociations across SSL encoder layers."""
    
    def __init__(self, ssl_model):
        self.ssl_model = ssl_model
        self.ot_analyzer = MultiSpaceOTAnalyzer(
            spaces={"acoustic": {}, "linguistic": {}},
            regularization=0.05,
        )
    
    def analyze_dissociation(self, patient_audio, control_audio):
        # Early layers = more acoustic
        # Late layers = more linguistic/abstract
        patient_features, patient_layers = self.ssl_model.encode(
            patient_audio, return_all_layers=True
        )
        control_features, control_layers = self.ssl_model.encode(
            control_audio, return_all_layers=True
        )
        
        patient = {
            "acoustic": UtteranceTrajectory(features=patient_layers[2]),
            "linguistic": UtteranceTrajectory(features=patient_layers[-2]),
        }
        reference = {
            "acoustic": UtteranceTrajectory(features=control_layers[2]),
            "linguistic": UtteranceTrajectory(features=control_layers[-2]),
        }
        
        result = self.ot_analyzer.compute(patient, reference)
        
        # Interpret dissociation:
        # High acoustic / low linguistic = motor execution (dysarthria)
        # Low acoustic / high linguistic = planning (apraxia)
        return result.dissociation_scores
```

#### 6.3.7 WavLM-Phonet Integration (LOW PRIORITY)

**Idea:** Use SoftAlign's Phonet (phonological feature predictor) for additional feature space or as SSL target.

From SoftAlign docs:
> *"WavLM-Phonet predicts 23 binary phonological features per frame"*

```python
from soft_align.backbones.phonet import PhonetBackbone
from soft_align.training.phonet import FEATURE_NAMES

class PhonologicalSSL:
    def __init__(self, ssl_model, phonet_checkpoint):
        self.ssl_model = ssl_model
        self.phonet = PhonetBackbone(checkpoint=phonet_checkpoint)
    
    def extract_combined(self, audio, sample_rate):
        ssl_features = self.ssl_model.encode(audio)
        phonet_posteriors = self.phonet.predict_features(audio, sample_rate)
        
        return {
            "ssl_features": ssl_features,
            "phonet_features": phonet_posteriors,  # (T, 23)
            "feature_names": FEATURE_NAMES,
        }
```

### 6.4 Unified Pipeline Vision

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Unified Clinical Speech Pipeline                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────────┐                                                  │
│   │   Raw Waveform   │                                                  │
│   └────────┬─────────┘                                                  │
│            │                                                            │
│            ▼                                                            │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │              SoftAlign: UtteranceAnalysisPipeline               │   │
│   │  • Beam decode (what was said?)                                 │   │
│   │  • False start detection                                        │   │
│   │  • Multiple attempt handling                                    │   │
│   │  • Quality tier assignment                                      │   │
│   └────────────────────────────────────────────────────────────────┘   │
│            │                                                            │
│            ▼                                                            │
│   ┌──────────────────┐     ┌──────────────────────────────────────┐    │
│   │ SoftAlign        │     │  Clinical Speech SSL                  │    │
│   │ ┌──────────────┐ │     │  ┌──────────────────────────────────┐│    │
│   │ │PhonemeClassi-│ │     │  │ SSL Pretrained Encoder           ││    │
│   │ │fierBackbone  │ │────▶│  │ (Aug Pred + Mask + Contrastive)  ││    │
│   │ │(smooth gamma)│ │     │  └──────────────────────────────────┘│    │
│   │ └──────────────┘ │     │                                      │    │
│   │                  │     │  Outputs:                            │    │
│   │ ┌──────────────┐ │     │  • Frame embeddings                  │    │
│   │ │ CTCAligner   │ │     │  • Phoneme embeddings (gamma-pooled) │    │
│   │ │ (gamma)      │ │     │  • Utterance embedding               │    │
│   │ └──────────────┘ │     └──────────────────────────────────────┘    │
│   │                  │                    │                            │
│   │ ┌──────────────┐ │                    │                            │
│   │ │ Measure      │ │◀───────────────────┘                            │
│   │ │ Extractor    │ │     (SSL features fed to SoftAlign measures)    │
│   │ │ + GOP        │ │                                                  │
│   │ └──────────────┘ │                                                  │
│   │                  │                                                  │
│   │ ┌──────────────┐ │                                                  │
│   │ │ Trajectory   │ │     (Fallback for severe pathology)             │
│   │ │ Analysis     │ │                                                  │
│   │ │ (OT-based)   │ │                                                  │
│   │ └──────────────┘ │                                                  │
│   │                  │                                                  │
│   │ ┌──────────────┐ │                                                  │
│   │ │ DDK Analyzer │ │     (Diadochokinetic tasks)                     │
│   │ └──────────────┘ │                                                  │
│   │                  │                                                  │
│   │ ┌──────────────┐ │                                                  │
│   │ │ WavLM-Phonet │ │     (Phonological feature analysis)             │
│   │ └──────────────┘ │                                                  │
│   └──────────────────┘                                                  │
│            │                                                            │
│            ▼                                                            │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                    Downstream Tasks                           │     │
│   │  • MSD Diagnosis (classification)                             │     │
│   │  • Severity Scoring (regression)                              │     │
│   │  • Phoneme-level Error Detection (sequence labeling)          │     │
│   │  • Longitudinal Tracking (trajectory analysis)                │     │
│   │  • DDK Analysis (rate, regularity, breakdown)                 │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.5 Integration Checklist for Claude Code

| Task | Priority | Complexity | Description |
|------|----------|------------|-------------|
| **PhonemeClassifierBackbone integration** | HIGH | Medium | Use classifier posteriors for smoother gamma in dataset |
| **SoftAlign gamma computation in dataset** | HIGH | Low | Add `compute_gamma()` method to `ClinicalSpeechDataset` |
| **Transition detection alignment** | HIGH | Low | Ensure `RegionAugmentor.identify_transitions()` matches SoftAlign's GammaEntropy logic |
| **UtteranceAnalysisPipeline preprocessing** | MEDIUM | Medium | Filter/segment audio before SSL extraction |
| **SoftAlign measures as downstream targets** | MEDIUM | Medium | Use TransitionSharpness, nPVI, etc. as auxiliary regression targets |
| **Trajectory fallback** | MEDIUM | Medium | Use OT when alignment confidence is low |
| **GOP prediction head** | MEDIUM | Medium | Add 4th SSL objective for GOP prediction |
| **DDK integration** | LOW | Medium | Combine SSL features with DDK analysis |
| **Multi-space dissociation** | LOW | Medium | Compare across SSL encoder layers |
| **Phonet integration** | LOW | Low | Add phonological features as auxiliary |

---

## 7. Future Directions

### 7.1 Immediate Next Steps

1. **Validate on real data:** Train on healthy speakers from the Speech AI corpus, evaluate downstream on MSD classification.

2. **Ablation studies:** Run the full ablation suite (configs/ablations.yaml) to determine:
   - Which SSL objectives matter most
   - Optimal transition bias
   - Best frontend/encoder combination
   - Loss weight sensitivity

3. **SoftAlign integration:** 
   - Train PhonemeClassifierBackbone on Speech AI data
   - Replace CTC gamma with classifier gamma
   - Add SoftAlign measures to downstream evaluation

### 7.2 Medium-Term Enhancements

1. **Curriculum learning:** Start with easier augmentations (large magnitude, easy to detect), progressively make them subtler.

2. **Phoneme-stratified augmentation:** Different augmentation profiles for vowels vs. consonants vs. specific phonemes.

3. **Prototype-deviation learning:** Learn healthy-speech prototypes per phoneme, represent clinical speech as deviation from expected production.

4. **Cross-recording consistency:** Same patient/session = positive pairs for contrastive learning; track progression via embedding drift.

### 7.3 Long-Term Research Directions

1. **Multi-task pretraining:** Jointly pretrain with ASR (CTC), speaker verification, and clinical augmentation prediction.

2. **Articulatory feature prediction:** Predict SPARC-style articulatory features as additional SSL objective.

3. **Phonological feature space:** Use WavLM-Phonet posteriors as targets for SSL—predict the phonological feature decomposition.

4. **Bayesian extensions:** Uncertainty quantification over learned representations for clinical reliability.

---

## Appendix: Configuration Reference

### A.1 ClinicalSpeechSSLConfig

```python
@dataclass
class ClinicalSpeechSSLConfig:
    # Input
    input_type: Literal["waveform", "spectrogram"] = "waveform"
    sample_rate: int = 16000
    
    # Frontend
    frontend_type: Literal[
        "cnn_small", "cnn_base", "cnn_large",
        "patch_tall_narrow", "patch_vit"
    ] = "cnn_base"
    frontend_dropout: float = 0.0
    
    # Encoder
    encoder_type: Literal[
        "transformer_small", "transformer_base", "transformer_large",
        "conformer_small", "conformer_medium", "conformer_large"
    ] = "conformer_medium"
    embed_dim: int = 256
    encoder_dropout: float = 0.1
    
    # SSL Objectives
    use_augmentation_prediction: bool = True
    use_masked_reconstruction: bool = True
    use_contrastive: bool = True
    use_gop_prediction: bool = False  # Requires SoftAlign
    
    # Augmentation Prediction
    num_augmentation_types: int = 4
    augmentation_per_region: bool = True
    predict_magnitude: bool = True
    
    # Masked Reconstruction
    mask_prob: float = 0.15
    mask_span_length: int = 10
    transition_bias: float = 2.0
    
    # Contrastive
    contrastive_projection_dim: int = 256
    contrastive_temperature: float = 0.07
    
    # Loss Weights
    aug_loss_weight: float = 1.0
    mask_loss_weight: float = 1.0
    contrastive_loss_weight: float = 0.5
    gop_loss_weight: float = 0.5
    
    # Spectrogram-specific
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    patch_frames: int = 4
    patch_stride: int = 2
    vit_patch_size: Tuple[int, int] = (16, 16)
```

### A.2 TrainingConfig

```python
@dataclass
class TrainingConfig:
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Batching
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 5
    keep_n_checkpoints: int = 3
    
    # Logging
    log_every_n_steps: int = 100
    eval_every_n_epochs: int = 1
    
    # Early Stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"
    
    # Hardware
    use_amp: bool = True
    device: str = "cuda"
```

### A.3 SoftAlign Integration Config

```python
@dataclass
class SoftAlignConfig:
    """Configuration for SoftAlign integration."""
    
    # Gamma computation
    gamma_backend: Literal["ctc", "classifier"] = "classifier"
    classifier_checkpoint: Optional[str] = None
    
    # Preprocessing
    use_utterance_pipeline: bool = True
    min_quality_tier: str = "medium"  # "high", "medium", "low"
    
    # Trajectory fallback
    use_trajectory_fallback: bool = True
    alignment_confidence_threshold: float = 0.5
    ot_regularization: float = 0.05
    
    # Measures
    compute_soft_align_measures: bool = True
    measures: List[str] = field(default_factory=lambda: [
        "soft_duration",
        "transition_sharpness", 
        "gamma_entropy",
        "npvi_vocalic",
        "alignment_confidence",
    ])
```

---

*Document version: 1.0*  
*Last updated: April 2026*  
*Authors: Speech AI Lab, Mayo Clinic*
