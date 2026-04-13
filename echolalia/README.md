# Clinical Speech SSL

**Transition-Aware Multi-Objective Self-Supervised Learning for Clinical Speech Analysis**

A modular PyTorch framework for self-supervised learning on clinical speech recordings, designed to capture clinically-relevant acoustic variations while being robust to recording conditions.

## Overview

Standard speech SSL models (wav2vec 2.0, HuBERT, WavLM) learn invariance to acoustic perturbations—but in clinical speech, those perturbations *are* the diagnostic signal. This package inverts the paradigm: instead of learning to ignore augmentations, we make detecting them a primary learning objective.

### Key Features

- **Multiple input modalities**: Raw waveform (CNN or **pretrained WavLM**), or spectrogram patches (tall-narrow or ViT-style)
- **Multiple encoder architectures**: Transformer and Conformer variants
- **Four complementary SSL objectives**:
  - **Augmentation Prediction**: Detect 5 clinical-relevant perturbations (pitch, timing, formants, tremor, **repetition/echolalia**)
  - **Masked Reconstruction**: Predict masked frames with transition bias
  - **Contrastive Learning**: Invariance to recording conditions via safe augmentations
  - **GOP Prediction** (optional): Per-phoneme Goodness of Pronunciation supervision
- **SoftAlign integration**: Offline preprocessing pipeline using `soft_align` for smooth gamma matrices, GOP targets, quality tiers, and clinical measures
- **Transition-focused learning**: Bias toward phoneme boundaries using CTC soft alignments
- **Downstream fine-tuning**: Classification, regression, and multi-task learning

## Installation

```bash
# soft_align must be installed locally first (not on PyPI):
git clone https://github.com/HugoBothaMD/soft_align.git
cd soft_align && pip install -e . && cd ..

# Then install echolalia in development mode
git clone https://github.com/HugoBothaMD/echolalia.git
cd echolalia/echolalia
pip install -e ".[dev]"
```

The package depends on `transformers` (for WavLM), `soft_align` (gamma + GOP + measures), `torch`, `torchaudio`, and `pyyaml` among others.

## Quick Start

### 1. Create an SSL Model

```python
from clinical_speech_ssl import ClinicalSpeechSSL, ClinicalSpeechSSLConfig, create_ssl_model

# Using a preset
model = create_ssl_model("base")

# Or with custom configuration
config = ClinicalSpeechSSLConfig(
    input_type="waveform",
    encoder_type="conformer_medium",
    embed_dim=256,
    use_augmentation_prediction=True,
    use_masked_reconstruction=True,
    use_contrastive=True,
    transition_bias=2.0,
)
model = ClinicalSpeechSSL(config)
```

### 2. Prepare Your Data

The dataset expects:
- Audio files (`.wav`, `.flac`, etc.)
- CTC alignment gamma matrices (`.pt` or `.npy`)

```python
from clinical_speech_ssl.data import ClinicalSpeechDataset, create_dataloaders

# From directory structure
dataset = ClinicalSpeechDataset(
    data_root="/path/to/data",  # Contains audio/ and alignments/ subdirs
    sample_rate=16000,
)

# Or from manifest file
dataset = ClinicalSpeechDataset(
    manifest_path="/path/to/manifest.json",
)

# Create train/val/test loaders
train_loader, val_loader, test_loader = create_dataloaders(
    dataset,
    batch_size=32,
    train_ratio=0.8,
    val_ratio=0.1,
)
```

### 3. SSL Pretraining

```python
from clinical_speech_ssl.training import SSLTrainer, TrainingConfig

training_config = TrainingConfig(
    learning_rate=1e-4,
    max_epochs=100,
    warmup_epochs=5,
    checkpoint_dir="./checkpoints/ssl_pretrain",
)

trainer = SSLTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=training_config,
)

history = trainer.train()
```

### 4. Downstream Fine-tuning

```python
from clinical_speech_ssl.downstream import DownstreamClassifier, DownstreamModel
from clinical_speech_ssl.training import DownstreamTrainer

# Load pretrained encoder
checkpoint = torch.load("./checkpoints/ssl_pretrain/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Create classification head
classifier = DownstreamClassifier(
    embed_dim=256,
    num_classes=5,
    hidden_dims=[128],
    pooling="attention",
    task_type="multiclass",
)

# Create downstream model
downstream_model = DownstreamModel(
    encoder=model,
    classifier=classifier,
    freeze_encoder=False,  # Set True for linear probing
)

# Train
trainer = DownstreamTrainer(
    model=downstream_model,
    train_loader=downstream_train_loader,
    val_loader=downstream_val_loader,
    config=TrainingConfig(learning_rate=1e-5, max_epochs=50),
)
history = trainer.train()
```

### 5. Extract Representations

```python
# Get utterance-level representations
representations = model.get_representations(
    waveform,
    lengths,
    layer=-1,  # Last layer
    pooling="mean",  # or "attention", "first", "last", "none"
)

# Get frame-level representations
frame_reps = model.encode(waveform, lengths)
```

## Recommended Workflow

The full pipeline initializes from pretrained WavLM, pretrains the conformer + SSL heads on a large clean corpus (LibriSpeech), optionally adapts to pathological speech, and finally fine-tunes on labeled clinical data.

```
WavLM (pretrained, frozen)
  ↓
Conformer encoder (random init)
  ↓
SSL heads: aug-pred (5 types) + masked-recon + contrastive + GOP
  ↓
Stage 1: Pretrain on LibriSpeech (clean, healthy, ~460h)
  ↓
Stage 2 (optional): Continue on unlabeled clinical speech (domain adapt)
  ↓
Stage 3: Fine-tune downstream head on labeled clinical data
```

### Stage 0 — Preprocess LibriSpeech with SoftAlign

The preprocessing script runs your trained `PhonemeClassifierBackbone` over each audio file and produces gamma matrices (smooth, not peaky), per-phoneme GOP targets, utterance-level clinical measure vectors, and quality tiers — all in one backbone pass.

```bash
# 1. Build a transcripts.json from LibriSpeech's .trans.txt files
python -m scripts.librispeech_transcripts \
    --librispeech-root /data/LibriSpeech/train-clean-100 \
    --output /data/LibriSpeech/train-clean-100.transcripts.json

# 2. Run preprocessing
python -m scripts.compute_alignments \
    --audio-dir /data/LibriSpeech/train-clean-100 \
    --transcripts /data/LibriSpeech/train-clean-100.transcripts.json \
    --output-dir /data/LibriSpeech/train-clean-100.processed \
    --backbone phoneme_classifier \
    --classifier-checkpoint /path/to/your/best_classifier.pt \
    --compute-gop --compute-measures --compute-quality
```

The script writes `manifest.json`, `alignments/*.pt`, `gop/*.pt`, and `measures/*.npy` to the output dir.

### Stage 1 — Pretrain on LibriSpeech

Point the training config's `data.manifest_path` at the processed manifest and run:

```bash
python -m clinical_speech_ssl.train --config configs/ssl_pretraining.yaml
```

Key config fields for stage 1:
```yaml
model:
  frontend_type: "wavlm_base"   # or "wavlm_large"
  freeze_frontend: true          # freeze WavLM, train conformer+heads
  use_gop_prediction: true       # 4th objective with LibriSpeech-derived targets
  num_augmentation_types: 5      # includes REPETITION

data:
  manifest_path: "/data/LibriSpeech/train-clean-100.processed/manifest.json"
```

### Stage 2 (optional) — Domain-adapt on unlabeled clinical speech

Run the same preprocessing on your unlabeled clinical recordings, then continue training from the stage-1 checkpoint with the WavLM unfrozen:

```python
trainer.load_checkpoint("checkpoints/stage1/best_model.pt")
trainer.model.unfreeze_frontend(num_layers=4)  # unfreeze top 4 WavLM layers
trainer.train()
```

You can also filter out low-quality recordings:
```python
dataset = ClinicalSpeechDataset(
    manifest_path="...",
    min_quality_tier=2,           # MEDIUM and above (3=HIGH, 2=MED, 1=LOW, 0=FAILED)
    load_gop_targets=True,
    load_measure_targets=True,
)
```

### Stage 3 — Fine-tune for downstream tasks

Standard supervised fine-tuning on your labeled clinical task (see [Quick Start step 4](#4-downstream-fine-tuning)).

## Architecture

### Model Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ClinicalSpeechSSL                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐   │
│  │   Frontend  │ → │   Encoder   │ → │   SSL Heads     │   │
│  └─────────────┘   └─────────────┘   └─────────────────┘   │
│        │                 │                   │              │
│    Waveform CNN     Transformer       • Aug Prediction     │
│    or               or                • Masked Recon       │
│    Spectrogram      Conformer         • Contrastive        │
│    Patcher                                                  │
└─────────────────────────────────────────────────────────────┘
```

### Frontend Options

| Frontend | Description | Best For |
|----------|-------------|----------|
| `cnn_small` | 4-layer CNN, 256 dim | Fast experimentation |
| `cnn_base` | 7-layer CNN, 512 dim | General use |
| `cnn_large` | 7-layer CNN, 1024 dim | Best performance from scratch |
| `wavlm_base` | Pretrained WavLM-base-plus (94M, 768 dim) | **Recommended starting point** |
| `wavlm_large` | Pretrained WavLM-large (316M, 1024 dim) | Best results, larger memory |
| `patch_tall_narrow` | Full freq, few time steps | Preserves spectral structure |
| `patch_vit` | Square/rectangular patches | ViT-style, more general |

**WavLM frontends** wrap pretrained `transformers.WavLMModel` and project to `embed_dim`. By default WavLM is frozen so only the conformer + SSL heads train, which is fast and data-efficient. Use `freeze_frontend: false` (or call `model.unfreeze_frontend(num_layers=N)`) to unfreeze for staged domain adaptation.

```python
config = ClinicalSpeechSSLConfig(
    frontend_type="wavlm_base",
    wavlm_model_name="microsoft/wavlm-base-plus",
    freeze_frontend=True,  # frozen during stage 1
    encoder_type="conformer_medium",
    embed_dim=256,
)
model = ClinicalSpeechSSL(config)
# Later, for clinical adaptation:
model.unfreeze_frontend(num_layers=4)  # unfreeze top 4 WavLM layers
```

### Encoder Options

| Encoder | Layers | Heads | Dim | Description |
|---------|--------|-------|-----|-------------|
| `transformer_small` | 4 | 4 | 256 | Minimal, fast |
| `transformer_base` | 12 | 12 | 768 | Standard Transformer |
| `conformer_small` | 4 | 4 | 256 | Minimal Conformer |
| `conformer_medium` | 12 | 4 | 256 | Recommended default |
| `conformer_large` | 17 | 8 | 512 | Maximum capacity |

### SSL Objectives

#### 1. Augmentation Prediction
Applies five clinical-relevant augmentations to phoneme regions and trains the model to detect and quantify them per region.

| Augmentation | Mimics | Clinical correlate |
|--------------|--------|---------------------|
| `TIME_STRETCH` | Slowed/scanning speech | Bradykinesia, rate disorders |
| `PITCH_SHIFT` | Monotone, pitch breaks | Prosodic disorders |
| `FORMANT_SHIFT` | Vowel distortion, undershoot | Articulatory imprecision |
| `AMPLITUDE_MOD` | Tremor, flutter | Laryngeal instability |
| `REPETITION` | Stuttering, echolalia | Repetitive speech disorders |

```python
config = ClinicalSpeechSSLConfig(
    use_augmentation_prediction=True,
    num_augmentation_types=5,
    predict_magnitude=True,  # Regress magnitude (e.g., repeat count, semitones)
    augmentation_per_region=True,  # Per-region predictions
)
```

#### 2. Masked Reconstruction
Masks frames with bias toward phoneme transitions and reconstructs them, encouraging learning of coarticulation patterns.

```python
config = ClinicalSpeechSSLConfig(
    use_masked_reconstruction=True,
    mask_prob=0.15,
    mask_span_length=10,
    transition_bias=2.0,  # 2x more likely to mask transitions
)
```

#### 3. Contrastive Learning
Creates positive pairs using "safe" augmentations (noise, gain, filtering) that don't affect clinical signal, learning recording-condition invariance.

```python
config = ClinicalSpeechSSLConfig(
    use_contrastive=True,
    contrastive_projection_dim=256,
    contrastive_temperature=0.07,
)
```

#### 4. GOP Prediction (optional, requires SoftAlign preprocessing)
Predicts per-phoneme Goodness of Pronunciation scores (`soft_align.compute_gop` with `GOPMode.SOFT`) from the encoder. Targets are computed offline; the head soft-pools encoder features by gamma and applies an MLP per phoneme.

```python
config = ClinicalSpeechSSLConfig(
    use_gop_prediction=True,
    gop_loss_weight=0.5,
)
# Dataset must be configured with load_gop_targets=True
```

## Data Format

### Directory Structure

```
data_root/
├── audio/
│   ├── patient001_session01_utt01.wav
│   ├── patient001_session01_utt02.wav
│   └── ...
├── alignments/
│   ├── patient001_session01_utt01.pt  # CTC gamma matrix
│   └── ...
└── metadata.json  # Optional: clinical labels
```

### Manifest Format

```json
{
  "metadata": {"version": "1.0"},
  "samples": [
    {
      "audio_path": "/path/to/audio.wav",
      "gamma_path": "/path/to/gamma.pt",
      "patient_id": "patient001",
      "session_id": "session01",
      "clinical_labels": {
        "diagnosis": 1,
        "severity": 0.75
      }
    }
  ]
}
```

### CTC Gamma Matrix

The gamma matrix contains soft phoneme alignments from a CTC-based aligner:
- Shape: `[T_frames, num_phonemes]`
- Each row sums to 1 (posterior probability over phonemes)
- Higher entropy at rows indicates transitions

## Experiments

### Running Ablation Studies

```bash
python -m clinical_speech_ssl.run_experiments \
    --config configs/ablations.yaml \
    --output_dir ./experiments
```

### Configuration Files

See `configs/` for example configurations:
- `ssl_pretraining.yaml`: Full SSL pretraining setup
- `downstream_classification.yaml`: Classification fine-tuning
- `ablations.yaml`: Systematic ablation studies

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_ssl_model.py -v

# Run with coverage
pytest tests/ --cov=clinical_speech_ssl --cov-report=html
```

## API Reference

### Main Classes

| Class | Description |
|-------|-------------|
| `ClinicalSpeechSSL` | Main SSL model |
| `ClinicalSpeechSSLConfig` | Model configuration |
| `ClinicalSpeechDataset` | Data loading |
| `SSLTrainer` | SSL pretraining |
| `DownstreamTrainer` | Fine-tuning |
| `DownstreamClassifier` | Classification head |
| `DownstreamRegressor` | Regression head |
| `MultiTaskHead` | Multi-task head |

### Model Presets

```python
# Available presets
model = create_ssl_model("tiny")          # Minimal for testing
model = create_ssl_model("small")         # Fast experimentation
model = create_ssl_model("base")          # Recommended (from-scratch CNN)
model = create_ssl_model("large")         # Maximum capacity (from-scratch CNN)
model = create_ssl_model("wavlm_base")    # Pretrained WavLM-base + conformer (recommended)
model = create_ssl_model("wavlm_large")   # Pretrained WavLM-large + conformer (best results)
```

## Citation

If you use this code, please cite:

```bibtex
@software{clinical_speech_ssl,
  title={Clinical Speech SSL: Transition-Aware Multi-Objective Self-Supervised Learning},
  author={Mayo Clinic Speech AI Lab},
  year={2024},
  url={https://github.com/mayo-speech-ai/clinical-speech-ssl}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
