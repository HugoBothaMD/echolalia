"""
Training entry point for Clinical Speech SSL.

Usage:
    python -m clinical_speech_ssl.train --config configs/ssl_pretraining.yaml
    python -m clinical_speech_ssl.train --config configs/ssl_pretraining.yaml --dry-run
    python -m clinical_speech_ssl.train --config configs/ssl_pretraining.yaml --resume checkpoints/latest.pt
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from clinical_speech_ssl.models.ssl_model import ClinicalSpeechSSL, ClinicalSpeechSSLConfig
from clinical_speech_ssl.data.dataset import ClinicalSpeechDataset, SSLCollator, create_dataloaders
from clinical_speech_ssl.training.trainer import SSLTrainer, TrainingConfig


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model_config(raw: Dict[str, Any]) -> ClinicalSpeechSSLConfig:
    """Build ClinicalSpeechSSLConfig from the 'model' section of the YAML."""
    model_cfg = raw.get("model", {})
    spec_cfg = raw.get("spectrogram", {})

    kwargs = {}
    # Map YAML keys to dataclass fields (names match 1:1)
    for key in (
        "input_type", "frontend_type", "encoder_type", "embed_dim",
        "frontend_dropout", "encoder_dropout",
        "use_augmentation_prediction", "use_masked_reconstruction", "use_contrastive",
        "num_augmentation_types", "augmentation_per_region", "predict_magnitude",
        "mask_prob", "mask_span_length", "transition_bias",
        "contrastive_projection_dim", "contrastive_temperature",
        "aug_loss_weight", "mask_loss_weight", "contrastive_loss_weight",
    ):
        if key in model_cfg:
            kwargs[key] = model_cfg[key]

    # Spectrogram settings
    for key in ("n_mels", "n_fft", "hop_length", "patch_frames", "patch_stride"):
        if key in spec_cfg:
            kwargs[key] = spec_cfg[key]
    if "vit_patch_size" in spec_cfg:
        kwargs["vit_patch_size"] = tuple(spec_cfg["vit_patch_size"])

    return ClinicalSpeechSSLConfig(**kwargs)


def build_training_config(raw: Dict[str, Any]) -> TrainingConfig:
    """Build TrainingConfig from the 'training' section of the YAML."""
    t = raw.get("training", {})
    requested_device = t.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    use_amp = t.get("use_amp", False)

    # Fall back to CPU if CUDA was requested but unavailable
    if "cuda" in requested_device and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        requested_device = "cpu"
        use_amp = False

    return TrainingConfig(
        num_epochs=t.get("max_epochs", 100),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        max_grad_norm=t.get("max_grad_norm", 1.0),
        log_every_n_steps=t.get("log_every_n_steps", 100),
        eval_every_n_epochs=t.get("eval_every_n_epochs", 1),
        save_every_n_epochs=t.get("save_every_n_epochs", 10),
        save_best=True,
        checkpoint_dir=t.get("checkpoint_dir", "checkpoints"),
        early_stopping_patience=t.get("early_stopping_patience"),
        early_stopping_metric=t.get("early_stopping_metric", "val_total_loss"),
        early_stopping_mode=t.get("early_stopping_mode", "min"),
        use_amp=use_amp,
        device=requested_device,
    )


def build_optimizer_and_scheduler(
    model: ClinicalSpeechSSL,
    raw: Dict[str, Any],
    steps_per_epoch: int,
) -> tuple:
    """Build optimizer and cosine-warmup scheduler from config."""
    t = raw.get("training", {})
    lr = t.get("learning_rate", 1e-4)
    weight_decay = t.get("weight_decay", 0.01)
    warmup_epochs = t.get("warmup_epochs", 5)
    min_lr = t.get("min_lr", 1e-6)
    max_epochs = t.get("max_epochs", 100)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = max_epochs * steps_per_epoch

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / lr, cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def dry_run(model: ClinicalSpeechSSL, model_config: ClinicalSpeechSSLConfig):
    """Run one forward/backward step on synthetic data to verify the pipeline.

    Uses a tiny model override to keep the dry run fast and avoid
    architecture-specific edge cases on short sequences.
    """
    from clinical_speech_ssl.models.ssl_model import create_ssl_model
    model = create_ssl_model("tiny")
    device = torch.device("cpu")
    model.to(device)
    print("Dry run uses 'tiny' preset on CPU for speed.")
    B, T = 2, 16000  # 2 samples, 1 second each
    waveforms = torch.randn(B, T, device=device)
    lengths = torch.tensor([T, T], device=device)

    # Synthetic gamma: T_frames = T // 320, P = 7 (CTC sequence length)
    T_frames = T // 320
    P = 7
    gammas = [torch.softmax(torch.randn(T_frames, P, device=device), dim=-1) for _ in range(B)]

    model.train()
    losses = model(waveforms, gammas, lengths)
    total_loss = losses["total_loss"]
    total_loss.backward()

    # Check gradients exist
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())

    print(f"Dry run results:")
    for k, v in sorted(losses.items()):
        print(f"  {k}: {v.item():.6f}")
    print(f"  gradients_nonzero: {has_grad}")

    if not has_grad:
        print("WARNING: No gradients detected — check model wiring.")
        sys.exit(1)

    print("Dry run passed.")


def main():
    parser = argparse.ArgumentParser(description="Clinical Speech SSL Pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--dry-run", action="store_true", help="Run one step on synthetic data and exit")
    parser.add_argument("--output-dir", type=str, default=None, help="Override checkpoint dir")
    args = parser.parse_args()

    # Load config
    raw_config = load_config(args.config)
    model_config = build_model_config(raw_config)
    train_config = build_training_config(raw_config)

    if args.output_dir:
        train_config.checkpoint_dir = args.output_dir

    # Build model
    model = ClinicalSpeechSSL(model_config)
    model.to(train_config.device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_config.encoder_type} | {param_count / 1e6:.1f}M parameters")

    # Dry run mode
    if args.dry_run:
        dry_run(model, model_config)
        return

    # Build dataset
    data_cfg = raw_config.get("data", {})
    train_cfg = raw_config.get("training", {})

    dataset = ClinicalSpeechDataset(
        data_root=data_cfg.get("data_root"),
        manifest_path=data_cfg.get("manifest_path"),
        sample_rate=data_cfg.get("sample_rate", 16000),
        max_length_sec=data_cfg.get("max_length_sec", 10.0),
        min_length_sec=data_cfg.get("min_length_sec", 0.5),
    )

    print(f"Dataset: {len(dataset)} samples")

    train_loader, val_loader, _ = create_dataloaders(
        dataset,
        batch_size=train_cfg.get("batch_size", 32),
        train_ratio=data_cfg.get("train_ratio", 0.8),
        val_ratio=data_cfg.get("val_ratio", 0.1),
        num_workers=train_cfg.get("num_workers", 4),
        seed=train_cfg.get("seed", 42),
    )

    # Build optimizer + scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, raw_config, steps_per_epoch=len(train_loader),
    )

    # Build trainer
    trainer = SSLTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=train_config,
    )

    # Resume
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    history = trainer.train()

    print("Training complete.")
    if history["val"]:
        final_val = history["val"][-1]
        print(f"Final validation: {final_val}")


if __name__ == "__main__":
    main()
