#!/usr/bin/env python
"""
Example script demonstrating SSL pretraining and downstream fine-tuning.

This script shows the complete workflow:
1. Create synthetic data for demonstration
2. SSL pretraining with all objectives
3. Downstream fine-tuning for classification
4. Evaluation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse

# Import from our package
from clinical_speech_ssl import (
    ClinicalSpeechSSL,
    ClinicalSpeechSSLConfig,
    create_ssl_model,
    DownstreamClassifier,
)
from clinical_speech_ssl.downstream import DownstreamModel
from clinical_speech_ssl.data import SSLCollator, SpeechSample


class SyntheticClinicalSpeechDataset(Dataset):
    """
    Synthetic dataset for demonstration purposes.
    
    In practice, you would use ClinicalSpeechDataset with real data.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        sample_rate: int = 16000,
        duration_sec: float = 2.0,
        num_phonemes: int = 10,
        num_classes: int = 3,
    ):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.seq_length = int(duration_sec * sample_rate)
        self.num_phonemes = num_phonemes
        self.num_classes = num_classes
        
        # Pre-generate data
        self.waveforms = [
            torch.randn(self.seq_length) * 0.1
            for _ in range(num_samples)
        ]
        
        # Generate CTC gamma matrices
        num_frames = self.seq_length // 320  # Approximate frame count
        self.gammas = [
            torch.softmax(torch.randn(num_frames, num_phonemes), dim=-1)
            for _ in range(num_samples)
        ]
        
        # Generate labels for downstream task
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
        # Patient IDs
        self.patient_ids = [f"patient_{i:03d}" for i in range(num_samples)]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return SpeechSample(
            waveform=self.waveforms[idx],
            sample_rate=self.sample_rate,
            gamma=self.gammas[idx],
            patient_id=self.patient_ids[idx],
            clinical_labels={"diagnosis": self.labels[idx].item()},
        )


def create_dataloaders(dataset, batch_size=8, train_ratio=0.8):
    """Create train and validation dataloaders."""
    n_train = int(len(dataset) * train_ratio)
    n_val = len(dataset) - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    collator = SSLCollator()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    
    return train_loader, val_loader


def ssl_pretrain(model, train_loader, val_loader, num_epochs=5, device="cpu"):
    """Simple SSL pretraining loop."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print("\n" + "="*60)
    print("SSL PRETRAINING")
    print("="*60)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            waveforms = batch['waveforms'].to(device)
            lengths = batch['lengths'].to(device)
            gammas = batch['gammas']
            
            optimizer.zero_grad()
            losses = model(waveforms, gammas, lengths)
            loss = losses['total_loss']
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                waveforms = batch['waveforms'].to(device)
                lengths = batch['lengths'].to(device)
                gammas = batch['gammas']
                
                losses = model(waveforms, gammas, lengths)
                val_loss += losses['total_loss'].item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    return model


def downstream_finetune(
    encoder,
    train_loader,
    val_loader,
    num_classes=3,
    num_epochs=5,
    freeze_encoder=True,
    device="cpu",
):
    """Downstream fine-tuning for classification."""
    
    print("\n" + "="*60)
    print(f"DOWNSTREAM FINE-TUNING ({'Frozen' if freeze_encoder else 'Full'})")
    print("="*60)
    
    # Create classification head
    classifier = DownstreamClassifier(
        embed_dim=encoder.config.embed_dim,
        num_classes=num_classes,
        hidden_dims=[64],
        pooling="mean",
        task_type="multiclass",
    )
    
    # Create downstream model
    model = DownstreamModel(
        encoder=encoder,
        classifier=classifier,
        freeze_encoder=freeze_encoder,
    )
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4 if freeze_encoder else 1e-5,
    )
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            waveforms = batch['waveforms'].to(device)
            lengths = batch['lengths'].to(device)
            labels = batch['clinical_labels']['diagnosis'].to(device)
            
            optimizer.zero_grad()
            result = model.compute_loss(waveforms, labels, lengths)
            loss = result['loss']
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Compute accuracy
            with torch.no_grad():
                logits = model(waveforms, lengths)
                preds = logits.argmax(dim=-1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.shape[0]
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                waveforms = batch['waveforms'].to(device)
                lengths = batch['lengths'].to(device)
                labels = batch['clinical_labels']['diagnosis'].to(device)
                
                result = model.compute_loss(waveforms, labels, lengths)
                val_loss += result['loss'].item()
                
                logits = model(waveforms, lengths)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.shape[0]
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2%} - "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2%}"
        )
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Clinical Speech SSL Example")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ssl_epochs", type=int, default=3)
    parser.add_argument("--downstream_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()
    
    print("="*60)
    print("CLINICAL SPEECH SSL - DEMONSTRATION")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Synthetic samples: {args.num_samples}")
    
    # Create synthetic dataset
    print("\nCreating synthetic dataset...")
    dataset = SyntheticClinicalSpeechDataset(
        num_samples=args.num_samples,
        num_classes=3,
    )
    
    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size=args.batch_size,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create SSL model
    print("\nCreating SSL model...")
    config = ClinicalSpeechSSLConfig(
        input_type="waveform",
        frontend_type="cnn_small",
        encoder_type="transformer_small",
        embed_dim=128,
        use_augmentation_prediction=True,
        use_masked_reconstruction=True,
        use_contrastive=True,
        transition_bias=2.0,
    )
    model = ClinicalSpeechSSL(config)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # SSL Pretraining
    model = ssl_pretrain(
        model,
        train_loader,
        val_loader,
        num_epochs=args.ssl_epochs,
        device=args.device,
    )
    
    # Downstream fine-tuning (frozen encoder)
    downstream_model_frozen = downstream_finetune(
        model,
        train_loader,
        val_loader,
        num_classes=3,
        num_epochs=args.downstream_epochs,
        freeze_encoder=True,
        device=args.device,
    )
    
    # Downstream fine-tuning (full)
    downstream_model_full = downstream_finetune(
        model,
        train_loader,
        val_loader,
        num_classes=3,
        num_epochs=args.downstream_epochs,
        freeze_encoder=False,
        device=args.device,
    )
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("\nThis demonstration used synthetic data.")
    print("For real experiments, use ClinicalSpeechDataset with your data.")


if __name__ == "__main__":
    main()
