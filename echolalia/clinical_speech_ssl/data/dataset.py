"""
Dataset classes for clinical speech SSL.

Handles loading of audio files, CTC alignment gamma matrices,
and optional phoneme labels for downstream tasks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np


@dataclass
class SpeechSample:
    """A single speech sample with metadata."""
    waveform: torch.Tensor  # [C, T] or [T]
    sample_rate: int
    gamma: torch.Tensor  # [T_frames, num_phonemes] CTC alignment posteriors
    phoneme_labels: Optional[List[str]] = None  # Phoneme sequence
    patient_id: Optional[str] = None
    session_id: Optional[str] = None
    utterance_id: Optional[str] = None

    # Optional downstream labels
    clinical_labels: Optional[Dict[str, Union[int, float]]] = None

    # Phase 2: SoftAlign integration targets
    gop_targets: Optional[torch.Tensor] = None  # [num_phonemes] per-phoneme GOP scores
    measure_targets: Optional[torch.Tensor] = None  # [D] utterance-level measure vector


class ClinicalSpeechDataset(Dataset):
    """
    Dataset for clinical speech recordings with CTC alignments.
    
    Expected directory structure:
        data_root/
            audio/
                patient001_session01_utt01.wav
                patient001_session01_utt02.wav
                ...
            alignments/
                patient001_session01_utt01.pt  # gamma matrix
                patient001_session01_utt02.pt
                ...
            metadata.json  # Optional: clinical labels, phoneme info
    
    Or provide a manifest file listing all samples.
    """
    
    # Waveform CNN default downsampling factor (20ms frames at 16kHz)
    SAMPLES_PER_FRAME = 320

    def __init__(
        self,
        data_root: Optional[Union[str, Path]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        sample_rate: int = 16000,
        max_length_sec: float = 10.0,
        min_length_sec: float = 0.5,
        load_clinical_labels: bool = True,
        transform: Optional[Callable] = None,
        gamma_mismatch_policy: str = "truncate",
        min_quality_tier: Optional[int] = None,
        load_gop_targets: bool = False,
        load_measure_targets: bool = False,
    ):
        """
        Args:
            data_root: Root directory containing audio/ and alignments/ subdirs
            manifest_path: Path to JSON manifest file (alternative to data_root)
            sample_rate: Target sample rate (will resample if needed)
            max_length_sec: Maximum audio length in seconds
            min_length_sec: Minimum audio length in seconds
            load_clinical_labels: Whether to load clinical labels from metadata
            transform: Optional transform to apply to samples
            gamma_mismatch_policy: How to handle gamma/waveform length mismatches.
                "raise" — error, "truncate" — fix silently, "warn" — fix with warning.
            min_quality_tier: Minimum quality tier (3=HIGH, 2=MEDIUM, 1=LOW, 0=FAILED).
                Samples below this threshold are dropped at load time.
            load_gop_targets: If True, load GOP score files from manifest "gop_path" fields.
            load_measure_targets: If True, load measure vectors from "measure_path" fields.
        """
        self.sample_rate = sample_rate
        self.gamma_mismatch_policy = gamma_mismatch_policy
        self.min_quality_tier = min_quality_tier
        self.load_gop_targets = load_gop_targets
        self.load_measure_targets = load_measure_targets
        self.max_length = int(max_length_sec * sample_rate)
        self.min_length = int(min_length_sec * sample_rate)
        self.load_clinical_labels = load_clinical_labels
        self.transform = transform
        
        self.samples: List[Dict] = []
        self.metadata: Dict = {}
        
        if manifest_path is not None:
            self._load_from_manifest(Path(manifest_path))
        elif data_root is not None:
            self._load_from_directory(Path(data_root))
        else:
            raise ValueError("Must provide either data_root or manifest_path")
    
    def _load_from_manifest(self, manifest_path: Path):
        """Load dataset from a manifest JSON file."""
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.metadata = manifest.get("metadata", {})
        raw_samples = manifest.get("samples", [])

        # Validate required fields and apply quality filter
        filtered_count = 0
        self.samples = []
        for sample in raw_samples:
            if "audio_path" not in sample:
                raise ValueError("Each sample must have 'audio_path'")
            if "gamma_path" not in sample:
                raise ValueError("Each sample must have 'gamma_path'")

            # Quality filtering
            if self.min_quality_tier is not None:
                tier = sample.get("quality_tier")
                if tier is not None and tier < self.min_quality_tier:
                    filtered_count += 1
                    continue

            self.samples.append(sample)

        if filtered_count > 0:
            print(f"Filtered {filtered_count} samples below quality tier {self.min_quality_tier}")
    
    def _load_from_directory(self, data_root: Path):
        """Load dataset from directory structure."""
        audio_dir = data_root / "audio"
        alignment_dir = data_root / "alignments"
        
        if not audio_dir.exists():
            raise ValueError(f"Audio directory not found: {audio_dir}")
        if not alignment_dir.exists():
            raise ValueError(f"Alignment directory not found: {alignment_dir}")
        
        # Load metadata if exists
        metadata_path = data_root / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        
        # Find all audio files
        audio_extensions = {".wav", ".flac", ".mp3", ".ogg"}
        for audio_path in sorted(audio_dir.iterdir()):
            if audio_path.suffix.lower() not in audio_extensions:
                continue
            
            # Find corresponding alignment
            gamma_path = alignment_dir / f"{audio_path.stem}.pt"
            if not gamma_path.exists():
                gamma_path = alignment_dir / f"{audio_path.stem}.npy"
            
            if not gamma_path.exists():
                print(f"Warning: No alignment found for {audio_path.name}, skipping")
                continue
            
            # Parse filename for IDs
            parts = audio_path.stem.split("_")
            sample_info = {
                "audio_path": str(audio_path),
                "gamma_path": str(gamma_path),
                "patient_id": parts[0] if len(parts) > 0 else None,
                "session_id": parts[1] if len(parts) > 1 else None,
                "utterance_id": parts[2] if len(parts) > 2 else None,
            }
            
            # Add clinical labels from metadata
            if self.load_clinical_labels and audio_path.stem in self.metadata:
                sample_info["clinical_labels"] = self.metadata[audio_path.stem]
            
            self.samples.append(sample_info)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_gamma(self, gamma_path: str) -> torch.Tensor:
        """Load gamma matrix from file."""
        path = Path(gamma_path)
        if path.suffix == ".pt":
            return torch.load(gamma_path, weights_only=True)
        elif path.suffix == ".npy":
            return torch.from_numpy(np.load(gamma_path))
        else:
            raise ValueError(f"Unsupported gamma format: {path.suffix}")

    def _validate_gamma(
        self, gamma: torch.Tensor, waveform: torch.Tensor, sample_id: str = "",
    ) -> torch.Tensor:
        """Validate and fix gamma shape relative to waveform length.

        Expected: gamma.shape[0] ≈ waveform.shape[0] // SAMPLES_PER_FRAME.
        Allows ±2 frame tolerance for rounding.

        Returns:
            Possibly truncated/padded gamma tensor.
        """
        assert gamma.dim() == 2, f"Gamma must be 2D (T, P), got shape {gamma.shape}"
        expected_frames = waveform.shape[0] // self.SAMPLES_PER_FRAME
        actual_frames = gamma.shape[0]
        diff = abs(actual_frames - expected_frames)

        if diff <= 2:
            # Within rounding tolerance — truncate or pad to match exactly
            if actual_frames > expected_frames:
                gamma = gamma[:expected_frames]
            elif actual_frames < expected_frames:
                pad = torch.zeros(expected_frames - actual_frames, gamma.shape[1])
                gamma = torch.cat([gamma, pad], dim=0)
            return gamma

        msg = (
            f"Gamma frame count mismatch for '{sample_id}': "
            f"gamma has {actual_frames} frames but waveform implies "
            f"{expected_frames} (waveform_len={waveform.shape[0]}, "
            f"spf={self.SAMPLES_PER_FRAME})"
        )

        if self.gamma_mismatch_policy == "raise":
            raise ValueError(msg)
        if self.gamma_mismatch_policy == "warn":
            import warnings
            warnings.warn(msg)
        # truncate policy (or warn+continue)
        if actual_frames > expected_frames:
            gamma = gamma[:expected_frames]
        else:
            pad = torch.zeros(expected_frames - actual_frames, gamma.shape[1])
            gamma = torch.cat([gamma, pad], dim=0)
        return gamma
    
    def __getitem__(self, idx: int) -> SpeechSample:
        sample_info = self.samples[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(sample_info["audio_path"])
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Squeeze to 1D
        waveform = waveform.squeeze(0)
        
        # Handle length constraints
        if waveform.shape[0] > self.max_length:
            # Random crop
            start = np.random.randint(0, waveform.shape[0] - self.max_length)
            waveform = waveform[start:start + self.max_length]
        elif waveform.shape[0] < self.min_length:
            # Pad with zeros
            pad_length = self.min_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        
        # Load gamma matrix and validate shape
        gamma = self._load_gamma(sample_info["gamma_path"])
        gamma = self._validate_gamma(
            gamma, waveform, sample_id=sample_info.get("audio_path", str(idx)),
        )
        
        # Load optional GOP targets
        gop_targets = None
        if self.load_gop_targets and "gop_path" in sample_info:
            gop_targets = torch.load(
                sample_info["gop_path"], weights_only=True,
            ).float()

        # Load optional measure targets
        measure_targets = None
        if self.load_measure_targets and "measure_path" in sample_info:
            measure_targets = torch.from_numpy(
                np.load(sample_info["measure_path"])
            ).float()

        sample = SpeechSample(
            waveform=waveform,
            sample_rate=self.sample_rate,
            gamma=gamma,
            phoneme_labels=sample_info.get("phoneme_labels"),
            patient_id=sample_info.get("patient_id"),
            session_id=sample_info.get("session_id"),
            utterance_id=sample_info.get("utterance_id"),
            clinical_labels=sample_info.get("clinical_labels"),
            gop_targets=gop_targets,
            measure_targets=measure_targets,
        )

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
    
    def get_clinical_label_info(self) -> Dict[str, Dict]:
        """
        Get information about available clinical labels.
        
        Returns:
            Dict mapping label name to info (type, classes for categorical, range for continuous)
        """
        label_info = {}
        
        for sample in self.samples:
            if "clinical_labels" not in sample:
                continue
            
            for label_name, value in sample["clinical_labels"].items():
                if label_name not in label_info:
                    label_info[label_name] = {
                        "values": [],
                        "type": "categorical" if isinstance(value, (str, bool)) else "continuous"
                    }
                label_info[label_name]["values"].append(value)
        
        # Compute stats
        for label_name, info in label_info.items():
            values = info["values"]
            if info["type"] == "categorical":
                info["classes"] = sorted(set(values))
                info["num_classes"] = len(info["classes"])
            else:
                info["min"] = min(values)
                info["max"] = max(values)
                info["mean"] = np.mean(values)
                info["std"] = np.std(values)
            del info["values"]
        
        return label_info


class SSLCollator:
    """
    Collator for SSL pretraining that handles variable-length sequences
    and prepares augmentation targets.
    """
    
    def __init__(
        self,
        pad_to_max: bool = True,
        max_length: Optional[int] = None,
    ):
        """
        Args:
            pad_to_max: If True, pad all sequences to max length in batch
            max_length: Optional fixed max length to pad to
        """
        self.pad_to_max = pad_to_max
        self.max_length = max_length
    
    def __call__(self, batch: List[SpeechSample]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Returns:
            Dict with keys:
                - waveforms: [B, T] padded waveforms
                - lengths: [B] original lengths
                - gammas: List of gamma matrices (variable length)
                - patient_ids: List of patient IDs
                - clinical_labels: Dict of stacked labels (if present)
        """
        waveforms = [s.waveform for s in batch]
        lengths = torch.tensor([w.shape[0] for w in waveforms])
        
        # Determine padding length
        if self.max_length is not None:
            max_len = self.max_length
        elif self.pad_to_max:
            max_len = max(lengths).item()
        else:
            max_len = max(lengths).item()
        
        # Pad waveforms
        padded_waveforms = torch.zeros(len(batch), max_len)
        for i, w in enumerate(waveforms):
            padded_waveforms[i, :w.shape[0]] = w
        
        # Gammas stay as list (variable phoneme counts, frame lengths)
        gammas = [s.gamma for s in batch]
        
        result = {
            "waveforms": padded_waveforms,
            "lengths": lengths,
            "gammas": gammas,
            "patient_ids": [s.patient_id for s in batch],
        }

        # GOP targets: variable length per sample, keep as list
        if batch[0].gop_targets is not None:
            result["gop_targets"] = [s.gop_targets for s in batch]

        # Measure targets: fixed length, stack into [B, D]
        if batch[0].measure_targets is not None:
            result["measure_targets"] = torch.stack(
                [s.measure_targets for s in batch]
            )

        # Collate clinical labels if present
        if batch[0].clinical_labels is not None:
            clinical_labels = {}
            for key in batch[0].clinical_labels.keys():
                values = [s.clinical_labels[key] for s in batch]
                if isinstance(values[0], (int, float)):
                    clinical_labels[key] = torch.tensor(values)
                else:
                    clinical_labels[key] = values
            result["clinical_labels"] = clinical_labels
        
        return result


def create_dataloaders(
    dataset: ClinicalSpeechDataset,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders with stratification by patient.
    
    Args:
        dataset: The full dataset
        batch_size: Batch size
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        num_workers: Number of data loading workers
        seed: Random seed for split
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Group samples by patient for stratified split
    patient_to_indices = {}
    for idx, sample in enumerate(dataset.samples):
        patient_id = sample.get("patient_id", f"unknown_{idx}")
        if patient_id not in patient_to_indices:
            patient_to_indices[patient_id] = []
        patient_to_indices[patient_id].append(idx)
    
    # Split patients
    patients = list(patient_to_indices.keys())
    np.random.seed(seed)
    np.random.shuffle(patients)
    
    n_train = int(len(patients) * train_ratio)
    n_val = int(len(patients) * val_ratio)
    
    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]
    
    # Get indices for each split
    train_indices = [i for p in train_patients for i in patient_to_indices[p]]
    val_indices = [i for p in val_patients for i in patient_to_indices[p]]
    test_indices = [i for p in test_patients for i in patient_to_indices[p]]
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    collator = SSLCollator()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
