#!/usr/bin/env python
"""
Offline preprocessing script: compute gamma matrices, GOP targets,
quality tiers, and SoftAlign measure vectors.

Usage:
    python -m scripts.compute_alignments \
        --audio-dir /data/audio \
        --transcripts /data/transcripts.json \
        --output-dir /data/processed \
        --backbone phoneme_classifier \
        --classifier-checkpoint /models/classifier/best.pt \
        --compute-gop --compute-measures --compute-quality

Output structure:
    output_dir/
        alignments/         # gamma .pt files
        gop/                # GOP score .pt files
        measures/           # measure vector .npy files
        measure_keys.json   # shared key names for measure vectors
        manifest.json       # dataset manifest loadable by ClinicalSpeechDataset
        failed.json         # samples that failed processing
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from clinical_speech_ssl.softalign_bridge.alignment import (
    get_alignment_backbone,
    compute_gamma,
)
from clinical_speech_ssl.softalign_bridge.gop import compute_gop_targets
from clinical_speech_ssl.softalign_bridge.measures import compute_measure_vector
from clinical_speech_ssl.softalign_bridge.quality import compute_quality_tier


def load_transcripts(path: str) -> Dict[str, str]:
    """Load transcript mapping from JSON file.

    Expected format: {"stem": "transcript text", ...}
    or: [{"audio": "name.wav", "transcript": "text"}, ...]
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        return {
            Path(item["audio"]).stem: item["transcript"]
            for item in data
        }
    else:
        raise ValueError(f"Unexpected transcripts format: {type(data)}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute alignments, GOP, measures, and quality tiers"
    )
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--transcripts", type=str, required=True,
                        help="JSON mapping audio stems to transcripts")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="phoneme_classifier",
                        choices=["phoneme_classifier", "phoneme_ctc"])
    parser.add_argument("--classifier-checkpoint", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None,
                        help="HuggingFace model name for CTC backbone")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--compute-gop", action="store_true")
    parser.add_argument("--compute-measures", action="store_true")
    parser.add_argument("--compute-quality", action="store_true")
    parser.add_argument("--control-only-gop", action="store_true",
                        help="Only compute GOP for control speakers "
                             "(requires 'control_speakers.json' in audio-dir)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output subdirectories
    align_dir = output_dir / "alignments"
    align_dir.mkdir(exist_ok=True)
    if args.compute_gop:
        (output_dir / "gop").mkdir(exist_ok=True)
    if args.compute_measures:
        (output_dir / "measures").mkdir(exist_ok=True)

    # Load transcripts
    transcripts = load_transcripts(args.transcripts)

    # Load control speaker list (for control-only GOP)
    control_speakers = None
    if args.control_only_gop:
        control_path = audio_dir / "control_speakers.json"
        if control_path.exists():
            with open(control_path) as f:
                control_speakers = set(json.load(f))
            print(f"Loaded {len(control_speakers)} control speakers")
        else:
            print(f"Warning: --control-only-gop set but {control_path} not found")

    # Load backbone
    print(f"Loading backbone: {args.backbone}")
    backbone = get_alignment_backbone(
        backbone_type=args.backbone,
        classifier_checkpoint=args.classifier_checkpoint,
        model_name=args.model_name,
    )

    # Find audio files
    audio_extensions = {".wav", ".flac", ".mp3", ".ogg"}
    audio_files = sorted(
        p for p in audio_dir.iterdir()
        if p.suffix.lower() in audio_extensions and p.stem in transcripts
    )
    print(f"Found {len(audio_files)} audio files with transcripts")

    # Process
    manifest_samples = []
    failed = []
    measure_keys = None

    for audio_path in tqdm(audio_files, desc="Processing"):
        stem = audio_path.stem
        transcript = transcripts[stem]

        try:
            # Load audio
            waveform, sr = torchaudio.load(str(audio_path))
            if sr != args.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, args.sample_rate)(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            audio = waveform.squeeze(0)

            # 1. Compute gamma
            gamma, alignment = compute_gamma(
                audio, transcript, backbone, args.sample_rate,
            )
            gamma_path = str(align_dir / f"{stem}.pt")
            torch.save(gamma, gamma_path)

            sample_info = {
                "audio_path": str(audio_path),
                "gamma_path": gamma_path,
            }

            # Parse patient/session/utterance from filename
            parts = stem.split("_")
            sample_info["patient_id"] = parts[0] if len(parts) > 0 else None
            sample_info["session_id"] = parts[1] if len(parts) > 1 else None
            sample_info["utterance_id"] = parts[2] if len(parts) > 2 else None

            # 2. Compute GOP (optionally control-only)
            if args.compute_gop:
                is_control = (
                    control_speakers is None
                    or sample_info.get("patient_id") in control_speakers
                )
                if is_control:
                    gop_result = compute_gop_targets(
                        audio, transcript, backbone, args.sample_rate,
                    )
                    gop_path = str(output_dir / "gop" / f"{stem}.pt")
                    torch.save(gop_result.scores, gop_path)
                    sample_info["gop_path"] = gop_path

            # 3. Compute measures
            if args.compute_measures:
                feat_vec, keys = compute_measure_vector(
                    audio, transcript, backbone, args.sample_rate,
                )
                measure_path = str(output_dir / "measures" / f"{stem}.npy")
                np.save(measure_path, feat_vec)
                sample_info["measure_path"] = measure_path

                if measure_keys is None:
                    measure_keys = keys

            # 4. Compute quality tier
            if args.compute_quality:
                tier = compute_quality_tier(
                    audio, transcript, backbone, args.sample_rate,
                    patient_id=sample_info.get("patient_id", "unknown"),
                )
                sample_info["quality_tier"] = tier

            manifest_samples.append(sample_info)

        except Exception as e:
            failed.append({
                "audio_path": str(audio_path),
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            continue

    # Write manifest
    manifest = {
        "metadata": {
            "backbone": args.backbone,
            "sample_rate": args.sample_rate,
            "num_samples": len(manifest_samples),
            "num_failed": len(failed),
        },
        "samples": manifest_samples,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Write measure keys
    if measure_keys is not None:
        with open(output_dir / "measure_keys.json", "w") as f:
            json.dump(measure_keys, f)

    # Write failed list
    if failed:
        with open(output_dir / "failed.json", "w") as f:
            json.dump(failed, f, indent=2)

    print(f"\nDone: {len(manifest_samples)} succeeded, {len(failed)} failed")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
