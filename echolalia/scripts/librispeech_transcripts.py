#!/usr/bin/env python
"""
Helper script to build a transcripts.json file from a LibriSpeech directory.

LibriSpeech organizes transcripts as .trans.txt files, one per chapter:
    LibriSpeech/train-clean-100/<speaker>/<chapter>/<speaker>-<chapter>.trans.txt

Each line has the format:
    <utterance_id> <transcript>

This script walks the directory and produces a flat JSON dict mapping
utterance_id -> transcript, suitable for passing to compute_alignments.py.

Usage:
    python -m scripts.librispeech_transcripts \
        --librispeech-root /data/LibriSpeech/train-clean-100 \
        --output /data/LibriSpeech/train-clean-100.transcripts.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Build transcripts.json from a LibriSpeech directory"
    )
    parser.add_argument("--librispeech-root", type=str, required=True,
                        help="Root dir containing speaker subdirs")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path")
    parser.add_argument("--lowercase", action="store_true",
                        help="Lowercase transcripts (LibriSpeech defaults to uppercase)")
    args = parser.parse_args()

    root = Path(args.librispeech_root)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    transcripts = {}
    trans_files = list(root.rglob("*.trans.txt"))
    print(f"Found {len(trans_files)} transcript files")

    for trans_file in trans_files:
        with open(trans_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                utt_id, text = parts
                if args.lowercase:
                    text = text.lower()
                transcripts[utt_id] = text

    print(f"Collected {len(transcripts)} transcripts")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(transcripts, f, indent=2)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
