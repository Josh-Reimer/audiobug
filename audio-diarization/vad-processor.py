#!/usr/bin/env python3
"""
silero_vad_long_audio.py

Analyze a long FLAC (or other) audio file for voice activity using Silero VAD.
Detected voice segments are saved to ./voices/ as individual audio files.
Segment metadata is saved to voices/segments.csv.

Requirements:
- python >= 3.8
- pip install torch soundfile scipy numpy tqdm python-dotenv

Usage:
    python silero_vad_long_audio.py --file path/to/audio.flac
"""

import os
import math
import csv
import argparse

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from dotenv import load_dotenv

import torch
from tqdm import tqdm

# --------------------------- Utility Functions --------------------------- #

def ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def resample_if_needed(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return x
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    y = resample_poly(x, up, down)
    return y.astype(np.float32)


def load_silero_model(device: str = "cpu"):
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
    model.to(device)
    return model, utils


# --------------------------- Voice Activity Detection --------------------------- #

def stream_vad(
    infile: str,
    model,
    utils,
    sr: int = 16000,
    chunk_duration_ms: int = 32,
    threshold: float = 0.3,
    min_silence_ms: int = 100,
    speech_pad_ms: int = 30,
    device: str = "cpu"
):
    (_, _, _, VADIterator, _) = utils
    vad_iterator = VADIterator(model, sampling_rate=sr, threshold=threshold,
                               min_silence_duration_ms=min_silence_ms,
                               speech_pad_ms=speech_pad_ms)

    with sf.SoundFile(infile, 'r') as f:
        orig_sr = f.samplerate
        in_chunk_size = int(orig_sr * (chunk_duration_ms / 1000.0))
        chunk_size = int(sr * (chunk_duration_ms / 1000.0))
        pbar = tqdm(total=f.frames, unit='frames', unit_scale=True, desc='Analyzing')

        try:
            while True:
                block = f.read(frames=in_chunk_size, dtype='float32')
                if block.size == 0:
                    break
                pbar.update(block.shape[0])

                # Mono + resample
                mono = ensure_mono(block)
                mono_resampled = resample_if_needed(mono, orig_sr, sr)

                # Batch chunks including last partial chunk
                chunks = [
                    mono_resampled[i:i + chunk_size]
                    for i in range(0, len(mono_resampled), chunk_size)
                ]
                if not chunks:
                    continue

                batch = torch.from_numpy(np.stack([np.pad(c, (0, chunk_size - len(c))) if len(c)<chunk_size else c for c in chunks])).to(device)
                for chunk in batch:
                    speech_dict = vad_iterator(chunk, return_seconds=True)
                    if speech_dict:
                        yield speech_dict

        except KeyboardInterrupt:
            print("\nInterrupted. Merging processed segments...")

        finally:
            pbar.close()
            vad_iterator.reset_states()


# --------------------------- Post-Processing --------------------------- #

def merge_segments(segments, merge_window=0.5):
    valid = [s for s in segments if isinstance(s, dict) and 'start' in s and 'end' in s]
    if not valid:
        print("No valid segments to merge.")
        return []

    segs = sorted(valid, key=lambda x: x['start'])
    merged = [segs[0]]

    for curr in segs[1:]:
        prev = merged[-1]
        if curr['start'] - prev['end'] <= merge_window:
            prev['end'] = curr['end']
        else:
            merged.append(curr)

    return merged


def save_segments(infile: str, segments, out_dir: str, sr: int = 16000):
    os.makedirs(out_dir, exist_ok=True)
    with sf.SoundFile(infile, 'r') as f:
        orig_sr = f.samplerate
        for idx, seg in enumerate(segments, start=1):
            start_frame = int(seg['start'] * orig_sr)
            end_frame = int(seg['end'] * orig_sr)
            if end_frame <= start_frame:
                continue
            f.seek(start_frame)
            block = f.read(frames=(end_frame - start_frame), dtype='float32')
            mono = ensure_mono(block)
            if orig_sr != sr:
                mono = resample_if_needed(mono, orig_sr, sr)
            out_path = os.path.join(out_dir, f"voice_{idx:05d}_{seg['start']:.2f}_{seg['end']:.2f}.wav")
            sf.write(out_path, mono, samplerate=sr)


def write_csv(segments, out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end", "duration"])
        for seg in segments:
            writer.writerow([seg['start'], seg['end'], seg['end'] - seg['start']])


# --------------------------- Main --------------------------- #

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to input audio file")
    parser.add_argument("--out", type=str, default="voices", help="Output directory")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    infile = args.file or os.getenv("OCT15LONG")
    if not infile or not os.path.exists(infile):
        raise FileNotFoundError("No valid input file found. Use --file or set OCT15LONG in .env")

    print(f"Analyzing file: {infile}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, utils = load_silero_model(device=device)

    raw_segments = list(stream_vad(infile, model, utils, device=device))
    merged = merge_segments(raw_segments)

    print(f"Detected {len(merged)} voice segments")
    save_segments(infile, merged, out_dir=args.out)
    write_csv(merged, out_csv=os.path.join(args.out, "segments.csv"))
    print(f"All voice segments saved in {args.out}/")
    print("Segments metadata written to segments.csv")


if __name__ == "__main__":
    main()
