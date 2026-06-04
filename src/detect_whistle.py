#!/usr/bin/env python3
"""
RefTracker - detect_whistle.py
Cheap, non-ML referee-WHISTLE detector — the proposed Step-2 decision-moment
trigger. A ref's pea whistle is a sustained, narrow-band tone (~2-4.5 kHz);
crowd noise is broadband. We decode the audio with ffmpeg, take an STFT, and
flag frames where the 2-4.5 kHz band both (a) holds a large share of energy and
(b) is sharply PEAKED (tonal, not flat), then require PERSISTENCE (a real blast
lasts ~0.2-1.0s) to reject claps/shouts. Output: candidate whistle timestamps.

Usage:
    python detect_whistle.py video.mp4 [--band 2000 4500] [--min-ms 180]
        [--ratio 0.15] [--peak 12] [--json out.json]

Note: needs an AUDIO stream. Veo panorama EXPORTS are often video-only — check
with `ffprobe`; re-export retaining audio if so.
"""
import argparse
import json
import subprocess
import sys

import numpy as np
from scipy.signal import stft

SR = 16000


def load_audio(path):
    """Decode any media file to mono float32 PCM at 16 kHz via ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        print("ERROR: no audio decoded — does this file have an audio stream? "
              "(ffprobe it; Veo panorama exports are often video-only.)")
        sys.exit(2)
    return audio


def detect(audio, band=(2000, 4500), ratio_thr=0.15, peak_thr=12.0,
           min_ms=180, gap_ms=150):
    """Return list of (start_s, end_s, peak_ratio, peak_peakiness) whistle events."""
    nperseg, hop = 1024, 256                      # ~16 ms hop at 16 kHz
    f, t, Z = stft(audio, fs=SR, nperseg=nperseg, noverlap=nperseg - hop)
    power = (np.abs(Z) ** 2)                       # [freq, frame]
    bandmask = (f >= band[0]) & (f <= band[1])
    total = power.sum(0) + 1e-12
    band_ratio = power[bandmask].sum(0) / total    # share of energy in band
    band_pow = power[bandmask]
    peakiness = band_pow.max(0) / (band_pow.mean(0) + 1e-12)  # tone vs flat

    gate = (band_ratio >= ratio_thr) & (peakiness >= peak_thr)

    # group gated frames into events, bridging short gaps
    min_frames = max(1, int((min_ms / 1000) * SR / hop))
    gap_frames = max(1, int((gap_ms / 1000) * SR / hop))
    events, run_start, gap = [], None, 0
    for i, g in enumerate(gate):
        if g:
            if run_start is None:
                run_start = i
            gap = 0
        elif run_start is not None:
            gap += 1
            if gap > gap_frames:
                if i - gap - run_start >= min_frames:
                    events.append((run_start, i - gap))
                run_start, gap = None, 0
    if run_start is not None and len(gate) - run_start >= min_frames:
        events.append((run_start, len(gate) - 1))

    out = []
    for a, b in events:
        out.append((round(float(t[a]), 2), round(float(t[b]), 2),
                    round(float(band_ratio[a:b + 1].max()), 3),
                    round(float(peakiness[a:b + 1].max()), 1)))
    return out, band_ratio, peakiness


def main():
    ap = argparse.ArgumentParser(description="Referee whistle detector (Step-2 trigger)")
    ap.add_argument("video")
    ap.add_argument("--band", type=float, nargs=2, default=[2000, 4500])
    ap.add_argument("--ratio", type=float, default=0.15, help="band energy-share gate")
    ap.add_argument("--peak", type=float, default=12.0, help="band peakiness gate (tonal)")
    ap.add_argument("--min-ms", type=int, default=180, help="min blast duration")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    audio = load_audio(args.video)
    print(f"Decoded {audio.size / SR:.1f}s of audio @ {SR}Hz")
    events, br, pk = detect(audio, tuple(args.band), args.ratio, args.peak, args.min_ms)
    print(f"band_ratio: median={np.median(br):.3f} p95={np.percentile(br,95):.3f} "
          f"max={br.max():.3f} | peakiness p95={np.percentile(pk,95):.1f}")
    print(f"\n{len(events)} candidate whistle event(s):")
    for s, e, r, p in events:
        m, sec = divmod(int(s), 60)
        print(f"  {m}:{sec:02d}  ({s:.1f}-{e:.1f}s, {e-s:.1f}s)  ratio={r} peak={p}")
    if args.json:
        json.dump({"events": [{"start": s, "end": e, "ratio": r, "peak": p}
                              for s, e, r, p in events]}, open(args.json, "w"), indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
