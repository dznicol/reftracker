#!/usr/bin/env python3
"""
RefTracker - track_ref_colour.py
A SINGLE-OBJECT, colour-first referee tracker — for the common case where the
ref's shirt is a colour no player wears. We DON'T use a multi-object identity
tracker (BoTSORT): when the target colour is unique there is no identity to
maintain, and MOT identity-drift is exactly what made the ref jump to the wrong
(navy) player. Instead: every frame, detect people, pick the one whose torso
best matches the ref's auto-calibrated kit colour (with a light smoothness
gate), then interpolate across short gaps where he's occluded/turned.

Usage:
    python track_ref_colour.py clip.mp4 --output tracked.mp4 --hue 75 99
        [--imgsz 1280] [--max-gap 30] [--start 0] [--end 0]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from discmark import smooth_disc_track, draw_disc


def torso_region(frame, box):
    x1, y1, x2, y2 = map(int, box[:4])
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    return crop[int(ch * 0.10):int(ch * 0.55), int(cw * 0.25):int(cw * 0.75)]


def main():
    ap = argparse.ArgumentParser(description="Colour-first single-object ref tracker")
    ap.add_argument("video")
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--tracking-json", default=None)
    ap.add_argument("--hue", type=int, nargs=2, default=[75, 99],
                    help="wide hue band to FIND the ref initially (default 75 99)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--max-gap", type=int, default=30,
                    help="interpolate occlusion gaps up to N frames (default 30)")
    ap.add_argument("--spectator-y", type=int, default=870,
                    help="ignore detections with centre below this y (foreground spectators)")
    ap.add_argument("--min-ratio", type=float, default=0.10,
                    help="min torso colour-match ratio to accept a detection")
    args = ap.parse_args()

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wide = (np.array([args.hue[0], 25, 80]), np.array([args.hue[1], 255, 255]))

    # ── Pass 1: detect the unique-coloured ref per frame, auto-calibrate ──
    print(f"Pass 1/2: detecting the ref by colour over {N} frames...")
    pos = {}                 # frame -> (cx, cy, bbox)
    kit_px = []              # accumulate his torso HSV for tight calibration
    last = None
    for fi in range(N):
        ret, frame = cap.read()
        if not ret:
            break
        best, best_r = None, args.min_ratio
        for b in model(frame, classes=[0], imgsz=args.imgsz, verbose=False)[0].boxes.xyxy.cpu().numpy():
            cy = (b[1] + b[3]) / 2
            if cy >= args.spectator_y or (b[3] - b[1]) < 16:
                continue
            reg = torso_region(frame, b)
            if reg is None:
                continue
            hsv = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV)
            ratio = np.count_nonzero(cv2.inRange(hsv, *wide)) / hsv[:, :, 0].size
            if ratio < args.min_ratio:
                continue
            cx, cyy = int((b[0] + b[2]) / 2), int(cy)
            # smoothness preference: penalise jumps from last known position
            score = ratio
            if last is not None:
                d = ((cx - last[0]) ** 2 + (cyy - last[1]) ** 2) ** 0.5
                score -= min(0.4, d / 1500)
            if score > best_r:
                best_r, best = score, (cx, cyy, b, hsv)
        if best is not None:
            cx, cyy, b, hsv = best
            pos[fi] = (cx, cyy, [float(v) for v in b[:4]])
            last = (cx, cyy)
            m = cv2.inRange(hsv, *wide) > 0
            if m.any():
                kit_px.append(cv2.cvtColor(torso_region(frame, b), cv2.COLOR_BGR2HSV)[m])
        if fi % 200 == 0:
            print(f"  frame {fi}/{N} ({len(pos)} hits)")
    cap.release()

    found = len(pos)
    print(f"\nRef found by colour in {found}/{N} frames ({100*found/max(N,1):.0f}%)")
    if found < 5:
        print("Too few colour hits — wrong hue band? Aborting.")
        sys.exit(1)
    px = np.concatenate(kit_px)
    print(f"Calibrated kit hue: median={np.median(px[:,0]):.0f} "
          f"S={np.median(px[:,1]):.0f} V={np.median(px[:,2]):.0f}")

    # ── interpolate short occlusion gaps ──
    frames_sorted = sorted(pos)
    filled = dict(pos)
    interp = 0
    for a, c in zip(frames_sorted, frames_sorted[1:]):
        if 1 < c - a <= args.max_gap:
            (xa, ya, ba), (xc, yc, bc) = pos[a], pos[c]
            for f in range(a + 1, c):
                t = (f - a) / (c - a)
                filled[f] = (int(xa + t*(xc-xa)), int(ya + t*(yc-ya)),
                             [ba[i] + t*(bc[i]-ba[i]) for i in range(4)])
                interp += 1
    print(f"Interpolated {interp} frames across gaps <= {args.max_gap}; "
          f"disc shown in {len(filled)}/{N} ({100*len(filled)/max(N,1):.0f}%)")

    # ── Pass 2: render the disc ──
    print("Pass 2/2: rendering disc-annotated video...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    track = smooth_disc_track({f: filled[f][2] for f in filled}, fps)
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi in track:
            cx, fy, rx, a = track[fi]
            draw_disc(frame, cx, fy, rx, alpha=a)
        out.write(frame)
        fi += 1
    cap.release()
    out.release()
    print(f"Saved: {args.output}")

    jpath = args.tracking_json or str(Path(args.output).with_suffix(".json"))
    json.dump({"video": args.video, "fps": fps, "width": W, "height": H,
               "total_frames": N, "tracker": "colour-first",
               "positions": [{"frame": f, "x": filled[f][0], "y": filled[f][1],
                              "bbox": filled[f][2], "interpolated": f not in pos}
                             for f in sorted(filled)]}, open(jpath, "w"), indent=2)
    print(f"Saved: {jpath}")


if __name__ == "__main__":
    main()
