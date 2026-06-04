#!/usr/bin/env python3
"""
RefTracker - evaluate.py
A tiny, low-tech evaluation harness so changes to the pipeline are MEASURABLE
instead of eyeballed. Two things get scored:

  1. TRACKING accuracy — what % of sampled frames have the marker on the real ref.
  2. DECISION accuracy — precision/recall/F1 of Gemini's decisions vs. ground truth.

Typical maker-day flow:

  # --- Tracking ---
  uv run python src/evaluate.py tracking-template output/tracked.mp4 --frames 40
  #   -> extracts 40 stills (marker already drawn) to output/eval/tracking_frames/
  #   -> writes output/eval/tracking_labels.csv with a blank `correct` column
  # Eyeball each still, put 1 (on the ref) / 0 (wrong person) in `correct`, save.
  uv run python src/evaluate.py score-tracking output/eval/tracking_labels.csv

  # --- Decisions ---
  uv run python src/evaluate.py decisions-template
  #   -> writes output/eval/decisions_truth.json — fill in the REAL decisions
  uv run python src/evaluate.py score-decisions output/decisions.json \
      output/eval/decisions_truth.json --tolerance 3

Requirements: opencv-python (already a project dep).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2


DEFAULT_EVAL_DIR = Path("output/eval")


# ──────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────
def _to_seconds(ts, offset=0.0):
    """
    Parse a timestamp into float seconds. Accepts "M:SS", "MM:SS", "H:MM:SS",
    "SS", "M:SS.s", or a bare number. Returns offset on failure (best effort).
    """
    if ts is None:
        return float(offset)
    if isinstance(ts, (int, float)):
        return float(ts) + float(offset)
    s = str(ts).strip()
    if not s:
        return float(offset)
    # Allow "1:02.5" style by keeping the decimal on the last part.
    try:
        parts = s.split(":")
        parts = [float(p) for p in parts]
    except ValueError:
        return float(offset)
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds + float(offset)


# ──────────────────────────────────────────────────────────────────────
# TRACKING: template + score
# ──────────────────────────────────────────────────────────────────────
def tracking_template(video_path, frames=40, out_dir=None):
    """Extract evenly-spaced stills from the tracked video + a CSV to label.
    By default the eval files land in an `eval/` folder NEXT TO the input video
    (e.g. output/<video>/<range>/eval/), so each sequence keeps its own labels
    instead of sharing one output/eval that would clobber across videos."""
    video_path = Path(video_path)
    eval_dir = Path(out_dir) if out_dir else (video_path.parent / "eval")
    out_dir = eval_dir / "tracking_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print("Error: video reports 0 frames")
        sys.exit(1)

    frames = min(frames, total)
    # Evenly spaced indices across the whole clip.
    idxs = [int(round(i * (total - 1) / max(frames - 1, 1))) for i in range(frames)]
    idxs = sorted(set(idxs))

    rows = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        ts = round(fi / fps, 2)
        name = f"frame_{fi:06d}.png"
        cv2.imwrite(str(out_dir / name), frame)
        rows.append({"frame": fi, "timestamp": ts, "image": name, "correct": ""})
    cap.release()

    csv_path = eval_dir / "tracking_labels.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "timestamp", "image", "correct"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} stills to {out_dir}/")
    print(f"Wrote label sheet: {csv_path}")
    print("\nNext: open the stills, and in the CSV set `correct` to:")
    print("   1  = green marker is on the actual referee")
    print("  -1  = marker is on the WRONG person")
    print("   0  = nothing tracked (no box drawn)")
    print("  (leave blank to skip a frame; an optional `reason` column is allowed)")
    print(f"Then: uv run python src/evaluate.py score-tracking {csv_path}")


def score_tracking(csv_path):
    """
    Read a filled-in label CSV and report tracking accuracy.

    Three-valued `correct` column:
      1  = green marker is on the actual referee
     -1  = marker is on the WRONG person (actively misleading)
      0  = nothing tracked (a miss — no box drawn)
      (blank/other = skipped)

    We report two rates because the failures differ:
      - Strict accuracy   = on_ref / all labelled  (the honest headline)
      - When-it-marks-someone = on_ref / (on_ref + wrong)  (excludes misses)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    on_ref, wrong_person, nothing, skipped = 0, 0, 0, 0
    wrong_frames, miss_frames = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            val = (row.get("correct") or "").strip().lower()
            frame = row.get("frame")
            if val in ("1", "y", "yes", "true"):
                on_ref += 1
            elif val in ("-1", "wrong"):
                wrong_person += 1
                wrong_frames.append(frame)
            elif val in ("0", "n", "no", "false"):
                nothing += 1
                miss_frames.append(frame)
            else:
                skipped += 1

    labeled = on_ref + wrong_person + nothing
    print(f"Tracking accuracy ({csv_path.name})")
    print("=" * 44)
    if labeled == 0:
        print("No labelled rows yet — fill in the `correct` column first.")
        return
    strict = on_ref / labeled * 100
    present = on_ref + wrong_person
    when_present = (on_ref / present * 100) if present else 0.0
    print(f"  Frames labelled        : {labeled}")
    print(f"  On the ref        (= 1): {on_ref}")
    print(f"  Wrong person      (=-1): {wrong_person}")
    print(f"  Nothing tracked   (= 0): {nothing}")
    print(f"  Skipped (blank)        : {skipped}")
    print(f"  STRICT ACCURACY        : {strict:.1f}%   (on-ref / all labelled)")
    print(f"  When it marks someone  : {when_present:.1f}%   (on-ref / on-ref+wrong)")
    if wrong_frames:
        print(f"  Wrong-person frames    : {', '.join(map(str, wrong_frames))}")
    if miss_frames:
        print(f"  Nothing-tracked frames : {', '.join(map(str, miss_frames))}")


# ──────────────────────────────────────────────────────────────────────
# DECISIONS: template + score
# ──────────────────────────────────────────────────────────────────────
def decisions_template(out_path=None):
    """Write a stub ground-truth file for the human to fill with real decisions."""
    out_path = Path(out_path) if out_path else DEFAULT_EVAL_DIR / "decisions_truth.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"{out_path} already exists — not overwriting.")
        return
    stub = {
        "_comment": (
            "Hand-label the REAL referee decisions in the clip. timestamp = 'M:SS' "
            "(in the same time-base the classifier reports, i.e. seconds into the "
            "clip). decision_type must match the classifier's vocabulary."
        ),
        "decisions": [
            {"timestamp": "0:55", "decision_type": "knock_on",
             "team_penalised": "", "note": "EXAMPLE — replace with real ones"},
            {"timestamp": "0:58", "decision_type": "scrum_awarded",
             "team_penalised": "", "note": "EXAMPLE — replace with real ones"},
        ],
    }
    with open(out_path, "w") as f:
        json.dump(stub, f, indent=2)
    print(f"Wrote ground-truth stub: {out_path}")
    print("Edit it to list the actual decisions, then run score-decisions.")


def _load_decisions(path, is_truth):
    with open(path) as f:
        data = json.load(f)
    decisions = data.get("decisions", [])
    out = []
    for d in decisions:
        dtype = d.get("decision_type", "unknown")
        if dtype == "play_on":
            continue
        if is_truth:
            secs = _to_seconds(d.get("timestamp"))
        else:
            secs = _to_seconds(d.get("timestamp_approx"),
                               d.get("segment_offset_seconds", 0))
        out.append({"t": secs, "type": dtype, "raw": d})
    return out


def score_decisions(pred_path, truth_path, tolerance=3.0, detect_tolerance=None):
    """
    Match predicted decisions against ground truth, and report the two stages
    SEPARATELY (detection vs classification):
      - detection recall    = (tp + timing_only) / truth   (found the moment, any label)
      - detection precision = (tp + timing_only) / preds   (preds that hit a real moment)
      - classification acc   = tp / (tp + timing_only)       (given detected, label correct)
    `tolerance` matches type+time; `detect_tolerance` (>= tolerance) is the looser
    window for "found the moment". Also checks the truth file's `_not_decisions`
    false-positive traps (over-call diagnostics).
    """
    detect_tolerance = detect_tolerance or tolerance
    preds = _load_decisions(pred_path, is_truth=False)
    truth = _load_decisions(truth_path, is_truth=True)

    if not truth:
        print("No ground-truth decisions found — fill in the truth file first.")
        return

    # ── time-base sanity guard ────────────────────────────────────────
    if preds and truth:
        near = sum(1 for g in truth for p in preds
                   if abs(p["t"] - g["t"]) <= max(detect_tolerance, tolerance))
        if near == 0:
            tr = (min(d["t"] for d in truth), max(d["t"] for d in truth))
            pr = (min(d["t"] for d in preds), max(d["t"] for d in preds))
            print(f"⚠️  TIME-BASE WARNING: no pred lands near any truth decision. "
                  f"truth spans {tr[0]:.0f}-{tr[1]:.0f}s, preds {pr[0]:.0f}-{pr[1]:.0f}s "
                  f"— are they on the SAME clip/clock? (segment_offset_seconds set?)\n")

    preds_sorted = sorted(preds, key=lambda d: d["t"])
    used = [False] * len(preds_sorted)

    tp, timing_only = 0, 0
    matched_truth = []
    missed = []
    for g in truth:
        best_j, best_dt, best_same = None, None, False
        for j, p in enumerate(preds_sorted):
            if used[j]:
                continue
            dt = abs(p["t"] - g["t"])
            if dt > detect_tolerance:
                continue
            same = (p["type"] == g["type"])
            # Prefer a same-type match; otherwise keep the closest in time.
            if best_j is None or (same and not best_same) or \
               (same == best_same and dt < best_dt):
                best_j, best_dt, best_same = j, dt, same
        if best_j is None:
            missed.append(g)
        else:
            used[best_j] = True
            if best_same:
                tp += 1
                matched_truth.append((g, preds_sorted[best_j], best_dt))
            else:
                timing_only += 1
                matched_truth.append((g, preds_sorted[best_j], best_dt))

    fp = sum(1 for j, p in enumerate(preds_sorted) if not used[j])
    fn = len(missed)
    nt, npred = len(truth), len(preds)
    detected = tp + timing_only                     # found the moment (any label)
    # Two-stage split (the metric that answers detection vs classification):
    det_recall = detected / nt if nt else 0.0
    det_prec = detected / npred if npred else 0.0
    clf_acc = tp / detected if detected else 0.0
    # Overall strict precision/recall/F1 (same-type true positives).
    precision = tp / (tp + fp + timing_only) if (tp + fp + timing_only) else 0.0
    recall = tp / (tp + fn + timing_only) if (tp + fn + timing_only) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    # ── false-positive traps (_not_decisions): over-call diagnostics ──
    import json as _json
    not_dec = _json.load(open(truth_path)).get("_not_decisions", [])
    trap_hits = []
    for nd in not_dec:
        tt = _to_seconds(nd.get("timestamp"))
        hit = next((p for p in preds_sorted if abs(p["t"] - tt) <= detect_tolerance), None)
        if hit:
            trap_hits.append((tt, hit["type"]))

    print(f"Decision accuracy  (detect ±{detect_tolerance:g}s)")
    print("=" * 46)
    print(f"  Ground-truth decisions : {nt}")
    print(f"  Predicted decisions    : {npred}")
    print(f"  True positives (type ✓): {tp}")
    print(f"  Right time, wrong type : {timing_only}")
    print(f"  False positives        : {fp}")
    print(f"  Missed (no detection)  : {fn}")
    print(f"  -- STAGE SPLIT --")
    print(f"  DETECTION recall       : {det_recall:.2f}  ({detected}/{nt} moments found)")
    print(f"  DETECTION precision    : {det_prec:.2f}  ({detected}/{npred} preds on a real moment)")
    print(f"  CLASSIFICATION acc     : {clf_acc:.2f}  ({tp}/{detected} found moments labelled right)")
    print(f"  -- overall (strict) --")
    print(f"  Precision              : {precision:.2f}")
    print(f"  Recall                 : {recall:.2f}")
    print(f"  F1                     : {f1:.2f}")
    if not_dec:
        print(f"  FP-trap hits           : {len(trap_hits)}/{len(not_dec)}  "
              f"(over-calls at known NON-decision moments)")
        for tt, ty in trap_hits:
            print(f"      ✗ called {ty} at {tt:.0f}s (should be play_on)")

    if matched_truth:
        print("\n  Matches:")
        for g, p, dt in matched_truth:
            flag = "✓" if g["type"] == p["type"] else f"✗ (got {p['type']})"
            print(f"    {g['t']:6.1f}s  {g['type']:<18} {flag}  Δ{dt:.1f}s")
    if missed:
        print("\n  Missed:")
        for g in missed:
            print(f"    {g['t']:6.1f}s  {g['type']}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RefTracker evaluation harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("tracking-template",
                       help="Extract stills from a tracked video + a CSV to label")
    p.add_argument("video", help="Tracked video (output of track_ref.py)")
    p.add_argument("--frames", type=int, default=40, help="How many stills (default 40)")
    p.add_argument("--out-dir", default=None, help="Where to write stills")

    p = sub.add_parser("score-tracking", help="Score a filled-in tracking CSV")
    p.add_argument("csv", help="tracking_labels.csv with the `correct` column filled")

    p = sub.add_parser("decisions-template",
                       help="Write a ground-truth stub for decisions")
    p.add_argument("--out", default=None, help="Output path for the stub JSON")

    p = sub.add_parser("score-decisions", help="Score predicted decisions vs. truth")
    p.add_argument("pred", help="decisions.json from classify_decisions.py")
    p.add_argument("truth", help="hand-labelled ground-truth JSON")
    p.add_argument("--tolerance", type=float, default=3.0,
                   help="Same-type match slack, seconds (default 3)")
    p.add_argument("--detect-tolerance", type=float, default=None,
                   help="Looser 'found the moment' window for detection metrics "
                        "(>= --tolerance; defaults to it). Gemini timestamps drift "
                        "several seconds, so try 5 on real clips.")

    args = parser.parse_args()
    if args.cmd == "tracking-template":
        tracking_template(args.video, frames=args.frames, out_dir=args.out_dir)
    elif args.cmd == "score-tracking":
        score_tracking(args.csv)
    elif args.cmd == "decisions-template":
        decisions_template(args.out)
    elif args.cmd == "score-decisions":
        score_decisions(args.pred, args.truth, tolerance=args.tolerance,
                        detect_tolerance=args.detect_tolerance)


if __name__ == "__main__":
    main()
