#!/usr/bin/env python3
"""
RefTracker - merge_output.py
Merges Gemini decision classifications onto the tracked video as text labels.
Takes the annotated video from track_ref.py and the decisions from classify_decisions.py.

Usage:
    python merge_output.py tracked_video.mp4 decisions.json [--output final.mp4]

Requirements:
    pip install opencv-python numpy
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def _disc_positions(tracking_json, max_gap):
    """Per-frame ref bbox from a tracking JSON. Uses only REAL detections
    (interpolated==False, or all if the flag is absent) and bridges gaps of up
    to `max_gap` frames — so genuine occlusions (longer gaps) show NO disc."""
    js = json.load(open(tracking_json))
    real = {p["frame"]: p["bbox"] for p in js["positions"]
            if not p.get("interpolated", False)}
    disc = dict(real)
    fr = sorted(real)
    for a, c in zip(fr, fr[1:]):
        if 1 < c - a <= max_gap:
            ba, bc = real[a], real[c]
            for f in range(a + 1, c):
                t = (f - a) / (c - a)
                disc[f] = [ba[i] + t * (bc[i] - ba[i]) for i in range(4)]
    return disc


def merge_decisions_onto_video(video_path, decisions_path, output_path,
                               display_duration=5.0, tracking_json=None,
                               max_gap=14, follow=False, follow_size=(760, 620),
                               follow_zoom=1.0, banner_offset=0.0):
    """
    Overlay decision banners (and optionally the ref disc) onto a video.

    - tracking_json: if given, draw the underfoot disc from it (raw clip in,
      finished clip out — one command). Genuine occlusions show no disc.
    - follow: render a ref-centred zoom ("follow-cam") instead of the full field.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    data = json.load(open(decisions_path))
    decisions = data.get("decisions", [])

    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    timed_decisions = []
    for d in decisions:
        if d.get("decision_type") == "play_on":
            continue
        ts = d.get("timestamp_approx") or d.get("timestamp") or "0:00"  # accept both formats
        offset = d.get("segment_offset_seconds", 0)
        try:
            parts = str(ts).replace(".", ":").split(":")
            seconds = int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 else int(parts[0])
        except (ValueError, IndexError):
            seconds = 0
        start_frame = int((seconds + offset + banner_offset) * fps)
        timed_decisions.append({
            "start_frame": start_frame,
            "end_frame": min(int((seconds + offset + banner_offset + display_duration) * fps), total_frames),
            "type": d.get("decision_type", "unknown").replace("_", " ").title(),
            "explanation": d.get("explanation") or d.get("note", ""),
            "team_against": d.get("team_penalised", ""),
            "team_for": d.get("team_benefiting", ""),
        })
    print(f"Overlaying {len(timed_decisions)} decisions"
          + (f" + disc (gap<={max_gap}f)" if tracking_json else "")
          + (" as FOLLOW-CAM" if follow else "") + "...")

    disc, ell, lab = {}, None, None
    if tracking_json:
        import supervision as sv
        disc = _disc_positions(tracking_json, max_gap)
        ell = sv.EllipseAnnotator(thickness=3, color=sv.Color.from_hex("#00FF00"),
                                  color_lookup=sv.ColorLookup.INDEX)
        lab = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_color=sv.Color.WHITE,
                                text_position=sv.Position.BOTTOM_CENTER,
                                color=sv.Color.from_hex("#00FF00"), color_lookup=sv.ColorLookup.INDEX)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ow, oh = follow_size if follow else (width, height)
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))

    # ── Pre-compute a SMOOTH follow-cam path (avoids per-frame jitter/zoom) ──
    cam_cx = cam_cy = None
    win_w = win_h = 0
    if follow and disc:
        kf = sorted(disc)
        kx = [(disc[f][0] + disc[f][2]) / 2 for f in kf]
        ky = [(disc[f][1] + disc[f][3]) / 2 for f in kf]
        allf = np.arange(total_frames)
        # glide the CAMERA across all frames (incl. long occlusions); the DISC
        # is still only drawn where it exists, so honesty is unaffected.
        cam_cx = np.interp(allf, kf, kx)
        cam_cy = np.interp(allf, kf, ky)
        k = max(3, int(fps * 0.8) | 1)            # ~0.8s moving-average, odd
        ker = np.ones(k) / k
        cam_cx = np.convolve(cam_cx, ker, "same")
        cam_cy = np.convolve(cam_cy, ker, "same")
        # CONSTANT zoom (median ref height x 8 x follow_zoom) — higher follow_zoom
        # = wider view (more context, ref smaller). No per-frame zoom flicker.
        med_bh = float(np.median([disc[f][3] - disc[f][1] for f in kf]))
        win_h = int(np.clip(med_bh * 8 * follow_zoom, 320, height))
        win_w = min(width, int(win_h * ow / oh))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # draw disc on the FULL frame (so it stays aligned with the ref after any crop)
        if frame_idx in disc:
            bb = disc[frame_idx]
            det = sv.Detections(
                xyxy=np.array([bb]), confidence=np.array([1.0]), tracker_id=np.array([0]))
            frame = ell.annotate(frame, det)
            frame = lab.annotate(frame, det, labels=["REF"])

        if follow and cam_cx is not None:
            cx, cy = cam_cx[frame_idx], cam_cy[frame_idx]
            x1 = int(np.clip(cx - win_w / 2, 0, max(0, width - win_w)))
            y1 = int(np.clip(cy - 0.12 * win_h - win_h / 2, 0, max(0, height - win_h)))
            crop = frame[y1:y1 + win_h, x1:x1 + win_w]
            if crop.size:
                frame = cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_CUBIC)

        active = [d for d in timed_decisions
                  if d["start_frame"] <= frame_idx <= d["end_frame"]]
        if active:
            frame = draw_decision_banner(frame, active[-1], ow, oh)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"  Frame {frame_idx}/{total_frames}")

    cap.release()
    out.release()
    print(f"Final video saved: {output_path}")


def _wrap_text(text, font, scale, thickness, max_width, max_lines=2):
    """Greedily wrap `text` into <= max_lines that each fit within max_width px.
    If it still overflows, the last line is ellipsised."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        (tw, _), _ = cv2.getTextSize(trial, font, scale, thickness)
        if tw <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
            if len(lines) == max_lines:      # no room for another full line
                break
    if len(lines) < max_lines and cur:
        lines.append(cur)
        cur = ""
    if cur:  # leftover didn't fit — ellipsise the final line
        last = lines[-1] if lines else ""
        while " " in last:
            cand = last + " ..."
            (tw, _), _ = cv2.getTextSize(cand, font, scale, thickness)
            if tw <= max_width:
                break
            last = last.rsplit(" ", 1)[0]
        if lines:
            lines[-1] = last + " ..."
    return lines


def draw_decision_banner(frame, decision, width, height):
    """
    Draw banners at top and bottom of frame with decision info.
    Top banner is fully opaque to prevent ghost text from previous decisions.
    """
    banner_height = 70

    # Fully opaque dark banner at top — no blending, no ghost text
    cv2.rectangle(frame, (0, 0), (width, banner_height), (0, 0, 0), -1)

    # Build top line: "Decision Type  against: X | for: Y"
    type_text = decision["type"]
    team_parts = []
    if decision.get("team_against") and decision["team_against"] != "N/A":
        team_parts.append(f"against: {decision['team_against']}")
    if decision.get("team_for") and decision["team_for"] != "N/A":
        team_parts.append(f"for: {decision['team_for']}")
    team_str = " | ".join(team_parts)

    if team_str:
        top_text = f"{type_text}   [{team_str}]"
    else:
        top_text = type_text

    # Shrink the title font so it never overflows the frame width (matters on
    # the narrower follow-cam render).
    ts = 1.0
    (tw, _), _ = cv2.getTextSize(top_text, cv2.FONT_HERSHEY_SIMPLEX, ts, 2)
    if tw > width - 30:
        ts = max(0.45, ts * (width - 30) / tw)
    cv2.putText(frame, top_text, (15, 42),
                cv2.FONT_HERSHEY_SIMPLEX, ts, (255, 255, 255), 2)

    # Explanation — bottom banner, wrapped to up to 2 lines so it never
    # bleeds off the right edge.
    if decision["explanation"] and len(decision["explanation"]) > 10:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.6, 1
        margin = 15
        line_h = 26
        lines = _wrap_text(decision["explanation"], font, scale, thick,
                           width - 2 * margin, max_lines=2)
        exp_banner_height = line_h * len(lines) + 18
        exp_banner_y = height - exp_banner_height
        cv2.rectangle(frame, (0, exp_banner_y), (width, height), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            y = exp_banner_y + 20 + i * line_h
            cv2.putText(frame, line, (margin, y), font, scale, (255, 255, 255), thick)

    return frame


def get_decision_color(decision_type):
    """Return a colour for each decision type."""
    type_lower = decision_type.lower()
    if "card" in type_lower:
        if "yellow" in type_lower:
            return (0, 255, 255)  # Yellow
        if "red" in type_lower:
            return (0, 0, 255)  # Red
    if "penalty" in type_lower:
        return (0, 140, 255)  # Orange
    if "try" in type_lower:
        return (0, 255, 0)  # Green
    if "advantage" in type_lower:
        return (255, 200, 0)  # Light blue
    if "scrum" in type_lower:
        return (255, 255, 0)  # Cyan
    return (255, 255, 255)  # White default


def main():
    parser = argparse.ArgumentParser(description="Merge decisions (and disc) onto a video")
    parser.add_argument("video", help="Tracked video, OR the raw clip if --tracking-json is given")
    parser.add_argument("decisions", help="Decisions JSON (classify_decisions or ground-truth format)")
    parser.add_argument("--output", "-o", default=None, help="Output video path")
    parser.add_argument("--duration", type=float, default=6.0,
                        help="How long each decision banner stays on screen (seconds)")
    parser.add_argument("--tracking-json", default=None,
                        help="Draw the ref disc from this tracking JSON (e.g. track_ref_colour.py "
                             "output) onto the RAW clip — one-command finished render.")
    parser.add_argument("--max-gap", type=int, default=14,
                        help="Bridge disc gaps up to N frames; longer = true occlusion, no disc "
                             "(default 14 ≈ 0.5s)")
    parser.add_argument("--follow", action="store_true",
                        help="Render a ref-centred zoom (follow-cam) instead of the full field — "
                             "more watchable when the ref is small. Needs --tracking-json.")
    parser.add_argument("--follow-zoom", type=float, default=1.0,
                        help="Follow-cam zoom: >1 = WIDER (more context, ref smaller), "
                             "<1 = tighter. (default 1.0)")
    parser.add_argument("--banner-offset", type=float, default=0.0,
                        help="Delay banners by N seconds after the classified timestamp — "
                             "useful when the classifier timestamps the whistle but you want "
                             "the overlay to appear once the signal is visible (default 0)")
    args = parser.parse_args()

    output = args.output or str(Path(args.video).stem) + "_final.mp4"
    merge_decisions_onto_video(args.video, args.decisions, output, args.duration,
                               tracking_json=args.tracking_json, max_gap=args.max_gap,
                               follow=args.follow, follow_zoom=args.follow_zoom,
                               banner_offset=args.banner_offset)


if __name__ == "__main__":
    main()
