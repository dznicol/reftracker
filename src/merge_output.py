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


def merge_decisions_onto_video(video_path, decisions_path, output_path, display_duration=5.0):
    """
    Overlay decision labels onto the tracked video at the appropriate timestamps.

    Each decision appears as a text banner for `display_duration` seconds.
    """
    video_path = Path(video_path)
    decisions_path = Path(decisions_path)
    output_path = Path(output_path)

    # Load decisions
    with open(decisions_path) as f:
        data = json.load(f)
    decisions = data.get("decisions", [])

    # Parse decision timestamps into frame numbers
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Convert timestamp strings to frame numbers
    timed_decisions = []
    for d in decisions:
        if d.get("decision_type") == "play_on":
            continue

        ts = d.get("timestamp_approx", "0:00")
        offset = d.get("segment_offset_seconds", 0)

        # Parse "M:SS" format
        try:
            parts = ts.replace(".", ":").split(":")
            if len(parts) == 2:
                seconds = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 1:
                seconds = int(parts[0])
            else:
                seconds = 0
        except (ValueError, IndexError):
            seconds = 0

        total_seconds = seconds + offset
        start_frame = int(total_seconds * fps)
        end_frame = int((total_seconds + display_duration) * fps)

        timed_decisions.append({
            "start_frame": start_frame,
            "end_frame": min(end_frame, total_frames),
            "type": d.get("decision_type", "unknown").replace("_", " ").title(),
            "explanation": d.get("explanation", ""),
            "team_against": d.get("team_penalised", ""),
            "team_for": d.get("team_benefiting", ""),
        })

    print(f"Overlaying {len(timed_decisions)} decisions onto video...")

    # Setup output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Show only the latest active decision (avoids ghost text overlap)
        active = [d for d in timed_decisions
                  if d["start_frame"] <= frame_idx <= d["end_frame"]]
        if active:
            frame = draw_decision_banner(frame, active[-1], width, height)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 200 == 0:
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

    cv2.putText(frame, top_text, (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

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
    parser = argparse.ArgumentParser(description="Merge decisions onto tracked video")
    parser.add_argument("video", help="Tracked video from track_ref.py")
    parser.add_argument("decisions", help="Decisions JSON from classify_decisions.py")
    parser.add_argument("--output", "-o", default=None, help="Output video path")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="How long each decision label stays on screen (seconds)")
    args = parser.parse_args()

    output = args.output or str(Path(args.video).stem) + "_final.mp4"

    merge_decisions_onto_video(args.video, args.decisions, output, args.duration)


if __name__ == "__main__":
    main()
