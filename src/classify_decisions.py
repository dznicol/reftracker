#!/usr/bin/env python3
"""
RefTracker - classify_decisions.py
Sends video segments to Gemini 2.5 Flash to classify referee decisions.
Outputs timestamped decision JSON that can be merged with tracking data.

Usage:
    python classify_decisions.py video_segment.mp4 [--output decisions.json]
    python classify_decisions.py videos/segments/ [--output decisions.json]  # process directory

Requirements:
    pip install google-generativeai

Environment:
    export GOOGLE_API_KEY=your_key_here
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

REFEREE_PROMPT = """You are analysing rugby union match footage. The referee wears a GREEN
shirt with black short sleeves and black shorts. Find and watch ONLY the referee.

ABSOLUTE RULES — violations will produce wrong output:
1. A decision REQUIRES the referee to BLOW THE WHISTLE and make a HAND SIGNAL.
   No whistle = no decision. The referee must stop or pause play.
2. The referee RUNNING while watching play is NOT a decision, even if an arm is out.
3. Do NOT guess or infer what happened in the game. ONLY report what you SEE the
   referee's hands and arms doing. Fewer accurate decisions beats many wrong ones.

NOT DECISIONS — do NOT report these:
- Arm extended PARALLEL to the gain line while play continues = the referee is
  showing the offside line to the backs. This is game management, not a decision.
  The arm runs along the field width, not pointing at either team. NO WHISTLE.
- Referee running with one arm slightly out = balance/pointing while running
- Referee talking, pointing at players, or repositioning = game management
- Ball still in play and referee has not blown whistle = no decision happening

ACTUAL DECISIONS — the referee STOPS, BLOWS WHISTLE, then signals:

Decisions often come in a SEQUENCE of two signals:
  First: the INFRINGEMENT signal (what went wrong)
  Then: the RESTART signal (what happens next)

Example sequence for a knock forward:
  1. Ref blows whistle, stops play
  2. One arm raised with hand bent, moving back and forth = KNOCK FORWARD
  3. Then one arm extended HORIZONTALLY out to the side = SCRUM AWARDED
     (the horizontal arm points the direction of the put-in)
  Report these as TWO separate decisions at their respective timestamps.

CRITICAL — do not confuse these two, they look different:
- SCRUM AWARDED: ONE arm extended roughly HORIZONTAL, out to the side, pointing
  the put-in direction (toward the team awarded the feed). The arm is LOW/level,
  NOT above the head. Whistle in the other hand. (Both-arms-above-the-head is NOT
  a scrum signal — do not report a scrum just because an arm is raised high.)
- PENALTY: ONE arm raised straight and HIGH (upper arm up, around 45-90 degrees),
  angled TOWARD the non-offending team being awarded the penalty. Whistle in the
  other hand. A single high, raised arm held up + whistle = PENALTY, not a scrum.

Key signals:
- KNOCK FORWARD: arm raised, hand/forearm bent, sweeping/rolling forward motion
- TRY: arm raised straight VERTICAL above the head, usually at/near the goal line
  while running toward the posts (context: at the try line, not mid-field)
- ADVANTAGE: arm extended horizontally toward the team gaining advantage, held steady
- FREE KICK: arm bent at elbow, raised upward
- NOT ROLLING AWAY: rolling hand gesture near the ground
- HIGH TACKLE: hand sweeping across neck/shoulder area
- NOT RELEASING: arms mimicking hugging/holding motion

I have attached official World Rugby signal reference images for comparison.

For each decision, return a JSON object with:
- timestamp_approx: time in segment (e.g. "0:55" or "1:02"). Be PRECISE — watch
  the exact moment the referee makes each signal, not when you think something happened.
- signal_observed: describe EXACTLY what you see the ref doing with arms/hands/body
- decision_type: one of:
  penalty_awarded, penalty_advantage, scrum_awarded, scrum_reset, scrum_penalty,
  lineout_call, try_awarded, no_try, held_up, tmo_review, yellow_card, red_card,
  knock_on, forward_pass, offside, not_releasing, not_rolling, high_tackle,
  ruck_infringement, advantage_over, play_on, conversion, free_kick, kick_off, other
  NOTE: a try is only AWARDED if the grounding is clear. If the ref raises an arm as
  if signalling a try but then crosses/waves the arms or inspects and waves it away,
  that is NO_TRY (or HELD_UP if the defenders held the ball up) — report no_try, NOT
  try_awarded. Watch for the REVERSAL after an initial raised arm.
- team_penalised: which team was penalised (by kit colour), or "N/A"
- team_benefiting: which team benefited, or "N/A"
- ref_position: "at the breakdown", "5m away", "10m+ away", "far side", "behind play", "touchline"
- line_of_sight: "clear" / "partially_obstructed" / "obstructed" / "behind_play"
- reaction_time: "immediate" / "slight_delay" / "delayed" / "after_consultation"
- explanation: plain-English explanation for someone watching their first rugby match.
  Describe the signal and what it means. Be specific about what you SEE.
- confidence: float 0-1

If no clear decision signals are visible, return:
[{"decision_type": "play_on", "explanation": "Open play, no referee decisions.", "confidence": 0.9}]

Return ONLY valid JSON, no markdown formatting or code blocks."""


# Prepended to the prompt when the input is a *tracked* video (output of track_ref.py),
# where the referee has already been located and highlighted. Lets Gemini skip the
# "find the ref" step and focus entirely on reading the hand signals.
MARKER_NOTES = {
    "disc": (
        "IMPORTANT: The referee has ALREADY been identified for you. In this video the "
        "referee is highlighted with a bright GREEN ELLIPSE/DISC drawn at their feet and a "
        "'REF' label, plus a trail of green dots showing where they have been. The person "
        "with this green disc IS the referee — follow ONLY them. Do not try to find the "
        "referee yourself; trust the marker. The disc is at the feet and never covers the "
        "hands, so read the arm/hand signals normally.\n\n"
    ),
    "box": (
        "IMPORTANT: The referee has ALREADY been identified for you. In this video the "
        "referee is highlighted with a bright GREEN BOUNDING BOX and a 'REF' label, plus a "
        "trail of green dots showing where they have been. The person inside this green box "
        "IS the referee — follow ONLY them. Do not try to find the referee yourself; trust "
        "the marker.\n\n"
    ),
}


# Signal images to include as visual references for Gemini
# Primary signals (always include) + key secondary signals
SIGNAL_IMAGES = [
    # Primary
    "primary/penalty.png",
    "primary/try_and_penalty_try.png",
    "primary/scrum.png",
    "primary/advantage.png",
    "primary/free_kick.png",
    "primary/drop_out.png",
    "primary/no_try.png",
    # Key secondary
    "secondary/knock_forward.png",
    "secondary/high_tackle_foul_play.png",
    "secondary/not_releasing_ball_tackle.png",
    "secondary/not_rolling_away.png",
    "secondary/offside_scrum_ruck_maul.png",
    "secondary/forward_pass.png",
    "secondary/collapsing_ruck_or_maul.png",
    # Key other
    "other/yellow_card.png",
    "other/red_card.png",
]


def classify_segment(video_path, model, segment_offset=0.0, prompt=REFEREE_PROMPT):
    """
    Upload a video segment to Gemini and get referee decision classifications.

    Args:
        video_path: Path to the video segment
        model: Gemini model instance
        segment_offset: Time offset in seconds (for multi-segment processing)
        prompt: Instruction prompt to send with the video

    Returns:
        List of decision dicts with adjusted timestamps
    """
    video_path = Path(video_path)
    print(f"  Uploading {video_path.name}...")

    # Upload video file to Gemini
    video_file = genai.upload_file(str(video_path), mime_type="video/mp4")

    # Wait for processing
    print(f"  Waiting for Gemini to process video...")
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        print(f"  Error: Gemini failed to process {video_path.name}")
        return []

    # Load signal reference images
    signals_dir = Path(__file__).parent.parent / "signals"
    signal_files = []
    for img_rel in SIGNAL_IMAGES:
        img_path = signals_dir / img_rel
        if img_path.exists():
            signal_files.append(genai.upload_file(str(img_path), mime_type="image/png"))
    if signal_files:
        print(f"  Attached {len(signal_files)} signal reference images")

    # Send to model: video + signal images + prompt
    print(f"  Classifying referee decisions...")
    content = [video_file] + signal_files + [prompt]
    response = model.generate_content(
        content,
        generation_config=genai.GenerationConfig(
            temperature=0.0,  # Deterministic — reruns are comparable for eval
            max_output_tokens=8192,
        ),
    )

    # Parse response
    try:
        text = response.text.strip()
        # Handle potential markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        decisions = json.loads(text)
    except (json.JSONDecodeError, IndexError) as e:
        print(f"  Warning: Could not parse Gemini response as JSON: {e}")
        print(f"  Raw response: {response.text[:500]}")
        return []

    # Adjust timestamps for segment offset
    for decision in decisions:
        decision["segment_file"] = video_path.name
        decision["segment_offset_seconds"] = segment_offset

    # Clean up uploaded files
    for f in [video_file] + signal_files:
        try:
            genai.delete_file(f.name)
        except Exception:
            pass  # Best effort cleanup

    print(f"  Found {len(decisions)} decision(s) in {video_path.name}")
    return decisions


def process_video(video_path, output_path, model_name="gemini-2.5-flash",
                  ref_marker="none", ref_colour="green"):
    """
    Process a single video file or directory of segments.

    Args:
        video_path: Video file or directory of segments
        output_path: Where to write the decisions JSON
        model_name: Gemini model to use (e.g. "gemini-2.5-flash", "gemini-3-flash")
        ref_marker: "none" for raw video, or "disc"/"box" when feeding a *tracked*
            video where the ref is already highlighted (adds a hint to the prompt)
    """
    # Configure Gemini
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Get a free key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    genai.configure(api_key=api_key)
    print(f"Using model: {model_name}")
    model = genai.GenerativeModel(model_name)

    # Build the prompt — prepend a marker hint when the input is a tracked video
    prompt = MARKER_NOTES.get(ref_marker, "") + REFEREE_PROMPT
    if ref_marker in MARKER_NOTES:
        print(f"Ref is pre-marked ({ref_marker}) — added marker hint to prompt")
    # Tell Gemini the ref's actual kit colour (prompt is written for green)
    if ref_colour and ref_colour.lower() != "green":
        prompt = prompt.replace("a GREEN shirt", f"a {ref_colour.upper()} shirt")
        print(f"Ref kit colour set to {ref_colour.upper()} in prompt")

    video_path = Path(video_path)
    all_decisions = []

    if video_path.is_dir():
        # Process directory of segments
        segments = sorted(video_path.glob("*.mp4"))
        if not segments:
            print(f"No .mp4 files found in {video_path}")
            sys.exit(1)

        print(f"Processing {len(segments)} video segments...")
        segment_duration = 300  # Assume 5-min segments

        for i, segment in enumerate(segments):
            offset = i * segment_duration
            print(f"\nSegment {i + 1}/{len(segments)}: {segment.name}")
            decisions = classify_segment(segment, model, segment_offset=offset,
                                         prompt=prompt)
            all_decisions.extend(decisions)

            # Rate limiting - be nice to free tier
            if i < len(segments) - 1:
                print("  Waiting 3s before next segment...")
                time.sleep(3)
    else:
        # Process single video file
        print(f"Processing: {video_path.name}")
        all_decisions = classify_segment(video_path, model, prompt=prompt)

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "source": str(video_path),
        "model": model_name,
        "ref_marker": ref_marker,
        "total_decisions": len(all_decisions),
        "decisions": all_decisions,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved: {output_path}")
    print(f"Total decisions classified: {len(all_decisions)}")

    # Print summary
    decision_types = {}
    for d in all_decisions:
        dt = d.get("decision_type", "unknown")
        decision_types[dt] = decision_types.get(dt, 0) + 1

    print("\nDecision summary:")
    for dt, count in sorted(decision_types.items(), key=lambda x: -x[1]):
        print(f"  {dt}: {count}")

    return all_decisions


def main():
    parser = argparse.ArgumentParser(
        description="RefTracker - Gemini Referee Decision Classification"
    )
    parser.add_argument(
        "video",
        help="Video file or directory of segments to classify"
    )
    parser.add_argument(
        "--output", "-o",
        default="decisions.json",
        help="Output JSON file path (default: decisions.json)"
    )
    parser.add_argument(
        "--model", default="gemini-2.5-flash",
        help="Gemini model name, e.g. gemini-2.5-flash, gemini-3-flash, "
             "gemini-3.5-flash (default: gemini-2.5-flash). NOTE: the installed "
             "google-generativeai SDK is deprecated; if a gemini-3.x string 404s, "
             "the API version it targets may not serve that model yet — migrate to "
             "the google-genai SDK."
    )
    parser.add_argument(
        "--ref-marker", default="none", choices=["none", "disc", "box"],
        help="Set to 'disc' or 'box' when the input is a TRACKED video from "
             "track_ref.py (ref already highlighted). Adds a prompt hint so Gemini "
             "trusts the marker instead of re-finding the ref. (default: none)"
    )
    parser.add_argument(
        "--ref-colour", default="green",
        help="Referee kit colour, injected into the prompt (default: green). "
             "Use e.g. 'cyan' for the Aberdeen match."
    )
    args = parser.parse_args()

    process_video(args.video, args.output, model_name=args.model,
                  ref_marker=args.ref_marker, ref_colour=args.ref_colour)


if __name__ == "__main__":
    main()
