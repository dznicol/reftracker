# RefTracker

AI referee tracking and decision classification for rugby union. Combines local computer vision with LLM-based decision analysis.

**The pitch:** "I love watching rugby but half the time I don't understand why the ref blew the whistle. So I built a tool that watches the ref and explains what they're doing."

## What it does

1. **Tracks the referee** in wide-angle match footage using YOLOv8 + BoTSORT with CLIP-ReID appearance matching
2. **Classifies decisions** by sending video segments to Gemini 2.5 Flash with official World Rugby signal reference images
3. **Overlays explanations** onto the video with decision banners and plain-English descriptions
4. **Generates a heatmap** of referee movement across the pitch

## Architecture

```
Video Input (wide-angle Veo footage)
    |
    +-> track_ref.py (LOCAL - runs on laptop)
    |   +-- YOLOv8n person detection
    |   +-- BoTSORT tracking with CLIP-ReID appearance matching
    |   +-- Grass-masked HSV colour identification (referee kit colour)
    |   +-- Continuous primary colour verification
    |   +-- Supervision: bounding box + dot trail
    |   +-- Output: annotated .mp4, heatmap .png, tracking .json
    |
    +-> classify_decisions.py (CLOUD API - Gemini 2.5 Flash)
    |   +-- Uploads video segments to Gemini
    |   +-- Includes 16 official signal reference images
    |   +-- Returns timestamped decision JSON
    |
    +-> merge_output.py (LOCAL - OpenCV)
        +-- Overlays decision banners onto tracked video
        +-- Top banner: decision type + team info
        +-- Bottom banner: plain-English explanation
        +-- Output: final annotated .mp4
```

## Key design decisions

- **Local CV is the hero, Gemini is enrichment.** The tracking pipeline works without Gemini. Real engineering is in the CV pipeline, not API calls.
- **$0 total cost.** YOLOv8, OpenCV, Supervision, BoTSORT are free/open-source. Gemini free tier gives 250 requests/day.
- **Wide-angle footage required.** Broadcast highlights don't work because the camera follows the ball. Need an elevated fixed camera like Veo.
- **CLIP-ReID over OSNet.** CLIP's semantic understanding of appearance outperforms surveillance-trained ReID models for sports footage where kit colour is the key differentiator.

## Tracking approach

Referee identification uses a multi-signal approach:
- **Grass-masked HSV colour detection** — masks out bright pitch-green pixels before scoring kit colour, preventing the green grass visible in every bounding box from fooling the detector
- **Dual-colour scoring** — scores for primary colour (green) AND secondary colour (black) in 70/30 weighting
- **Spatial isolation bonus** — refs tend to be alone, not clustered with teammates
- **Continuous primary colour verification** — rejects BoTSORT track assignments where the tracked person doesn't show the referee's primary kit colour
- **Proximity-gated re-identification** — when the track is lost, searches within a velocity-scaled radius

## Setup

```bash
# Clone and install
git clone git@github.com:dznicol/reftracker.git
cd reftracker
uv sync

# Set up Gemini API key (free at https://aistudio.google.com/apikey)
echo "GOOGLE_API_KEY=your-key-here" > .env

# Model weights auto-download on first run
```

## Usage

```bash
# Track referee in video
uv run python src/track_ref.py videos/your_match.mp4 \
  --output output/tracked.mp4 \
  --heatmap output/heatmap.png \
  --ref-colour green

# Classify decisions with Gemini
uv run python src/classify_decisions.py videos/your_match.mp4 \
  --output output/decisions.json

# Merge decisions onto tracked video
uv run python src/merge_output.py output/tracked.mp4 output/decisions.json \
  --output output/final.mp4
```

### Referee colour options

`--ref-colour` supports: `green`, `black`, `yellow`, `red`, `blue`, `white`

## Current accuracy

### Tracking
- ~87% of frames tracked on test footage (62s, 1920x1080, 29fps)
- Handles referee turning sideways, entering player clusters
- Known limitation: loses lock when referee enters dense breakdowns and re-ID can select wrong player

### Decision classification
- Gemini correctly identifies **when** decisions happen (timestamps within ~3s)
- Signal classification needs improvement — struggles to distinguish knock forward from penalty at wide-angle camera distance
- Team assignment unreliable — Gemini doesn't inherently know which team occupies which side of the pitch

## Planned improvements

- **Double-pass architecture**: Use referee velocity from tracking data to identify "ref stopped" windows, extract short clips around each stop, send only those to Gemini. Reduces hallucination and improves classification accuracy.
- **Player clustering for team assignment**: In rugby, players stay behind the ball (forward passing is illegal). The majority of same-coloured players clustered on one side reveals which team occupies that half, enabling correct team assignment for decisions.

## Tech stack

- [YOLOv8](https://github.com/ultralytics/ultralytics) — person detection
- [BoxMOT](https://github.com/mikel-brostrom/boxmot) — BoTSORT tracking with CLIP-ReID
- [Supervision](https://github.com/roboflow/supervision) — visual annotations
- [OpenCV](https://opencv.org/) — video processing
- [Gemini 2.5 Flash](https://ai.google.dev/) — decision classification
- [uv](https://github.com/astral-sh/uv) — Python package management

## Documentation

- [Technical Glossary](docs/glossary.md) — definitions of CV, tracking, and classification terms used in this project
