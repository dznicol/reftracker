# RefTracker

AI referee tracking tool for rugby. Uses YOLOv8 + BoTSORT (CLIP-ReID) + Supervision for local CV, Gemini 2.5 Flash for decision classification.

## Project structure
- `src/` — Python source code
  - `track_ref_colour.py` — COLOUR-FIRST tracker (recommended for colour-unique refs; no BoTSORT)
  - `track_ref.py` — BoTSORT tracker (fallback) + auto-calibrated tight kit colour
  - `classify_decisions.py` — Gemini decision classification with signal reference images
  - `merge_output.py` — overlay disc (from tracking JSON) + decision banners; `--follow` follow-cam
  - `discmark.py` — broadcast-quality disc renderer (offline zero-lag smoothing + sub-pixel AA layered ellipse + YOLOv8n-seg person occlusion so the disc draws BEHIND players); used by merge_output + track_ref_colour
  - `evaluate.py` — score tracking + decisions (detection-vs-classification split, FP traps)
  - `detect_whistle.py` — non-ML whistle detector (two-pass Step-2 trigger; needs audio)
- `signals/` — 52 official World Rugby referee signal reference images
- `docs/` — glossary and documentation
- `output/` — generated outputs (not committed)
- `videos/` — source video (not committed)

## Running
```bash
cd ~/sandbox/reftracker

# Track referee  (--marker disc|box|both; disc = TV-style ellipse at the feet, default)
#   --imgsz 1280   detect small/distant refs in wide shots (slower)
#   --ref-start X,Y  seed the ref by location when colour auto-ID is ambiguous
#   --hold-frames 15  hold the disc through brief occlusions;  --trail  re-enable dot trail
uv run python src/track_ref.py videos/veo_sample.mp4 --output output/tracked.mp4 --heatmap output/heatmap.png --ref-colour green --marker disc

# Track referee — COLOUR-FIRST (no BoTSORT): use when the ref's shirt colour is
# UNIQUE (no player wears it). Detects that colour every frame + interpolates gaps.
# More robust than track_ref.py for colour-unique refs (no MOT identity drift).
uv run python src/track_ref_colour.py videos/clip.mp4 --output output/tracked.mp4 --hue 75 99 --imgsz 1280

# Classify decisions with Gemini
#   --model       pick the Gemini model (e.g. gemini-3-flash, gemini-3.5-flash)
#   --ref-marker  set disc/box when feeding the TRACKED video (ref pre-highlighted)
uv run python src/classify_decisions.py videos/veo_sample.mp4 --output output/decisions.json
# Experiment: feed the tracked video so Gemini doesn't have to re-find the ref
uv run python src/classify_decisions.py output/tracked.mp4 --output output/decisions_marked.json --ref-marker disc --model gemini-3.5-flash

# Finished render: disc (from tracking JSON) + decision banners onto the RAW clip.
#   --tracking-json  draw the disc from a track JSON (e.g. track_ref_colour output)
#   --follow [--follow-zoom 1.4]  smooth ref-centred follow-cam (zoom >1 = wider)
#   --max-gap N      bridge only short gaps; true occlusion shows no disc
#   --banner-offset N  delay banners N seconds after classified timestamp (aligns overlay with visible signal)
#   --disc-colour HEX  disc colour (default #2EE86E)
#   --no-occlude       skip person-masking (drops the behind-players compositing;
#                      falls back to an open-top arc ring; faster render)
uv run python src/merge_output.py output/tracked.mp4 output/decisions.json --output output/final.mp4
uv run python src/merge_output.py videos/clip.mp4 output/decisions.json --tracking-json output/tracked.json --follow --output output/final_follow.mp4
```

## Evaluation (measure changes instead of eyeballing)
```bash
# Tracking accuracy: extract labelled stills, fill `correct` (1/0) in the CSV, score
uv run python src/evaluate.py tracking-template output/tracked.mp4 --frames 40
uv run python src/evaluate.py score-tracking output/eval/tracking_labels.csv

# Decision accuracy: write a ground-truth stub, fill in the REAL decisions, score
uv run python src/evaluate.py decisions-template
uv run python src/evaluate.py score-decisions output/decisions.json output/eval/decisions_truth.json --tolerance 3
```

## Model / SDK note
The installed `google-generativeai` SDK is **deprecated** but still works. Verified
2026-05-30: `gemini-3.5-flash` works on it (GA) — no migration needed for the bump.
Gotchas: `gemini-3-flash` 404s (use `gemini-3-flash-preview`); `gemini-2.5-flash`
still works. Migrating to `google-genai` is optional housekeeping, not urgent.

## Key notes
- Needs `GOOGLE_API_KEY` in `.env` at project root for Gemini
- Model weights (`yolov8n.pt`, `clip_market1501.pt`) auto-download on first run
- Uses MPS (Apple Silicon GPU) when available
