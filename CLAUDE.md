# RefTracker

AI referee tracking tool for rugby. Uses YOLOv8 + BoTSORT (CLIP-ReID) + Supervision for local CV, Gemini 2.5 Flash for decision classification.

## Project structure
- `src/` — Python source code
  - `track_ref.py` — core tracking pipeline (YOLOv8 + BoTSORT + colour verification)
  - `classify_decisions.py` — Gemini decision classification with signal reference images
  - `merge_output.py` — overlay decisions onto tracked video
- `signals/` — 52 official World Rugby referee signal reference images
- `docs/` — glossary and documentation
- `output/` — generated outputs (not committed)
- `videos/` — source video (not committed)

## Running
```bash
cd ~/sandbox/reftracker

# Track referee
uv run python src/track_ref.py videos/veo_sample.mp4 --output output/tracked.mp4 --heatmap output/heatmap.png --ref-colour green

# Classify decisions with Gemini
uv run python src/classify_decisions.py videos/veo_sample.mp4 --output output/decisions.json

# Merge decisions onto tracked video
uv run python src/merge_output.py output/tracked.mp4 output/decisions.json --output output/final.mp4
```

## Key notes
- Needs `GOOGLE_API_KEY` in `.env` at project root for Gemini
- Model weights (`yolov8n.pt`, `clip_market1501.pt`) auto-download on first run
- Uses MPS (Apple Silicon GPU) when available
