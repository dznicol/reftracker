#!/bin/bash
# Regenerate the three README GIFs (docs/images/) from the Glasgow-vs-Aberdeen
# showcase sequence. Run from the repo root. The dznicol profile repo uses a
# copy of tracking_levels.gif as images/reftracker-tracking.gif.
#
# Windows below were recovered by frame-matching the original GIFs against the
# source renders (29 fps):
#   follow_demo       frames  363-537  (12.5s, 6s)  - penalty banner second half
#   decision_freekick frames 2637-2782 (90.9s, 5s)  - banner fires ~2s in (the
#                     render uses --banner-offset 2 so it lands on the arm signal)
#   tracking_levels   frames  261-492  (8s realtime sampled at 8 fps)
set -euo pipefail

D=output/glasgow-2s-vs-aberdeen-2025-10-29-2026-01-25/01-31-20_01-33-00
TMP=$(mktemp -d)
PAL="split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"

# 1. Finished follow-cam renders (disc + banners on the raw clip)
uv run python src/merge_output.py "$D/clip.mp4" "$D/decisions.json" \
    --tracking-json "$D/tracked_colour.json" --follow --output "$TMP/follow.mp4"
uv run python src/merge_output.py "$D/clip.mp4" "$D/decisions.json" \
    --tracking-json "$D/tracked_colour.json" --follow --banner-offset 2 \
    --output "$TMP/follow_offset2.mp4"

# 2. Three-panel tracking composite frames
uv run python scripts/make_tracking_levels.py "$D/clip.mp4" \
    "$D/tracked_colour.json" "$TMP/levels" --start 261 --end 492

# 3. Cut + palette-optimise the GIFs
ffmpeg -v error -ss 12.5172 -t 6 -i "$TMP/follow.mp4" \
    -filter_complex "[0:v]fps=10,scale=380:-1:flags=lanczos,$PAL" \
    -y docs/images/follow_demo.gif
ffmpeg -v error -ss 90.931 -t 5 -i "$TMP/follow_offset2.mp4" \
    -filter_complex "[0:v]fps=12,scale=420:343:flags=lanczos,$PAL" \
    -y docs/images/decision_freekick.gif
ffmpeg -v error -framerate 8 -i "$TMP/levels/f%03d.png" \
    -filter_complex "[0:v]$PAL" -y docs/images/tracking_levels.gif

rm -rf "$TMP"
ls -la docs/images/*.gif
