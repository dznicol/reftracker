#!/usr/bin/env python3
"""
Build the README `tracking_levels.gif` composite frames: one 8-second window
rendered at three zoom levels (original wide view, match view, ref close-up),
all from a single colour-first tracking pass. Panels use smooth follow-crops;
the disc is drawn natively at each panel's resolution with person-occlusion
masking (disc composites behind players).

Usage:
    uv run python scripts/make_tracking_levels.py <clip.mp4> <tracked.json> \
        <outdir> [--start 261] [--end 492] [--frames 64]
Then assemble with ffmpeg (see scripts/make_readme_gifs.sh).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from discmark import smooth_disc_track, draw_disc, PersonMasker, _gauss  # noqa: E402
from merge_output import _disc_positions  # noqa: E402

CANVAS_W, CANVAS_H = 480, 464
WIDE_H, PANEL_W, PANEL_H, GUTTER = 270, 238, 190, 4


def label(img, text):
    for c, t in [((0, 0, 0), 3), ((255, 255, 255), 1)]:
        cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, t,
                    cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("tracking_json")
    ap.add_argument("outdir")
    ap.add_argument("--start", type=int, default=261)
    ap.add_argument("--end", type=int, default=492)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--fps", type=int, default=29)
    args = ap.parse_args()

    track = smooth_disc_track(_disc_positions(args.tracking_json, 14), args.fps)
    masker = PersonMasker()

    kf = sorted(f for f in track if args.start - 40 <= f <= args.end + 40)
    allf = np.arange(args.start, args.end + 1)
    cam_x = np.interp(allf, kf, [track[f][0] for f in kf])
    cam_y = np.interp(allf, kf, [track[f][1] for f in kf])
    # (name, cam smoothing sigma, source-crop w/h)
    views = {"Match view": (_gauss(cam_x, 20), _gauss(cam_y, 20), 360, 288),
             "Ref close-up": (_gauss(cam_x, 10), _gauss(cam_y, 10), 150, 119)}

    cap = cv2.VideoCapture(args.clip)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    for k in range(args.frames):
        n = args.start + round(k * (args.end - args.start) / (args.frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, fr = cap.read()
        assert ok, f"frame {n}"
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)

        wide = cv2.resize(fr, (CANVAS_W, WIDE_H), interpolation=cv2.INTER_AREA)
        if n in track:
            cx, fy, rx, a = track[n]
            draw_disc(wide, cx * CANVAS_W / W, fy * WIDE_H / H,
                      max(rx * CANVAS_W / W, 4.5), alpha=a)
        label(wide, "Original (wide) view")
        canvas[0:WIDE_H] = wide

        for i, (name, (cxs, cys, cw, chh)) in enumerate(views.items()):
            x1 = int(np.clip(cxs[n - args.start] - cw / 2, 0, W - cw))
            y1 = int(np.clip(cys[n - args.start] - 0.62 * chh, 0, H - chh))
            panel = cv2.resize(fr[y1:y1 + chh, x1:x1 + cw], (PANEL_W, PANEL_H),
                               interpolation=cv2.INTER_CUBIC)
            if n in track:
                cx, fy, rx, a = track[n]
                cxd, fyd = (cx - x1) * PANEL_W / cw, (fy - y1) * PANEL_H / chh
                rxd = rx * PANEL_W / cw
                occ = masker.mask(panel, cxd, fyd, rxd)
                draw_disc(panel, cxd, fyd, rxd, alpha=a, occlusion=occ,
                          gap_deg=0 if occ is not None else 80)
            label(panel, name)
            canvas[WIDE_H + GUTTER:, i * (PANEL_W + GUTTER):
                   i * (PANEL_W + GUTTER) + PANEL_W] = panel

        cv2.imwrite(str(out / f"f{k:03d}.png"), canvas)
    cap.release()
    print(f"{args.frames} composite frames -> {out}")


if __name__ == "__main__":
    main()
