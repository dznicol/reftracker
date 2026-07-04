#!/usr/bin/env python3
"""
RefTracker - discmark.py
Broadcast-quality underfoot disc marker: offline track smoothing + a layered,
sub-pixel anti-aliased ellipse (soft glow, translucent fill, dark contact rim,
crisp bright ring). Replaces supervision's EllipseAnnotator/LabelAnnotator,
whose raw per-frame integer rendering is what made the old disc jitter.

Two pieces:
  smooth_disc_track(bboxes, fps) -> {frame: (cx, feet_y, rx, alpha)}
      Median-despike + centered Gaussian smoothing per contiguous segment
      (zero-lag, since rendering happens after tracking), heavy smoothing on
      SIZE so the disc never pulses, and a short fade in/out at gaps.
  draw_disc(frame, cx, feet_y, rx, ...)
      Draw one disc at float coordinates (OpenCV fixed-point `shift` sub-pixel).
"""

import cv2
import numpy as np

_SHIFT = 6           # fixed-point bits for sub-pixel cv2 drawing
_F = 1 << _SHIFT
_ASPECT = 0.36       # ellipse minor/major axis ratio (TV perspective flattening)


# ─────────────────────────── track smoothing ───────────────────────────

def _medfilt(x, k):
    """Centered median filter (edge-padded) — kills 1-2 frame bbox pops."""
    if len(x) < k or k < 3:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(xp, k)
    return np.median(win, axis=1)


def _gauss(x, sigma):
    """Centered Gaussian smoothing (edge-padded). Zero lag on linear motion."""
    r = int(3 * sigma)
    if r < 1 or len(x) < 3:
        return x
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, "valid")


def _repair_feet(fy, hh, fps):
    """
    The bbox bottom stops being the ref's feet when he passes behind another
    player: YOLO either merges the two into one tall box (bottom = THAT
    player's feet — the disc visibly 'switches' to them) or crops him to his
    visible torso (bottom = his knees). Both leave a sustained height anomaly
    that the short despike filter can't catch, so: flag frames whose height
    strays >18% from a rolling ~4s median and re-interpolate feet_y/height
    across them like an occlusion gap. cx is left alone — the horizontal
    position stays good throughout.
    """
    n = len(hh)
    if n < 9:
        return fy, hh
    k = min((int(fps * 4)) | 1, n - (1 - n % 2))      # odd, <= segment length
    base = _medfilt(hh, k)
    bad = np.abs(hh - base) > 0.18 * base
    if not bad.any() or bad.all():
        return fy, hh
    idx = np.arange(n)
    good = np.flatnonzero(~bad)
    return np.interp(idx, good, fy[good]), np.interp(idx, good, hh[good])


def smooth_disc_track(bboxes, fps, pos_smooth=0.12, size_smooth=0.7, fade=0.25):
    """
    bboxes: {frame: [x1, y1, x2, y2]}  (gaps allowed — each contiguous run is
    smoothed independently so the disc never bleeds across true occlusions).
    Returns {frame: (cx, feet_y, rx, alpha)} with alpha ramping 0->1 over
    `fade` seconds at the start/end of every run.
    """
    if not bboxes:
        return {}
    frames = sorted(bboxes)
    segs, cur = [], [frames[0]]
    for f in frames[1:]:
        if f == cur[-1] + 1:
            cur.append(f)
        else:
            segs.append(cur)
            cur = [f]
    segs.append(cur)

    out = {}
    nfade = max(1, int(fps * fade))
    for seg in segs:
        bb = np.array([bboxes[f] for f in seg], np.float64)
        fy_r, hh_r = _repair_feet(bb[:, 3], bb[:, 3] - bb[:, 1], fps)
        cx = _gauss(_medfilt((bb[:, 0] + bb[:, 2]) / 2, 3), fps * pos_smooth)
        fy = _gauss(_medfilt(fy_r, 5), fps * pos_smooth)
        # size: despike hard + smooth long — bbox height pops 50px when limbs
        # clip in/out, and a pulsing disc is the most visible jitter of all
        hh = _gauss(_medfilt(hh_r, 9), fps * size_smooth)
        rx = np.clip(hh * 0.42, 6.0, None)
        for i, f in enumerate(seg):
            a = min(1.0, (i + 1) / nfade, (len(seg) - i) / nfade)
            out[f] = (float(cx[i]), float(fy[i]), float(rx[i]), a)
    return out


# ─────────────────────────── disc rendering ───────────────────────────

def _hex_to_bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def _ellipse_mask(shape, ctr, axes, thickness, a0=0, a1=360):
    """Anti-aliased ellipse mask as float32 0..1 at sub-pixel coordinates."""
    m = np.zeros(shape, np.uint8)
    cv2.ellipse(m, ctr, axes, 0, a0, a1, 255, thickness, cv2.LINE_AA, _SHIFT)
    return m.astype(np.float32) / 255.0


def draw_disc(frame, cx, feet_y, rx, colour=(110, 232, 46), alpha=1.0,
              gap_deg=0, occlusion=None):
    """
    Draw a broadcast-style underfoot disc on `frame` (in place).
    cx, feet_y, rx are FLOATS (sub-pixel); colour is BGR; alpha 0..1 fades the
    whole disc (used at track gaps so it never pops in/out).

    Two ways to keep the graphic off the player, broadcast-style:
    - occlusion: full-frame float32 mask 0..1 where 1 = person pixels; the
      disc is composited BEHIND them (the real broadcast look).
    - gap_deg: leave an open arc of this many degrees at the TOP of the
      ring (the supervision/EA-cursor trick) — for when no mask is available.
    """
    if alpha <= 0.01:
        return frame
    h, w = frame.shape[:2]
    ry = rx * _ASPECT
    t = max(2, int(round(rx * 0.09)))                # ring thickness
    pad = 3 * t + 6
    x0, y0 = int(cx - rx) - pad, int(feet_y - ry) - pad
    x1, y1 = int(np.ceil(cx + rx)) + pad, int(np.ceil(feet_y + ry)) + pad
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return frame
    roi = frame[y0:y1, x0:x1]
    base = roi.astype(np.float32)
    art = base.copy()
    shape = roi.shape[:2]

    ctr = (int(round((cx - x0) * _F)), int(round((feet_y - y0) * _F)))
    axes = (int(round(rx * _F)), int(round(ry * _F)))
    col = np.array(colour, np.float32)
    dark = col * 0.22                                # deep rim shade of the same hue
    # open arc: gap centred at the TOP (270 deg in OpenCV's y-down frame)
    a0, a1 = (270 + gap_deg / 2, 630 - gap_deg / 2) if gap_deg else (0, 360)

    # 1. translucent fill, weighted toward the rim (radial-gradient look)
    fill = _ellipse_mask(shape, ctr, axes, -1)
    inner = _ellipse_mask(shape, ctr,
                          (int(axes[0] * 0.55), int(axes[1] * 0.55)), -1)
    grad = fill * (1.0 - 0.75 * inner)
    art += (col - art) * (0.20 * grad)[..., None]

    # 2. soft outer glow under the ring
    glow = cv2.GaussianBlur(_ellipse_mask(shape, ctr, axes, 3 * t, a0, a1), (0, 0), t)
    art += (col - art) * (0.40 * glow)[..., None]

    # 3. dark contact rim (contrast against grass), slightly wider than the ring
    rim = cv2.GaussianBlur(_ellipse_mask(shape, ctr, axes, t + 3, a0, a1), (0, 0), 0.7)
    art += (dark - art) * (0.55 * rim)[..., None]

    # 4. crisp bright ring on top
    ring = _ellipse_mask(shape, ctr, axes, t, a0, a1)
    bright = np.clip(col * 1.12 + 28, 0, 255)
    art += (bright - art) * (0.95 * ring)[..., None]

    a = alpha
    if occlusion is not None:
        a = a * (1.0 - occlusion[y0:y1, x0:x1, None])
    out = base + (art - base) * a
    roi[:] = np.clip(out, 0, 255).astype(np.uint8)
    return frame


class PersonMasker:
    """
    Person-pixel occlusion masks so the disc composites BEHIND players —
    the real broadcast look. Runs YOLOv8n-seg on a small region around the
    disc (cheap enough for offline rendering); model loads lazily on first use.
    """

    def __init__(self, model_name="yolov8n-seg.pt"):
        self._model = None
        self._name = model_name

    def mask(self, frame, cx, feet_y, rx):
        """Float32 HxW mask (0..1, lightly feathered) of person pixels near
        the disc, or None if nobody is segmented there (= nothing occludes)."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self._name)
        h, w = frame.shape[:2]
        s = int(min(max(10 * rx, 320), 640))
        x0 = int(np.clip(cx - s / 2, 0, max(0, w - s)))
        y0 = int(np.clip(feet_y - s * 0.6, 0, max(0, h - s)))
        crop = frame[y0:y0 + s, x0:x0 + s]
        if crop.size == 0:
            return None
        r = self._model(crop, classes=[0], imgsz=640, verbose=False,
                        retina_masks=True)[0]
        if r.masks is None or not len(r.masks.xy):
            return None
        sub = np.zeros(crop.shape[:2], np.uint8)
        # depth cull: only people standing ON or IN FRONT of the disc occlude
        # it — someone behind the far edge is farther from camera than the
        # disc, so the disc should draw over their pixels, not vice versa
        far_edge = (feet_y - rx * _ASPECT) - y0
        for poly in r.masks.xy:
            if len(poly) >= 3 and float(poly[:, 1].max()) >= far_edge:
                cv2.fillPoly(sub, [poly.astype(np.int32)], 255)
        m = np.zeros((h, w), np.float32)
        m[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = sub.astype(np.float32) / 255
        return cv2.GaussianBlur(m, (0, 0), 1.0)
