#!/usr/bin/env python3
"""
RefTracker - track_ref.py
Detects and tracks the referee in rugby match footage using YOLOv8 + BoTSORT
(boxmot, with CLIP-ReID appearance features) + Supervision. Outputs an annotated
video with a TV-style underfoot disc (or box), and an optional heatmap.

BoTSORT keeps appearance + motion + camera-motion-compensated (ECC) track IDs.
On top of it we identify the ref by auto-calibrated kit colour and re-acquire by
colour each frame.

NOTE: when the ref's shirt is a colour NO player wears, the multi-object identity
tracker is unnecessary (and its identity-drift can stick the lock to the wrong
player) — use the simpler colour-first `track_ref_colour.py` instead.

Usage:
    python track_ref.py input_video.mp4 [--output output.mp4] [--heatmap heatmap.png]

Requirements:
    pip install ultralytics opencv-python supervision numpy boxmot
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from boxmot import BotSort
from pathlib import Path
import supervision as sv


def _get_grass_mask(hsv):
    """
    Create a mask of bright outdoor grass pixels.
    Grass is typically high-saturation, high-value green — brighter and more
    vivid than kit fabric green.  We exclude these pixels before scoring kit colour.
    """
    grass_lower = np.array([35, 80, 100])
    grass_upper = np.array([80, 255, 255])
    return cv2.inRange(hsv, grass_lower, grass_upper)


def _colour_ratio_excluding_grass(hsv, colour_bounds):
    """
    Return the fraction of non-grass pixels that match `colour_bounds`.
    This prevents pitch grass from inflating green kit scores.
    """
    grass_mask = _get_grass_mask(hsv)
    non_grass = cv2.bitwise_not(grass_mask)
    non_grass_count = np.count_nonzero(non_grass)
    if non_grass_count == 0:
        return 0.0

    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in colour_bounds:
        m = cv2.inRange(hsv, lower, upper)
        combined = cv2.bitwise_or(combined, m)

    kit_hits = cv2.bitwise_and(combined, non_grass)
    return np.count_nonzero(kit_hits) / non_grass_count


def track_referee(video_path, output_path, heatmap_path=None, tracking_json_path=None,
                  ref_colour="green", num_calibration_frames=60, marker="disc",
                  imgsz=640, ref_start=None, hold_frames=0, draw_trail=False):
    """
    Main tracking pipeline using BoTSORT (boxmot, CLIP-ReID) for tracking.

    Single-pass architecture:
      1. First N frames: calibrate — score each BoTSORT track by colour, then
         auto-tighten the kit-colour range from the identified ref's own pixels
      2. Identify referee from scores (or --ref-start to seed by location)
      3. Continue tracking with BoTSORT (same instance, persistent IDs)
      4. Colour re-acquire each frame (green-lock) with a camera-aware clamp
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    # ── Colour ranges ─────────────────────────────────────────────
    COLOUR_RANGES = {
        "green": [(np.array([35, 30, 30]), np.array([85, 255, 200]))],
        "black": [(np.array([0, 0, 0]), np.array([180, 255, 60]))],
        "yellow": [(np.array([20, 80, 80]), np.array([35, 255, 255]))],
        "red": [(np.array([0, 80, 80]), np.array([10, 255, 255])),
                (np.array([170, 80, 80]), np.array([180, 255, 255]))],
        "blue": [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
        "cyan": [(np.array([83, 40, 50]), np.array([99, 255, 255]))],
        "white": [(np.array([0, 0, 180]), np.array([180, 30, 255]))],
    }
    SECONDARY_COLOUR = {"green": "black", "yellow": "black", "red": "black",
                        "blue": "black", "cyan": "black", "black": None,
                        "white": "black"}

    if ref_colour not in COLOUR_RANGES:
        print(f"Warning: Unknown ref colour '{ref_colour}', falling back to green")
        ref_colour = "green"

    colour_bounds = COLOUR_RANGES[ref_colour]
    sec_name = SECONDARY_COLOUR.get(ref_colour)
    sec_bounds = COLOUR_RANGES[sec_name] if sec_name else None

    def _has_player_number(frame, box):
        """Detect bright white patches = player shirt numbers. Ref has none."""
        x1, y1, x2, y2 = map(int, box[:4])
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return False
        # Upper back area (top 50%)
        back_h = max(1, int(crop.shape[0] * 0.5))
        margin = max(1, int(crop.shape[1] * 0.10))
        back = crop[:back_h, margin:crop.shape[1] - margin]
        if back.size == 0:
            return False
        gray = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        white_ratio = np.count_nonzero(white_mask) / white_mask.size
        return white_ratio > 0.03  # >3% bright white = player number

    def _score_primary(frame, box):
        """Score primary colour only — used for verification."""
        x1, y1, x2, y2 = map(int, box[:4])
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return 0.0
        shirt_h = max(1, int(crop.shape[0] * 0.35))
        margin = max(1, int(crop.shape[1] * 0.10))
        shirt = crop[:shirt_h, margin:crop.shape[1] - margin]
        if shirt.size == 0:
            return 0.0
        hsv = cv2.cvtColor(shirt, cv2.COLOR_BGR2HSV)
        return _colour_ratio_excluding_grass(hsv, colour_bounds)

    def _score_person(frame, box):
        """Score a bounding box for referee colour match (primary + secondary)."""
        x1, y1, x2, y2 = map(int, box[:4])
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return 0.0
        shirt_h = max(1, int(crop.shape[0] * 0.35))
        margin = max(1, int(crop.shape[1] * 0.10))
        shirt = crop[:shirt_h, margin:crop.shape[1] - margin]
        if shirt.size == 0:
            return 0.0
        hsv = cv2.cvtColor(shirt, cv2.COLOR_BGR2HSV)
        primary = _colour_ratio_excluding_grass(hsv, colour_bounds)
        sec = 0.0
        if sec_bounds:
            cmb = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in sec_bounds:
                cmb = cv2.bitwise_or(cmb, cv2.inRange(hsv, lo, hi))
            sec = np.count_nonzero(cmb) / cmb.size
        return primary * 0.7 + sec * 0.3 if sec_bounds else primary

    # ── Load model and tracker ────────────────────────────────────
    print("Loading YOLOv8n model + BoTSORT tracker (with ReID)...")
    project_root = Path(__file__).parent.parent
    model = YOLO(str(project_root / "yolov8n.pt"))
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tracker = BotSort(
        reid_weights=project_root / "clip_market1501.pt",  # CLIP-ReID for better appearance matching
        device=torch.device(device),
        half=False,
        track_high_thresh=0.3,   # Lower threshold — ref often has low confidence
        track_low_thresh=0.1,    # Keep low-confidence detections for association
        new_track_thresh=0.4,    # Create new tracks more easily
        track_buffer=60,         # Keep lost tracks for ~2s at 29fps
        match_thresh=0.85,       # Generous matching threshold
        appearance_thresh=0.25,  # ReID appearance matching
        with_reid=True,          # Enable appearance features
    )

    # ── Open video ────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        sys.exit(1)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames "
          f"({total_frames / fps:.1f}s)")
    print(f"Looking for ref wearing: {ref_colour}"
          + (f" (+ {sec_name} secondary)" if sec_name else ""))

    # ── Setup Supervision annotators ──────────────────────────────
    # marker: "disc"  -> TV-style ellipse at the ref's feet (never covers hands)
    #         "box"   -> classic bounding rectangle
    #         "both"  -> draw both
    box_annotator = sv.BoxAnnotator(
        thickness=3,
        color=sv.Color.from_hex("#00FF00"),
        color_lookup=sv.ColorLookup.INDEX,
    )
    ellipse_annotator = sv.EllipseAnnotator(
        thickness=3,
        color=sv.Color.from_hex("#00FF00"),
        color_lookup=sv.ColorLookup.INDEX,
    )
    # Disc puts the label by the feet (near the ellipse); box keeps it on top.
    label_position = (sv.Position.BOTTOM_CENTER if marker == "disc"
                      else sv.Position.TOP_LEFT)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.8,
        text_thickness=2,
        text_color=sv.Color.WHITE,
        text_position=label_position,
        color=sv.Color.from_hex("#00FF00"),
        color_lookup=sv.ColorLookup.INDEX,
    )
    # Custom dot trail instead of Supervision's line trace
    TRACE_LENGTH = fps * 5  # 5 seconds of dots
    DOT_RADIUS = 3
    DOT_COLOUR = (0, 255, 0)  # BGR green
    trail_positions = []  # list of (cx, cy) for dot trail

    def _draw_dot_trail(img, positions_list):
        """Draw fading green dots for the ref's recent positions."""
        n = len(positions_list)
        for i, (px, py) in enumerate(positions_list):
            # Fade: older dots are dimmer
            alpha = (i + 1) / n  # 0→1
            r = max(2, int(DOT_RADIUS * (0.5 + 0.5 * alpha)))
            intensity = int(128 + 127 * alpha)
            colour = (0, intensity, 0)
            cv2.circle(img, (px, py), r, colour, -1)
        return img

    def _annotate_ref(img, bbox, conf, tid):
        """Draw the chosen marker (disc/box/both) + REF label, optional trail."""
        det = sv.Detections(
            xyxy=np.array([bbox]),
            confidence=np.array([conf]),
            tracker_id=np.array([tid]),
        )
        if draw_trail:  # off by default — the dots clutter/obscure the ref
            img = _draw_dot_trail(img, trail_positions[-TRACE_LENGTH:])
        if marker in ("box", "both"):
            img = box_annotator.annotate(img, det)
        if marker in ("disc", "both"):
            img = ellipse_annotator.annotate(img, det)
        img = label_annotator.annotate(img, det, labels=["REF"])
        return img

    # ── Setup output ──────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # ── Tracking state ────────────────────────────────────────────
    ref_track_id = None
    ref_positions = []
    tracking_data = []
    calibration_scores = {}     # track_id -> list of colour scores
    calibration_positions = {}  # track_id -> list of (cx, cy)
    frame_buffer = []           # Buffer calibration frames for retroactive annotation
    reid_count = 0
    last_ref_cx = None          # Last known ref centroid
    last_ref_cy = None
    last_ref_bbox = None        # Last known ref bbox (for gap-fill hold)
    last_ref_conf = 0.0
    frames_since_ref_seen = 0   # Grace period before re-ID
    PER_FRAME_PX = 35          # max ref pixel move/frame = run (~12) + pan
                               # allowance (~23); Veo pans gently (~7px p90)
    STRONG_GREEN = 0.45        # unmistakable ref shirt — may override the clamp
    LOST_FRAMES = 3            # allow the strong-green far-jump after this many
                               # lost frames. Safe to be small: the auto-tuned
                               # tight range means only the ref scores green
                               # (everyone else ~0), so a far green = the ref.
    PRIMARY_COLOUR_MIN = 0.20  # Minimum primary colour ratio to accept track

    print(f"Calibrating over {num_calibration_frames} frames...")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Detect with YOLOv8 (no built-in tracking) ─────────────
        results = model(frame, classes=[0], imgsz=imgsz, verbose=False)

        confirmed = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy()

            # BoxMOT format: [x1, y1, x2, y2, conf, cls]
            dets = np.hstack([boxes, confs.reshape(-1, 1), cls_ids.reshape(-1, 1)])
            tracks = tracker.update(dets, frame)

            # tracks: [[x1, y1, x2, y2, track_id, conf, cls, ...], ...]
            for t in tracks:
                bbox = np.array(t[:4], dtype=np.float32)
                tid = int(t[4])
                conf = float(t[5])
                confirmed.append((tid, bbox, conf))
        else:
            tracker.update(np.empty((0, 6)), frame)

        # ── Phase 1: Calibration ──────────────────────────────────
        if frame_idx < num_calibration_frames:
            for tid, bbox, conf in confirmed:
                score = _score_person(frame, bbox)
                if tid not in calibration_scores:
                    calibration_scores[tid] = []
                calibration_scores[tid].append(score)

                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                if tid not in calibration_positions:
                    calibration_positions[tid] = []
                calibration_positions[tid].append((cx, cy))

            # Buffer frame + tracks for retroactive annotation
            frame_buffer.append((frame.copy(), list(confirmed)))
            frame_idx += 1
            continue

        # ── Identify referee (once, at end of calibration) ────────
        if ref_track_id is None and calibration_scores:
            # Seed override: when the user gives the ref's start location, trust
            # it over the colour/isolation heuristic (needed when the ref's kit
            # isn't colour-distinct enough to auto-identify — e.g. a pale-blue
            # ref among navy teams on dim, wide footage).
            if ref_start is not None:
                sx, sy = ref_start
                best, bestd = None, float("inf")
                for tid, plist in calibration_positions.items():
                    if len(plist) < 3:
                        continue
                    mx = float(np.mean([p[0] for p in plist]))
                    my = float(np.mean([p[1] for p in plist]))
                    d = ((mx - sx) ** 2 + (my - sy) ** 2) ** 0.5
                    if d < bestd:
                        bestd, best = d, tid
                if best is not None:
                    ref_track_id = best
                    print(f"Referee seeded -> Track {best} "
                          f"(nearest ({sx},{sy}), dist={bestd:.0f}px)")

            # Auto-identify by colour + isolation only if no seed locked it.
            candidates = {}
            for tid, scores in calibration_scores.items():
                if len(scores) < max(3, num_calibration_frames * 0.2):
                    continue
                avg_score = np.mean(scores)

                # Spatial isolation bonus
                positions = calibration_positions.get(tid, [])
                if positions and len(calibration_positions) > 1:
                    iso_list = []
                    for pos in positions:
                        min_dist = float('inf')
                        for oid, opos_list in calibration_positions.items():
                            if oid == tid:
                                continue
                            for opos in opos_list[-3:]:
                                d = ((pos[0] - opos[0])**2 + (pos[1] - opos[1])**2)**0.5
                                min_dist = min(min_dist, d)
                        if min_dist < float('inf'):
                            iso_list.append(min_dist)
                    if iso_list:
                        avg_score += min(0.15, np.mean(iso_list) / 2000)

                candidates[tid] = avg_score

            if not candidates:
                # Fallback: relax frame requirement
                for tid, scores in calibration_scores.items():
                    candidates[tid] = np.mean(scores)

            if ref_track_id is None and candidates:
                sorted_cands = sorted(candidates.items(), key=lambda x: -x[1])
                print("\nTop candidates:")
                for tid, score in sorted_cands[:5]:
                    n = len(calibration_scores[tid])
                    print(f"  Track {tid}: score={score:.3f} (frames={n})")
                ref_track_id = sorted_cands[0][0]
                ref_score = sorted_cands[0][1]
                print(f"\nReferee identified: BoTSORT Track {ref_track_id} "
                      f"(score={ref_score:.3f})")
            elif ref_track_id is None:
                print("Warning: Could not identify referee.")

            # ── Auto-calibrate a TIGHT colour range from the ref's kit ──
            # The generic colour range also matches grass and dark kits in this
            # washed-out footage (a black-shirt player on grass scores as green
            # as the ref). So sample the IDENTIFIED ref's own shirt pixels and
            # build a tight range around their actual HSV — this is what truly
            # separates the ref from everyone else. (Hue wraps for red, so skip
            # the auto-tighten there.)
            if ref_track_id is not None and ref_colour != "red":
                samp = []
                for buf_frame, buf_tracks in frame_buffer:
                    for tid, bbox, conf in buf_tracks:
                        if tid != ref_track_id:
                            continue
                        x1, y1, x2, y2 = map(int, bbox[:4])
                        crop = buf_frame[max(0, y1):y2, max(0, x1):x2]
                        if crop.size == 0:
                            continue
                        ch, cw = crop.shape[:2]
                        # Torso-CENTRE strip: kit-dominated. (The top-35% bbox
                        # band is mostly grass for a distant ref, which biases
                        # the sampled hue toward grass — measured.)
                        strip = crop[int(ch * 0.15):int(ch * 0.50),
                                     int(cw * 0.32):int(cw * 0.68)]
                        if strip.size == 0:
                            continue
                        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV).reshape(-1, 3)
                        keep = (hsv[:, 1] >= 60) & (hsv[:, 2] >= 60)  # colourful
                        if keep.any():
                            samp.append(hsv[keep])
                if samp:
                    px = np.concatenate(samp, axis=0)
                    # The ref's kit is the DOMINANT hue in his shirt region;
                    # grass at the bbox edges is a minority at a slightly lower
                    # hue. Centre on the median (kit) hue with a tight window,
                    # then derive S/V from the kit-hue pixels ONLY, so edge
                    # grass can't widen the range (the bug on the first try).
                    h_med = float(np.median(px[:, 0]))
                    near = np.abs(px[:, 0] - h_med) <= 6
                    kit = px[near] if int(near.sum()) >= 20 else px
                    slo, shi = np.percentile(kit[:, 1], [10, 90])
                    vlo, vhi = np.percentile(kit[:, 2], [10, 90])
                    colour_bounds = [(
                        np.array([max(0, h_med - 6), max(40, slo - 10),
                                  max(40, vlo - 10)]),
                        np.array([min(179, h_med + 7), min(255, shi + 10),
                                  min(255, vhi + 10)]))]
                    ref_scores = [_score_primary(bf, bb)
                                  for bf, bts in frame_buffer
                                  for t, bb, c in bts if t == ref_track_id]
                    ref_typ = float(np.median(ref_scores)) if ref_scores else 0.2
                    PRIMARY_COLOUR_MIN = max(0.05, 0.40 * ref_typ)
                    STRONG_GREEN = max(0.10, 0.75 * ref_typ)
                    lo0, hi0 = colour_bounds[0]
                    print(f"Tight kit range H[{lo0[0]:.0f}-{hi0[0]:.0f}] "
                          f"S[{lo0[1]:.0f}-{hi0[1]:.0f}] V[{lo0[2]:.0f}-{hi0[2]:.0f}]"
                          f"  ref_typ={ref_typ:.3f}  MIN={PRIMARY_COLOUR_MIN:.3f}"
                          f"  STRONG={STRONG_GREEN:.3f}")

            # Write buffered calibration frames with retroactive annotation
            for buf_idx, (buf_frame, buf_tracks) in enumerate(frame_buffer):
                if ref_track_id is not None:
                    for tid, bbox, conf in buf_tracks:
                        if tid == ref_track_id:
                            cx = int((bbox[0] + bbox[2]) / 2)
                            cy = int((bbox[1] + bbox[3]) / 2)
                            last_ref_cx = cx
                            last_ref_cy = cy
                            last_ref_bbox, last_ref_conf = bbox, conf
                            trail_positions.append((cx, cy))
                            ref_positions.append((cx, cy))
                            tracking_data.append({
                                "frame": buf_idx,
                                "timestamp": round(buf_idx / fps, 2),
                                "x": cx, "y": cy,
                                "bbox": bbox.tolist(),
                                "confidence": conf,
                            })
                            buf_frame = _annotate_ref(buf_frame, bbox, conf, tid)
                            break
                out.write(buf_frame)
            frame_buffer = []
            # Fall through to process current frame

        # ── Phase 2: Normal tracking ──────────────────────────────
        ref_found = False
        if ref_track_id is not None:
            # ── v3: green re-acquire with camera-aware proximity clamp ──
            # The ref wears a distinct green shirt. Pick the greenest upright
            # detection and adopt it, but CLAMP how far it may sit from the
            # last known ref position. Budget per frame = run-speed + a pan
            # allowance (Veo pans gently), and it grows the longer we've been
            # lost so a long occlusion can search wider. A *strong* green match
            # may jump beyond the clamp once it saturates (genuine
            # re-acquisition: we lost him, but now clearly see the green shirt).
            radius = PER_FRAME_PX * max(1, frames_since_ref_seen + 1)
            radius = min(radius, max(width, height) // 2)
            best_in, best_in_s = None, 0.0     # greenest WITHIN the clamp
            best_any, best_any_s = None, 0.0   # greenest anywhere (escape only)
            for tid, bbox, conf in confirmed:
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                if bw <= 0 or (bh / bw) < 1.3:   # upright person only
                    continue
                g = _score_primary(frame, bbox)
                if g < PRIMARY_COLOUR_MIN:
                    continue
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                if g > best_any_s:
                    best_any_s, best_any = g, (tid, bbox, conf, cx, cy)
                near = (last_ref_cx is None or
                        ((cx - last_ref_cx) ** 2
                         + (cy - last_ref_cy) ** 2) ** 0.5 <= radius)
                if near and g > best_in_s:
                    best_in_s, best_in = g, (tid, bbox, conf, cx, cy)

            # Prefer the greenest candidate within the clamp. Only fall back to
            # a distant strong-green match once we've genuinely been lost for a
            # while (otherwise a green touchline official hijacks a good lock).
            pick = best_in
            if pick is None and frames_since_ref_seen >= LOST_FRAMES \
                    and best_any_s >= STRONG_GREEN:
                pick = best_any
            if pick is not None:
                tid, bbox, conf, cx, cy = pick
                if tid != ref_track_id and reid_count < 20:
                    print(f"  Green-lock: -> Track {tid} (green="
                          f"{best_in_s if best_in else best_any_s:.3f}, "
                          f"frame={frame_idx})")
                if tid != ref_track_id:
                    reid_count += 1
                ref_track_id = tid
                last_ref_cx, last_ref_cy = cx, cy
                last_ref_bbox, last_ref_conf = bbox, conf
                trail_positions.append((cx, cy))
                ref_positions.append((cx, cy))
                tracking_data.append({
                    "frame": frame_idx,
                    "timestamp": round(frame_idx / fps, 2),
                    "x": cx, "y": cy,
                    "bbox": bbox.tolist(),
                    "confidence": conf,
                })
                frame = _annotate_ref(frame, bbox, conf, tid)
                ref_found = True
                frames_since_ref_seen = 0

            # No green within reach this frame — trust BoTSORT's (camera-
            # compensated) track to carry the ref through a side-on / occluded
            # phase, rather than dropping him just because his box isn't green.
            if not ref_found:
                for tid, bbox, conf in confirmed:
                    if tid == ref_track_id:
                        cx = int((bbox[0] + bbox[2]) / 2)
                        cy = int((bbox[1] + bbox[3]) / 2)
                        last_ref_cx, last_ref_cy = cx, cy
                        last_ref_bbox, last_ref_conf = bbox, conf
                        trail_positions.append((cx, cy))
                        ref_positions.append((cx, cy))
                        tracking_data.append({
                            "frame": frame_idx,
                            "timestamp": round(frame_idx / fps, 2),
                            "x": cx, "y": cy,
                            "bbox": bbox.tolist(),
                            "confidence": conf,
                        })
                        frame = _annotate_ref(frame, bbox, conf, tid)
                        ref_found = True
                        frames_since_ref_seen = 0
                        break

            # ── Lost this frame — widen the clamp next frame ──────
            if not ref_found:
                frames_since_ref_seen += 1
                # Gap-fill: hold the disc at the last known spot for a short
                # window so brief occlusions/dropouts don't make it flicker.
                # (Held frames are NOT recorded as tracking data — render only.)
                if (hold_frames > 0 and last_ref_bbox is not None
                        and frames_since_ref_seen <= hold_frames):
                    frame = _annotate_ref(frame, last_ref_bbox,
                                          last_ref_conf, ref_track_id)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            pct = (frame_idx / total_frames) * 100
            print(f"  Frame {frame_idx}/{total_frames} ({pct:.0f}%)")

    cap.release()
    out.release()

    print(f"\nAnnotated video saved: {output_path}")
    pct = len(ref_positions) / max(frame_idx, 1) * 100
    print(f"Referee tracked in {len(ref_positions)}/{frame_idx} frames ({pct:.1f}%)")
    if reid_count > 0:
        print(f"Colour re-ID triggered {reid_count} time(s)")

    # Save tracking JSON
    if tracking_json_path is None:
        tracking_json_path = output_path.with_suffix(".json")
    with open(tracking_json_path, "w") as f:
        json.dump({
            "video": str(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": frame_idx,
            "tracker": "botsort",
            "ref_track_id": ref_track_id,
            "positions": tracking_data,
        }, f, indent=2)
    print(f"Tracking data saved: {tracking_json_path}")

    # Generate heatmap
    if heatmap_path and ref_positions:
        generate_heatmap(ref_positions, width, height, heatmap_path)

    return tracking_data


def generate_heatmap(positions, width, height, output_path):
    """Generate a 2D heatmap of referee positions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    heatmap = np.zeros((height, width), dtype=np.float32)
    for (cx, cy) in positions:
        if 0 <= cx < width and 0 <= cy < height:
            heatmap[cy, cx] += 1

    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=width // 30, sigmaY=height // 30)
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)

    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    background = np.zeros((height, width, 3), dtype=np.uint8)
    background[:] = (20, 20, 20)
    mask = heatmap > 10
    mask_3d = np.stack([mask] * 3, axis=-1)
    result = np.where(mask_3d,
                      cv2.addWeighted(background, 0.3, heatmap_colored, 0.7, 0),
                      background)

    cv2.putText(result, "Referee Movement Heatmap", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(str(output_path), result)
    print(f"Heatmap saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RefTracker - AI Referee Tracking")
    parser.add_argument("video", help="Input video file path")
    parser.add_argument("--output", "-o", default=None, help="Output annotated video path")
    parser.add_argument("--heatmap", default=None, help="Output heatmap image path")
    parser.add_argument("--tracking-json", default=None, help="Output tracking JSON path")
    parser.add_argument("--ref-colour", default="green",
                        choices=["green", "black", "yellow", "red", "blue",
                                 "cyan", "white"],
                        help="Referee kit colour to look for (default: green)")
    parser.add_argument("--calibration-frames", type=int, default=60,
                        help="Number of frames for referee calibration (default: 60)")
    parser.add_argument("--marker", default="disc",
                        choices=["disc", "box", "both"],
                        help="Ref marker style: 'disc' = TV-style ellipse at the feet "
                             "(won't cover hand signals), 'box' = bounding rectangle, "
                             "'both' (default: disc)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO inference resolution. Raise to 1280/1920 for "
                             "wide shots with small/distant players so they're "
                             "detected at all (slower). (default: 640)")
    parser.add_argument("--ref-start", default=None, metavar="X,Y",
                        help="Approx pixel location of the ref at the start "
                             "(e.g. 800,440). Seeds identification by LOCATION "
                             "instead of colour — use when the ref's kit isn't "
                             "colour-distinct enough to auto-identify.")
    parser.add_argument("--hold-frames", type=int, default=0,
                        help="Hold the disc at its last spot for up to N frames "
                             "when the ref drops out of detection, so brief "
                             "occlusions don't flicker. ~15 (0.5s) suits a "
                             "watchable clip; 0 = off (default).")
    parser.add_argument("--trail", action="store_true",
                        help="Draw the fading green dot trail of the ref's recent "
                             "path. Off by default — the dots clutter/obscure the "
                             "ref (the heatmap shows movement better).")
    args = parser.parse_args()

    ref_start = None
    if args.ref_start:
        try:
            ref_start = tuple(int(v) for v in args.ref_start.split(",")[:2])
        except ValueError:
            print(f"Error: --ref-start must be 'X,Y', got {args.ref_start!r}")
            sys.exit(1)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    output = args.output or str(video_path.stem) + "_tracked.mp4"
    heatmap = args.heatmap or str(video_path.stem) + "_heatmap.png"

    track_referee(
        video_path=str(video_path),
        output_path=output,
        heatmap_path=heatmap,
        tracking_json_path=args.tracking_json,
        ref_colour=args.ref_colour,
        num_calibration_frames=args.calibration_frames,
        marker=args.marker,
        imgsz=args.imgsz,
        ref_start=ref_start,
        hold_frames=args.hold_frames,
        draw_trail=args.trail,
    )


if __name__ == "__main__":
    main()
