#!/usr/bin/env python3
"""GAZE3D preview and smoke test: the camera (or a .bag, or a still image) through FaceLandmarker + ptgaze + depth,
with the irises, their depth source, the gaze ray and the hit on the monitor drawn on the frame.

  python run_gaze3d.py                                  live window (q / Esc quit)
  python run_gaze3d.py --image face.jpg --snapshot o.png   wiring check without a camera (dummy intrinsics, pnp origin)
  python run_gaze3d.py --frames 90 --snapshot o.png     headless: 90 frames, the last one saved, a summary printed
  python run_gaze3d.py --bag session.bag --csv hits.csv replay a recording, one CSV row per frame
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from gaze3d.capture import ImageCapture, add_capture_args, capture_from_args  # noqa: E402
from gaze3d.geometry import PlaneMapper, parse_pair  # noqa: E402
from gaze3d.model import add_model_args, add_pose_args, draw_overlay, model_from_args, pose_from_args  # noqa: E402

CSV_FIELDS = ("seq", "t", "face", "blink", "valid", "sx_mm", "sy_mm", "px", "py", "eye_l_x", "eye_l_y", "eye_l_z",
              "eye_r_x", "eye_r_y", "eye_r_z", "src_l", "src_r", "gaze_yaw", "gaze_pitch", "head_yaw", "head_pitch",
              "dist_mm", "ear", "model_ms")


def put_lines(img, lines, org=(10, 60), colour=(255, 255, 255)):
    import cv2
    x, y = org
    if lines:                                           # a dark band so the text reads on any background
        y1 = min(img.shape[0], y + 22 * len(lines) - 6)
        band = img[max(0, y - 20):y1, :, :]
        band[:] = (band * 0.45).astype(band.dtype)
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
        y += 22
    return img


def row_for(s, mapper, seq):
    r = {k: "" for k in CSV_FIELDS}
    r.update(seq=seq, t="%.3f" % s.timestamp, face=int(s.face_found), blink=int(s.blink), valid=int(s.valid),
             ear="%.3f" % s.ear, model_ms="%.1f" % s.model_ms)
    if not s.face_found:
        return r
    r["src_l"], r["src_r"] = s.depth_src
    for name, p in (("l", s.eye_mm[0]), ("r", s.eye_mm[1])):
        if p is not None:
            r["eye_%s_x" % name], r["eye_%s_y" % name], r["eye_%s_z" % name] = ("%.1f" % v for v in p)
    if s.direction is not None:
        from gaze3d.geometry import yaw_pitch_deg
        r["gaze_yaw"], r["gaze_pitch"] = ("%.2f" % v for v in yaw_pitch_deg(s.direction))
    hyp = s.head_yaw_pitch()
    if hyp is not None:
        r["head_yaw"], r["head_pitch"] = ("%.1f" % v for v in hyp)
    if s.sx is not None:
        px = mapper.predict((s.sx, s.sy))
        r.update(sx_mm="%.1f" % s.sx, sy_mm="%.1f" % s.sy, px="%.0f" % px[0], py="%.0f" % px[1], dist_mm="%.0f" % s.dist_mm)
    return r


def hud(s, mapper, fps, model, cap):
    from gaze3d.geometry import yaw_pitch_deg
    lines = ["fps %.1f  landmarks %.0f ms  total %.0f ms  %s %s" % (fps, model.landmark_ms, s.model_ms, model.device, cap.usb)]
    if not s.face_found:
        lines.append("NO FACE" + ("  (profile mesh, ear %.2f)" % s.ear if s.ear else ""))
        return lines
    state = "BLINK" if s.blink else ("tracking" if s.valid else "no hit")
    if s.sx is not None:
        px = mapper.predict((s.sx, s.sy))
        lines.append("%s  hit %+.0f, %+.0f mm -> %.0f, %.0f px  dist %.0f mm" % (state, s.sx, s.sy, px[0], px[1], s.dist_mm))
    else:
        lines.append(state + ("  ray misses the panel" if s.direction is not None else "  no direction"))
    eyes = []
    for name, p, src in (("L", s.eye_mm[0], s.depth_src[0]), ("R", s.eye_mm[1], s.depth_src[1])):
        eyes.append("%s %s" % (name, "none" if p is None else "(%.0f, %.0f, %.0f) %s" % (p[0], p[1], p[2], src)))
    lines.append("eyes mm  " + "   ".join(eyes))
    g = yaw_pitch_deg(s.direction) if s.direction is not None else (0.0, 0.0)
    h = s.head_yaw_pitch() or (0.0, 0.0)
    lines.append("gaze yaw %+.1f pitch %+.1f   head %+.0f/%+.0f   ear %.2f" % (g[0], g[1], h[0], h[1], s.ear))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_capture_args(ap)
    add_model_args(ap)
    add_pose_args(ap, screen=True)
    ap.add_argument("--image", default=None, help="run on a still image instead of a camera")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = until q)")
    ap.add_argument("--snapshot", default=None, help="save the last annotated frame here (implies headless)")
    ap.add_argument("--csv", default=None, help="write one row per frame here")
    ap.add_argument("--headless", action="store_true", help="no window")
    ap.add_argument("--scale", type=float, default=1.0, help="window scale")
    args = ap.parse_args()

    import cv2
    cap = ImageCapture(args.image, repeat=max(1, args.frames or 3)) if args.image else capture_from_args(args, realtime=not (args.snapshot or args.headless))
    cap.start()
    pose = pose_from_args(args)
    mapper = PlaneMapper(parse_pair(args.screen_px), parse_pair(args.screen_mm))
    model = model_from_args(args, cap.intrinsics, pose)
    print("[gaze3d] %s; %s on %s; intrinsics fx %.0f fy %.0f pp (%.0f, %.0f) %s -> %s" % (
        pose.describe(), args.mode, model.device, cap.intrinsics.fx, cap.intrinsics.fy, cap.intrinsics.ppx,
        cap.intrinsics.ppy, cap.intrinsics.model, model.camera_yaml))
    headless = args.headless or bool(args.snapshot) or not os.environ.get("DISPLAY")
    writer = None
    if args.csv:
        fh = open(args.csv, "w", newline="")
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
    n = faces = valid = 0
    fps, last_t, last_img = 0.0, None, None
    hits, srcs = [], {}
    try:
        while not (args.frames and n >= args.frames):
            f = cap.read()
            if f is None:
                if getattr(cap, "ended", False):
                    break
                continue
            s = model.process(f)
            n += 1
            now = time.time()
            if last_t is not None and now > last_t:
                inst = 1.0 / (now - last_t)
                fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
            last_t = now
            faces += int(s.face_found)
            valid += int(s.valid)
            if s.face_found:
                for src in s.depth_src:
                    srcs[src] = srcs.get(src, 0) + 1
            if s.valid:
                hits.append((s.sx, s.sy))
            img = draw_overlay(f.bgr, s, full=True, intr=f.intrinsics)
            put_lines(img, hud(s, mapper, fps, model, cap))
            last_img = img
            if writer:
                writer.writerow(row_for(s, mapper, f.seq))
            if not headless:
                show = img if args.scale == 1.0 else cv2.resize(img, None, fx=args.scale, fy=args.scale)
                cv2.imshow("gaze3d", show)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        model.close()
        cap.stop()
        if writer:
            fh.close()
        if not headless:
            cv2.destroyAllWindows()
    if args.snapshot and last_img is not None:
        cv2.imwrite(args.snapshot, last_img)
        print("[gaze3d] snapshot %s" % args.snapshot)
    mean = np.mean(hits, axis=0) if hits else (float("nan"), float("nan"))
    print("[gaze3d] %d frames, face %d, valid %d, mean hit (%.0f, %.0f) mm, depth sources %s, timeouts %d" % (
        n, faces, valid, mean[0], mean[1], srcs, cap.timeouts))


if __name__ == "__main__":
    main()
