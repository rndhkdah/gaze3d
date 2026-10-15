"""GAZE3D as a six-squares task tracker (the contract in taskenv/api.py): RealSense D435i colour + depth aligned to
it, MediaPipe irises, ptgaze (ETH-XGaze) gaze direction in the camera frame, the ray from the depth-measured eye
midpoint intersected with the monitor plane. Features = the hit on the panel in mm (sx_mm, sy_mm); the default
mapper scales mm to pixels with the task's --screen-mm, the 9-dot calibration refines that.

Camera pose in the screen frame: --cam-pos x,y,z mm (auto = centred, 45 mm under the active area, 20 mm in front)
and --cam-tilt degrees looking up (25). Rough values are fine: the calibration absorbs them.

GAZE3D/trackers/gaze3d.py (this folder's own task) and TASK_TEMP/trackers/gaze3d.py (the shared task) both
re-export this module."""
from __future__ import annotations

from .capture import add_capture_args, capture_from_args
from .geometry import PlaneMapper, ScreenPose, parse_cam_pos, parse_pair, yaw_pitch_deg
from .model import add_model_args, add_pose_args, draw_overlay, model_from_args
from .tracker import GazeTracker

NAME = "gaze3d"
FEATURE_NAMES = ("sx_mm", "sy_mm")
LOG_FIELDS = ("eye_l_x", "eye_l_y", "eye_l_z", "eye_r_x", "eye_r_y", "eye_r_z", "src_l", "src_r", "gaze_yaw",
              "gaze_pitch", "head_yaw", "head_pitch", "dist_mm", "ear", "model_ms")


def add_args(ap):
    add_capture_args(ap)
    add_model_args(ap)
    add_pose_args(ap, screen=False)          # --screen-mm / --screen-px are the task's own options


def create(args):
    screen_mm = parse_pair(getattr(args, "screen_mm", "598x337"))
    pose = ScreenPose(screen_mm, parse_cam_pos(args.cam_pos), args.cam_tilt)
    capture = capture_from_args(args).start()          # fail on a missing camera first; gives the intrinsics
    model = model_from_args(args, capture.intrinsics, pose)
    print("[gaze3d] %s; %s on %s; depth %s" % (pose.describe(), args.mode, model.device,
                                               "yes" if capture.__class__.__name__ == "RealSenseDepth" else "NO (pnp origin)"))
    return Gaze3DAdapter(GazeTracker(capture, model), pose, screen_mm)


class Gaze3DAdapter:
    """gaze3d.tracker.GazeTracker runs the camera and the model on its thread; this adds the describe / eye_image /
    extra_images / default_mapper side of the contract."""

    def __init__(self, inner, pose=None, screen_mm=(598.0, 337.0)):
        self.inner, self.pose, self.screen_mm = inner, pose, tuple(screen_mm)

    def start(self):
        self.inner.start()
        return self

    def stop(self):
        self.inner.stop()

    def latest(self):
        return self.inner.latest()

    def set_mapper(self, mapper):
        self.inner.set_mapper(mapper)

    def reset(self):
        """The task's R key: drop the direction filter window (nothing else is stateful)."""
        ex = getattr(self.inner, "extractor", None)
        if ex is not None and hasattr(ex, "_dirs"):
            ex._dirs.clear()

    @property
    def fps(self):
        return self.inner.fps

    @property
    def error(self):
        return self.inner.error

    def default_mapper(self, screen_px):
        return PlaneMapper(screen_px, self.screen_mm)

    def describe(self, s):
        log = {k: "" for k in LOG_FIELDS}
        if s is None or not s.face_found:
            return {"lines": ["NO FACE: MediaPipe needs the face", "face the camera"], "hud": "", "log": log}
        log["ear"], log["model_ms"] = "%.2f" % s.ear, "%.1f" % s.model_ms
        log["src_l"], log["src_r"] = s.depth_src
        for name, p in (("l", s.eye_mm[0]), ("r", s.eye_mm[1])):
            if p is not None:
                log["eye_%s_x" % name], log["eye_%s_y" % name], log["eye_%s_z" % name] = ("%.1f" % v for v in p)
        hyp = s.head_yaw_pitch()
        if hyp is not None:
            log["head_yaw"], log["head_pitch"] = ("%.1f" % v for v in hyp)
        gyp = yaw_pitch_deg(s.direction) if s.direction is not None else None
        if gyp is not None:
            log["gaze_yaw"], log["gaze_pitch"] = ("%.2f" % v for v in gyp)
        blink = "   BLINK" if s.blink else ""
        zl = "%.0f" % s.eye_mm[0][2] if s.eye_mm[0] is not None else "-"
        zr = "%.0f" % s.eye_mm[1][2] if s.eye_mm[1] is not None else "-"
        if s.sx is None:
            lines = ["no hit: " + ("ray misses the panel" if s.direction is not None else "no direction") + blink]
            hud = "gaze3d: no hit" + blink
        else:
            log["dist_mm"] = "%.0f" % s.dist_mm
            lines = ["hit %+.0f, %+.0f mm  d %.0f" % (s.sx, s.sy, s.dist_mm) + blink]
            hud = "gaze3d: %+.0f, %+.0f mm" % (s.sx, s.sy) + blink
        lines.append("eye z L %s %s  R %s %s" % (zl, s.depth_src[0], zr, s.depth_src[1]))
        g = gyp or (0.0, 0.0)
        h = hyp or (0.0, 0.0)
        lines.append("gaze %+.1f/%+.1f  head %+.0f/%+.0f  ear %.2f" % (g[0], g[1], h[0], h[1], s.ear))
        return {"lines": lines, "hud": hud, "log": log}

    def eye_image(self, s, eye_crop):
        if eye_crop is None:
            return None
        crop, x0, y0 = eye_crop
        return draw_overlay(crop.copy(), s, origin=(x0, y0))

    def extra_images(self, s):
        return []
