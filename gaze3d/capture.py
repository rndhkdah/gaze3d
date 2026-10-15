"""Capture for GAZE3D: the RealSense D435i colour stream plus depth aligned to it (metres, 0 = no data) with the
colour intrinsics, or a plain webcam (colour only, dummy intrinsics). Frames carry `.bgr` and `.timestamp` so
gt.tracker.GazeTracker (New) can drive them unchanged; `.depth` and `.intrinsics` are extra.

open_capture("auto" | "rs" | index | /dev/videoN, size, fps, depth_size, record=None, bag=None)   not started
RealSenseDepth(...).start()  turns the IR emitter on (depth on skin needs it; colour has an IR-cut filter), aligns
depth to colour, reads the colour intrinsics; falls back to 640x480 (USB 2) with a FALLBACK line; --record writes a
.bag, --bag replays one (no camera). read(timeout_ms) -> Frame | None (timeout, or the end of a bag: .ended).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from .geometry import Intrinsics

USB2_SIZE = (640, 480)


@dataclass
class Frame:
    bgr: np.ndarray                       # (H, W, 3) uint8
    timestamp: float                      # seconds, host clock (time.time())
    seq: int
    t_host: float
    depth: np.ndarray | None = None       # (H, W) float32 metres aligned to bgr; None without a depth camera
    intrinsics: Intrinsics | None = None


def parse_size(text):
    w, h = str(text).lower().split("x")
    return int(w), int(h)


class RealSenseDepth:
    def __init__(self, size=(1280, 720), fps=30, depth_size=(848, 480), record=None, bag=None, realtime=True,
                 serial=None):
        self.size, self.fps, self.depth_size = tuple(size), int(fps), tuple(depth_size)
        self.record, self.bag, self.realtime, self.serial = record, bag, realtime, serial
        self.pipe = self.align = None
        self.intrinsics = None
        self.usb, self.degraded, self.timeouts, self.seq, self.ended = "", False, 0, 0, False
        self._rs = None
        self._scale = 0.001

    def _config(self, rs, size, depth_size):
        cfg = rs.config()
        if self.bag:
            cfg.enable_device_from_file(self.bag, repeat_playback=False)
        else:
            if self.serial:
                cfg.enable_device(self.serial)
            cfg.enable_stream(rs.stream.color, size[0], size[1], rs.format.bgr8, self.fps)
            cfg.enable_stream(rs.stream.depth, depth_size[0], depth_size[1], rs.format.z16, self.fps)
            if self.record:
                cfg.enable_record_to_file(self.record)
        return cfg

    def start(self):
        if self.pipe is not None:
            return self
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise SystemExit("pyrealsense2 is not installed in this venv (bash stack.sh setup): %r" % (e,))
        self._rs = rs
        pipe = rs.pipeline()
        try:
            prof = pipe.start(self._config(rs, self.size, self.depth_size))
        except RuntimeError as e:
            if self.bag:
                raise SystemExit("cannot replay %s: %s" % (self.bag, e))
            try:
                prof = pipe.start(self._config(rs, USB2_SIZE, USB2_SIZE))
            except RuntimeError as e2:
                raise SystemExit("RealSense: could not start %dx%d+depth (%s) nor %dx%d (%s); is it on USB 3 and free?"
                                 % (*self.size, e, *USB2_SIZE, e2))
            self.degraded = True
            print("[capture] FALLBACK: %dx%d colour + depth; the requested %dx%d needs USB 3" % (*USB2_SIZE, *self.size))
        self.pipe = pipe
        dev = prof.get_device()
        try:
            self.usb = dev.get_info(rs.camera_info.usb_type_descriptor)
        except RuntimeError:
            self.usb = "bag" if self.bag else "?"
        if self.bag:
            dev.as_playback().set_real_time(bool(self.realtime))
        else:
            ds = dev.first_depth_sensor()
            if ds.supports(rs.option.emitter_enabled):
                ds.set_option(rs.option.emitter_enabled, 1.0)
        try:
            self._scale = float(dev.first_depth_sensor().get_depth_scale())
        except RuntimeError:
            self._scale = 0.001
        self.align = rs.align(rs.stream.color)
        vsp = prof.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = Intrinsics.from_rs(vsp.get_intrinsics())
        self.size = (self.intrinsics.width, self.intrinsics.height)
        print("[capture] RealSense %s colour %dx%d + depth aligned, usb %s, depth scale %g%s" % (
            dev.get_info(rs.camera_info.name) if not self.bag else "bag", *self.size, self.usb, self._scale,
            ", recording to %s" % self.record if self.record else ""))
        return self

    def stop(self):
        if self.pipe is not None:
            try:
                self.pipe.stop()
            except RuntimeError:
                pass
            self.pipe = None

    def read(self, timeout_ms=1000):
        if self.pipe is None:
            return None
        rs = self._rs
        try:
            frames = self.pipe.wait_for_frames(int(timeout_ms))
        except RuntimeError:
            if self.bag:
                self.ended = True
            self.timeouts += 1
            return None
        t_host = time.time()
        frames = self.align.process(frames)
        c, d = frames.get_color_frame(), frames.get_depth_frame()
        if not c:
            return None
        bgr = np.asanyarray(c.get_data()).copy()
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self._scale if d else None
        ts = t_host
        try:
            if c.get_frame_timestamp_domain() == rs.timestamp_domain.global_time:
                ts = c.get_timestamp() / 1000.0
        except RuntimeError:
            pass
        if self.bag:
            ts = t_host                      # a bag's clock is the recording time; the task wants the host clock
        self.seq += 1
        return Frame(bgr, ts, self.seq, t_host, depth, self.intrinsics)


class Webcam:
    """cv2.VideoCapture colour only; depth None, dummy intrinsics (focal = width)."""

    def __init__(self, source=0, size=None, fps=None):
        self.source, self.size, self.fps = source, size, fps
        self.cap, self.seq, self.timeouts, self.intrinsics, self.ended = None, 0, 0, None, False
        self.usb, self.degraded = "webcam", False

    def start(self):
        if self.cap is not None:
            return self
        import cv2
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise SystemExit("cannot open webcam %r" % (self.source,))
        if self.size:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap, self.size = cap, (w, h)
        self.intrinsics = Intrinsics.dummy(w, h)
        print("[capture] webcam %r %dx%d, no depth (dummy intrinsics)" % (self.source, w, h))
        return self

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read(self, timeout_ms=1000):
        if self.cap is None:
            return None
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            self.timeouts += 1
            return None
        t = time.time()
        self.seq += 1
        return Frame(bgr, t, self.seq, t, None, self.intrinsics)


class ImageCapture:
    """One still image served `repeat` times (the --image wiring check)."""

    def __init__(self, path, repeat=1):
        import cv2
        self.bgr = cv2.imread(str(path))
        if self.bgr is None:
            raise SystemExit("cannot read image %s" % path)
        self.repeat, self.seq, self.timeouts, self.ended = int(repeat), 0, 0, False
        h, w = self.bgr.shape[:2]
        self.intrinsics, self.size, self.usb, self.degraded = Intrinsics.dummy(w, h), (w, h), "image", False

    def start(self):
        return self

    def stop(self):
        pass

    def read(self, timeout_ms=1000):
        if self.seq >= self.repeat:
            self.ended = True
            return None
        t = time.time()
        self.seq += 1
        return Frame(self.bgr.copy(), t, self.seq, t, None, self.intrinsics)


def realsense_present():
    try:
        import pyrealsense2 as rs
        return rs.context().query_devices().size() > 0
    except Exception:
        return False


def open_capture(spec="auto", size=(1280, 720), fps=30, depth_size=(848, 480), record=None, bag=None, realtime=True):
    """Not started. auto = RealSense when one is plugged in, else webcam 0; rs = RealSense; N or /dev/videoN = webcam."""
    if bag:
        return RealSenseDepth(size, fps, depth_size, None, bag, realtime)
    spec = str(spec).strip().lower()
    if spec == "auto":
        if realsense_present():
            return RealSenseDepth(size, fps, depth_size, record)
        print("[capture] no RealSense found: webcam 0 without depth")
        return Webcam(0, size, fps)
    if spec in ("rs", "realsense"):
        return RealSenseDepth(size, fps, depth_size, record)
    return Webcam(int(spec) if spec.isdigit() else spec, size, fps)


def add_capture_args(ap, size="1280x720", depth_size="848x480"):
    ap.add_argument("--camera", default="auto", help="auto | rs | webcam index or /dev/videoN")
    ap.add_argument("--size", default=size, help="colour size WxH")
    ap.add_argument("--depth-size", default=depth_size, help="depth stream size WxH (aligned to colour afterwards)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--record", default=None, help="write the RealSense streams to this .bag while tracking")
    ap.add_argument("--bag", default=None, help="replay this .bag instead of a camera")


def capture_from_args(args, realtime=True):
    return open_capture(args.camera, parse_size(args.size), args.fps, parse_size(args.depth_size),
                        getattr(args, "record", None), getattr(args, "bag", None), realtime)
