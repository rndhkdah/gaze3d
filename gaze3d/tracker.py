"""Background thread that owns the camera and the model: read -> process -> (mapper) and publishes the latest
TrackerSample without blocking (pygame stays on the main thread). Copied from New/gt/tracker.py with the eye-crop
helper, so GAZE3D stands alone. The extractor is anything with process(frame) -> sample (.as_vector(),
.eye_points) and optionally close(); the capture anything with start/stop/read(timeout_ms) -> frame (.bgr,
.timestamp) | None."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class TrackerSample:
    t: float                    # frame timestamp, host clock seconds
    t_done: float               # host time when the features were ready
    seq: int                    # counts delivered samples
    sample: object              # the model's per-frame object (.valid, .face_found, .blink)
    vec: np.ndarray | None      # sample.as_vector(), None when invalid
    screen: tuple | None        # mapper.predict(vec) when a mapper is set and the sample is valid
    eye_crop: tuple | None = None           # (BGR crop, x0, y0) around both eyes, for the preview
    frame_small: np.ndarray | None = None   # the whole frame, 320 px wide, when there is no eye crop (framing check)


def eye_box(points, w: int, h: int, margin: float = 0.6, aspect: float = 8.0 / 3.0):
    """(x0, y0, x1, y1) int pixels around the eye landmarks: widened by `margin` x the eye span on each side, at
    least `aspect` wide, clamped to the image. The crop for the eye preview."""
    p = np.asarray(points, float)
    (xa, ya), (xb, yb) = p.min(axis=0), p.max(axis=0)
    cx, cy = (xa + xb) / 2.0, (ya + yb) / 2.0
    bw = max(xb - xa, 1.0) * (1.0 + 2.0 * margin)
    bh = max(bw / aspect, (yb - ya) * (1.0 + 2.0 * margin))
    x0, x1 = int(round(cx - bw / 2.0)), int(round(cx + bw / 2.0))
    y0, y1 = int(round(cy - bh / 2.0)), int(round(cy + bh / 2.0))
    return max(0, x0), max(0, y0), min(int(w), x1), min(int(h), y1)


class GazeTracker:
    def __init__(self, capture, extractor, mapper=None):
        self.capture, self.extractor = capture, extractor
        self._mapper = mapper
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.seq = 0
        self.fps = 0.0
        self.latency_ms = 0.0       # frame timestamp -> features ready, EMA
        self.error = None           # exception that stopped the loop, if any

    def start(self) -> "GazeTracker":
        self.capture.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gaze-tracker", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.capture.stop()
        close = getattr(self.extractor, "close", None)
        if close is not None:
            close()

    def latest(self) -> TrackerSample | None:
        with self._lock:
            return self._latest

    def set_mapper(self, mapper):
        with self._lock:
            self._mapper = mapper

    def _run(self):
        t_last = None
        try:
            while not self._stop.is_set():
                frame = self.capture.read()
                if frame is None:
                    if getattr(self.capture, "ended", False):
                        break
                    continue
                sample = self.extractor.process(frame)
                vec = sample.as_vector()
                crop = None
                eye_points = getattr(sample, "eye_points", None)
                if eye_points is not None:
                    x0, y0, x1, y1 = eye_box(eye_points, frame.bgr.shape[1], frame.bgr.shape[0])
                    if x1 > x0 and y1 > y0:
                        crop = (frame.bgr[y0:y1, x0:x1].copy(), x0, y0)
                small = None
                if crop is None:
                    import cv2
                    h0, w0 = frame.bgr.shape[:2]
                    small = cv2.resize(frame.bgr, (320, max(1, int(round(320 * h0 / w0)))), interpolation=cv2.INTER_AREA)
                with self._lock:
                    mapper = self._mapper
                screen = tuple(mapper.predict(vec)) if (mapper is not None and vec is not None) else None
                t_done = time.time()
                self.seq += 1
                now = time.monotonic()
                if t_last is not None:
                    self.fps += 0.1 * (1.0 / max(1e-6, now - t_last) - self.fps)
                t_last = now
                self.latency_ms += 0.1 * ((t_done - frame.timestamp) * 1000.0 - self.latency_ms)
                with self._lock:
                    self._latest = TrackerSample(frame.timestamp, t_done, self.seq, sample, vec, screen, crop, small)
        except Exception as e:                      # keep the error for the main thread to show
            self.error = e
            print(f"[tracker] stopped: {e!r}", flush=True)
