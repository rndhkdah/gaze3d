"""Synthetic tracker: a figure-eight gaze path that pauses on a tile every few seconds and blinks now and then.
No camera, no model; the reference implementation of the taskenv.api contract and the way to check the task
screens anywhere (`--tracker synthetic`). Its feature vector is the normalised point (u, v) in 0..1, so the
default mapper is just a scale and a calibration fits an identity-like affine map."""
from __future__ import annotations

import math
import time

import numpy as np

from taskenv.api import BasicSample, ScaledMapper, TrackerSample

NAME = "synthetic"
FEATURE_NAMES = ("u", "v")
LOG_FIELDS = ("phase",)


def add_args(ap):
    ap.add_argument("--synthetic-period", type=float, default=8.0, help="seconds per figure-eight")


def create(args):
    return SyntheticTracker(getattr(args, "synthetic_period", 8.0))


class SyntheticTracker:
    def __init__(self, period_s: float = 8.0, cols: int = 3, rows: int = 2):
        self.period, self.cols, self.rows = float(period_s), cols, rows
        self.seq, self.fps, self.error, self._mapper = 0, 60.0, None, None
        self._t0 = time.time()
        self.phase = "path"

    def start(self):
        return self

    def stop(self):
        pass

    def set_mapper(self, mapper):
        self._mapper = mapper

    def default_mapper(self, screen_px):
        return ScaledMapper(screen_px, "synthetic path (u, v) x screen")

    def latest(self):
        t = time.time()
        a = t - self._t0
        phase, cycle = a % 6.0, int(a // 6.0)
        if phase < 2.0:                                   # hold on a tile centre to exercise the dwell
            col, row = (cycle * 5) % self.cols, (cycle * 3) % self.rows
            u, v = (col + 0.5) / self.cols, (row + 0.5) / self.rows
            self.phase = "hold"
        else:
            u = 0.5 + 0.42 * math.sin(2 * math.pi * a / self.period)
            v = 0.5 + 0.38 * math.sin(4 * math.pi * a / self.period)
            self.phase = "path"
        blink = (a % 5.0) < 0.2
        self.seq += 1
        sample = BasicSample(valid=not blink, face_found=True, blink=blink, timestamp=t)
        vec = None if blink else np.array([u, v], float)
        screen = tuple(self._mapper.predict(vec)) if (vec is not None and self._mapper is not None) else None
        return TrackerSample(t, t, self.seq, sample, vec, screen)

    def describe(self, sample):
        return {"lines": ["synthetic gaze path", f"phase {self.phase}" + ("   BLINK" if sample.blink else "")],
                "hud": f"synthetic {self.phase}", "log": {"phase": self.phase}}

    def eye_image(self, sample, eye_crop):
        return None

    def extra_images(self, sample):
        return []
