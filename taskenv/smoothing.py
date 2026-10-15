"""One Euro filter and the hold/lost logic for the gaze point. Pure Python, no hardware.

OneEuro (Casiez, Roussel, Vogel 2012): a low-pass filter whose cutoff rises with the signal's speed, so
the point is steady during fixations (min_cutoff) and still follows saccades (beta).
GazeSmoother wraps two of them (x, y) and handles invalid samples: a blink or a lost face holds the
last point, and after `lost_after` seconds the gaze is reported as lost. The first valid sample after a
loss resets the filters, so the point jumps to the new fixation instead of sliding from the stale one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


class OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = float(min_cutoff), float(beta), float(d_cutoff)
        self.reset()

    def reset(self):
        self._x = None
        self._dx = 0.0
        self._t = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        x, t = float(x), float(t)
        if self._x is None:
            self._x, self._t = x, t
            return x
        dt = t - self._t
        if dt <= 0.0:
            return self._x
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx += a_d * ((x - self._x) / dt - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x += self._alpha(cutoff, dt) * (x - self._x)
        self._t = t
        return self._x


@dataclass(frozen=True)
class SmoothedGaze:
    x: float
    y: float
    valid: bool        # this update had a valid sample
    lost: bool         # no valid sample for longer than lost_after
    hold_age: float    # seconds since the last valid sample (0 when valid)


class GazeSmoother:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, lost_after: float = 0.3):
        self.fx = OneEuro(min_cutoff, beta)
        self.fy = OneEuro(min_cutoff, beta)
        self.lost_after = float(lost_after)
        self.reset()

    def reset(self):
        self.fx.reset()
        self.fy.reset()
        self._last = None
        self._t_valid = None
        self._lost = False

    def update(self, point, t: float) -> SmoothedGaze | None:
        """point: (x, y) of a valid sample, or None for a blink / no face. Returns None until the first
        valid point has been seen."""
        if point is None:
            if self._last is None:
                return None
            age = float(t) - self._t_valid
            if age > self.lost_after:
                self._lost = True
            return SmoothedGaze(self._last[0], self._last[1], False, self._lost, age)
        if self._lost:
            self.fx.reset()
            self.fy.reset()
            self._lost = False
        x, y = self.fx(point[0], t), self.fy(point[1], t)
        self._last, self._t_valid = (x, y), float(t)
        return SmoothedGaze(x, y, True, False, 0.0)
