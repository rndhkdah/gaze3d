"""The contract between the task (taskenv.squares, taskenv.calibrate) and a gaze tracker.

A tracker is a Python module: a built-in one under trackers/ (synthetic, gazetracking), any .py file, or an
importable module, named with --tracker. The module defines

  NAME            short name; written into calibration files and the per-sample CSV
  FEATURE_NAMES   names of the calibration feature vector TrackerSample.vec holds (length >= 1)
  LOG_FIELDS      extra per-sample CSV columns that describe(sample)["log"] fills (may be empty)
  add_args(ap)    its own argparse options (camera, thresholds ...); the task parses --tracker first, then these
  create(args)    -> the tracker object, not started (open the camera here to fail fast)

and the tracker object offers

  start() -> self, stop(), latest() -> TrackerSample | None, set_mapper(mapper), fps (float), error (Exception | None)
  reset()                                    optional: the task's R key (the eyetracker drops its eye-sphere lock)
  default_mapper(screen_px) -> Mapper        used while there is no (usable) calibration file
  describe(sample) -> {"lines": [str, ...], "hud": str, "log": {field: value}}
                                             readout under the eye preview, a HUD suffix, CSV extras (LOG_FIELDS)
  eye_image(sample, eye_crop) -> BGR ndarray | None   the eye crop with the tracker's marks; unmirrored, the task mirrors
  extra_images(sample) -> [ndarray, ...]     small grey or BGR images shown under the eye crop (iris masks ...)

TrackerSample.sample is the tracker's own per-frame object; the task reads only .valid, .face_found and .blink from it
(BasicSample is enough). A Mapper has predict(vec) -> (x, y) screen pixels, .description (str) and .report (dict of
the fit, or None when uncalibrated).

Thread rule: the tracker runs its camera and model on its own thread; latest() must not block. pygame stays on
the task's main thread.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass

import numpy as np

REQUIRED = ("NAME", "FEATURE_NAMES", "LOG_FIELDS", "add_args", "create")
TRACKER_METHODS = ("start", "stop", "latest", "set_mapper", "default_mapper", "describe", "eye_image", "extra_images")


@dataclass
class BasicSample:
    """The minimum per-frame object; trackers may use their own class with (at least) these fields."""
    valid: bool
    face_found: bool
    blink: bool
    timestamp: float


@dataclass
class TrackerSample:
    t: float                    # frame timestamp, host clock seconds
    t_done: float               # host time when the features were ready
    seq: int                    # counts delivered samples
    sample: object              # the tracker's per-frame object (.valid, .face_found, .blink)
    vec: np.ndarray | None      # calibration features (FEATURE_NAMES order), None when invalid
    screen: tuple | None        # mapper.predict(vec) when a mapper is set and the sample is valid
    eye_crop: tuple | None = None       # (BGR crop, x0, y0) around the eyes, for the preview
    frame_small: np.ndarray | None = None   # the whole frame, ~320 px wide, when there is no eye crop (framing check)


class ScaledMapper:
    """Normalised (u, v) in 0..1 -> screen pixels; the default mapper for trackers that already output a point."""

    def __init__(self, screen_px, description="normalised (u, v) x screen size"):
        self.screen_px, self._description, self.report = tuple(int(v) for v in screen_px), description, None

    @property
    def description(self) -> str:
        return self._description

    def predict(self, vec):
        v = np.asarray(vec, float)
        return float(v[0] * self.screen_px[0]), float(v[1] * self.screen_px[1])


def load_tracker(spec: str, search_dirs=()):
    """The tracker module for a --tracker value: a built-in name (trackers/<name>.py), a path to a .py file, or an
    importable module name. Raises SystemExit with the missing names when the module is not a tracker."""
    spec = str(spec).strip()
    try:
        if spec.endswith(".py") or os.sep in spec:
            path = os.path.abspath(spec)
            if not os.path.isfile(path):
                raise SystemExit(f"tracker file not found: {path}")
            ms = importlib.util.spec_from_file_location("tracker_" + os.path.splitext(os.path.basename(path))[0], path)
            module = importlib.util.module_from_spec(ms)
            ms.loader.exec_module(module)
        else:
            try:
                module = importlib.import_module(f"trackers.{spec}")
            except ModuleNotFoundError as e:
                if e.name not in (f"trackers.{spec}", "trackers"):
                    raise
                module = importlib.import_module(spec)
    except ModuleNotFoundError as e:
        raise SystemExit(f"no tracker '{spec}': {e} (built-in: {', '.join(builtin_trackers(search_dirs))}; "
                         "or give a path to a .py file)")
    missing = [n for n in REQUIRED if not hasattr(module, n)]
    if missing:
        raise SystemExit(f"tracker '{spec}' lacks {', '.join(missing)}: see taskenv/api.py for the contract")
    if len(module.FEATURE_NAMES) < 1:
        raise SystemExit(f"tracker '{spec}': FEATURE_NAMES must name at least one feature")
    return module


def builtin_trackers(search_dirs=()):
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trackers")
    names = []
    for d in (here, *search_dirs):
        if os.path.isdir(d):
            names += sorted(f[:-3] for f in os.listdir(d) if f.endswith(".py") and not f.startswith("_"))
    return names


def check_tracker(tracker):
    """SystemExit naming the methods a created tracker object lacks (called once after create)."""
    missing = [m for m in TRACKER_METHODS if not callable(getattr(tracker, m, None))]
    if missing:
        raise SystemExit(f"tracker object {type(tracker).__name__} lacks {', '.join(missing)}: see taskenv/api.py")
    return tracker
