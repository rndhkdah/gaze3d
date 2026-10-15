"""Screen calibration for any tracker: learns tracker feature vector (FEATURE_NAMES) -> monitor pixel.

Per axis, standardise the features -> ridge-regularised affine fit, alpha chosen by leave-one-target-out error.
Within each target, samples more than 3.5 MAD from the target's median are dropped (blinks, glances). Every dot
must collect samples (undersampled), and a saved fit whose LOO error exceeds MAX_LOO_FRACTION of the screen
diagonal is ignored at load time (usable) in favour of the tracker's default_mapper. The JSON file records the
tracker name and feature names; a file from another tracker is ignored too.

Reported errors (leave-one-target-out): loo_mean_px = error of the mean prediction per held-out target (what you
get after smoothing); loo_sample_px = per-sample error (includes jitter). cm and degrees use --screen-mm and
--viewing-mm (the task has no depth).

Standalone:  python -m taskenv.calibrate --tracker gaze3d --display auto --points 9 --out calib/calibration_gaze3d.json
Needs:       numpy, pygame, a tracker module (taskenv.api)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import time

import numpy as np

ALPHAS = (0.01, 0.1, 1.0, 10.0)
VERSION = 2
MAX_LOO_FRACTION = 0.15      # a calibration whose LOO error exceeds this fraction of the screen diagonal is ignored


# ---------------------------------------------------------------------- pure fit
def reject_outliers(X, target_id, k: float = 3.5) -> np.ndarray:
    """Boolean keep-mask: per target, drop samples whose features deviate > k robust sigmas (MAD)."""
    X, target_id = np.asarray(X, float), np.asarray(target_id)
    keep = np.ones(len(X), bool)
    for t in np.unique(target_id):
        idx = np.flatnonzero(target_id == t)
        med = np.median(X[idx], axis=0)
        sigma = 1.4826 * np.median(np.abs(X[idx] - med), axis=0) + 1e-9
        keep[idx[np.max(np.abs(X[idx] - med) / sigma, axis=1) > k]] = False
    return keep


def errors_in_units(err_px: float, screen_px, screen_mm, viewing_mm: float):
    """(cm, degrees of visual angle) for an on-screen error in pixels."""
    mm_per_px = (screen_mm[0] / screen_px[0] + screen_mm[1] / screen_px[1]) / 2.0
    mm = err_px * mm_per_px
    return mm / 10.0, math.degrees(math.atan2(mm, viewing_mm))


class Affine:
    """Standardised features -> ridge weights + intercept, closed form."""

    def __init__(self, mean, scale, w, b):
        self.mean, self.scale, self.w, self.b = (np.asarray(mean, float), np.asarray(scale, float), np.asarray(w, float),
                                                 float(b))

    @classmethod
    def fit(cls, F, y, alpha: float) -> "Affine":
        F, y = np.asarray(F, float), np.asarray(y, float)
        mean, scale = F.mean(axis=0), F.std(axis=0) + 1e-9
        Z = (F - mean) / scale                                    # centred, so the intercept is y's mean
        w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
        return cls(mean, scale, w, y.mean())

    def predict(self, F):
        return ((np.asarray(F, float) - self.mean) / self.scale) @ self.w + self.b

    def to_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "w": self.w.tolist(), "b": self.b}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["scale"], d["w"], d["b"])


def fit_calibration(X, Y, target_id, screen_px, screen_mm, viewing_mm=600.0, alphas=ALPHAS, outlier_k=3.5,
                    tracker="", feature_names=()):
    """Returns (GazeMapper, report). X: (N, D) feature vectors, Y: (N, 2) target pixels, target_id: (N,)."""
    X, Y, target_id = np.asarray(X, float), np.asarray(Y, float), np.asarray(target_id)
    targets = np.unique(target_id)
    if len(targets) < 4:
        raise ValueError(f"need at least 4 calibration targets, got {len(targets)}")
    keep = reject_outliers(X, target_id, outlier_k)
    X, Y, target_id = X[keep], Y[keep], target_id[keep]
    best = None
    for alpha in alphas:
        mean_err, sample_err = [], []
        for t in targets:
            tr, te = target_id != t, target_id == t
            pred = np.c_[Affine.fit(X[tr], Y[tr, 0], alpha).predict(X[te]), Affine.fit(X[tr], Y[tr, 1], alpha).predict(X[te])]
            mean_err.append(float(np.linalg.norm(pred.mean(axis=0) - Y[te][0])))
            sample_err.append(float(np.linalg.norm(pred - Y[te], axis=1).mean()))
        score = float(np.mean(mean_err))
        if best is None or score < best[0]:
            best = (score, alpha, mean_err, sample_err)
    loo_mean, alpha, per_target, per_target_sample = best
    axes = (Affine.fit(X, Y[:, 0], alpha), Affine.fit(X, Y[:, 1], alpha))
    cm, deg = errors_in_units(loo_mean, screen_px, screen_mm, viewing_mm)
    report = {"loo_mean_px": loo_mean, "loo_max_px": float(max(per_target)), "per_target_px": per_target,
              "loo_sample_px": float(np.mean(per_target_sample)), "loo_mean_cm": cm, "loo_mean_deg": deg,
              "alpha": float(alpha), "viewing_mm": float(viewing_mm), "n_samples": int(len(X)),
              "n_rejected": int((~keep).sum()), "n_targets": int(len(targets))}
    mapper = GazeMapper(axes, tuple(screen_px), tuple(screen_mm), float(viewing_mm), report, X, Y, target_id,
                        tracker=tracker, feature_names=tuple(feature_names))
    return mapper, report


class GazeMapper:
    """predict(vec) -> (x, y) screen pixels. Saved as JSON with the raw calibration samples so it can be refit."""

    def __init__(self, axes, screen_px, screen_mm, viewing_mm, report, X=None, Y=None, target_id=None,
                 created=None, display=None, tracker="", feature_names=()):
        self.axes, self.screen_px, self.screen_mm, self.viewing_mm = axes, tuple(screen_px), tuple(screen_mm), viewing_mm
        self.report, self.X, self.Y, self.target_id = report, X, Y, target_id
        self.created = created or dt.datetime.now().isoformat(timespec="seconds")
        self.display, self.tracker, self.feature_names = display, tracker, tuple(feature_names)

    @property
    def description(self) -> str:
        return f"calibrated: {self.report['loo_mean_px']:.0f} px / {self.report['loo_mean_deg']:.1f} deg LOO"

    def predict(self, vec):
        f = np.asarray(vec, float)[None, :]
        return float(self.axes[0].predict(f)[0]), float(self.axes[1].predict(f)[0])

    def save(self, path):
        tolist = lambda a: None if a is None else np.asarray(a).tolist()
        with open(path, "w") as f:
            json.dump({"version": VERSION, "tracker": self.tracker, "features": list(self.feature_names),
                       "axes": [a.to_dict() for a in self.axes], "screen_px": list(self.screen_px),
                       "screen_mm": list(self.screen_mm), "viewing_mm": self.viewing_mm, "report": self.report,
                       "X": tolist(self.X), "Y": tolist(self.Y), "target_id": tolist(self.target_id),
                       "created": self.created, "display": self.display}, f, indent=1)

    @classmethod
    def load(cls, path) -> "GazeMapper":
        with open(path) as f:
            d = json.load(f)
        if d.get("version") != VERSION:
            raise SystemExit(f"{path} is not a taskenv v{VERSION} calibration (version {d.get('version')}); delete it and press C")
        arr = lambda a: None if a is None else np.asarray(a)
        return cls(tuple(Affine.from_dict(a) for a in d["axes"]), d["screen_px"], d["screen_mm"], d["viewing_mm"],
                   d["report"], arr(d.get("X")), arr(d.get("Y")), arr(d.get("target_id")), d.get("created"), d.get("display"),
                   d.get("tracker", ""), tuple(d.get("features", ())))


def usable(report, screen_px, max_fraction=MAX_LOO_FRACTION) -> bool:
    """False for a fit too inaccurate for the tiles: LOO mean error above `max_fraction` of the screen diagonal."""
    return float(report["loo_mean_px"]) <= max_fraction * math.hypot(*screen_px)


def undersampled(target_id, n_targets: int, min_samples: int = 5) -> list:
    """Indices of calibration targets with fewer than `min_samples` valid samples (the tracker lost the operator
    there). A fit over the remaining dots looks fine in its own numbers and is useless on screen."""
    tid = np.asarray(target_id)
    return [k for k in range(n_targets) if int((tid == k).sum()) < min_samples]


def load_mapper(path, module, tracker, screen_px, tag="calibrate"):
    """The GazeMapper at `path` when it exists, belongs to this tracker and is usable (one line about it), else the
    tracker's default_mapper(screen_px)."""
    if path and os.path.isfile(path):
        mapper = GazeMapper.load(path)
        if mapper.tracker != module.NAME or mapper.feature_names != tuple(module.FEATURE_NAMES):
            print(f"[{tag}] IGNORING calibration {path}: made for tracker '{mapper.tracker}' with features "
                  f"{list(mapper.feature_names)}, this is '{module.NAME}' {list(module.FEATURE_NAMES)}. Press C to calibrate.", flush=True)
        elif usable(mapper.report, mapper.screen_px):
            print(f"[{tag}] calibration {path}: {mapper.report['loo_mean_px']:.0f} px / {mapper.report['loo_mean_deg']:.1f} deg "
                  f"leave-one-target-out, from {mapper.created}", flush=True)
            return mapper
        else:
            limit = MAX_LOO_FRACTION * math.hypot(*mapper.screen_px)
            print(f"[{tag}] IGNORING calibration {path}: {mapper.report['loo_mean_px']:.0f} px leave-one-target-out error on a "
                  f"{mapper.screen_px[0]}x{mapper.screen_px[1]} screen (limit {limit:.0f} px); from {mapper.created}, "
                  f"{mapper.report['n_samples']} samples, {mapper.report['n_rejected']} rejected. Press C to recalibrate "
                  "with the operator tracked on every dot and the head still.", flush=True)
        return tracker.default_mapper(screen_px)
    print(f"[{tag}] no calibration at {path}: using the tracker's default mapping; press C to calibrate", flush=True)
    return tracker.default_mapper(screen_px)


# ---------------------------------------------------------------------- targets
def target_points(n: int, w: int, h: int, margin: float = 0.08):
    """9: a 3x3 grid inside the margin; 13: plus the four quadrant centres."""
    xs = [margin * w, w / 2, (1 - margin) * w]
    ys = [margin * h, h / 2, (1 - margin) * h]
    pts = [(x, y) for y in ys for x in xs]
    if n == 13:
        qx, qy = (xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2
        pts += [(qx, qy), (w - qx, qy), (qx, h - qy), (w - qx, h - qy)]
    elif n != 9:
        raise ValueError("points must be 9 or 13")
    return [(float(x), float(y)) for x, y in pts]


def validation_points(w: int, h: int):
    return [(0.3 * w, 0.3 * h), (0.7 * w, 0.3 * h), (0.5 * w, 0.75 * h), (0.15 * w, 0.6 * h)]


def sample_status(s) -> str:
    if s is None or not s.sample.face_found:
        return "NO FACE"
    if s.sample.blink:
        return "blink"
    return "tracking" if s.sample.valid else "no features"


# ---------------------------------------------------------------------- pygame routine
def _text(screen, font, lines, x=20, y=20, color=(230, 230, 230)):
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, color), (x, y + i * (font.get_height() + 4)))


def _wait_key(screen, font, lines, keys):
    """Show `lines`, wait for one of `keys` (pygame key constants); Esc returns None."""
    import pygame
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                return None
            if ev.type == pygame.KEYDOWN and ev.key in keys:
                return ev.key
        screen.fill((0, 0, 0))
        _text(screen, font, lines, x=screen.get_width() // 2 - 300, y=screen.get_height() // 2 - 60)
        pygame.display.flip()
        pygame.time.wait(20)


def _collect(screen, tracker, pts, order, settle_s, collect_s, font, label):
    """Shrinking dots; returns (X, Y, tid) or None when Esc was pressed."""
    import pygame
    X, Y, tid = [], [], []
    clock = pygame.time.Clock()
    total = settle_s + collect_s
    for n, k in enumerate(order):
        tx, ty = pts[k]
        t0, last_seq, got = time.monotonic(), -1, 0
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    return None
            now = time.monotonic() - t0
            if now > total:
                break
            s = tracker.latest()
            if s is not None and s.seq != last_seq:
                last_seq = s.seq
                if now >= settle_s and s.vec is not None and s.sample.valid:
                    X.append(np.asarray(s.vec, float))
                    Y.append((tx, ty))
                    tid.append(k)
                    got += 1
            screen.fill((0, 0, 0))
            r = int(40 - 32 * min(1.0, now / total))
            pygame.draw.circle(screen, (60, 60, 60), (int(tx), int(ty)), 44, 2)
            pygame.draw.circle(screen, (255, 255, 255) if now >= settle_s else (140, 140, 140), (int(tx), int(ty)), r)
            pygame.draw.circle(screen, (0, 0, 0), (int(tx), int(ty)), 3)
            _text(screen, font, [f"{label}  target {n + 1}/{len(order)}  samples {got}  {sample_status(s)}  "
                                 f"tracker {getattr(tracker, 'fps', 0.0):.0f} fps", "Esc: abort"])
            pygame.display.flip()
            clock.tick(60)
    return X, Y, tid


def _validate(screen, tracker, mapper, report, font, screen_mm, show_s=2.5):
    """Draws the live prediction on new dots, then the summary. Returns 'accept', 'redo' or None."""
    import pygame
    from taskenv.smoothing import GazeSmoother
    w, h = screen.get_size()
    errs = []
    clock = pygame.time.Clock()
    for tx, ty in validation_points(w, h):
        sm = GazeSmoother(min_cutoff=1.0, beta=0.02)
        t0, last_seq, preds, g = time.monotonic(), -1, [], None
        while time.monotonic() - t0 < show_s:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    return None
            s = tracker.latest()
            if s is not None and s.seq != last_seq:
                last_seq = s.seq
                ok = s.vec is not None and s.sample.valid
                p = mapper.predict(s.vec) if ok else None
                g = sm.update(p, s.t)
                if ok and time.monotonic() - t0 > 0.8:
                    preds.append(p)
            else:
                g = sm.update(None, time.time()) or g
            screen.fill((0, 0, 0))
            pygame.draw.circle(screen, (255, 255, 255), (int(tx), int(ty)), 12)
            if g is not None:
                pygame.draw.circle(screen, (0, 200, 255) if g.valid else (255, 160, 0), (int(g.x), int(g.y)), 18, 3)
            _text(screen, font, ["validation: look at the dot", "Esc: abort"])
            pygame.display.flip()
            clock.tick(60)
        if preds:
            errs.append(float(np.linalg.norm(np.mean(preds, axis=0) - np.array((tx, ty)))))
    val_px = float(np.mean(errs)) if errs else float("nan")
    val_cm, val_deg = errors_in_units(val_px, (w, h), screen_mm, report["viewing_mm"]) if errs else (float("nan"),) * 2
    lines = [f"leave-one-target-out: mean {report['loo_mean_px']:.0f} px = {report['loo_mean_cm']:.1f} cm = "
             f"{report['loo_mean_deg']:.1f} deg   (max {report['loo_max_px']:.0f} px, per-sample {report['loo_sample_px']:.0f} px)",
             f"validation dots:      mean {val_px:.0f} px = {val_cm:.1f} cm = {val_deg:.1f} deg",
             f"samples {report['n_samples']} (rejected {report['n_rejected']}), alpha {report['alpha']}, "
             f"viewing distance {report['viewing_mm']:.0f} mm (--viewing-mm)", "",
             "Enter: accept and save     R: redo     Esc: cancel"]
    print("[calibrate] " + " | ".join(lines[:3]), flush=True)
    key = _wait_key(screen, font, lines, (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_r))
    return None if key is None else ("redo" if key == pygame.K_r else "accept")


def run_calibration(screen, tracker, module, n_points=9, settle_s=0.5, collect_s=1.0, screen_mm=(598.0, 337.0),
                    viewing_mm=600.0, out=None, display=None):
    """Full calibration on an open pygame surface. Returns the accepted GazeMapper, or None on Esc."""
    import pygame
    font = pygame.font.SysFont(None, 30)
    w, h = screen.get_size()
    pts = target_points(n_points, w, h)
    while True:
        order = list(range(len(pts)))
        random.shuffle(order)
        got = _collect(screen, tracker, pts, order, settle_s, collect_s, font, "calibration")
        if got is None:
            return None
        X, Y, tid = got
        missing = undersampled(tid, len(pts))
        if missing:
            key = _wait_key(screen, font, [f"{len(missing)} of {len(pts)} dots got fewer than 5 valid samples ({len(X)} in all): "
                                           "the tracker lost you there.",
                                           "Keep the head still and face the camera.", "R: redo     Esc: abort"], (pygame.K_r,))
            if key is None:
                return None
            continue
        mapper, report = fit_calibration(X, Y, tid, (w, h), screen_mm, viewing_mm, tracker=module.NAME,
                                         feature_names=module.FEATURE_NAMES)
        mapper.display = display
        verdict = _validate(screen, tracker, mapper, report, font, screen_mm)
        if verdict is None:
            return None
        if verdict == "accept":
            if out:
                os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
                mapper.save(out)
                print(f"[calibrate] saved {out}", flush=True)
            return mapper


# ---------------------------------------------------------------------- standalone
def add_task_args(ap):
    ap.add_argument("--display", default="auto", help="pygame display index, or auto = the --screen-px one")
    ap.add_argument("--screen-px", default="1920x1080", help="resolution of the screen under the camera (auto-pick, snapshot size)")
    ap.add_argument("--screen-mm", default="598x337", help="visible size of that screen in mm (cm / degree readouts only)")
    ap.add_argument("--viewing-mm", type=float, default=600.0, help="eye-screen distance for the degree readout")


def parse_with_tracker(ap, argv, default_tracker="synthetic"):
    """Two-stage parse: --tracker first, then the tracker's own options. Returns (args, module)."""
    from taskenv.api import load_tracker
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--tracker", default=default_tracker)
    known, _ = pre.parse_known_args(argv)
    module = load_tracker(known.tracker)
    ap.add_argument("--tracker", default=default_tracker, help="built-in name under trackers/, a .py file, or a module")
    group = ap.add_argument_group(f"tracker '{module.NAME}'")
    module.add_args(group)
    return ap.parse_args(argv), module


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_task_args(ap)
    ap.add_argument("--points", type=int, default=9, choices=(9, 13))
    ap.add_argument("--settle", type=float, default=0.5)
    ap.add_argument("--collect", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="default calib/calibration_<tracker>.json")
    args, module = parse_with_tracker(ap, argv, default_tracker="gaze3d")
    from taskenv.api import check_tracker
    from taskenv.display import open_fullscreen, parse_size
    import pygame
    screen_mm = tuple(float(v) for v in args.screen_mm.lower().split("x"))
    out = args.out or f"calib/calibration_{module.NAME}.json"
    tracker = check_tracker(module.create(args)).start()
    try:
        screen, display = open_fullscreen(args.display, parse_size(args.screen_px), caption="gaze calibration")
        mapper = run_calibration(screen, tracker, module, args.points, args.settle, args.collect, screen_mm, args.viewing_mm,
                                 out, display)
        print("[calibrate] " + ("done" if mapper else "aborted"), flush=True)
    finally:
        tracker.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
