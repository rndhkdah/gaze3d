"""Gaze testbed, tracker-agnostic: six coloured squares (3x2, filling the screen). The square under the smoothed
gaze lights up, and a guided trial run measures how often the tracker picks the square the operator was asked to
look at. The tracker is a plug-in (taskenv.api): --tracker synthetic | gazetracking | path/to/module.py.

Free-look:  the gazed square is bright with a white border and its name is on the HUD; a dwell selection
            prints `[squares] selected BLUE`.
Trials:     key T (automatic in snapshot runs): `trial 3/12: look at BLUE`. After a short gap (time to read the
            prompt) the first dwell selection is the answer (hit / miss), else a timeout. The per-colour hit rate
            is shown and written to --csv-dir/squares_<time>.csv (columns trial, target, selected, hit, time_s,
            timeout; names, not indices). Every tracker sample goes to --csv-dir/samples_<time>.csv (SAMPLE_FIELDS,
            then the tracker's feature names, then its LOG_FIELDS).
Eye preview: key V: the tracker's eye crop with its marks, top-right, like a mirror, its extra images (iris masks)
            and its readout lines; while no face is tracked, the whole camera frame with a red border (framing).
Mapping:    calib/calibration_<tracker>.json when it exists and is usable (key C makes one), else the tracker's
            default mapper.

Keys: T start / abort trials, V eye preview, C calibrate in place, R reset the filter (and the tracker, when it has
      reset(): the eyetracker re-locks its eye spheres), Space back to free-look, Esc quit.
Run:  python -m taskenv.squares --tracker gazetracking [--camera auto --dwell 0.8 --trials 12 --display auto]
      python -m taskenv.squares --tracker synthetic --snapshot out.png     (headless check, no camera)
Needs: pygame, numpy, taskenv.*; the tracker module brings its own needs
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
import random
import time

import numpy as np

from taskenv.display import soft_dot, text_box
from taskenv.dwell import DwellTimer, cell_at, clamp_to_screen
from taskenv.smoothing import GazeSmoother

# row-major: top row red, green, blue; bottom row yellow, magenta, cyan. Mid-saturation so the white
# gaze dot and the HUD stay readable on every tile.
SQUARES = (("red", (205, 60, 60)), ("green", (60, 170, 80)), ("blue", (60, 100, 220)),
           ("yellow", (220, 190, 50)), ("magenta", (190, 70, 180)), ("cyan", (50, 180, 190)))
NAMES = tuple(name for name, _ in SQUARES)
COLS, ROWS = 3, 2
SAMPLE_FIELDS = ("t", "seq", "face", "blink", "valid", "x", "y", "sm_x", "sm_y", "lost", "tile", "trial", "target")


# ---------------------------------------------------------------------- pure parts
def square_at(x: float, y: float, w: int, h: int):
    """Index 0..5 of the tile under a screen point, or None when off screen."""
    cell = cell_at(x, y, w, h, COLS, ROWS)
    return None if cell is None else cell[1] * COLS + cell[0]


def trial_sequence(n: int, k: int = len(SQUARES), rng=None) -> list:
    """n targets: each square n // k times plus a random remainder, shuffled without immediate repeats
    (when a shuffle allows it)."""
    rng = rng or random.Random()
    seq = list(range(k)) * (n // k) + rng.sample(range(k), n % k)
    for _ in range(200):
        rng.shuffle(seq)
        if all(a != b for a, b in zip(seq, seq[1:])):
            break
    return seq


class TrialRun:
    """Guided trials over `targets`. After `gap_s` (reading the prompt) the first dwell selection is the
    answer; `timeout_s` without one records a timeout. Pure: driven by update(selected, t)."""

    def __init__(self, targets, timeout_s: float = 5.0, gap_s: float = 1.0):
        self.targets, self.timeout_s, self.gap_s = list(targets), float(timeout_s), float(gap_s)
        self.rows = []
        self._i, self._t0 = None, None       # current trial index; time it starts accepting answers

    @property
    def done(self) -> bool:
        return self._i is not None and self._i >= len(self.targets)

    @property
    def current(self):
        """Target index of the current trial, None before begin() and after the last trial."""
        return None if self._i is None or self.done else self.targets[self._i]

    @property
    def index(self) -> int:
        return 0 if self._i is None else self._i

    def begin(self, t: float):
        self._i, self._t0, self.rows = 0, float(t) + self.gap_s, []

    def in_gap(self, t: float) -> bool:
        return self.current is not None and float(t) < self._t0

    def update(self, selected, t: float):
        """selected: square index when the dwell fired this frame, else None. Returns None, "hit",
        "miss" or "timeout"."""
        if self.current is None or self.in_gap(t):
            return None
        elapsed = float(t) - self._t0
        if selected is not None:
            result, timeout = ("hit" if selected == self.current else "miss"), False
        elif elapsed > self.timeout_s:
            result, timeout = "timeout", True
        else:
            return None
        self.rows.append(dict(trial=self._i + 1, target=self.current, selected=selected, hit=result == "hit",
                              time_s=elapsed, timeout=timeout))
        self._i += 1
        self._t0 = float(t) + self.gap_s
        return result


def summarize(rows) -> dict:
    per = {i: (0, 0) for i in range(len(SQUARES))}
    for r in rows:
        h, n = per[r["target"]]
        per[r["target"]] = (h + int(bool(r["hit"])), n + 1)
    hits = sum(int(bool(r["hit"])) for r in rows)
    times = [float(r["time_s"]) for r in rows if r["hit"]]
    return {"per_square": per, "hits": hits, "total": len(rows), "rate": hits / len(rows) if rows else 0.0,
            "timeouts": sum(int(bool(r["timeout"])) for r in rows),
            "mean_time_s": sum(times) / len(times) if times else float("nan")}


def summary_lines(summary: dict) -> list:
    t = "--" if math.isnan(summary["mean_time_s"]) else f"{summary['mean_time_s']:.2f} s"
    lines = [f"trials: {summary['hits']}/{summary['total']} hit = {100.0 * summary['rate']:.0f}%   "
             f"{summary['timeouts']} timeout(s)   mean time to select {t}"]
    lines += [f"  {NAMES[i]:8s} {h}/{n}" for i, (h, n) in summary["per_square"].items() if n]
    return lines


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("trial", "target", "selected", "hit", "time_s", "timeout"))
        for r in rows:
            w.writerow((r["trial"], NAMES[r["target"]], "" if r["selected"] is None else NAMES[r["selected"]],
                        int(bool(r["hit"])), f"{float(r['time_s']):.3f}", int(bool(r["timeout"]))))


def sample_row(s, sm, gazed, trials, feature_names=()) -> dict:
    """One row of the per-sample log: what the tracker saw and where it landed (SAMPLE_FIELDS, then the feature
    vector under `feature_names`). The tracker's LOG_FIELDS are merged in by the caller. Pure."""
    g = s.sample
    f = lambda v, nd=3: "" if v is None else f"{v:.{nd}f}"
    active = trials is not None and trials.current is not None
    row = {"t": f"{s.t:.3f}", "seq": s.seq, "face": int(g.face_found), "blink": int(g.blink), "valid": int(g.valid),
           "x": f(None if s.screen is None else s.screen[0], 0), "y": f(None if s.screen is None else s.screen[1], 0),
           "sm_x": f(None if sm is None else sm.x, 0), "sm_y": f(None if sm is None else sm.y, 0),
           "lost": "" if sm is None else int(sm.lost), "tile": "" if gazed is None else NAMES[gazed],
           "trial": trials.index + 1 if active else "", "target": NAMES[trials.current] if active else ""}
    vec = None if s.vec is None else np.asarray(s.vec, float)
    for i, name in enumerate(feature_names):
        row[name] = "" if vec is None or i >= len(vec) else f"{vec[i]:.4f}"
    return row


# ---------------------------------------------------------------------- drawing
def _dim(rgb, k: float):
    return tuple(int(c * k) for c in rgb)


def tile_rect(i: int, w: int, h: int):
    """(x, y, width, height) of tile i on a w x h screen."""
    cw, ch = w / COLS, h / ROWS
    return int((i % COLS) * cw), int((i // COLS) * ch), int(cw) + 1, int(ch) + 1


def draw_squares(screen, font, gazed, flash, k: float = 1.0):
    """Six tiles; the gazed one bright with a white border, the others dimmed; `flash` = (index, rgb)
    draws a thick border on a just-answered trial tile. `k` scales the line widths (1 = 1080p)."""
    import pygame
    w, h = screen.get_size()
    for i, (_, rgb) in enumerate(SQUARES):
        pygame.draw.rect(screen, rgb if i == gazed else _dim(rgb, 0.55), tile_rect(i, w, h))
    for i, (name, _) in enumerate(SQUARES):
        x, y, tw, th = tile_rect(i, w, h)
        if flash is not None and flash[0] == i:
            pygame.draw.rect(screen, flash[1], (x, y, tw, th), max(2, int(14 * k)))
        elif i == gazed:
            pygame.draw.rect(screen, (255, 255, 255), (x, y, tw, th), max(2, int(6 * k)))
        text_box(screen, font, [name.upper()], x + tw / 2, y + th / 2, center=True)


def draw_gaze(screen, sm, progress: float, k: float = 1.0):
    """White soft dot (orange while holding through a blink), dwell arc, red X when lost; `k` scales sizes."""
    import pygame
    if sm is None:
        return
    x, y = sm.x, sm.y
    d, r, lw = 18 * k, 40 * k, max(2, int(4 * k))
    if sm.lost:
        pygame.draw.line(screen, (255, 40, 40), (x - d, y - d), (x + d, y + d), lw)
        pygame.draw.line(screen, (255, 40, 40), (x - d, y + d), (x + d, y - d), lw)
        return
    soft_dot(screen, x, y, (255, 255, 255) if sm.valid else (255, 170, 60), radius=int(28 * k))
    if progress > 0:
        rect = pygame.Rect(int(x - r), int(y - r), int(2 * r), int(2 * r))
        pygame.draw.arc(screen, (20, 20, 20), rect, math.pi / 2 - 2 * math.pi * progress, math.pi / 2, max(2, int(6 * k)))


def to_surface(img, width: int, mirror: bool = True):
    """A BGR or grey image as an RGB pygame Surface `width` px wide (aspect kept), mirrored like a mirror."""
    import cv2
    import pygame
    img = np.asarray(img)
    img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if mirror:
        img = img[:, ::-1]
    h = max(1, int(round(img.shape[0] * width / max(1, img.shape[1]))))
    rgb = np.ascontiguousarray(cv2.resize(np.ascontiguousarray(img), (width, h), interpolation=cv2.INTER_NEAREST)[:, :, ::-1])
    return pygame.image.frombuffer(rgb.tobytes(), (width, h), "RGB")


def strip_surface(images, width: int, gap: int = 6):
    """The tracker's extra images side by side (each mirrored, in reverse order so the strip mirrors too), padded to
    one height, as a Surface `width` px wide; None without images."""
    import cv2
    imgs = [np.asarray(i) for i in images if i is not None and np.asarray(i).size]
    if not imgs:
        return None
    imgs = [i if i.ndim == 3 else cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in imgs]
    h = max(i.shape[0] for i in imgs)
    pad = lambda i: np.pad(i, ((0, h - i.shape[0]), (0, 0), (0, 0)), constant_values=128)
    parts = []
    for i in reversed(imgs):
        parts += [pad(i)[:, ::-1], np.full((h, gap, 3), 128, np.uint8)]
    return to_surface(np.hstack(parts[:-1]), width, mirror=False)


def draw_eye_preview(screen, font, surf, extra_surf, lines, width: int = 360, frame_surf=None):
    """Top-right: the eye crop, or the whole camera frame (`frame_surf`, red border) while no face is tracked, the
    tracker's extra images, and its readout lines under it."""
    import pygame
    w, _ = screen.get_size()
    x, y = w - width - 16, 12
    shown = surf if surf is not None else frame_surf
    if shown is None:
        text_box(screen, font, ["eye preview: no camera frame yet"], x, y, pad=8)
        y += font.get_height() + 4 + 48 + 6
    else:
        colour = (255, 255, 255) if surf is not None else (255, 80, 80)
        pygame.draw.rect(screen, colour, (x - 2, y - 2, width + 4, shown.get_height() + 4), 2)
        screen.blit(shown, (x, y))
        y += shown.get_height() + 8
    if extra_surf is not None:
        screen.blit(extra_surf, (x, y))
        y += extra_surf.get_height() + 6
    text_box(screen, font, lines or ["no samples yet"], x, y, pad=8)


# ---------------------------------------------------------------------- main
def parse_args(argv=None):
    from taskenv.calibrate import add_task_args, parse_with_tracker
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_task_args(ap)
    ap.add_argument("--dwell", type=float, default=0.8, help="seconds on one square to select it")
    ap.add_argument("--min-cutoff", type=float, default=1.0, help="One Euro: lower = steadier fixations")
    ap.add_argument("--beta", type=float, default=0.02, help="One Euro: higher = less lag on fast moves")
    ap.add_argument("--lost-after", type=float, default=0.3, help="seconds without a valid sample -> lost")
    ap.add_argument("--clamp-margin", type=float, default=0.25,
                    help="a smoothed point up to this fraction of the screen past an edge counts for the edge tile (0 = off)")
    ap.add_argument("--calib", default=None, help="calibration file; default calib/calibration_<tracker>.json")
    ap.add_argument("--trials", type=int, default=12, help="targets per trial run (key T)")
    ap.add_argument("--trial-timeout", type=float, default=5.0, help="seconds per trial before a timeout")
    ap.add_argument("--trial-gap", type=float, default=1.0, help="seconds to read the prompt before answers count")
    ap.add_argument("--csv-dir", default="calib", help="results: <csv-dir>/squares_<time>.csv and samples_<time>.csv")
    ap.add_argument("--no-eyes", action="store_true", help="start with the eye preview off (key V toggles)")
    ap.add_argument("--seconds", type=float, default=0.0, help="quit after this many seconds (0 = until Esc)")
    ap.add_argument("--snapshot", default=None,
                    help="headless: write the last frame to this PNG, no window; a trial run starts at once")
    ap.add_argument("--screen-record", default=None,
                    help="record the task window (gaze point, tiles, HUD) to this video: .avi = MJPG, .mp4 = mp4v; "
                         "writes <file>.times.csv (frame, t) to line it up with the samples CSV and a camera --record")
    ap.add_argument("--screen-fps", type=float, default=30.0, help="frames per second written by --screen-record")
    return parse_with_tracker(ap, argv, default_tracker="synthetic")


def main(argv=None):
    args, module = parse_args(argv)
    from taskenv.api import check_tracker
    from taskenv.calibrate import load_mapper, run_calibration
    from taskenv.display import ScreenRecorder, open_fullscreen, parse_size
    screen_mm = tuple(float(v) for v in args.screen_mm.lower().split("x"))
    screen_px = parse_size(args.screen_px)
    headless = screen_px if args.snapshot else None
    calib_path = args.calib or f"calib/calibration_{module.NAME}.json"

    tracker = check_tracker(module.create(args))            # a real tracker opens its camera here: fail fast
    screen, display = open_fullscreen(args.display, screen_px, caption=f"gaze squares [{module.NAME}]", headless_size=headless)
    w, h = screen.get_size()
    recorder = ScreenRecorder(args.screen_record, (w, h), args.screen_fps) if args.screen_record else None
    mapper = load_mapper(calib_path, module, tracker, (w, h), tag="squares")
    tracker.set_mapper(mapper)
    tracker.start()

    import pygame
    k = h / 1080.0                                        # UI scale: the layout was drawn for 1080p
    font, tile_font, banner_font = (pygame.font.SysFont(None, int(n * k)) for n in (28, 64, 52))
    eye_w = int(360 * k)
    smoother = GazeSmoother(args.min_cutoff, args.beta, args.lost_after)
    dwell = DwellTimer(args.dwell)
    clock = pygame.time.Clock()
    show_eyes = not args.no_eyes
    last_seq, selections = -1, 0
    sm, sample, sample_t, gazed, progress, blink = None, None, None, None, 0.0, False

    def clamped(sm):
        """The smoothed point pulled onto the screen edge when it sits just outside (--clamp-margin); the raw
        mapped point stays in the CSV as x, y."""
        if sm is None or args.clamp_margin <= 0:
            return sm
        cx, cy = clamp_to_screen(sm.x, sm.y, w, h, args.clamp_margin)
        return sm if (cx, cy) == (sm.x, sm.y) else dataclasses.replace(sm, x=cx, y=cy)
    eye_surf, extra_surf, frame_surf, desc = None, None, None, None
    trials, summary, flash = None, None, None            # flash = (tile index, rgb, until)
    t_start = time.monotonic()

    def start_trials(t):
        nonlocal trials, summary, dwell
        trials = TrialRun(trial_sequence(args.trials), args.trial_timeout, args.trial_gap)
        trials.begin(t)
        summary, dwell = None, DwellTimer(args.dwell)
        print(f"[squares] trial run: {len(trials.targets)} targets, {args.trial_timeout:.0f} s timeout", flush=True)

    def finish_trials():
        """Summary + CSV of the rows so far (a full run, an abort or the exit of a snapshot run)."""
        nonlocal trials, summary
        if trials is None:
            return
        if trials.rows:
            os.makedirs(args.csv_dir, exist_ok=True)
            path = os.path.join(args.csv_dir, time.strftime("squares_%Y%m%d_%H%M%S.csv"))
            write_csv(path, trials.rows)
            lines = summary_lines(summarize(trials.rows))
            summary = lines + [f"csv: {path}", "", "SPACE: back to free-look     T: run again"]
            print("[squares] " + " | ".join(lines) + f" | csv {path}", flush=True)
        trials = None

    os.makedirs(args.csv_dir, exist_ok=True)
    samples_path = os.path.join(args.csv_dir, time.strftime("samples_%Y%m%d_%H%M%S.csv"))
    samples_file = open(samples_path, "w", newline="")
    fields = SAMPLE_FIELDS + tuple(module.FEATURE_NAMES) + tuple(module.LOG_FIELDS)
    samples = csv.DictWriter(samples_file, fields, extrasaction="ignore")
    samples.writeheader()
    print(f"[squares] tracker {module.NAME}; per-sample log: {samples_path}", flush=True)
    if args.snapshot:
        start_trials(time.time())
    try:
        while True:
            now = time.time()
            if args.seconds and time.monotonic() - t_start > args.seconds:
                break
            quit_now = False
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    quit_now = True
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                    smoother.reset()
                    if callable(getattr(tracker, "reset", None)):
                        tracker.reset()
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_v:
                    show_eyes = not show_eyes
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                    summary = None
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_t:
                    if trials is None:
                        start_trials(now)
                    else:
                        print("[squares] trial run aborted", flush=True)
                        finish_trials()
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_c:
                    trials, summary = None, None
                    new = run_calibration(screen, tracker, module, screen_mm=screen_mm, viewing_mm=args.viewing_mm,
                                          out=calib_path, display=display)
                    if new is not None:
                        mapper = new
                        tracker.set_mapper(mapper)
                    smoother.reset()
                    dwell = DwellTimer(args.dwell)
            if quit_now:
                break
            if tracker.error is not None:
                raise SystemExit(f"tracker stopped: {tracker.error!r}")
            s = tracker.latest()
            fired_on, t_event = None, now
            if s is not None and s.seq != last_seq:
                last_seq, sample, sample_t, blink, t_event = s.seq, s.sample, s.t, s.sample.blink, s.t
                point = s.screen if (s.sample.valid and s.screen is not None) else None
                sm = clamped(smoother.update(point, s.t))
                if sm is not None:
                    gazed = None if sm.lost else square_at(sm.x, sm.y, w, h)
                    waiting = trials is not None and trials.in_gap(s.t)      # the dwell holds during the prompt gap
                    progress, fired = dwell.update(None if waiting else gazed, s.t)
                    if fired:
                        selections += 1
                        fired_on = gazed
                        if trials is None:
                            print(f"[squares] selected {NAMES[gazed].upper()}", flush=True)
                desc = tracker.describe(s.sample)
                row = sample_row(s, sm, gazed if (sm is not None and not sm.lost) else None, trials, module.FEATURE_NAMES)
                row.update({k: desc.get("log", {}).get(k, "") for k in module.LOG_FIELDS})
                samples.writerow(row)
                if show_eyes and s.eye_crop is not None:
                    img = tracker.eye_image(s.sample, s.eye_crop)
                    eye_surf = to_surface(img, eye_w) if img is not None else None
                    extra_surf, frame_surf = strip_surface(tracker.extra_images(s.sample), eye_w), None
                elif s.eye_crop is None:
                    eye_surf, extra_surf = None, None
                    if show_eyes and s.frame_small is not None:
                        frame_surf = to_surface(s.frame_small, eye_w)
            elif sm is not None and not sm.lost:
                sm = clamped(smoother.update(None, now))     # no new sample: age the hold / mark lost
                if sm.lost:
                    gazed, progress = None, 0.0
                    dwell.update(None, now)
            if trials is not None:
                result = trials.update(fired_on, t_event)
                if result is not None:
                    row = trials.rows[-1]
                    colour = {"hit": (40, 220, 90), "miss": (255, 40, 40), "timeout": (170, 170, 170)}[result]
                    flash = (row["target"] if row["selected"] is None else row["selected"], colour, now + 0.5)
                    dwell = DwellTimer(args.dwell)
                    chosen = "" if row["selected"] is None else f" ({NAMES[row['selected']].upper()} after {row['time_s']:.2f} s)"
                    print(f"[squares] trial {row['trial']}: look at {NAMES[row['target']].upper()} -> {result.upper()}{chosen}",
                          flush=True)
                    if trials.done:
                        finish_trials()
            if flash is not None and now > flash[2]:
                flash = None

            on_screen = sm is not None and not sm.lost
            draw_squares(screen, tile_font, gazed if on_screen else None, flash[:2] if flash else None, k)
            draw_gaze(screen, sm, progress, k)
            latency = f"{max(0.0, now - sample_t) * 1000:4.0f} ms" if sample_t else "  -- "
            if sample is not None and not sample.face_found:
                state = "NO FACE"
            elif sample is not None and not sample.valid and not sample.blink:
                state = "NO FEATURES"
            else:
                state = "no samples yet" if sm is None else ("LOST" if sm.lost else ("blink/hold" if not sm.valid else "tracking"))
            looking = NAMES[gazed].upper() if (on_screen and gazed is not None) else "--"
            extra = f"   {desc['hud']}" if desc and desc.get("hud") else ""
            hud = [f"{module.NAME} {tracker.fps:4.1f} fps   end-to-end {latency}   {state}" + ("   BLINK" if blink else ""),
                   f"looking at: {looking:8s}   dwell {progress:4.2f}   selections {selections}" + extra,
                   "mapping: " + mapper.description + (", press C to calibrate" if mapper.report is None else ""),
                   f"T trials ({args.trials})   V eye preview   C calibrate   R reset filter   Esc quit"]
            text_box(screen, font, hud, 16 * k, 12 * k, pad=int(12 * k))
            if show_eyes:
                draw_eye_preview(screen, font, eye_surf, extra_surf, desc["lines"] if desc else None, eye_w, frame_surf)
            if trials is not None and trials.current is not None:
                ready = "   (get ready)" if trials.in_gap(now) else ""
                text_box(screen, banner_font, [f"trial {trials.index + 1}/{len(trials.targets)}:   look at "
                                               f"{NAMES[trials.current].upper()}{ready}"], w / 2, h / 2, center=True, alpha=200)
            elif summary is not None:
                text_box(screen, font, summary, w / 2, h / 2, center=True, alpha=200)
            pygame.display.flip()
            if recorder is not None:
                recorder.write(screen, time.time())
            clock.tick(30)
    except KeyboardInterrupt:
        pass
    finally:
        finish_trials()
        samples_file.close()
        print(f"[squares] per-sample log closed: {samples_path}", flush=True)
        if args.snapshot:
            pygame.image.save(screen, args.snapshot)
            print(f"[squares] snapshot -> {args.snapshot}  (selections {selections})", flush=True)
        if recorder is not None:
            recorder.close()
            print(f"[squares] screen recording -> {recorder.path} ({recorder.frames} frames at <= {recorder.fps:g} fps; "
                  f"times in {recorder.times_path})", flush=True)
        tracker.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
