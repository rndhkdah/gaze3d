import csv
import math
import random

import numpy as np
import pytest

from taskenv.api import BasicSample, TrackerSample
from taskenv.squares import (SAMPLE_FIELDS, SQUARES, TrialRun, sample_row, square_at, strip_surface, summarize, summary_lines,
                             to_surface, trial_sequence, write_csv)

ROWS = [dict(trial=1, target=0, selected=0, hit=True, time_s=1.0, timeout=False),
        dict(trial=2, target=0, selected=3, hit=False, time_s=2.0, timeout=False),
        dict(trial=3, target=5, selected=None, hit=False, time_s=5.0, timeout=True),
        dict(trial=4, target=5, selected=5, hit=True, time_s=1.5, timeout=False)]


def test_six_named_squares_in_row_major_order():
    assert [name for name, _ in SQUARES] == ["red", "green", "blue", "yellow", "magenta", "cyan"]


def test_square_at_maps_screen_points_to_the_3x2_tiles():
    assert square_at(0, 0, 1920, 1080) == 0 and square_at(960, 100, 1920, 1080) == 1 and square_at(1919, 0, 1920, 1080) == 2
    assert square_at(10, 1079, 1920, 1080) == 3 and square_at(960, 900, 1920, 1080) == 4 and square_at(1919, 1079, 1920, 1080) == 5
    assert square_at(-1, 5, 1920, 1080) is None and square_at(5, 1080, 1920, 1080) is None


def test_trial_sequence_is_balanced_and_avoids_repeats():
    for seed in range(10):
        seq = trial_sequence(12, rng=random.Random(seed))
        assert len(seq) == 12 and all(seq.count(i) == 2 for i in range(6))
        assert all(a != b for a, b in zip(seq, seq[1:]))
    assert len(trial_sequence(8, rng=random.Random(1))) == 8


def test_trial_run_records_hit_miss_and_timeout():
    run = TrialRun([2, 4, 0], timeout_s=5.0, gap_s=0.0)
    run.begin(10.0)
    assert run.update(2, 11.0) == "hit" and run.update(1, 12.0) == "miss" and run.update(None, 17.5) == "timeout"
    assert run.done and run.update(0, 18.0) is None
    assert [r["selected"] for r in run.rows] == [2, 1, None] and [r["timeout"] for r in run.rows] == [False, False, True]


def test_trial_run_ignores_selections_during_the_gap():
    run = TrialRun([1, 2], timeout_s=5.0, gap_s=1.0)
    run.begin(0.0)
    assert run.in_gap(0.5) and run.update(1, 0.5) is None
    assert run.update(1, 1.5) == "hit" and run.rows[0]["time_s"] == pytest.approx(0.5)
    assert run.update(None, 7.4) is None and run.update(None, 7.6) == "timeout"


def test_summarize_and_lines():
    s = summarize(ROWS)
    assert s["hits"] == 2 and s["total"] == 4 and s["timeouts"] == 1 and s["mean_time_s"] == pytest.approx(1.25)
    assert math.isnan(summarize([])["mean_time_s"])
    text = "\n".join(summary_lines(s))
    assert "2/4" in text and "red" in text and "cyan" in text


def test_write_csv_uses_square_names(tmp_path):
    path = tmp_path / "trials.csv"
    write_csv(path, ROWS)
    with open(path, newline="") as f:
        got = list(csv.DictReader(f))
    assert [r["target"] for r in got] == ["red", "red", "cyan", "cyan"]
    assert [r["selected"] for r in got] == ["red", "yellow", "", "cyan"] and got[2]["timeout"] == "1"


def test_sample_row_has_the_fixed_fields_then_the_feature_vector():
    from taskenv.smoothing import SmoothedGaze
    g = BasicSample(True, True, False, 12.5)
    s = TrackerSample(12.5, 12.6, 7, g, np.array([0.25, 0.75]), (480.0, 810.0))
    run = TrialRun([3], gap_s=0.0)
    run.begin(12.0)
    row = sample_row(s, SmoothedGaze(482.0, 805.0, True, False, 0.0), 3, run, ("u", "v"))
    assert tuple(row)[:len(SAMPLE_FIELDS)] == SAMPLE_FIELDS and tuple(row)[len(SAMPLE_FIELDS):] == ("u", "v")
    assert row["face"] == 1 and row["x"] == "480" and row["sm_y"] == "805" and row["u"] == "0.2500" and row["v"] == "0.7500"
    assert row["tile"] == "yellow" and row["trial"] == 1 and row["target"] == "yellow" and row["lost"] == 0
    empty = sample_row(TrackerSample(1.0, 1.0, 1, BasicSample(False, False, False, 1.0), None, None), None, None, None, ("u", "v"))
    assert empty["face"] == 0 and empty["x"] == "" and empty["u"] == "" and empty["target"] == ""


def test_surfaces_mirror_and_scale():
    import pygame
    img = np.zeros((40, 120, 3), np.uint8)
    img[:, :60] = (0, 0, 255)                                     # BGR red on the image-left half
    surf = to_surface(img, 300)
    assert isinstance(surf, pygame.Surface) and surf.get_size() == (300, 100)
    assert surf.get_at((290, 5))[:3] == (255, 0, 0) and surf.get_at((10, 5))[:3] == (0, 0, 0)     # mirrored, as RGB
    a, b = np.zeros((20, 40), np.uint8), np.full((30, 50), 255, np.uint8)
    strip = strip_surface([a, b], 300)
    assert strip.get_width() == 300 and strip.get_at((20, 5))[:3] == (255, 255, 255) and strip.get_at((280, 5))[:3] == (0, 0, 0)
    assert strip_surface([], 300) is None and strip_surface([np.zeros((0, 0), np.uint8)], 300) is None
