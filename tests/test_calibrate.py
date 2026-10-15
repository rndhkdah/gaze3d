import argparse

import numpy as np
import pytest

from taskenv.api import ScaledMapper
from taskenv.calibrate import (Affine, GazeMapper, errors_in_units, fit_calibration, load_mapper, reject_outliers, target_points,
                               undersampled, usable)

SCREEN_PX = (1920, 1080)
SCREEN_MM = (598.0, 337.0)


def _synthetic(dim=4, n_per_target=25, noise=0.004, seed=0):
    """Samples of an affine feature -> screen mapping at a 3x3 grid of targets, for 2 or 4 features."""
    rng = np.random.default_rng(seed)
    targets = target_points(9, *SCREEN_PX)
    X, Y, tid = [], [], []
    for k, (tx, ty) in enumerate(targets):
        sx, sy = (tx - 960) / 960, (ty - 540) / 540
        for _ in range(n_per_target):
            if dim == 2:
                f = (0.5 + 0.5 * sx + rng.normal(0, noise), 0.5 + 0.5 * sy + rng.normal(0, noise))
            else:
                f = (0.5 - 0.12 * sx + rng.normal(0, noise), 0.5 + 0.05 * sy + rng.normal(0, noise),
                     0.52 - 0.14 * sx + rng.normal(0, noise), 0.49 + 0.06 * sy + rng.normal(0, noise))
            X.append(f)
            Y.append((tx, ty))
            tid.append(k)
    return np.array(X), np.array(Y, float), np.array(tid), targets


class FakeModule:
    NAME, FEATURE_NAMES, LOG_FIELDS = "fake", ("a", "b", "c", "d"), ()


class FakeTracker:
    def default_mapper(self, screen_px):
        return ScaledMapper(screen_px, "fake default")


@pytest.mark.parametrize("dim", [2, 4])
def test_fit_recovers_synthetic_mapping_for_any_feature_dimension(dim):
    X, Y, tid, targets = _synthetic(dim)
    mapper, report = fit_calibration(X, Y, tid, SCREEN_PX, SCREEN_MM, tracker="fake", feature_names=("f",) * dim)
    assert report["loo_mean_px"] < 30 and report["n_targets"] == 9
    for k, (tx, ty) in enumerate(targets):
        x, y = mapper.predict(X[tid == k].mean(axis=0))
        assert abs(x - tx) < 30 and abs(y - ty) < 30
    assert "calibrated" in mapper.description and mapper.tracker == "fake"


def test_affine_is_exact_without_noise_and_round_trips():
    F = np.random.default_rng(1).normal(size=(50, 3))
    y = F @ np.array([3.0, -2.0, 0.5]) + 7.0
    a = Affine.fit(F, y, alpha=1e-9)
    assert a.predict(F) == pytest.approx(y, abs=1e-6)
    assert Affine.from_dict(a.to_dict()).predict(F[:3]) == pytest.approx(a.predict(F[:3]))


def test_reject_outliers_and_errors_in_units():
    X, Y, tid, _ = _synthetic()
    X = X.copy()
    X[0, 0] += 0.3
    keep = reject_outliers(X, tid)
    assert not keep[0] and keep.sum() >= 0.95 * len(X)
    cm, deg = errors_in_units(100.0, SCREEN_PX, SCREEN_MM, viewing_mm=600.0)
    mm_per_px = (598 / 1920 + 337 / 1080) / 2
    assert cm == pytest.approx(100 * mm_per_px / 10) and deg == pytest.approx(np.degrees(np.arctan(100 * mm_per_px / 600)))


def test_fit_needs_four_targets_and_every_dot_sampled():
    X, Y, tid, _ = _synthetic()
    keep = tid < 3
    with pytest.raises(ValueError):
        fit_calibration(X[keep], Y[keep], tid[keep], SCREEN_PX, SCREEN_MM)
    assert undersampled([0] * 5 + [1] * 5 + [2] * 4, 3) == [2] and undersampled([0, 0, 0, 0] * 9, 9) == list(range(1, 9))


def test_load_mapper_uses_the_file_only_when_it_matches_and_is_usable(tmp_path, capsys):
    X, Y, tid, _ = _synthetic()
    mapper, report = fit_calibration(X, Y, tid, SCREEN_PX, SCREEN_MM, tracker="fake", feature_names=FakeModule.FEATURE_NAMES)
    path = tmp_path / "calibration_fake.json"
    mapper.save(path)
    loaded = GazeMapper.load(path)
    assert loaded.predict(X[0]) == pytest.approx(mapper.predict(X[0])) and loaded.feature_names == FakeModule.FEATURE_NAMES
    assert isinstance(load_mapper(str(path), FakeModule, FakeTracker(), SCREEN_PX), GazeMapper)
    other = type("M", (), {"NAME": "other", "FEATURE_NAMES": ("x", "y"), "LOG_FIELDS": ()})
    assert isinstance(load_mapper(str(path), other, FakeTracker(), SCREEN_PX), ScaledMapper)      # another tracker's file
    assert "made for tracker 'fake'" in capsys.readouterr().out
    mapper.report = dict(report, loo_mean_px=900.0)
    mapper.save(path)
    assert not usable(mapper.report, SCREEN_PX)
    assert isinstance(load_mapper(str(path), FakeModule, FakeTracker(), SCREEN_PX), ScaledMapper)   # too inaccurate
    assert isinstance(load_mapper(str(tmp_path / "missing.json"), FakeModule, FakeTracker(), SCREEN_PX), ScaledMapper)


def test_target_points():
    assert len(target_points(9, 1920, 1080)) == 9 and len(target_points(13, 1920, 1080)) == 13
    with pytest.raises(ValueError):
        target_points(5, 1920, 1080)
