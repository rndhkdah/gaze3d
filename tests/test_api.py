import argparse
import os

import pytest

from taskenv.api import REQUIRED, BasicSample, ScaledMapper, TrackerSample, builtin_trackers, check_tracker, load_tracker

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_builtin_trackers_are_listed_and_loadable_by_name():
    names = builtin_trackers()
    assert set(names) == {"gaze3d", "synthetic"}
    m = load_tracker("synthetic")
    assert m.NAME == "synthetic" and all(hasattr(m, n) for n in REQUIRED)


def test_tracker_loads_from_a_file_path():
    m = load_tracker(os.path.join(HERE, "trackers", "synthetic.py"))
    assert m.NAME == "synthetic" and m.FEATURE_NAMES == ("u", "v")


def test_bad_specs_fail_with_a_message(tmp_path):
    with pytest.raises(SystemExit, match="no tracker"):
        load_tracker("does_not_exist")
    with pytest.raises(SystemExit, match="not found"):
        load_tracker(str(tmp_path / "missing.py"))
    bad = tmp_path / "bad.py"
    bad.write_text("NAME = 'bad'\n")
    with pytest.raises(SystemExit, match="lacks FEATURE_NAMES, LOG_FIELDS, add_args, create"):
        load_tracker(str(bad))


def test_check_tracker_names_missing_methods():
    class Half:
        def start(self): ...

    with pytest.raises(SystemExit, match="lacks stop, latest"):
        check_tracker(Half())


def test_scaled_mapper_and_sample_types():
    m = ScaledMapper((1920, 1080))
    assert m.predict((0.5, 1.0)) == (960.0, 1080.0) and m.report is None and m.description
    s = TrackerSample(1.0, 1.1, 3, BasicSample(True, True, False, 1.0), None, None)
    assert s.eye_crop is None and s.frame_small is None and s.sample.valid


def test_synthetic_tracker_fulfils_the_contract():
    m = load_tracker("synthetic")
    ap = argparse.ArgumentParser()
    m.add_args(ap)
    tr = check_tracker(m.create(ap.parse_args([])))
    tr.set_mapper(tr.default_mapper((1920, 1080)))
    tr.start()
    s = tr.latest()
    assert s.seq == 1 and s.sample.face_found and tr.error is None and tr.fps > 0
    if s.sample.valid:
        assert s.vec.shape == (2,) and 0 <= s.screen[0] <= 1920 and 0 <= s.screen[1] <= 1080
    else:
        assert s.screen is None and s.sample.blink
    d = tr.describe(s.sample)
    assert set(d) == {"lines", "hud", "log"} and set(d["log"]) == set(m.LOG_FIELDS)
    assert tr.eye_image(s.sample, None) is None and tr.extra_images(s.sample) == []
    tr.stop()


def test_gaze3d_adapter_describes_samples_without_a_camera():
    pytest.importorskip("gaze3d.model")
    import numpy as np
    from gaze3d.geometry import Intrinsics, ScreenPose
    from gaze3d.model import Sample, fuse
    m = load_tracker("gaze3d")
    from trackers.gaze3d import Gaze3DAdapter
    assert m.FEATURE_NAMES == ("sx_mm", "sy_mm")
    tr = Gaze3DAdapter(inner=None, screen_mm=(598.0, 337.0))
    nf = tr.describe(Sample(False, False, False, 0.0))
    assert nf["lines"][0].startswith("NO FACE") and nf["hud"] == "" and set(nf["log"]) == set(m.LOG_FIELDS)
    intr, pose = Intrinsics(1280, 720, 900.0, 900.0, 640.0, 360.0), ScreenPose((598, 337), tilt_deg=25.0)
    P = np.full((478, 2), (640.0, 420.0))
    for eye, x in (({"a": 33, "b": 133, "up": 159, "low": 145, "iris": 468, "ear": ((160, 144), (158, 153))}, 560.0),
                   ({"a": 362, "b": 263, "up": 386, "low": 374, "iris": 473, "ear": ((385, 380), (387, 373))}, 720.0)):
        P[eye["a"]], P[eye["b"]] = (x - 30, 330.0), (x + 30, 330.0)
        P[eye["up"]], P[eye["low"]], P[eye["iris"]] = (x, 322.0), (x, 338.0), (x, 330.0)
        for u, l in eye["ear"]:
            P[u], P[l] = (x - 9, 322.0), (x - 9, 338.0)
    P[168] = (640.0, 340.0)
    s = fuse(P, np.array([0.0, 0.0, -1.0]), np.full((720, 1280), 0.56, np.float32), intr, pose, 1.0, head_rot=np.eye(3))
    assert s.valid
    d = tr.describe(s)
    assert d["lines"][0].startswith("hit") and "mm" in d["hud"] and d["log"]["src_l"] == "iris"
    assert d["log"]["eye_l_z"] == "560.0" and d["log"]["head_yaw"] == "0.0" and d["log"]["gaze_pitch"] == "0.00"
    assert set(d["log"]) == set(m.LOG_FIELDS) and 2 <= len(d["lines"]) <= 4 and all(len(t) <= 44 for t in d["lines"])
    crop = np.zeros((80, 240, 3), np.uint8)
    img = tr.eye_image(s, (crop, 500, 290))
    assert img.shape == crop.shape and img[40, 60].any() and not crop.any()
    assert tr.eye_image(s, None) is None and tr.extra_images(s) == []
    mp = tr.default_mapper((1920, 1080))
    assert type(mp).__name__ == "PlaneMapper" and mp.predict((299.0, 168.5)) == pytest.approx((960.0, 540.0))
    assert mp.report is None and isinstance(mp.description, str)
    blink = fuse(P, np.array([0.0, 0.0, -1.0]), None, intr, pose, 1.0)        # no depth, no pnp: no origin
    db = tr.describe(blink)
    assert db["lines"][0].startswith("no hit") and db["log"]["src_l"] == "none"

    class Model:
        _dirs = [1, 2]

    class Inner:
        extractor = Model()

    Gaze3DAdapter(Inner()).reset()
    assert Inner.extractor._dirs == []
