import sys

import numpy as np
import pytest

from gaze3d.capture import Frame, ImageCapture, RealSenseDepth, Webcam, open_capture, parse_size


def test_parse_size():
    assert parse_size("1280x720") == (1280, 720) and parse_size("640X480") == (640, 480)


def test_open_capture_picks_the_backend(monkeypatch):
    monkeypatch.setattr("gaze3d.capture.realsense_present", lambda: False)
    assert isinstance(open_capture("auto"), Webcam)
    monkeypatch.setattr("gaze3d.capture.realsense_present", lambda: True)
    assert isinstance(open_capture("auto"), RealSenseDepth)
    assert isinstance(open_capture("rs"), RealSenseDepth)
    cam = open_capture("/dev/video6")
    assert isinstance(cam, Webcam) and cam.source == "/dev/video6"
    assert open_capture("2").source == 2
    bag = open_capture("auto", bag="x.bag")
    assert isinstance(bag, RealSenseDepth) and bag.bag == "x.bag"


def test_realsense_start_without_pyrealsense2_exits(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyrealsense2", None)
    with pytest.raises(SystemExit, match="pyrealsense2"):
        RealSenseDepth().start()
    assert RealSenseDepth().read() is None


def test_image_capture_serves_the_image_then_ends(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "f.png"
    cv2.imwrite(str(path), np.full((48, 64, 3), 7, np.uint8))
    cap = ImageCapture(path, repeat=2).start()
    f1, f2 = cap.read(), cap.read()
    assert isinstance(f1, Frame) and f1.bgr.shape == (48, 64, 3) and f1.depth is None
    assert f1.intrinsics.fx == 64 and (f1.seq, f2.seq) == (1, 2)
    assert cap.read() is None and cap.ended


def test_camera_yaml_loads_in_ptgaze(tmp_path):
    ptcam = pytest.importorskip("ptgaze.common.camera")
    from gaze3d.geometry import Intrinsics, write_camera_yaml
    intr = Intrinsics(1280, 720, 911.5, 910.2, 643.1, 357.9, (0.1, -0.2, 0.0, 0.0, 0.05), "brown_conrady")
    cam = ptcam.Camera.from_file(str(write_camera_yaml(tmp_path / "c.yaml", intr)))
    assert (cam.width, cam.height) == (1280, 720)
    assert np.allclose(cam.camera_matrix, intr.camera_matrix()) and cam.dist_coefficients.shape == (5, 1)
    assert cam.dist_coefficients.ravel()[0] == pytest.approx(0.1)
