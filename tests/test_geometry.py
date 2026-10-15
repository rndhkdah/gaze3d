import math

import numpy as np
import pytest

from gaze3d.geometry import (Intrinsics, PlaneMapper, ScreenPose, eye_point, gaze_hit, median_depth, parse_cam_pos,
                             parse_pair, ray_plane, write_camera_yaml, yaw_pitch_deg)

INTR = Intrinsics(1280, 720, 900.0, 900.0, 640.0, 360.0)


def test_deproject_project_roundtrip():
    p = INTR.deproject(640 + 90, 360 - 45, 900.0)
    assert np.allclose(p, (90.0, -45.0, 900.0))
    assert np.allclose(INTR.project(p), (730.0, 315.0))
    assert INTR.project((0, 0, 0)) is None


def test_dummy_intrinsics_like_ptgaze():
    d = Intrinsics.dummy(640, 480)
    assert (d.fx, d.fy, d.ppx, d.ppy) == (640.0, 640.0, 320.0, 240.0)
    assert np.allclose(d.camera_matrix(), [[640, 0, 320], [0, 640, 240], [0, 0, 1]])


def test_camera_yaml(tmp_path):
    path = write_camera_yaml(tmp_path / "cam.yaml", INTR)
    text = path.read_text()
    assert "image_width: 1280" in text and "900.000000, 0.000000, 640.000000" in text
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(text)
    assert data["camera_matrix"]["rows"] == 3 and len(data["distortion_coefficients"]["data"]) == 5


def test_median_depth_ignores_zeros_and_needs_enough_pixels():
    depth = np.zeros((720, 1280), np.float32)
    depth[100:110, 200:210] = 0.6
    depth[105, 205] = 0.0
    assert median_depth(depth, 205, 105, patch=9) == pytest.approx(0.6)
    assert median_depth(depth, 205, 105, patch=3, min_valid=10) is None       # 8 valid in a 3x3 with the hole
    assert median_depth(depth, 5000, 105) is None
    assert median_depth(None, 205, 105) is None


def test_eye_point_fallback_chain():
    depth = np.zeros((720, 1280), np.float32)
    iris, corner, nose = (700.0, 300.0), (690.0, 302.0), (640.0, 320.0)
    pnp = np.array([10.0, -20.0, 580.0])
    assert eye_point(depth, INTR, iris, corner, nose, pnp)[1] == "pnp"
    assert eye_point(depth, INTR, iris, corner, nose, None) == (None, "none")
    depth[315:325, 635:645] = 0.55                       # nose only
    p, src = eye_point(depth, INTR, iris, corner, nose, pnp)
    assert src == "nose" and np.allclose(p, INTR.deproject(700, 300, 550.0))
    depth[297:307, 685:695] = 0.57                       # corner
    assert eye_point(depth, INTR, iris, corner, nose, pnp)[1] == "corner"
    depth[295:305, 695:705] = 0.56                       # iris wins
    p, src = eye_point(depth, INTR, iris, corner, nose, pnp)
    assert src == "iris" and p[2] == pytest.approx(560.0)
    depth[295:305, 695:705] = 2.0                        # out of range -> next source
    assert eye_point(depth, INTR, iris, corner, nose, pnp, max_z=1.5)[1] == "corner"


def test_screen_pose_is_a_rotation_with_the_expected_axes():
    pose = ScreenPose((598, 337), tilt_deg=0.0)
    assert np.allclose(pose.cam_pos, (299.0, 337 + 45, -20.0))
    assert np.linalg.det(pose.R) == pytest.approx(1.0)
    assert np.allclose(pose.rotate((1, 0, 0)), (-1, 0, 0))     # image right = user's left
    assert np.allclose(pose.rotate((0, 1, 0)), (0, 1, 0))      # down stays down
    assert np.allclose(pose.rotate((0, 0, 1)), (0, 0, -1))     # camera forward = toward the user = out of the screen
    tilted = ScreenPose((598, 337), cam_pos=(0, 0, 0), tilt_deg=30.0)
    assert np.linalg.det(tilted.R) == pytest.approx(1.0)
    p = tilted.to_screen((0, 0, 1000.0))                       # a point on the tilted axis sits ABOVE the camera
    assert p[1] == pytest.approx(-1000 * math.sin(math.radians(30)))
    assert p[2] == pytest.approx(-1000 * math.cos(math.radians(30)))


def test_ray_plane():
    assert ray_plane((100, 50, -600), (0, 0, 1)) == pytest.approx((100.0, 50.0, 600.0))
    sx, sy, t = ray_plane((0, 0, -600), (0.1, -0.2, 1.0))
    assert (sx, sy) == pytest.approx((60.0, -120.0))
    assert ray_plane((0, 0, -600), (0, 0, -1)) is None          # away from the screen
    assert ray_plane((0, 0, -600), (1, 0, 0)) is None           # parallel
    assert ray_plane((0, 0, 600), (0, 0, 1)) is None            # behind the panel


def test_gaze_at_the_camera_hits_its_footprint():
    pose = ScreenPose((598, 337), tilt_deg=25.0)
    eye = np.array([0.0, -200.0, 560.0])                        # mm, camera frame
    sx, sy, t = gaze_hit(eye, -eye, pose)                        # looking straight at the optical centre
    # the ray passes through the camera centre (20 mm in front of the panel) and continues to the panel
    cam_s = pose.cam_pos
    d_s = pose.rotate(-eye / np.linalg.norm(eye))
    expect = cam_s + (-cam_s[2] / d_s[2]) * d_s
    assert (sx, sy) == pytest.approx((expect[0], expect[1]))
    assert abs(sx - cam_s[0]) < 30 and abs(sy - cam_s[1]) < 30


def test_head_translation_moves_the_hit_by_the_same_amount():
    pose = ScreenPose((598, 337), tilt_deg=25.0)
    d = np.array([0.0, 0.0, -1.0])                               # a fixed direction, toward the camera plane
    a = gaze_hit((0.0, -150.0, 560.0), d, pose)
    b = gaze_hit((50.0, -150.0, 560.0), d, pose)                 # the head moves 50 mm to the image's right
    assert b[0] - a[0] == pytest.approx(-50.0)                   # = the user's left on the screen
    assert b[1] == pytest.approx(a[1])


def test_yaw_pitch_matches_ptgaze_convention():
    assert yaw_pitch_deg((0, 0, -1)) == pytest.approx((0.0, 0.0))
    yaw, pitch = yaw_pitch_deg((-math.sin(math.radians(10)), 0, -math.cos(math.radians(10))))
    assert yaw == pytest.approx(10.0) and pitch == pytest.approx(0.0)
    assert yaw_pitch_deg((0, -0.5, -0.866))[1] == pytest.approx(30.0, abs=0.1)
    assert yaw_pitch_deg((0, 0, 0)) == (0.0, 0.0)


def test_plane_mapper_and_parsers():
    m = PlaneMapper((1920, 1080), (598, 337))
    assert m.predict((299.0, 168.5)) == pytest.approx((960.0, 540.0))
    assert m.report is None and "px/mm" in m.description
    assert parse_pair("598x337") == (598.0, 337.0) and parse_pair("1920X1080") == (1920.0, 1080.0)
    assert parse_cam_pos("auto") is None and parse_cam_pos("299,382,-20") == (299.0, 382.0, -20.0)
    with pytest.raises(ValueError):
        parse_cam_pos("1,2")
