import math

import numpy as np
import pytest

from gaze3d.geometry import Intrinsics, ScreenPose, gaze_hit
from gaze3d.model import (EYES, LEFT_EYE, MAX_EAR, N_MESH, RIGHT_EYE, Sample, draw_overlay, eye_aspect_ratio, fuse,
                          resolve_device)

INTR = Intrinsics(1280, 720, 900.0, 900.0, 640.0, 360.0)
POSE = ScreenPose((598, 337), tilt_deg=25.0)
AT_CAMERA = np.array([0.0, 0.0, -1.0])


def face(cx=640.0, cy=330.0, ipd=160.0, open_eyes=True, width=60.0):
    """478 landmarks (px) with the eye corners, lids, irises and the nose bridge at known places."""
    P = np.full((478, 2), (cx, cy + 80.0), float)                 # everything else: the chin area
    lid = 0.25 * width if open_eyes else 0.02 * width
    for eye, x in ((RIGHT_EYE, cx - ipd / 2), (LEFT_EYE, cx + ipd / 2)):
        P[eye["a"]] = (x - width / 2, cy)
        P[eye["b"]] = (x + width / 2, cy)
        P[eye["lid_up"]] = (x, cy - lid / 2)
        P[eye["lid_low"]] = (x, cy + lid / 2)
        for (u, l) in eye["ear"]:
            P[u] = (x - width * 0.15, cy - lid / 2)
            P[l] = (x - width * 0.15, cy + lid / 2)
        P[eye["iris"]] = (x, cy)
        for k, r in enumerate(eye["ring"]):
            P[r] = (x + (6.0 if k % 2 else -6.0) * (1 if k < 2 else 0), cy + (6.0 if k >= 2 else -6.0) * (1 if k >= 2 else 0))
    P[168] = (cx, cy + 10.0)
    return P


def plane_depth(z=0.56):
    return np.full((720, 1280), z, np.float32)


def test_eye_aspect_ratio_open_and_closed():
    assert eye_aspect_ratio(face(open_eyes=True), RIGHT_EYE) == pytest.approx(0.25)
    assert eye_aspect_ratio(face(open_eyes=False), LEFT_EYE) == pytest.approx(0.02)


def test_no_face_and_profile():
    s = fuse(None, AT_CAMERA, plane_depth(), INTR, POSE, 1.0)
    assert (s.valid, s.face_found, s.as_vector()) == (False, False, None)
    P = face()
    for eye in EYES:                                   # lids wide apart: EAR beyond MAX_EAR = profile mesh
        P[eye["lid_up"]][1] -= 40
        for (u, l) in eye["ear"]:
            P[u][1] -= 40
    s = fuse(P, AT_CAMERA, plane_depth(), INTR, POSE, 1.0)
    assert not s.face_found and s.ear > MAX_EAR


def test_depth_at_the_irises_places_both_eyes_and_hits_the_plane():
    P = face()
    s = fuse(P, AT_CAMERA, plane_depth(0.56), INTR, POSE, 2.0)
    assert s.valid and s.face_found and not s.blink
    assert s.depth_src == ("iris", "iris")
    left, right = s.eye_mm
    assert np.allclose(left, INTR.deproject(720, 330, 560.0)) and np.allclose(right, INTR.deproject(560, 330, 560.0))
    assert np.allclose(s.origin_mm, (left + right) / 2)
    sx, sy, dist = gaze_hit(s.origin_mm, AT_CAMERA, POSE)
    assert (s.sx, s.sy, s.dist_mm) == pytest.approx((sx, sy, dist))
    assert np.allclose(s.as_vector(), (sx, sy))
    assert s.eye_points.shape == (8, 2) and s.iris_px.shape == (2, 2)
    assert s.head_yaw_pitch() is None


def test_blink_keeps_the_hit_but_is_invalid():
    s = fuse(face(open_eyes=False), AT_CAMERA, plane_depth(), INTR, POSE, 3.0)
    assert s.face_found and s.blink and not s.valid and s.sx is not None and s.as_vector() is None


def test_without_depth_the_pnp_centres_are_the_origin():
    pnp = np.array([[80.0, -30.0, 600.0], [-80.0, -30.0, 600.0]])
    s = fuse(face(), AT_CAMERA, None, INTR, POSE, 4.0, pnp_eyes_mm=pnp)
    assert s.valid and s.depth_src == ("pnp", "pnp") and np.allclose(s.origin_mm, (0.0, -30.0, 600.0))
    s = fuse(face(), AT_CAMERA, None, INTR, POSE, 4.0)
    assert s.face_found and not s.valid and s.origin_mm is None and s.depth_src == ("none", "none")


def test_one_eye_without_depth_uses_the_other_fallbacks():
    depth = plane_depth(0.56)
    depth[300:360, 690:760] = 0.0                      # a hole over the left iris + corners; nose bridge still valid
    s = fuse(face(), AT_CAMERA, depth, INTR, POSE, 5.0)
    assert s.depth_src == ("nose", "iris") and s.valid
    assert s.eye_mm[0][2] == pytest.approx(560.0)


def test_no_direction_or_ray_away_is_invalid():
    s = fuse(face(), None, plane_depth(), INTR, POSE, 6.0)
    assert s.face_found and not s.valid and s.origin_mm is not None and s.direction is None
    s = fuse(face(), np.array([0.0, 0.0, 1.0]), plane_depth(), INTR, POSE, 6.0)   # looking away from the screen
    assert not s.valid and s.sx is None and s.direction is not None
    s = fuse(face(), np.zeros(3), plane_depth(), INTR, POSE, 6.0)
    assert not s.valid and s.direction is None


def test_head_translation_moves_the_hit():
    a = fuse(face(cx=640), AT_CAMERA, plane_depth(), INTR, POSE, 7.0)
    b = fuse(face(cx=700), AT_CAMERA, plane_depth(), INTR, POSE, 7.0)   # 60 px right in the image at 0.56 m
    assert b.sx - a.sx == pytest.approx(-(60 / 900.0) * 560.0, rel=1e-6)
    assert b.sy == pytest.approx(a.sy)


def test_head_yaw_pitch_from_rotation():
    R = np.eye(3)
    s = Sample(True, True, False, 0.0, head_rot=R)
    assert s.head_yaw_pitch() == pytest.approx((0.0, 0.0))
    c, si = math.cos(math.radians(20)), math.sin(math.radians(20))
    Ry = np.array([[c, 0, si], [0, 1, 0], [-si, 0, c]])            # turn the head
    yaw, pitch = Sample(True, True, False, 0.0, head_rot=Ry).head_yaw_pitch()
    assert abs(yaw) == pytest.approx(20.0) and pitch == pytest.approx(0.0)


def test_draw_overlay_marks_the_irises_in_the_crop():
    cv2 = pytest.importorskip("cv2")
    s = fuse(face(), AT_CAMERA, plane_depth(), INTR, POSE, 8.0)
    crop = np.zeros((120, 300, 3), np.uint8)
    out = draw_overlay(crop.copy(), s, origin=(500, 280), intr=INTR)
    assert out.shape == crop.shape and not crop.any()
    assert out[330 - 280, 560 - 500].tolist() == [60, 255, 60]       # right iris, green
    assert out[330 - 280, 720 - 500].tolist() == [255, 60, 60]       # left iris, blue
    assert draw_overlay(crop.copy(), Sample(False, False, False, 0.0), (0, 0)).sum() == 0
    full = draw_overlay(np.zeros((720, 1280, 3), np.uint8), s, full=True, intr=INTR)
    assert full.any()


def test_resolve_device():
    assert resolve_device("cpu") == "cpu" and resolve_device("auto") in ("cpu", "cuda")
