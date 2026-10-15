"""Pure geometry for the monitor-plane gaze: camera intrinsics and deprojection, depth sampling at a landmark, the
screen frame with the camera's pose in it, the ray-plane hit, and the mm -> pixel default mapper. NumPy only.

Frames. Camera (OpenCV): x right in the image, y down, z forward into the scene, millimetres here (RealSense depth is
metres; convert at the boundary). Screen: origin at the top-left of the active area, x right as the user sees it,
y down, z INTO the screen; right-handed, so the user sits at negative z. A face camera mirrors the user's right, so
with no tilt the camera's x is the screen's -x, y stays y, z (toward the user) is the screen's -z:
R0 = diag(-1, 1, -1). A tilt of `tilt_deg` > 0 (camera looking up) rotates that about the screen's x axis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------- intrinsics
@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics of the colour image (pixels). `coeffs` are kept for the record (ptgaze YAML) only: the
    D435i colour stream reports (near) zero distortion and the calibration absorbs the rest."""
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    coeffs: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)
    model: str = "none"

    @classmethod
    def dummy(cls, width, height):
        """ptgaze's dummy camera: focal length = image width, principal point at the centre, no distortion."""
        return cls(int(width), int(height), float(width), float(width), width / 2.0, height / 2.0)

    @classmethod
    def from_rs(cls, intr):
        """From a pyrealsense2 `intrinsics` object (as_video_stream_profile().get_intrinsics())."""
        return cls(int(intr.width), int(intr.height), float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy),
                   tuple(float(c) for c in intr.coeffs), str(intr.model).replace("distortion.", ""))

    def camera_matrix(self):
        return np.array([[self.fx, 0.0, self.ppx], [0.0, self.fy, self.ppy], [0.0, 0.0, 1.0]], float)

    def deproject(self, u, v, z):
        """Pixel (u, v) at depth z (the z coordinate, not the range) -> (x, y, z) in z's unit."""
        return np.array([(u - self.ppx) / self.fx * z, (v - self.ppy) / self.fy * z, z], float)

    def project(self, p):
        """(x, y, z) -> pixel; None when z <= 0."""
        p = np.asarray(p, float)
        if p[2] <= 1e-9:
            return None
        return (self.fx * p[0] / p[2] + self.ppx, self.fy * p[1] / p[2] + self.ppy)


def write_camera_yaml(path, intr: Intrinsics):
    """ptgaze's camera-parameter file (Camera.from_file): image size, 3x3 matrix, 5 distortion coefficients."""
    k = intr.camera_matrix().ravel()
    c = list(intr.coeffs)[:5] + [0.0] * max(0, 5 - len(intr.coeffs))
    text = ("image_width: %d\nimage_height: %d\ncamera_matrix:\n    rows: 3\n    cols: 3\n    data: [%s]\n"
            "distortion_coefficients:\n    rows: 1\n    cols: 5\n    data: [%s]\n"
            % (intr.width, intr.height, ", ".join("%.6f" % x for x in k), ", ".join("%.8f" % x for x in c)))
    with open(path, "w") as f:
        f.write(text)
    return path


# ---------------------------------------------------------------------- depth at a landmark
def median_depth(depth, u, v, patch=9, min_valid=10):
    """Median of the non-zero depths in a patch x patch window at pixel (u, v); None with fewer than min_valid
    valid pixels or outside the image. Same unit as `depth`."""
    if depth is None:
        return None
    h, w = depth.shape[:2]
    r = max(0, int(patch) // 2)
    u, v = int(round(float(u))), int(round(float(v)))
    if not (0 <= u < w and 0 <= v < h):
        return None
    win = depth[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
    vals = win[np.isfinite(win) & (win > 0)]
    if vals.size < min_valid:
        return None
    return float(np.median(vals))


def eye_point(depth, intr: Intrinsics, iris_px, corner_px=None, nose_px=None, pnp_mm=None,
              patch=9, min_valid=10, min_z=0.3, max_z=1.5):
    """3D position (mm, camera frame) of one eye and where its depth came from: the depth at the iris pixel, else at
    the eye-corner midpoint, else the nose-bridge depth applied at the iris pixel, else ptgaze's solvePnP eye
    centre (`pnp_mm`), else None. Depth is in metres (RealSense); z outside [min_z, max_z] m counts as missing."""
    def ok(z):
        return z is not None and min_z <= z <= max_z

    for src, px in (("iris", iris_px), ("corner", corner_px), ("nose", nose_px)):
        if px is None:
            continue
        z = median_depth(depth, px[0], px[1], patch, min_valid)
        if ok(z):
            return intr.deproject(iris_px[0], iris_px[1], z * 1000.0), src
    if pnp_mm is not None and np.all(np.isfinite(pnp_mm)):
        return np.asarray(pnp_mm, float), "pnp"
    return None, "none"


# ---------------------------------------------------------------------- screen frame
def _rx(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class ScreenPose:
    """The camera's pose in the screen frame: `cam_pos` (mm; the optical centre) and `tilt_deg` (> 0 = the camera
    looks up). `auto` position = horizontally centred, 45 mm below the active area, 20 mm in front of the panel."""

    BELOW_MM, FRONT_MM = 45.0, 20.0

    def __init__(self, screen_mm, cam_pos=None, tilt_deg=25.0):
        self.screen_mm = (float(screen_mm[0]), float(screen_mm[1]))
        if cam_pos is None:
            cam_pos = (self.screen_mm[0] / 2.0, self.screen_mm[1] + self.BELOW_MM, -self.FRONT_MM)
        self.cam_pos = np.asarray(cam_pos, float).reshape(3)
        self.tilt_deg = float(tilt_deg)
        self.R = _rx(-self.tilt_deg) @ np.diag([-1.0, 1.0, -1.0])

    def to_screen(self, p_cam):
        return self.R @ np.asarray(p_cam, float) + self.cam_pos

    def rotate(self, d_cam):
        return self.R @ np.asarray(d_cam, float)

    def describe(self):
        return "cam at (%.0f, %.0f, %.0f) mm, tilt %+.0f deg" % (*self.cam_pos, self.tilt_deg)


def ray_plane(origin_s, dir_s):
    """Where the ray (screen frame) meets the panel plane z = 0: (sx, sy, t) with t the distance along the unit
    direction, or None when the ray runs away from or parallel to the plane."""
    o, d = np.asarray(origin_s, float), np.asarray(dir_s, float)
    if abs(d[2]) < 1e-9:
        return None
    t = -o[2] / d[2]
    if t <= 0:
        return None
    hit = o + t * d
    return float(hit[0]), float(hit[1]), float(t)


def gaze_hit(origin_cam, dir_cam, pose: ScreenPose):
    """Camera-frame origin (mm) and unit direction -> (sx, sy, dist_mm) on the screen, or None."""
    n = np.linalg.norm(dir_cam)
    if not np.isfinite(n) or n < 1e-9:
        return None
    return ray_plane(pose.to_screen(origin_cam), pose.rotate(np.asarray(dir_cam, float) / n))


def yaw_pitch_deg(d):
    """ptgaze's angles of a camera-frame direction: pitch = asin(-y) (up > 0), yaw = atan2(-x, -z) (> 0 toward the
    image's left = the user's right); (0, 0) = straight at the camera."""
    d = np.asarray(d, float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return 0.0, 0.0
    x, y, z = d / n
    return math.degrees(math.atan2(-x, -z)) + 0.0, math.degrees(math.asin(max(-1.0, min(1.0, -y)))) + 0.0   # + 0.0: no -0.0


# ---------------------------------------------------------------------- default mapper
class PlaneMapper:
    """(sx_mm, sy_mm) on the panel -> pixels by the screen size; the mapper used before the task's calibration."""

    def __init__(self, screen_px, screen_mm):
        self.screen_px = (int(screen_px[0]), int(screen_px[1]))
        self.screen_mm = (float(screen_mm[0]), float(screen_mm[1]))
        self.report = None

    @property
    def description(self) -> str:
        return "plane hit in mm x (%d/%.0f, %d/%.0f) px/mm" % (self.screen_px[0], self.screen_mm[0],
                                                             self.screen_px[1], self.screen_mm[1])

    def predict(self, vec):
        v = np.asarray(vec, float)
        return (float(v[0] * self.screen_px[0] / self.screen_mm[0]), float(v[1] * self.screen_px[1] / self.screen_mm[1]))


def parse_pair(text, sep="x"):
    """'598x337' -> (598.0, 337.0); accepts X and ','."""
    t = str(text).lower().replace(",", sep)
    a, b = t.split(sep)
    return float(a), float(b)


def parse_cam_pos(text):
    """'auto' -> None, else 'x,y,z' mm -> (x, y, z)."""
    if text is None or str(text).strip().lower() == "auto":
        return None
    parts = [float(p) for p in str(text).replace("x", ",").split(",")]
    if len(parts) != 3:
        raise ValueError("--cam-pos wants x,y,z in mm or auto: %r" % text)
    return tuple(parts)
