"""The GAZE3D model: MediaPipe FaceLandmarker (irises) + ptgaze (gaze direction in the camera frame) + RealSense depth
at the irises -> the gaze ray -> its hit on the monitor plane (mm). The fusion `fuse()` is pure (NumPy) so the tests
drive it with synthetic landmarks; `Gaze3D` wraps the two networks and fits gt.tracker.GazeTracker as the extractor
(process(frame) -> Sample, close()).

Left / right are the subject's sides: 468 = the subject's right eye (on the image's left), 473 = the left eye;
eye index 0 = left, 1 = right everywhere here (ptgaze: face.leye / face.reye in the same order).
"""
from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .geometry import (Intrinsics, ScreenPose, eye_point, gaze_hit, parse_cam_pos, parse_pair, write_camera_yaml,
                       yaw_pitch_deg)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(HERE, "models", "face_landmarker.task")
WEIGHTS_DIR = os.path.join(HERE, "models", "hf")
CALIB_DIR = os.path.join(HERE, "calib")
MODES = {"eth-xgaze": ("eth-xgaze.yaml", "hysts/ptgaze-eth-xgaze-resnet18"),
         "mpiifacegaze": ("mpiifacegaze.yaml", "hysts/ptgaze-mpiifacegaze-resnet-simple"),
         "mpiigaze": ("mpiigaze.yaml", "hysts/ptgaze-mpiigaze-resnet-preact")}
RIGHT_EYE = dict(iris=468, ring=(469, 470, 471, 472), a=33, b=133, lid_up=159, lid_low=145, ear=((160, 144), (158, 153)))
LEFT_EYE = dict(iris=473, ring=(474, 475, 476, 477), a=362, b=263, lid_up=386, lid_low=374, ear=((385, 380), (387, 373)))
EYES = (LEFT_EYE, RIGHT_EYE)
EYE_LANDMARKS = tuple(eye[k] for eye in EYES for k in ("a", "b", "lid_up", "lid_low"))
NOSE_BRIDGE = 168
N_MESH = 468                 # ptgaze's 3D face template covers the mesh without the 10 iris points
MAX_EAR = 0.4                # a frontal eye opens to ~0.3; more = a mesh fitted to a profile view
FACE_FORWARD = np.array([0.0, 0.0, -1.0])   # ptgaze's template looks along -z; identity rotation = at the camera


def eye_aspect_ratio(px, eye) -> float:
    (u1, l1), (u2, l2) = eye["ear"]
    width = np.linalg.norm(px[eye["a"]] - px[eye["b"]])
    if width < 1e-6:
        return 0.0
    return float((np.linalg.norm(px[u1] - px[l1]) + np.linalg.norm(px[u2] - px[l2])) / (2.0 * width))


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


# ---------------------------------------------------------------------- sample + fusion
@dataclass
class Sample:
    valid: bool                 # face, direction, both eyes placed, ray hits the panel, no blink
    face_found: bool
    blink: bool
    timestamp: float
    landmarks_px: np.ndarray | None = None  # (478, 2)
    eye_points: np.ndarray | None = None    # (8, 2) corners + lids, for the eye crop
    iris_px: np.ndarray | None = None       # (2, 2) [left, right]
    eye_mm: list | None = None              # [left xyz, right xyz] mm camera frame (entries may be None)
    depth_src: tuple = ("none", "none")     # iris | corner | nose | pnp | none, per eye
    origin_mm: np.ndarray | None = None     # ray origin: mean of the placed eyes
    direction: np.ndarray | None = None     # unit gaze direction, camera frame (ptgaze)
    head_rot: np.ndarray | None = None      # (3, 3) head rotation (ptgaze solvePnP)
    head_mm: np.ndarray | None = None       # head position from solvePnP (generic face scale)
    sx: float | None = None                 # the hit on the panel, mm, screen frame
    sy: float | None = None
    dist_mm: float | None = None            # eye -> panel along the ray
    ear: float = 0.0
    model_ms: float = 0.0

    def as_vector(self):
        return np.array([self.sx, self.sy], float) if self.valid else None

    def head_yaw_pitch(self):
        if self.head_rot is None:
            return None
        return yaw_pitch_deg(self.head_rot @ FACE_FORWARD)


def fuse(points_px, gaze_vec, depth, intr: Intrinsics, pose: ScreenPose, t, pnp_eyes_mm=None, head_rot=None,
         head_mm=None, blink_ear=0.14, patch=9, min_valid=10, min_z=0.3, max_z=1.5) -> Sample:
    """Landmarks (478, 2) px (None = no face), the ptgaze direction (camera frame, None = not run), the aligned depth
    in metres (None = no depth camera) and the pnp eye centres [left, right] mm (the last-resort origin) -> Sample.
    The hit is computed whenever possible; `valid` additionally needs open eyes."""
    if points_px is None:
        return Sample(False, False, False, t)
    P = np.asarray(points_px, float)[:, :2]
    ear = float(np.mean([eye_aspect_ratio(P, e) for e in EYES]))
    if ear > MAX_EAR:
        return Sample(False, False, False, t, ear=ear)
    blink = ear < blink_ear
    iris = P[[LEFT_EYE["iris"], RIGHT_EYE["iris"]]]
    s = Sample(False, True, blink, t, landmarks_px=P, eye_points=P[list(EYE_LANDMARKS)], iris_px=iris, ear=ear,
               head_rot=head_rot, head_mm=head_mm)
    eyes, srcs = [], []
    for i, eye in enumerate(EYES):
        corner = (P[eye["a"]] + P[eye["b"]]) / 2.0
        pnp = pnp_eyes_mm[i] if pnp_eyes_mm is not None else None
        p, src = eye_point(depth, intr, iris[i], corner, P[NOSE_BRIDGE], pnp, patch, min_valid, min_z, max_z)
        eyes.append(p)
        srcs.append(src)
    s.eye_mm, s.depth_src = eyes, tuple(srcs)
    placed = [p for p in eyes if p is not None]
    if not placed:
        return s
    s.origin_mm = np.mean(placed, axis=0)
    if gaze_vec is None:
        return s
    d = np.asarray(gaze_vec, float)
    if not np.all(np.isfinite(d)) or np.linalg.norm(d) < 1e-9:
        return s
    s.direction = _unit(d)
    hit = gaze_hit(s.origin_mm, s.direction, pose)
    if hit is None:
        return s
    s.sx, s.sy, s.dist_mm = hit
    s.valid = not blink
    return s


# ---------------------------------------------------------------------- the networks
def resolve_device(device="auto"):
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def checkpoint_path(mode, weights_dir=WEIGHTS_DIR):
    """ptgaze's safetensors for the mode, from the local cache first, else downloaded from the Hugging Face Hub."""
    from huggingface_hub import hf_hub_download
    repo = MODES[mode][1]
    try:
        return hf_hub_download(repo, "model.safetensors", cache_dir=weights_dir, local_files_only=True)
    except Exception:
        return hf_hub_download(repo, "model.safetensors", cache_dir=weights_dir)


def build_estimator(mode, device, camera_yaml, model_path=MODEL_PATH, weights_dir=WEIGHTS_DIR):
    """ptgaze GazeEstimator from the package's own mode config, with our camera file, weights and device."""
    import ptgaze
    from omegaconf import OmegaConf
    from ptgaze.config_schema import AppConfig
    from ptgaze.gaze_estimator import GazeEstimator
    if mode not in MODES:
        raise SystemExit("unknown --mode %r (one of %s)" % (mode, ", ".join(MODES)))
    pkg = os.path.dirname(os.path.abspath(ptgaze.__file__))
    cfg = OmegaConf.merge(OmegaConf.structured(AppConfig), OmegaConf.load(os.path.join(pkg, "data", "configs", MODES[mode][0])))
    cfg.PACKAGE_ROOT = pkg
    cfg.device = device
    cfg.face_detector.mediapipe_model_path = model_path      # ptgaze builds (and we never use) its own detector
    cfg.face_detector.mediapipe_max_num_faces = 1
    cfg.gaze_estimator.camera_params = camera_yaml
    cfg.gaze_estimator.checkpoint = checkpoint_path(mode, weights_dir)
    return GazeEstimator(cfg)


def camera_yaml_for(intr: Intrinsics, calib_dir=CALIB_DIR):
    os.makedirs(calib_dir, exist_ok=True)
    return write_camera_yaml(os.path.join(calib_dir, "camera_%dx%d.yaml" % (intr.width, intr.height)), intr)


class Gaze3D:
    """FaceLandmarker (Tasks API, VIDEO mode) + ptgaze + fuse(): process(frame) -> Sample. `intr` is the colour
    camera; frames may carry their own `.intrinsics` (used when present) and `.depth` (metres, aligned)."""

    def __init__(self, intr: Intrinsics, pose: ScreenPose, mode="eth-xgaze", device="auto", model_path=MODEL_PATH,
                 weights_dir=WEIGHTS_DIR, calib_dir=CALIB_DIR, blink_ear=0.14, depth_patch=9, min_z=0.3, max_z=1.5,
                 filter_n=1, min_tracking_confidence=0.5):
        if not os.path.isfile(model_path):
            raise SystemExit("face model missing: %s (bash stack.sh setup downloads it)" % model_path)
        self.intr, self.pose, self.mode = intr, pose, mode
        self.blink_ear, self.patch, self.min_z, self.max_z = float(blink_ear), int(depth_patch), float(min_z), float(max_z)
        self.device = resolve_device(device)
        self._dirs = deque(maxlen=max(1, int(filter_n)))
        try:
            import mediapipe as mp
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1,
                min_tracking_confidence=min_tracking_confidence)
            self._lm = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        except (ImportError, AttributeError) as e:
            raise SystemExit("MediaPipe Tasks API not found (%r): written against mediapipe==1.1.0 (bash stack.sh setup)" % (e,))
        self._mp = mp
        self.camera_yaml = camera_yaml_for(intr, calib_dir)
        self.estimator = build_estimator(mode, self.device, self.camera_yaml, model_path, weights_dir)
        self.input_is_face = mode != "mpiigaze"
        from ptgaze.common import Face
        self._Face = Face
        self._last_ms = -1
        self.last_ms = 0.0
        self.landmark_ms = 0.0

    def close(self):
        if self._lm is not None:
            self._lm.close()
            self._lm = None
        est = getattr(self, "estimator", None)
        if est is not None:
            try:
                est.close()
            except Exception:
                pass
            self.estimator = None

    def _landmarks(self, bgr, timestamp):
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(bgr[:, :, ::-1]))
        ts = max(int(timestamp * 1000), self._last_ms + 1)      # VIDEO mode wants strictly increasing ms
        self._last_ms = ts
        res = self._lm.detect_for_video(image, ts)
        if not res.face_landmarks:
            return None
        h, w = bgr.shape[:2]
        return np.array([(lm.x * w, lm.y * h) for lm in res.face_landmarks[0]], float)

    def _gaze(self, bgr, pts):
        """ptgaze on our landmarks: (unit direction, head R (3,3), head position mm, [left, right] eye centres mm)."""
        mesh = pts[:N_MESH]
        bbox = np.round(np.vstack([mesh.min(axis=0), mesh.max(axis=0)])).astype(np.int32)
        face = self._Face(bbox, mesh)
        self.estimator.estimate_gaze(bgr, face)
        if self.input_is_face:
            d = face.gaze_vector
        else:
            d = _unit(face.reye.gaze_vector + face.leye.gaze_vector)
        pnp = np.array([face.leye.center, face.reye.center], float) * 1000.0
        return np.asarray(d, float), face.head_pose_rot.as_matrix(), np.asarray(face.head_position, float) * 1000.0, pnp

    def process(self, frame) -> Sample:
        t0 = time.perf_counter()
        bgr = frame.bgr
        pts = self._landmarks(bgr, frame.timestamp)
        self.landmark_ms = (time.perf_counter() - t0) * 1000
        d = R = head = pnp = None
        if pts is not None:
            d, R, head, pnp = self._gaze(bgr, pts)
            self._dirs.append(d)
            if len(self._dirs) > 1:
                d = _unit(np.mean(self._dirs, axis=0))
        else:
            self._dirs.clear()
        s = fuse(pts, d, getattr(frame, "depth", None), getattr(frame, "intrinsics", None) or self.intr, self.pose,
                 frame.timestamp, pnp, R, head, self.blink_ear, self.patch, 10, self.min_z, self.max_z)
        s.model_ms = (time.perf_counter() - t0) * 1000
        return s


# ---------------------------------------------------------------------- drawing + options
def draw_overlay(img, s: Sample, origin=(0, 0), full=False, intr: Intrinsics | None = None, ray_mm=120.0):
    """Marks on a BGR image whose top-left pixel is `origin` in the frame: iris dots (subject's left blue, right
    green) with the depth source letter, and, with intrinsics, the gaze ray projected from the eye midpoint; with
    `full` also the landmarks and the hit text. In place; returns img."""
    import cv2
    if s is None or not s.face_found:
        return img
    ox, oy = origin
    P = lambda p: (int(round(p[0] - ox)), int(round(p[1] - oy)))
    if full and s.landmarks_px is not None:
        for x, y in s.landmarks_px[:N_MESH]:
            cv2.circle(img, (int(x - ox), int(y - oy)), 0, (200, 200, 200), -1)
    for i, colour in enumerate(((255, 60, 60), (60, 255, 60))):
        p = s.iris_px[i]
        cv2.circle(img, P(p), 3, colour, -1)
        cv2.circle(img, P(p), 9, colour, 1)
        z = s.eye_mm[i][2] if s.eye_mm and s.eye_mm[i] is not None else None
        label = s.depth_src[i][0] + ("" if z is None else " %.0f" % z)
        cv2.putText(img, label, (P(p)[0] + 10, P(p)[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    if intr is not None and s.origin_mm is not None and s.direction is not None:
        a = intr.project(s.origin_mm)
        b = intr.project(s.origin_mm + s.direction * ray_mm)
        if a is not None and b is not None:
            cv2.line(img, P(a), P(b), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, P(a), 4, (0, 255, 255), -1)
    if full and s.sx is not None:
        cv2.putText(img, "hit %+.0f, %+.0f mm  d %.0f" % (s.sx, s.sy, s.dist_mm), (10 - ox + 0, 30 - oy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return img


def add_model_args(ap):
    ap.add_argument("--mode", default="eth-xgaze", choices=sorted(MODES), help="ptgaze model (default eth-xgaze)")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--model", default=MODEL_PATH, help="MediaPipe face_landmarker.task")
    ap.add_argument("--weights-dir", default=WEIGHTS_DIR, help="Hugging Face cache for the ptgaze weights")
    ap.add_argument("--blink-ear", type=float, default=0.14, help="eye aspect ratio below which a frame is a blink")
    ap.add_argument("--depth-patch", type=int, default=9, help="px window for the median depth at a landmark")
    ap.add_argument("--min-z", type=float, default=0.3, help="metres: depth below this is treated as missing")
    ap.add_argument("--max-z", type=float, default=1.5, help="metres: depth above this is treated as missing")
    ap.add_argument("--filter", type=int, default=1, help="frames averaged on the direction (the task smooths anyway)")
    ap.add_argument("--min-tracking-confidence", type=float, default=0.5)


def add_pose_args(ap, screen=False):
    ap.add_argument("--cam-pos", default="auto", help="camera optical centre in the screen frame, x,y,z mm (auto = "
                                                      "centred, 45 mm below the active area, 20 mm in front)")
    ap.add_argument("--cam-tilt", type=float, default=25.0, help="degrees the camera looks up (negative = down)")
    if screen:
        ap.add_argument("--screen-mm", default="598x337", help="active area WxH in mm")
        ap.add_argument("--screen-px", default="1920x1080", help="resolution WxH")


def pose_from_args(args) -> ScreenPose:
    return ScreenPose(parse_pair(args.screen_mm), parse_cam_pos(args.cam_pos), args.cam_tilt)


def model_from_args(args, intr: Intrinsics, pose: ScreenPose) -> Gaze3D:
    return Gaze3D(intr, pose, args.mode, args.device, args.model, args.weights_dir, CALIB_DIR, args.blink_ear,
                  args.depth_patch, args.min_z, args.max_z, args.filter, args.min_tracking_confidence)
