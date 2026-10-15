# GAZE3D: 3D gaze on the monitor from the RealSense D435i — design (2026-10-15)

## Goal

A tracker that measures *where on the monitor* the user looks, in millimetres, from the D435i on the desk under the
screen: a metric 3D eye position (RealSense depth at the iris landmarks), a 3D gaze direction (an appearance model
in the camera frame), the ray intersected with the monitor plane, and the task's multi-point calibration on top.
It plugs into the six-squares task (`TASK_TEMP`, `TRACKER=gaze3d`) so its hit rate compares row for row with
GazeTracking (58 %) and the EyeTracker ports.

This is the composite the alternatives survey (`reports/Open source 3D eye tracker alternatives.md`) recommended
for the remote setup: ptgaze (MIT) for the direction, MediaPipe (Apache) irises + RealSense depth for the origin,
a ray–plane intersection with the measured monitor, a 9-point regression to absorb kappa and model bias. The
near-eye material (pye3d, 3DTracker) does not apply to a desk camera.

## Why the earlier trackers plateau, and what this one changes

- GazeTracking (New) and the EyeTracker ports have **no head-pose or head-position term**: any head translation
  shifts their features, and their vertical range is flat (rows overlap).
- LEGACY (MediaPipe + depth) regressed 15 raw features to pixels with no gaze geometry; its 9.8° LOO came from the
  mapping, and both eyes got one skin depth from the nose bridge.
- Here the geometry does the work: the ray from the *measured* eye position in the *measured* direction hits the
  plane at a point that already moves correctly with the head. The calibration only has to fix a near-constant
  offset and scale (kappa, the model's degree-level bias, a wrong camera tilt), so a 2D affine on 9 dots suffices.

## Decisions

1. **Direction from ptgaze, ETH-XGaze mode** (`--mode eth-xgaze`; `mpiifacegaze` and `mpiigaze` selectable). ETH-XGaze
   was trained on 18 cameras around the subject, so a camera *below* the screen looking up is inside its training
   pose range; MPIIFaceGaze was laptop webcams above the screen. ResNet-18 on a 224×224 normalised face. Device
   `--device auto` (CUDA when `torch.cuda.is_available()`, else CPU). ptgaze's gaze vector is in the OpenCV camera
   frame (x right, y down, z forward); looking straight at the camera is (0, 0, −1).
2. **One FaceLandmarker, ours.** MediaPipe Tasks `FaceLandmarker` (VIDEO mode, `models/face_landmarker.task`, the file
   the other folders use) gives 478 points with irises 468/473; ptgaze's own detector drops the irises, so we hand
   ptgaze a `Face(bbox, points[:468])` and call `estimate_gaze(image, face)` ourselves. ptgaze still builds an unused
   detector of its own at start (its `GazeEstimator.__init__`); it costs memory only.
3. **Camera intrinsics are real.** The colour stream's `fx, fy, ppx, ppy, coeffs` from pyrealsense2 are written to
   `calib/camera_<WxH>.yaml` in ptgaze's format and loaded by its `Camera.from_file`; head pose (solvePnP on
   ptgaze's metric MediaPipe face template) and the normalisation warp use them. A webcam (no RealSense) gets the
   dummy camera (focal = width) and no depth.
4. **Eye position from depth at the iris pixel.** Depth is aligned to colour (`rs.align`), float32 metres. For each
   eye: median of the non-zero depth in a `--depth-patch 9` px window at the iris pixel (≥ 10 valid), else at the
   eye-corner midpoint, else the nose-bridge depth (landmark 168) applied at the iris pixel, else ptgaze's
   solvePnP eye centre (`face.reye.center`, metres, generic face scale). The source is logged (`depth_src` =
   iris / corner / nose / pnp). The pixel is deprojected with `rs2_deproject_pixel_to_point` (distortion-aware);
   depth outside `--min-z 0.3` .. `--max-z 1.5` m counts as missing. The gaze origin is the midpoint of the two eye
   points: moving the origin along the ray does not move the intersection, so the 12 mm cornea–eyeball-centre
   offset is irrelevant here.
5. **Screen frame and camera pose are explicit.** Screen frame: origin at the top-left of the active area, x right
   (as the user sees it), y down, z *into* the screen (right-handed; the user sits at negative z). Camera pose
   in that frame: `--cam-pos X,Y,Z` mm (default `auto` = horizontally centred, 45 mm below the bottom edge, 20 mm in
   front of the panel: `(W/2, H+45, −20)`) and `--cam-tilt` degrees, positive = camera looking up (default 25; LEGACY's
   median head pitch of −28° came from this placement). `R_cs = Rx(−tilt) · diag(−1, 1, −1)`: camera x is the
   screen's −x (a face camera mirrors the user's right), camera y is down, camera z points at the user.
   `p_screen = R_cs p_cam + cam_pos`. Both numbers are rough by design: the calibration absorbs them.
6. **Features = the intersection in mm.** `FEATURE_NAMES = ("sx_mm", "sy_mm")`, the ray–plane hit on z = 0, valid only
   when the ray goes toward the plane (t > 0). The default mapper (`PlaneMapper`) scales mm to pixels with the
   screen size from `--screen-mm`; the task's 9-dot ridge affine then refines it (ridge over standardised features,
   unit-free). Everything else goes to `LOG_FIELDS`: eye L/R xyz (mm), depth sources, gaze yaw/pitch (deg, camera
   frame), head yaw/pitch (deg), eye–screen distance along the ray, EAR, the model time.
7. **Blink from the eye aspect ratio** (`--blink-ear 0.14`, the under-monitor threshold found for the MediaPipe port):
   blink samples are invalid (dwell holds). A mesh with EAR > 0.4 (profile view) counts as no face, as in webcam3d.
8. **No smoothing in the tracker**: the task's One Euro filter does it; `--filter N` averages the direction over N
   frames if ever needed (default 1).
9. **Record and replay.** `--record file.bag` writes the RealSense stream while tracking; `--bag file.bag` replays it
   (no camera needed), so a session can be re-run offline after a change. `--image face.jpg` runs the model once on a
   picture (dummy camera, pnp origin) for a wiring check without any camera.
10. **Self-contained folder, task adapter in TASK_TEMP.** `GAZE3D/` has its own venv (torch 2.14.1+cu130, ptgaze
    0.3.0 with `--no-deps`, mediapipe 1.1.0, pyrealsense2, pygame), `stack.sh`, tests, models. The adapter
    `TASK_TEMP/trackers/gaze3d.py` reuses `gt.tracker.GazeTracker` (New) for the capture thread, as the EyeTracker
    adapters do; `TASK_TEMP/_env.sh` maps `TRACKER=gaze3d` to `GAZE3D/.venv` and puts `GAZE3D` on `PYTHONPATH`.

## Geometry (pure, tested)

```
depth z at iris pixel (u, v)  ─deproject (intrinsics)─▶  E_l, E_r  (mm, camera frame)
origin O = (E_l + E_r) / 2;  direction d = ptgaze gaze_vector (unit, camera frame)
screen frame:  O' = R_cs O + c,  d' = R_cs d;   plane z = 0:  t = −O'_z / d'_z  (valid if d'_z > 0 and t > 0)
hit (sx, sy) = (O' + t d')_xy  mm;   default pixels = (sx / W_mm · W_px, sy / H_mm · H_px)
```

Sanity cases the tests pin down: a camera tilted up sees a point on its axis *above* itself on the screen; a gaze
from the eye straight at the camera hits the plane at the camera's footprint; moving the head sideways while the
gaze direction is fixed moves the hit by the same amount; `det R_cs = +1`.

## Layout

```
GAZE3D/gaze3d/capture.py     Frame(bgr, depth, timestamp, seq, t_host, intrinsics); RealSenseDepth (colour + aligned depth,
                             emitter on, --record/--bag, USB 2 fallback), Webcam (colour only), open_capture(), write_camera_yaml()
GAZE3D/gaze3d/geometry.py    Intrinsics, deproject(), median_depth(), eye_point(), ScreenPose (cam pos + tilt -> R, t,
                             to_screen()), ray_plane(), PlaneMapper(screen_px, screen_mm)
GAZE3D/gaze3d/model.py       Sample; fuse() (pure: landmarks + gaze vector + depth -> Sample); Gaze3D (FaceLandmarker +
                             ptgaze GazeEstimator; process(frame) -> Sample; close()); eye_aspect_ratio; draw_overlay()
GAZE3D/run_gaze3d.py         preview: OpenCV window with landmarks, eye points + depth, gaze ray, hit in mm/px; --image,
                             --frames N --snapshot out.png (headless), --bag, --record
GAZE3D/tests/                test_geometry.py, test_model.py (fuse on synthetic landmarks, no nets), test_capture.py
GAZE3D/stack.sh _env.sh      setup | check | preview | squares | calibrate | status | kill
GAZE3D/requirements.txt models/face_landmarker.task models/hf/ (ptgaze weights, downloaded by setup)
TASK_TEMP/trackers/gaze3d.py adapter: NAME "gaze3d", FEATURE_NAMES ("sx_mm", "sy_mm"), LOG_FIELDS, add_args, create,
                             describe / eye_image / extra_images / default_mapper (PlaneMapper), reset (no-op)
TASK_TEMP/_env.sh            gaze3d -> GAZE3D_ROOT venv + PYTHONPATH;  TASK_TEMP/tests/test_api.py: adapter on fake samples
```

## Data flow

RealSenseDepth.read(): colour 1280×720 + depth 848×480 @ 30 → align to colour → Frame(bgr, depth m, intrinsics) →
Gaze3D.process: FaceLandmarker (VIDEO) → 478 px points → EAR, blink / profile gate → ptgaze `estimate_gaze(bgr,
Face(points[:468]))` → gaze_vector, head rot, pnp eye centres → fuse(): depth at irises → E_l, E_r → O, d →
ScreenPose → ray_plane → Sample(valid, sx_mm, sy_mm, …; as_vector() = [sx, sy]; eye_points for the crop) →
gt.tracker thread → TrackerSample(vec, screen = mapper(vec), eye_crop) → task (dwell, trials, CSV).

## Error handling

- `create(args)` opens the camera and loads both models before the fullscreen window, so a missing camera, a
  missing weight file or a CUDA failure exits with a one-line reason. RealSense on USB 2 falls back to 640×480
  with a FALLBACK warning (as the other folders).
- No face / profile mesh → `face_found False`; blink → `valid False`; ray away from the plane → `valid False`
  with the hit `None`; depth missing → pnp origin, logged, still valid.
- Exceptions in the capture thread are stored in `.error` and end the task (gt.tracker behaviour).

## Testing

Pure (`bash stack.sh check`, no camera, no torch): geometry sanity cases above, `median_depth` / `eye_point`
fallback chain, `PlaneMapper`, `fuse()` on synthetic landmarks + a synthetic depth plane + a given direction (valid,
hit position, blink, profile), `write_camera_yaml` round trip through ptgaze's `Camera.from_file` when ptgaze is
importable (skipped otherwise), the adapter's describe / eye_image / default_mapper on fake samples
(`TASK_TEMP/tests/test_api.py`, skipped when `gaze3d` is not importable).

With the models (`run_gaze3d.py --image <face.jpg> --snapshot out.png`): FaceLandmarker finds the face, ptgaze
returns a unit vector, the hit is finite — the end-to-end wiring without hardware.

With the camera (user): `bash stack.sh preview` (face, both eyes with a depth source, the ray on the overlay, the
hit moving with the head), then `bash stack.sh squares`: C (9 dots) then T (12 trials) → `TASK_TEMP/calib/`.

## Out of scope (v1)

Fitting the camera pose from the calibration samples (the raw rays are logged, so it can be done offline later);
per-eye gaze (ETH-XGaze gives one face direction); the eyeball-centre model; a LEGACY-style polynomial map.

## Addendum (later the same day): own task copy and screen recording

The user asked for "the same task domain as the others" and chose a self-contained copy: `GAZE3D/taskenv/` is
TASK_TEMP's task copied verbatim (calibrate's default tracker set to gaze3d), `GAZE3D/trackers/` holds `gaze3d`
and `synthetic`, `gaze3d/tracker.py` is the capture thread copied from New/gt (so GAZE3D no longer imports `gt`),
and the adapter body moved to `gaze3d/taskadapter.py`, which both `GAZE3D/trackers/gaze3d.py` and
`TASK_TEMP/trackers/gaze3d.py` re-export (one source, two tasks). `_env.sh` puts only GAZE3D on PYTHONPATH;
`stack.sh squares|calibrate|status|kill` run the local task (outputs in `GAZE3D/calib/`); `kill.sh` matches only
this folder's venv python. Decision 10 (adapter via gt.tracker) is superseded by this.

Screen recording (asked for in the same breath): `--screen-record file` on `taskenv.squares` in both copies,
`taskenv.display.ScreenRecorder` (cv2.VideoWriter, MJPG for .avi else mp4v, frames taken after each flip at most
every 1/`--screen-fps` s, a `<file>.times.csv` of (frame, t) on the time.time() clock) so the task video, the
samples CSV and the camera `--record .bag` can be lined up afterwards. Tested with a pygame surface and read back
through cv2.VideoCapture; exercised headless in the task on the laptop webcam.
