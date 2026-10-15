# GAZE3D: 3D gaze on the monitor from the RealSense D435i

Where on the screen the user looks, in millimetres: RealSense depth at the MediaPipe iris landmarks gives each eye's
3D position, ptgaze (ETH-XGaze, MIT) gives the gaze direction in the camera frame, the ray is intersected with the
monitor plane, and the six-squares task's 9-dot calibration refines the result. The camera sits under the monitor
looking up. Design: `docs/2026-10-15-gaze3d-realsense-monitor-design.md`. This folder comes from the VR_CAM
workspace, where sibling folders hold the other trackers measured on the same task (GazeTracking, the JEOresearch
EyeTracker ports, the MediaPipe + depth LEGACY pipeline); it runs on its own.

```
gaze3d/geometry.py   intrinsics + deprojection, depth at a landmark (iris > corner > nose > pnp), the screen frame
                     (camera position + tilt), ray-plane hit, PlaneMapper (mm -> px)           pure NumPy, tested
gaze3d/capture.py    RealSense colour + depth aligned to colour (+ intrinsics, --record/--bag), webcam fallback, still image
gaze3d/model.py      fuse() (pure: landmarks + direction + depth -> Sample), Gaze3D (FaceLandmarker + ptgaze), overlay
gaze3d/tracker.py    the capture thread (camera + model on their own thread, latest sample for pygame)
gaze3d/taskadapter.py the six-squares tracker contract for gaze3d: features (sx_mm, sy_mm), log fields, describe, overlay
run_gaze3d.py        preview / headless smoke test (--image, --frames N --snapshot, --bag, --csv)
taskenv/             this folder's own copy of the six-squares task (api, squares, calibrate, dwell, smoothing, display)
trackers/            its plug-ins: gaze3d (re-exports gaze3d.taskadapter) and synthetic (no camera, screen checks)
stack.sh _env.sh kill.sh   setup | check | preview | image | squares | calibrate | status | kill
calib/               calibration_gaze3d.json, squares_<time>.csv, samples_<time>.csv, camera_<WxH>.yaml, recordings
../TASK_TEMP/trackers/gaze3d.py   the same adapter on the shared task (TRACKER=gaze3d there), for row-by-row comparison
```

## Run

```bash
bash stack.sh setup                       # .venv: torch (CUDA), mediapipe 1.1.0, pyrealsense2, ptgaze; models + weights
bash stack.sh check                       # 75 tests: geometry, fusion, capture, thread, the task and the contract
bash stack.sh image some_face.jpg         # wiring check without a camera -> calib/image_check.png
bash stack.sh preview                     # live overlay: irises with depth source, gaze ray, hit in mm / px; q quits
bash stack.sh preview --record s.bag      # ... and record;  bash stack.sh preview --bag s.bag --csv hits.csv replays
bash stack.sh squares                     # the six-squares task here: C = 9-dot calibration, T = 12 trials -> calib/
bash stack.sh squares --record calib/s.bag --screen-record calib/s.avi   # ... recording the camera and the task window
TRACKER=synthetic bash stack.sh squares --snapshot out.png --seconds 3   # headless screen check without a camera
```

The task is the same as TASK_TEMP's (copied, not imported: same screens, dwell, One Euro smoothing, 9/13-dot ridge
affine calibration, `squares_<time>.csv` and `samples_<time>.csv` columns), so results compare row for row with the
other trackers; `TRACKER=gaze3d bash ../TASK_TEMP/stack.sh squares` runs the same tracker on the shared copy.
`--screen-record file.avi` (MJPG; `.mp4` = mp4v) writes the task window at up to `--screen-fps 30` with a
`file.avi.times.csv` (frame, t) sidecar, so the video lines up with the samples CSV and a camera `--record s.bag`
(both on the time.time() clock). Both options exist in TASK_TEMP's task too.

External screen recorders (GNOME's built-in screencast, OBS) see the task window too: `open_fullscreen` marks the
window "never bypass the compositor" (`_NET_WM_BYPASS_COMPOSITOR = 2` via xprop) so Mutter does not unredirect the
fullscreen window after its first 100 full-area frames, which otherwise records as black on X11 + GNOME.

Camera pose in the screen frame (origin top-left of the active area, x right, y down, z into the screen):
`--cam-pos x,y,z` mm (default auto = centred, 45 mm below the active area, 20 mm in front) and `--cam-tilt` degrees
looking up (default 25). Rough values are fine: the calibration absorbs a constant offset and scale; what the
geometry buys is that head movement no longer shifts the gaze point. `--mode mpiifacegaze` / `mpiigaze` switch the
ptgaze model, `--device cpu` avoids the GPU, `--blink-ear 0.14` is the blink threshold from under the monitor.

Each eye's depth is the median over a 9 px window at the iris pixel; a hole there (lashes, specular cornea) falls
back to the eye-corner midpoint, then the nose bridge, then ptgaze's solvePnP eye centre (generic face scale),
logged per sample as `src_l` / `src_r` in the task's `samples_*.csv`. The emitter stays on (depth on skin needs it;
the colour sensor has an IR-cut filter).

## Status (2026-10-15)

Written and tested without the camera (it was unplugged): the tests, the end-to-end wiring on a still face image
(FaceLandmarker + ptgaze on CUDA, hit near the camera footprint for a face looking at the camera, ~12 ms per frame
on the GPU), and the task headless with the laptop webcam (black frames, so NO FACE) including the screen
recording. Not yet done: a live run, a calibration and the 12 trials on the monitor.
