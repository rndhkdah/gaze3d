#!/usr/bin/env bash
usage() {
  cat <<'USAGE'
GAZE3D: 3D gaze on the monitor from the RealSense D435i (depth at the irises + ptgaze direction + monitor plane),
with its own copy of the six-squares task (taskenv/, same screens, dwell, trials, calibration and CSVs as the others).

usage: [TRACKER=gaze3d|synthetic] bash stack.sh <command> [flags]

  setup               .venv: torch (CUDA wheels), mediapipe, pyrealsense2, ptgaze (--no-deps) ...; face model; ETH-XGaze weights
  check               tests: geometry, fusion, capture, thread, the task (dwell, smoothing, calibration, squares) and the contract
  preview [flags]     OpenCV window: landmarks, eye depths, gaze ray, hit in mm/px (run_gaze3d.py; q quits)
  image <face.jpg>    wiring check on a still image, no camera: writes calib/image_check.png
  squares [flags]     the six-squares task here: C = 9-dot calibration, T = 12 trials -> calib/squares_<time>.csv (+ samples)
  calibrate [flags]   standalone 9/13-dot calibration -> calib/calibration_gaze3d.json
  status              venv, tracker, calibration file, screen, who holds the camera, running task modules
  kill                stop this folder's running task module

Flags go to the module and the tracker: --camera rs, --size 1280x720, --cam-tilt 25, --cam-pos 299,382,-20,
--mode eth-xgaze, --device cpu, --record s.bag, --bag s.bag, --blink-ear 0.14, --snapshot x.png --seconds 5 ...
USAGE
  exit "${1:-1}"
}
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/_env.sh"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

need_venv() { [ -x "$VENV_PY" ] || { echo "venv python not found: $VENV_PY (run: bash stack.sh setup)"; exit 1; }; }

setup() {
  [ -x "$VENV_PY" ] || /usr/bin/python3 -m venv "$HERE/.venv"
  "$VENV_PY" -m pip install -q --upgrade pip
  "$VENV_PY" -c "import torch" 2>/dev/null || "$VENV_PY" -m pip install -q torch torchvision
  "$VENV_PY" -m pip install -q -r "$HERE/requirements.txt"
  "$VENV_PY" -m pip install -q --no-deps ptgaze==0.3.0
  mkdir -p "$HERE/models" "$HERE/calib"
  if [ ! -s "$HERE/models/face_landmarker.task" ]; then
    if [ -s "$HERE/../EYETRACKER/models/face_landmarker.task" ]; then cp "$HERE/../EYETRACKER/models/face_landmarker.task" "$HERE/models/"
    else curl -fsSL -o "$HERE/models/face_landmarker.task" "$MODEL_URL"; fi
  fi
  (cd "$HERE" && "$VENV_PY" -c "
import torch, mediapipe, pyrealsense2, cv2, ptgaze, pygame, taskenv.api
from gaze3d.model import checkpoint_path
print('[setup] torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| mediapipe', mediapipe.__version__, '| cv2', cv2.__version__, '| pygame', pygame.__version__)
print('[setup] weights', checkpoint_path('eth-xgaze'))
") || { echo "[setup] import probe FAILED"; exit 1; }
  echo "[setup] done: $VENV_PY"
}

check() {
  need_venv
  (cd "$HERE" && "$VENV_PY" -m pytest -q -p no:cacheprovider) || exit 1
}

status() {
  printf "  %-14s %s\n" "venv" "$([ -x "$VENV_PY" ] && echo "$VENV_PY" || echo "MISSING (bash stack.sh setup)")"
  printf "  %-14s %s\n" "tracker" "$TRACKER"
  printf "  %-14s %s\n" "calibration" "$([ -s "$GAZE_CALIB" ] && echo "$GAZE_CALIB" || echo "none: the tracker's default mapping (key C)")"
  printf "  %-14s %s\n" "screen" "$SCREEN_PX $SCREEN_MM mm, viewing $VIEWING_MM mm"
  local holders; holders=$(fuser /dev/video* 2>/dev/null | tr -s ' ' '\n' | sort -u | tr '\n' ' ')
  printf "  %-14s %s\n" "camera held by" "${holders:-nobody}$( [ -n "$holders" ] && ps -o comm=,args= -p $holders 2>/dev/null | head -3 | cut -c1-100 | sed 's/^/ : /' | tr '\n' ' ')"
  local pids; pids=$(gaze_pids)
  printf "  %-14s %s\n" "task modules" "${pids:+running (pid $(echo $pids))}${pids:-none}"
}

run_module() { need_venv; cd "$HERE" && exec "$VENV_PY" -m "taskenv.$1" "${@:2}"; }

cmd="${1:-}"
[ $# -gt 0 ] && shift
case "$cmd" in
  setup) setup ;;
  check) check ;;
  preview) need_venv; cd "$HERE" && exec "$VENV_PY" run_gaze3d.py "$@" ;;
  image) need_venv; [ -n "${1:-}" ] || usage 1; img="$1"; shift
         cd "$HERE" && exec "$VENV_PY" run_gaze3d.py --image "$img" --snapshot "$HERE/calib/image_check.png" "$@" ;;
  squares) run_module squares --tracker "$TRACKER" --calib "$GAZE_CALIB" --screen-mm "$SCREEN_MM" --screen-px "$SCREEN_PX" --viewing-mm "$VIEWING_MM" "$@" ;;
  calibrate) run_module calibrate --tracker "$TRACKER" --out "$GAZE_CALIB" --screen-mm "$SCREEN_MM" --screen-px "$SCREEN_PX" --viewing-mm "$VIEWING_MM" "$@" ;;
  status) status ;;
  kill) exec bash "$HERE/kill.sh" ;;
  -h | --help | help | "") usage 0 ;;
  *) echo "unknown command: $cmd"; usage 1 ;;
esac
