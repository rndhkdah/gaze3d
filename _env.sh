#!/usr/bin/env bash
# Sourced by stack.sh and kill.sh: paths, the tracker, the screen, the calibration file.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export GAZE3D_ROOT="$REPO"
# The tracker this folder's task runs: gaze3d (default) or synthetic (no camera; screen checks).
export TRACKER=${TRACKER:-gaze3d}
VENV_PY="$REPO/.venv/bin/python"
export VENV_PY
export PYTHONPATH="$REPO"          # this folder only: its own taskenv/ and trackers/
export GAZE_CALIB=${GAZE_CALIB:-"$REPO/calib/calibration_$TRACKER.json"}
# The screen under the camera: the external display, read from xrandr as the first connected output that is not the
# primary (laptop panel). SCREEN_PX picks the display (--display auto); SCREEN_MM feeds the mm plane and the readouts.
external_screen() {
  DISPLAY=${DISPLAY:-:0} xrandr --query 2>/dev/null | awk '
    / connected/ && !/primary/ { if (match($0, /[0-9]+x[0-9]+\+[0-9]+\+[0-9]+/)) px = substr($0, RSTART, RLENGTH);
      sub(/\+.*/, "", px); mm = ""; if (match($0, /[0-9]+mm x [0-9]+mm/)) { mm = substr($0, RSTART, RLENGTH); gsub(/mm| /, "", mm) }
      if (px != "") { print px, mm; exit } }'
}
if [ -z "${SCREEN_PX:-}" ] || [ -z "${SCREEN_MM:-}" ]; then
  read -r _px _mm <<< "$(external_screen)"
  SCREEN_PX=${SCREEN_PX:-${_px:-1920x1080}}
  SCREEN_MM=${SCREEN_MM:-${_mm:-598x337}}
fi
export SCREEN_PX SCREEN_MM
# Eye-screen distance for the degree readouts only.
VIEWING_MM=${VIEWING_MM:-$([ "${SCREEN_MM%%x*}" -gt 1000 ] 2>/dev/null && echo 1500 || echo 600)}
export VIEWING_MM

gaze_pids() {
  # this folder's task modules only: anchored on the python executable and on this folder's path, so the shared
  # task in TASK_TEMP (or a shell that merely mentions the module) is not matched
  pgrep -f "^$REPO/\.venv/bin/python[^ ]* -m taskenv\.(squares|calibrate)( |$)" 2>/dev/null
}
