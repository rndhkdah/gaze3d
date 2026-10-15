#!/usr/bin/env bash
# Stop any running task module (SIGTERM, then SIGKILL after 2 s).
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/_env.sh"
pids=$(gaze_pids)
if [ -z "$pids" ]; then
  echo "[kill] no task module running"
  exit 0
fi
echo "[kill] SIGTERM $(echo $pids)"
kill $pids 2>/dev/null
sleep 2
left=$(gaze_pids)
[ -n "$left" ] && { echo "[kill] SIGKILL $(echo $left)"; kill -9 $left 2>/dev/null; }
exit 0
