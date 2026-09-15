#!/usr/bin/env bash
# Pick a working X display for the AI2-THOR 2.1.0 (EB-ALFRED) renderer.
#
# The 2018 Unity build presents through the NVIDIA Vulkan driver, which needs the
# machine's real X server (the physical console session, usually :0 or :1).
# A private Xvfb does NOT work: the driver cannot create a swapchain for it and
# the Unity player aborts (SIGABRT).
#
# Usage:  export DISPLAY="$(bash 00_pick_display.sh)"
set -euo pipefail
if [ -n "${DISPLAY:-}" ] && timeout 5 xdpyinfo >/dev/null 2>&1; then
  echo "$DISPLAY"
  exit 0
fi
for d in :0 :1 :2; do
  if DISPLAY="$d" timeout 5 xdpyinfo >/dev/null 2>&1; then
    echo "$d"
    exit 0
  fi
done
# last resort: report the sockets that exist
echo "no usable X display found; active displays:" >&2
ls /tmp/.X11-unix/ 2>/dev/null >&2 || true
exit 1
