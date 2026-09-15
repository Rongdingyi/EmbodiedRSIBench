#!/usr/bin/env bash
# 02_check_deepseek_api.sh -- Phase C / Gate G2.
#
# 1. env checks (key / base_url / model)
# 2. text smoke, 3. vision smoke (real simulator RGB frame)
# 4. pinned-OpenETA backend compatibility smoke (with request-body audit)
# Writes outputs/api_smoke/deepseek_flash.json; non-zero exit on any failure.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "[FAIL] DEEPSEEK_API_KEY is not set" >&2
  exit 1
fi
if [ "${DEEPSEEK_BASE_URL:-}" != "https://api.deepseek.com" ]; then
  echo "[FAIL] DEEPSEEK_BASE_URL must be https://api.deepseek.com (got '${DEEPSEEK_BASE_URL:-}')" >&2
  exit 1
fi
if [ "${DEEPSEEK_MODEL:-}" != "deepseek-flash" ]; then
  echo "[FAIL] DEEPSEEK_MODEL must be deepseek-flash (got '${DEEPSEEK_MODEL:-}')" >&2
  exit 1
fi
echo "[ok] env: base_url=${DEEPSEEK_BASE_URL} model=${DEEPSEEK_MODEL}"

IMAGE="$PROJECT/outputs/api_smoke/sample_rgb.png"
mkdir -p "$PROJECT/outputs/api_smoke"
if [ ! -f "$IMAGE" ]; then
  echo "[info] rendering a real simulator RGB frame for the vision smoke..."
  /data/users/rongdingyi/miniconda3/envs/RoboTwin/bin/python - "$IMAGE" <<'PY'
import sys
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from PIL import Image
c = Controller(scene="FloorPlan1", platform=CloudRendering, width=480, height=360, quality="Medium")
ev = c.step("Pass")
Image.fromarray(ev.frame).save(sys.argv[1])
c.stop()
print("saved", sys.argv[1])
PY
fi

export API_SMOKE_IMAGE="$IMAGE"
"$PROJECT/external/OpenETA/.venv/bin/python" "$PROJECT/benchmark/openeta_bridge/api_smoke.py"
EXIT=$?
if [ $EXIT -ne 0 ]; then
  echo "[FAIL] G2 Model Gate failed; see outputs/api_smoke/deepseek_flash.json" >&2
fi
exit $EXIT
