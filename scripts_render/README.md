# Rendering recipes — Embodied RSI Benchmark Core v1.0

All five sources were verified to render real observations on this machine
(2026-09-15). No data in `release/` stores images: observations are produced
on the fly by the simulators below.

| source | env | simulator | verified output |
|---|---|---|---|
| TVRBench iTHOR | RoboTwin | AI2-THOR 5.0.0 (CloudRendering) | 640×480×3 frame |
| TVRBench ProcTHOR | RoboTwin | AI2-THOR procedural house from `procthor-10k/train.jsonl.gz` | 640×480×3 frame |
| SpatialWorld ai2thor | RoboTwin | AI2-THOR 5.0.0 (CloudRendering) | 640×480×3 frame |
| SpatialWorld ProcTHOR | RoboTwin | AI2-THOR procedural house (`scene_index`) | 640×480×3 frame |
| AsgardBench | RoboTwin | AI2-THOR 5.0.0 (CloudRendering, runtime swap) | 1024×1024×3 frame |
| EB-ALFRED | bench_avp | AI2-THOR 2.1.0 (Vulkan, physical X) | 500×500×3 frame, Ladle restored |
| EB-Habitat | embench | Habitat-Sim 0.3.0 + habitat-lab 0.3.0 | 320×320×3 `head_rgb` |

## Environments (reused, no new ai2thor envs)

**RoboTwin** (`/data/users/rongdingyi/miniconda3/envs/RoboTwin`) — added here:
`ai2thor==5.0.0`, `botocore`, `progressbar2`, `python-xlib`, `prior`, `tiktoken`,
`tenacity`, `pygame`, `bokeh>=3.8.1`, `typer`, `openpyxl`. numpy/torch untouched.
The 797 MB `thor-CloudRendering-<hash>` build is cached in `~/.ai2thor`.
No DISPLAY needed (Vulkan offscreen).

**bench_avp** — added here: `ai2thor==2.1.0`, `numpy==1.23.5`,
`opencv-python==4.10.0.84`, `gym`, `networkx`, `pandas`, `scipy`, `Pillow`,
`flask==1.1.2`, `werkzeug==1.0.1`. Notes:
- Flask/Werkzeug MUST stay at the 2019-era versions; newer Werkzeug closes
  keep-alive connections in a way that breaks the old Unity build's raw-socket
  HTTP client (`SocketException: The socket has been shut down`).
- The old build defaults to software OpenGL; the smoke script patches the Unity
  command with `-force-vulkan` so rendering uses the NVIDIA Vulkan driver.
- Requires the machine's **physical X display** (`:0`/`:1`). A private Xvfb does
  not work: the NVIDIA Vulkan driver cannot create a swapchain for it and the
  player aborts. Use `00_pick_display.sh` to find a usable display.
- The build `thor-201909061227-Linux64` is cached in `~/.ai2thor`.

**embench** (new, python 3.9; the only env that had to be created) — `habitat-sim==0.3.0`
(`withbullet headless`, from the `aihabitat` channel), `habitat-lab==0.3.0`
(cloned to `runtime_assets/habitat-lab`, installed editable), `transformers`,
`torch` (CPU). ReplicaCAD/YCB assets (1.5 GB) were downloaded with
`python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets`
and placed at `sources/embodiedbench/embodiedbench/envs/eb_habitat/data/`
(their internal symlinks were repointed after the move).

## How to run the smoke tests

```bash
# TVRBench / SpatialWorld / AsgardBench (no display needed)
/data/users/rongdingyi/miniconda3/envs/RoboTwin/bin/python tvrbench_smoke.py
/data/users/rongdingyi/miniconda3/envs/RoboTwin/bin/python spatialworld_smoke.py
/data/users/rongdingyi/miniconda3/envs/RoboTwin/bin/python asgardbench_smoke.py

# EB-ALFRED (physical X display + NVIDIA Vulkan)
export DISPLAY="$(bash 00_pick_display.sh)"
DISPLAY="$DISPLAY" VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  /data/users/rongdingyi/miniconda3/envs/bench_avp/bin/python eb_alfred_smoke.py

# EB-Habitat
/data/users/rongdingyi/miniconda3/envs/embench/bin/python eb_habitat_smoke.py
```

## Notes

- EB-ALFRED task setup uses `ann_*.json` (scene / object_poses / object_toggles)
  exactly as the release registry records; `ThorEnv`/`EBAlfEnv` come from the
  pinned EmbodiedBench commit.
- AsgardBench's `Scenario` hardcodes `platform=Linux64`; the smoke script swaps it
  for `CloudRendering` at runtime (no repo edits), and it must run with CWD at the
  repo root (it writes `Generated/status.json`).
- `private_eval_metadata` from the release JSONL must never be fed to agents.
