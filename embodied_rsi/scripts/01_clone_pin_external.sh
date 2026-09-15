#!/usr/bin/env bash
# 01_clone_pin_external.sh -- Phase B: clone external method repos at their pinned
# commits. Never follow main; upgrades require a new benchmark version.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$PROJECT/external"
mkdir -p "$EXT"
cd "$EXT"

clone_pin() {
  local name="$1" repo="$2" commit="$3"
  if [ ! -d "$name/.git" ]; then
    echo "[clone] $name"
    git clone "$repo" "$name"
  else
    echo "[reuse] $name"
  fi
  cd "$name"
  git fetch --all --tags --quiet
  git checkout --quiet "$commit"
  local head
  head="$(git rev-parse HEAD)"
  if [ "$head" != "$commit" ]; then
    echo "[FAIL] $name HEAD $head != pin $commit" >&2
    exit 1
  fi
  if [ -n "$(git status --porcelain)" ]; then
    echo "[FAIL] $name worktree not clean" >&2
    exit 1
  fi
  echo "[ok] $name @ $head"
  cd "$EXT"
}

clone_pin OpenETA     https://github.com/OpenMOSS/OpenETA                 7d4a0a1522ba8ebbd362bde880bad81d2a98f15e
clone_pin ace         https://github.com/ace-agent/ace                    82709de050e1db6e6ef2f07bcb0393560b94992a
clone_pin WorldMind   https://github.com/zjunlp/WorldMind                 712b0fd53b4bd6a1948603f087a1e9f25a5adaea
clone_pin EmbodiSkill https://github.com/air-embodied-brain/EmbodiSkill   760126030eab1d33ec6a6f30988f0f1fb58df3a7

python3 - "$PROJECT" <<'PY'
import json, subprocess, sys
from pathlib import Path
project = Path(sys.argv[1])
sources = {
    "OpenETA": ("https://github.com/OpenMOSS/OpenETA", "7d4a0a1522ba8ebbd362bde880bad81d2a98f15e", "OpenETA"),
    "ACE": ("https://github.com/ace-agent/ace", "82709de050e1db6e6ef2f07bcb0393560b94992a", "ace"),
    "WorldMind": ("https://github.com/zjunlp/WorldMind", "712b0fd53b4bd6a1948603f087a1e9f25a5adaea", "WorldMind"),
    "EmbodiSkill": ("https://github.com/air-embodied-brain/EmbodiSkill", "760126030eab1d33ec6a6f30988f0f1fb58df3a7", "EmbodiSkill"),
}
manifest = {}
for name, (repo, commit, dirname) in sources.items():
    workdir = project / "external" / dirname
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workdir).decode().strip()
    manifest[name] = {"repo": repo, "commit": commit, "head": head}
    if head != commit:
        raise SystemExit(f"[FAIL] {name} head {head} != {commit}")
out = project / "manifests" / "external_sources.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"[ok] external_sources.json written: {out}")
PY

echo "[OK] all external repos pinned"
