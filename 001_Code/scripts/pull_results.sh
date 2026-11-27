#!/usr/bin/env bash
# Copy colab results from the google drive mount into the repo so they can be
# committed and analysed locally. Checkpoints stay on drive, way too big for git.
# Usage: bash scripts/pull_results.sh [path-to-drive-inm363-folder]
set -e

repo="$(cd "$(dirname "$0")/.." && pwd)"

drive="$1"
if [ -z "$drive" ]; then
  # try the usual spots the drive desktop app mounts to
  for cand in "$HOME"/Library/CloudStorage/GoogleDrive-*/"My Drive"/inm363 \
              "$HOME/Google Drive/My Drive/inm363" \
              "$HOME/Google Drive/inm363"; do
    if [ -d "$cand" ]; then drive="$cand"; break; fi
  done
fi
if [ -z "$drive" ] || [ ! -d "$drive" ]; then
  echo "cant find the drive inm363 folder, pass it as the first argument"
  exit 1
fi
echo "drive folder: $drive"

# Merge the run dirs and shared base evals into results/. The ledger is handled
# separately below because both machines append to their own copy
rsync -av --exclude '.*' --exclude 'compute_ledger.jsonl' \
  "$drive/results/" "$repo/results/"

# append only ledger lines not already present
python3 - "$drive/results/compute_ledger.jsonl" "$repo/results/compute_ledger.jsonl" <<'PYEOF'
import os
import sys

src, dst = sys.argv[1], sys.argv[2]
if os.path.exists(src):
    # dedupe the local file first, git merges of the committed ledger can leave
    # repeated lines and identical rows are never two real sessions
    local = []
    if os.path.exists(dst):
        seen = set()
        for l in open(dst).read().splitlines():
            if l and l not in seen:
                seen.add(l)
                local.append(l)
        open(dst, "w").write("".join(x + "\n" for x in local))
    have = set(local)
    new = [l for l in open(src).read().splitlines() if l and l not in have]
    with open(dst, "a") as f:
        for l in new:
            f.write(l + "\n")
    print(f"ledger: appended {len(new)} new line(s)")
else:
    print("no ledger on drive, skipped")
PYEOF

echo
git -C "$repo" status --short results/ | head -30
echo
echo "done. review the list above, then: git add results/ && git commit"
