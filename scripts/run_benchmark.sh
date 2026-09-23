#!/usr/bin/env bash
set -euo pipefail

# Config paths are relative to semi-seg-ecg/src.
BASE_CONFIG="../configs/base/resnet18/mean_teacher_boundary_aware.yaml"
BENCH_CONFIG="../configs/bench/isp/1over2.yaml"
MODE="single"  # single | distributed

if [[ $# -ne 0 ]]; then
    echo "Edit BASE_CONFIG, BENCH_CONFIG, and MODE in $0; run without arguments." >&2
    exit 1
fi

cd "$(dirname "$0")/../semi-seg-ecg/src"

LAUNCH=(python)
case "$MODE" in
    single) ;;
    distributed)
        WORLD_SIZE="$(python - "$BASE_CONFIG" "$BENCH_CONFIG" <<'PYTHON'
import sys

import mergedeep
import yaml

config = {}
for path in sys.argv[1:]:
    with open(path) as file:
        mergedeep.merge(config, yaml.safe_load(file))
print(config['ddp']['world_size'])
PYTHON
)"
        LAUNCH+=(-m torch.distributed.run --standalone --nproc_per_node "$WORLD_SIZE")
        ;;
    *)
        echo "MODE must be single or distributed" >&2
        exit 1
        ;;
esac

exec "${LAUNCH[@]}" train.py -f "$BASE_CONFIG" -o "$BENCH_CONFIG"
