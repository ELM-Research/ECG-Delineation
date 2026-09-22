#!/usr/bin/env bash
# Train / test / publish one boundary-MT benchmark cell.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

GPUS="0"
DATASET="qtdb"
LABEL_FRACTION="16"
SEED="0"
SMOKE="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --label-fraction)
            LABEL_FRACTION="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --smoke)
            SMOKE="1"
            shift 1
            ;;
        -h|--help)
            echo "Usage: bash scripts/run_benchmark.sh [options]"
            echo ""
            echo "ResNet-18 + U-Net + boundary-aware Mean Teacher."
            echo ""
            echo "Options:"
            echo "  --dataset {ludb|qtdb|isp|zhejiang|cross_domain}  (default: qtdb)"
            echo "  --label-fraction N   16|8|4|2 for in-domain (ignored for cross_domain)"
            echo "  --seed N             0|1|2 (default: 0)"
            echo "  --gpus IDS           GPU indices (default: 0)"
            echo "  --smoke              1-epoch debug run"
            echo ""
            echo "Example:"
            echo "  bash scripts/run_benchmark.sh --dataset ludb --label-fraction 16 --seed 0 --gpus 0"
            echo "  bash scripts/run_benchmark.sh --dataset cross_domain --seed 0 --gpus 0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

case "$DATASET" in
    ludb|qtdb|isp|zhejiang|cross_domain) ;;
    *)
        echo "Invalid --dataset: $DATASET"
        exit 1
        ;;
esac

if [[ "$DATASET" != "cross_domain" ]]; then
    case "$LABEL_FRACTION" in
        16|8|4|2) ;;
        *)
            echo "Invalid --label-fraction: $LABEL_FRACTION (expected 16|8|4|2)"
            exit 1
            ;;
    esac
fi

case "$SEED" in
    0|1|2) ;;
    *)
        echo "Invalid --seed: $SEED (expected 0|1|2)"
        exit 1
        ;;
esac

if [[ "$SMOKE" == "1" && "$DATASET" == "cross_domain" && "$SEED" != "0" ]]; then
    echo "Smoke mode only supports --seed 0 for cross_domain"
    exit 1
fi

SMOKE_FLAG=""
if [[ "$SMOKE" == "1" ]]; then
    SMOKE_FLAG="--smoke"
fi

PATHS_JSON="$(python3 baseline/benchmark_paths.py \
    --repo "$REPO_ROOT" \
    --dataset "$DATASET" \
    --label-fraction "$LABEL_FRACTION" \
    --seed "$SEED" \
    $SMOKE_FLAG \
    --json)"

eval "$(python3 - "$PATHS_JSON" <<'PY'
import json, shlex, sys
p = json.loads(sys.argv[1])
for k, v in p.items():
    if v is None:
        continue
    print(f"{k.upper()}={shlex.quote(str(v))}")
PY
)"

BASE_CONFIG="../configs/base/resnet18/mean_teacher_boundary_unet.yaml"
OUTPUT_DIR="$REPO_ROOT/baseline/exps/resnet18/mean_teacher_boundary_unet"
BENCH_CONFIG="$BENCH_YAML_REL"
RUN_DIR="$RUN_DIR"
RESULTS_DIR="$RESULTS_DIR"

if [[ ! -f "$BENCH_PATH" ]]; then
    echo "Missing bench config: $BENCH_PATH"
    exit 1
fi

if [[ ! -f "semi-seg-ecg/configs/base/resnet18/mean_teacher_boundary_unet.yaml" ]]; then
    echo "Missing base config: semi-seg-ecg/configs/base/resnet18/mean_teacher_boundary_unet.yaml"
    exit 1
fi

NEEDS_OVERRIDE="$(python3 - "$DATASET" "$LABEL_FRACTION" "$SEED" "$SMOKE" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "baseline")
from benchmark_paths import needs_override
dataset, frac, seed, smoke = sys.argv[1:5]
frac_i = None if dataset == "cross_domain" else int(frac)
print(int(needs_override(dataset, frac_i, int(seed), smoke == "1")))
PY
)"

TEMP_OVERRIDE=""
OVERRIDE_ARG="$BENCH_CONFIG"
if [[ "$NEEDS_OVERRIDE" == "1" ]]; then
    TEMP_OVERRIDE="$(mktemp "${TMPDIR:-/tmp}/bench_override.XXXXXX.yaml")"
    python3 baseline/benchmark_paths.py \
        --repo "$REPO_ROOT" \
        --dataset "$DATASET" \
        --label-fraction "$LABEL_FRACTION" \
        --seed "$SEED" \
        $SMOKE_FLAG \
        --write-override "$TEMP_OVERRIDE"
    OVERRIDE_ARG="$TEMP_OVERRIDE"
    trap 'rm -f "$TEMP_OVERRIDE"' EXIT
fi

echo "=== Boundary-MT benchmark cell ==="
echo "Repo root:     $REPO_ROOT"
echo "Dataset:       $DATASET"
if [[ "$DATASET" == "cross_domain" ]]; then
    echo "Label frac:    merged"
else
    echo "Label frac:    1/${LABEL_FRACTION}"
fi
echo "Seed:          $SEED"
echo "GPU(s):        $GPUS"
echo "Smoke:         $SMOKE"
echo "Base config:   $BASE_CONFIG"
echo "Bench config:  $BENCH_CONFIG"
echo "Override:      $OVERRIDE_ARG"
echo "Run dir:       $RUN_DIR/"
echo "Results dir:   $RESULTS_DIR/"
echo ""

mkdir -p "$REPO_ROOT/baseline/exps"

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "semi_seg_ecg" ]]; then
    echo "Warning: conda env 'semi_seg_ecg' is not active."
    echo "Run: conda activate semi_seg_ecg"
fi

echo "--- Preflight: index files ---"
python3 - "$BENCH_PATH" <<'PY' || exit 1
import sys
from pathlib import Path
sys.path.insert(0, "baseline")
from benchmark_paths import index_files_from_bench

bench = Path(sys.argv[1])
semi = Path("semi-seg-ecg")
missing = [p for p in index_files_from_bench(bench, semi) if not p.exists()]
if missing:
    print("Missing index files:")
    for p in missing:
        print(f"  {p}")
    print("Run: bash scripts/setup_data.sh")
    sys.exit(1)
print("Index files OK.")
PY

cd semi-seg-ecg

echo ""
echo "--- Step 1/4: Train ---"
bash scripts/train.sh \
    -f "$BASE_CONFIG" \
    -o "$OVERRIDE_ARG" \
    --output_dir "$OUTPUT_DIR" \
    --gpus "$GPUS"

echo ""
echo "--- Step 2/4: Test ---"
bash scripts/test.sh \
    -f "$BASE_CONFIG" \
    -o "$OVERRIDE_ARG" \
    --output_dir "$OUTPUT_DIR" \
    --gpu "$GPUS"

cd "$REPO_ROOT"

echo ""
echo "--- Step 3/4: Publish results ---"
if [[ -d "$RUN_DIR" ]]; then
    python baseline/plot_results.py \
        --run-dir "$RUN_DIR" \
        --publish \
        --results-dir "$RESULTS_DIR"
else
    echo "Warning: run dir not found: $RUN_DIR (skipping publish)"
fi

echo ""
echo "--- Step 4/4: Boundary metrics ---"
if [[ -d "$RUN_DIR" ]]; then
    python baseline/eval_boundaries.py \
        --run-dir "$RUN_DIR" \
        --out "$RESULTS_DIR/boundary_metrics.json"
else
    echo "Warning: run dir not found (skipping boundary metrics)"
fi

echo ""
echo "Benchmark cell complete."
echo "Full artifacts (gitignored): $RUN_DIR/"
echo "Git-tracked summary:         $RESULTS_DIR/"
