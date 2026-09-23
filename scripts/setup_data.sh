#!/usr/bin/env bash
# Download SemiSegECG waveforms, labels, and index splits used by the paper.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/semi-seg-ecg"

echo "=== Full benchmark data setup ==="
echo "Working dir: $(pwd)"

if ! command -v gdown &>/dev/null; then
    echo "Installing gdown..."
    pip install -q gdown
fi

download_unzip() {
    local file_id="$1"
    local out_dir="$2"
    local zip_name="$3"
    mkdir -p "$out_dir"
    if [[ -d "$out_dir/${zip_name%.zip}" ]] || [[ -f "$out_dir/${zip_name%.zip}" ]]; then
        echo "  skip (present): $out_dir/${zip_name%.zip}"
        return 0
    fi
    if [[ -f "$out_dir/$zip_name" ]]; then
        echo "  unzip existing: $out_dir/$zip_name"
    else
        echo "  gdown $zip_name -> $out_dir/"
        gdown "$file_id" -O "$out_dir/$zip_name"
    fi
    unzip -q -o "$out_dir/$zip_name" -d "$out_dir"
    rm -f "$out_dir/$zip_name"
}

echo ""
echo "--- Waveforms + labels (data/) ---"
declare -A DATA_IDS=(
    [ludb]=1qPAEmilpbSfCArhfDDKl1Vrqn4j89ZWK
    [qtdb]=1PjoRMw7ZpzmwPWAZq8e7KXpAOVBwucrZ
    [isp]=1NUPQ5rhGvkFPIYNvXz5nWBk8_wF0EIFE
    [zhejiang]=1EYTOrK5GhskO8ulwJyea6UKCFG4-UZ21
    [ptb-xl]=1zs8g5ivGImTctPYyfBx14b1JZhe0u53w
)
for name in ludb qtdb isp zhejiang ptb-xl; do
    echo "[$name]"
    download_unzip "${DATA_IDS[$name]}" data "${name}.zip"
done

echo ""
echo "--- Index splits (index/) ---"
declare -A INDEX_IDS=(
    [ludb]=1vWSola1ySAt5XI8jMoG6ZPAwcFn8OAjP
    [qtdb]=1VCpOrIC0B5V57Jc7b-kR26-T22Bq__nQ
    [isp]=1V2_YoolEnK8I8jRd0W8rixv1xHDBdvyd
    [zhejiang]=1Q8zTQcZDZToaN6L-BLOBoNEJol89xgNW
    [cross_domain]=1B_uCeuVS-eyUjWZ85AswGLn6wwFnfFvd
)
for name in ludb qtdb isp zhejiang cross_domain; do
    echo "[$name]"
    download_unzip "${INDEX_IDS[$name]}" index "${name}.zip"
done

echo ""
echo "Setup complete. Next:"
echo "  bash scripts/run_benchmark.sh"
