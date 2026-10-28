#!/usr/bin/env bash
# Downloads model weights and datasets into models/ and data/raw/ (both git-ignored).
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python -c "
from huggingface_hub import snapshot_download
for rid in ['mistralai/Timestral-209M', 'amazon/chronos-2']:
    snapshot_download(rid, local_dir='models/' + rid.split('/')[1])
"

# SDWPF / KDD Cup 2022 (CC BY 4.0), figshare article 24798654
mkdir -p data/raw/sdwpf
for pair in README:46005789 sdwpf_turb_location_elevation.csv:46005798 \
            sdwpf_baidukddcup2022_turb_location.csv:46005807 sdwpf_2001_2112_full.parquet:46005810 \
            sdwpf_245days_v1.csv:46005813 final_phase_test.zip:46005816; do
  curl -sSL --retry 3 -o "data/raw/sdwpf/${pair%%:*}" "https://ndownloader.figshare.com/files/${pair##*:}"
done
unzip -q -o data/raw/sdwpf/final_phase_test.zip -d data/raw/sdwpf -x "__MACOSX/*" "*.DS_Store"

# BDG2 / ASHRAE GEPIII (CC BY-SA 4.0), served through Git LFS
mkdir -p data/raw/bdg2
base=https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data
for p in meters/cleaned/electricity_cleaned.csv meters/raw/electricity.csv metadata/metadata.csv weather/weather.csv; do
  curl -sSL --retry 3 -o "data/raw/bdg2/$(echo "$p" | tr / _)" "$base/$p"
done
