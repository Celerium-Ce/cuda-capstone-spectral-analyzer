#!/usr/bin/env bash
# Builds the project (if needed) and runs the batch spectral analyzer over
# data/input, writing spectrogram CSVs to output/ and a timestamped
# proof-of-execution log to logs/.
#
# Usage: ./run.sh [extra spectral_analyzer args, e.g. --limit 5]
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d "data/input" ] || [ -z "$(ls -A data/input 2>/dev/null)" ]; then
  echo "data/input is empty - generating synthetic signals..."
  (cd data && python3 generate_signals.py)
fi

make all

mkdir -p logs
log_file="logs/run_$(date +%Y%m%d_%H%M%S).log"

./bin/spectral_analyzer --input data/input --output output \
    --log "$log_file" "$@"

echo "Done. Log written to $log_file"
