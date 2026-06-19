#!/usr/bin/env bash
# Parquet Capital — one-command setup + launch.
# Builds the dataset if it's missing, then starts the dashboard.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f "parquet_out/clean_roster.csv" ]; then
  echo "clean_roster.csv not found — building dataset from raw sources…"
  python build_dataset.py
fi

echo "Launching Parquet Capital dashboard…"
streamlit run app_NBA.py
