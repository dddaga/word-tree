#!/bin/bash
# Run from project root: bash ui/scripts/copy_data.sh
set -e
mkdir -p ui/public/data
cp results/sim/sim_step002_benchmarks.json ui/public/data/ 2>/dev/null || echo "WARN: sim_step002_benchmarks.json not found"
cp results/ts/ts_step010_cnn_lstm.json ui/public/data/ 2>/dev/null || echo "WARN: ts_step010_cnn_lstm.json not found"
cp results/ts/ts_step011_cnn_transformer.json ui/public/data/ 2>/dev/null || echo "WARN: ts_step011_cnn_transformer.json not found"
cp results/ts/ts_step020_mae_encoder.json ui/public/data/ 2>/dev/null || echo "WARN: ts_step020_mae_encoder.json not found"
cp results/ts/ts_step021_gan_encoder.json ui/public/data/ 2>/dev/null || echo "WARN: ts_step021_gan_encoder.json not found"
cp results/ts/ts_step030_sgnnet_ts.json ui/public/data/ 2>/dev/null || echo "WARN: ts_step030_sgnnet_ts.json not found"
echo "Data copied."
