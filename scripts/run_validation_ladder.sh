#!/usr/bin/env bash
set -e
export PYTHONPATH=/Volumes/T9/IndraAstra/dhiraj/neuro_graph
cd /Volumes/T9/IndraAstra/dhiraj/neuro_graph

echo "============================================"
echo "VALIDATION LADDER — CPU MODE"
echo "============================================"
echo ""

echo ">>> STEP 1: DONE — results confirmed in logs (l2 wins 23.9%, N=1024 16%)"
echo ""

echo ">>> STEP 2: W_phase receiver"
echo "    N=512 | baseline vs K_phase=8 vs K_phase=16 | 60 epochs"
python3 -u scripts/validate_step2_wphase.py --device cpu --epochs 60
echo "Step 2 complete."
echo ""

echo ">>> STEP 3: Two-scale Turing mechanism"
echo "    N=512 | 4 modes | 60 epochs each"
python3 -u scripts/validate_step3_turing.py --device cpu --epochs 60
echo "Step 3 complete."
echo ""

echo ">>> STEP 4: Simulated annealing"
echo "    N=512 | 4 schedules | 60 epochs each"
python3 -u scripts/validate_step4_anneal.py --device cpu --epochs 60
echo "Step 4 complete."
echo ""

echo "============================================"
echo "ALL STEPS COMPLETE"
echo "Results in results/validate_step*.json"
echo "============================================"
