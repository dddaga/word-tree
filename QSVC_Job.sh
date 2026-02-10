#!/bin/bash
#SBATCH --job-name=QSVC_qubits_samples
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=96:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --output=%J.out
#SBATCH --error=%J.err

source minconda3/bin/activate
conda activate qiskit_env
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
cd /home/ujjwal.phy21.itbhu/QIntern/
python3 QSVC.py
