# Quick Start Guide

## ✅ Setup Complete!

Your system is configured to use **MPS (Apple Silicon GPU)** for accelerated training.

## Run Training

### Option 1: Using the convenience script (Recommended)
```bash
cd /Volumes/T9/work/word-tree/fixed_io_nodes
./run.sh
```

### Option 2: Direct Python execution
```bash
cd /Volumes/T9/work/word-tree/fixed_io_nodes
source ~/.virtualenvs/word-tree/bin/activate
source ../.uvenv
python main.py
```

### Option 3: One-liner from project root
```bash
cd /Volumes/T9/work/word-tree && source activate.sh && cd fixed_io_nodes && python main.py
```

## What Happens

1. **Device Detection**: Automatically detects MPS (Apple Silicon GPU)
2. **Worker Optimization**: Uses 5 workers for GPU parallelism
3. **MNIST Download**: First run downloads MNIST to `./data/` (~11MB)
4. **Qdrant Init**: Initializes graph nodes in Qdrant database
5. **Training**: Trains for 1 epoch with real-time loss logging
6. **Output**: Logs saved to `training_logs/log0.csv`

## Expected Output

```
🚀 Using MPS (Apple Silicon GPU)
📊 Optimized workers for GPU: 5
📋 LookupTable created on CPU for multiprocessing compatibility
Logger: Writing logs to training_logs/log0.csv
Accumulator: Moved LookupTable to mps
Accumulator: Ready on mps.
Worker 0: Initializing on mps...
Worker 1: Initializing on mps...
...
Worker 0: Ready on mps.
Worker 1: Ready on mps.
...
Worker 0: Loss: 2.3456
Worker 1: Loss: 2.1234
...
```

## Stop Training

Press `Ctrl+C` - the script handles graceful shutdown:
- Flushes gradients to Qdrant
- Closes all worker processes  
- Saves final logs

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
system:
  device: "auto"  # Or "mps", "cuda", "cpu"

training:
  lr: 0.01
  worker_count: 5
  auto_workers: true  # Auto-adjust for device

model:
  iterations: 5
  vector_dim: 56
```

## Check Training Progress

```bash
# View logs in real-time
tail -f training_logs/log0.csv

# Or open in spreadsheet
open training_logs/log0.csv
```

## Troubleshooting

### Script won't start
```bash
# Check Qdrant
curl http://localhost:6333/health

# If not running:
cd /Volumes/T9/work/word-tree
docker-compose up -d
```

### Memory issues
Reduce workers in `configs/config.yaml`:
```yaml
training:
  worker_count: 2
```

### Switch to CPU
If MPS causes issues:
```yaml
system:
  device: "cpu"
```

## Performance

**Your System** (M-series Mac with MPS):
- Training speed: ~50-100 samples/sec
- 5 parallel workers
- Unified memory (GPU shares with RAM)

**Tip**: Monitor with Activity Monitor → GPU tab

## Next Steps

- Modify `configs/config.yaml` for your experiments
- Check `RUN.md` for detailed documentation
- Check `DEVICE_INFO.md` for device-specific optimization tips

