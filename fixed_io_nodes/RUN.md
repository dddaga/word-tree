# How to Run main.py

This is a distributed training script that uses PyTorch multiprocessing to train a Graph Neural Network on MNIST data using Qdrant vector database.

## Prerequisites

### 1. Activate Virtual Environment

From the project root:
```bash
cd /Volumes/T9/work/word-tree
source activate.sh
```

Or manually:
```bash
source ~/.virtualenvs/word-tree/bin/activate
source .uvenv  # Sets TMPDIR for ExFAT compatibility
```

### 2. Install Dependencies

```bash
cd fixed_io_nodes
uv pip install -r requirements.txt
```

Or if you prefer pip:
```bash
pip install -r requirements.txt
```

### 3. Start Qdrant Database

The script requires Qdrant running on `localhost:6333`.

**Option A: Using Docker (from project root)**
```bash
cd /Volumes/T9/work/word-tree
docker-compose up -d
```

**Option B: Using Docker manually**
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**Check if Qdrant is running:**
```bash
curl http://localhost:6333/health
```

Should return: `{"title":"qdrant - vector search engine","version":"x.x.x"}`

## Running the Script

### Basic Run

```bash
cd /Volumes/T9/work/word-tree/fixed_io_nodes
python main.py
```

### Configuration

The script uses `configs/config.yaml` by default. Key settings:

- **Qdrant URL**: `http://localhost:6333`
- **Collection**: `final6`
- **Device**: `cuda` (falls back to CPU if CUDA unavailable)
- **Workers**: 5 parallel training workers
- **Graph**: 100 total nodes, 14 input, 10 output
- **Log path**: `training_logs/log0.csv`

### Device Auto-Detection

The script automatically detects the best available device:
- **MPS** (Apple Silicon GPU) - Detected on your Mac! ✅
- **CUDA** (NVIDIA GPU)
- **CPU** (fallback)

To override automatic detection, edit `configs/config.yaml`:
```yaml
system:
  device: "auto"  # Default: auto-detect
  # Or manually set: "mps", "cuda", or "cpu"
  
training:
  worker_count: 5
  auto_workers: true  # Adjusts workers based on device
  # GPU: 5+ workers, CPU: 2 workers
```

## What the Script Does

1. **Downloads MNIST** dataset (first run only, saved to `./data/`)
2. **Initializes Qdrant** collection with graph nodes
3. **Spawns processes**:
   - **Logger**: Writes training metrics to CSV
   - **Gradient Accumulator**: Aggregates and applies gradients
   - **Data Loader**: Feeds MNIST samples to workers
   - **Workers** (5 by default): Train in parallel
4. **Trains** on MNIST for 1 epoch
5. **Logs** results to `training_logs/log0.csv`

## Output

During training you'll see:
```
Logger: Writing logs to training_logs/log0.csv
Accumulator: Ready.
Worker 0: Ready on cuda.
Worker 1: Ready on cuda.
...
Worker 0: Loss: 2.3456
Worker 1: Loss: 2.1234
...
```

Training logs are saved to: `training_logs/log0.csv`

## Stopping Training

Press `Ctrl+C` to gracefully shut down. The script will:
1. Stop data loading
2. Flush gradients to Qdrant
3. Close all worker processes
4. Save final logs

## Troubleshooting

### Error: Connection refused (Qdrant)
```bash
# Start Qdrant first
docker-compose up -d
# Or check if it's running
docker ps
```

### Error: CUDA out of memory
Edit `configs/config.yaml`:
```yaml
system:
  device: "cpu"
```

### Error: Operation not supported (os error 45)
Make sure you sourced `.uvenv`:
```bash
source /Volumes/T9/work/word-tree/.uvenv
```

### ModuleNotFoundError
Install missing dependencies:
```bash
uv pip install -r requirements.txt
```

## Quick Start Command

Complete startup sequence:
```bash
# 1. Navigate and activate
cd /Volumes/T9/work/word-tree
source activate.sh

# 2. Start Qdrant
docker-compose up -d

# 3. Run training
cd fixed_io_nodes
python main.py
```

