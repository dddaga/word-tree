# Device Configuration Guide

## Automatic Device Detection ✨

The training script automatically detects and uses the best available device.

### Your System
- **Device**: MPS (Apple Silicon GPU) ✅
- **CPU Threads**: 10
- **Optimal Workers**: 5

## How It Works

The `device_utils.py` module checks in this priority:

1. **CUDA** (NVIDIA GPUs)
   - Best for: High-end training, large models
   - Workers: 5-10+ recommended

2. **MPS** (Apple Silicon) ← **Your device!**
   - Best for: Mac M1/M2/M3 training
   - Workers: 5 recommended
   - Note: Unified memory architecture

3. **CPU** (Fallback)
   - Best for: Compatibility, debugging
   - Workers: 1-2 recommended

## Configuration Options

### Option 1: Auto (Recommended)
```yaml
# configs/config.yaml
system:
  device: "auto"  # Automatically detect best device

training:
  worker_count: 5
  auto_workers: true  # Optimize workers for device
```

### Option 2: Manual Override
```yaml
system:
  device: "mps"  # Force specific device

training:
  worker_count: 5
  auto_workers: false  # Use exact worker count
```

## Testing Device

Run the device detection utility:
```bash
cd /Volumes/T9/work/word-tree/fixed_io_nodes
source ~/.virtualenvs/word-tree/bin/activate
python device_utils.py
```

## Performance Tips

### For MPS (Your Setup)
- ✅ Use 5 workers (good parallelism)
- ✅ MPS handles memory efficiently
- ⚠️ Some operations may fall back to CPU (normal)

### For CUDA
- Use 5-10 workers for high throughput
- Monitor GPU memory usage
- Consider mixed precision training

### For CPU
- Use 1-2 workers (avoid overhead)
- Expect slower training
- Good for debugging

## Troubleshooting

### MPS Errors
If you see MPS-related errors:
```yaml
system:
  device: "cpu"  # Fallback to CPU
```

### Memory Issues
Reduce batch size or worker count:
```yaml
training:
  worker_count: 2  # Reduce parallelism
```

### Performance Check
Monitor with:
```bash
# Mac Activity Monitor
# Or terminal:
top -o cpu
```

