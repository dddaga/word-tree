# Known Issues

## MPS/CUDA with Multiprocessing

**Status**: Not currently supported  
**Affects**: macOS MPS and NVIDIA CUDA  
**Workaround**: Use CPU device

### Issue Description

The current architecture uses Python multiprocessing with shared queues to parallelize training across multiple worker processes. This design has compatibility issues with GPU devices (MPS/CUDA):

1. **Tensor Sharing**: MPS and CUDA tensors cannot be shared between processes using Python's multiprocessing queues
2. **LookupTable**: The LookupTable contains GPU tensors that need to be pickled and sent between processes
3. **Config Passing**: The configuration dict is passed to worker processes, which can contain GPU device references

### Error Messages

```
RuntimeError: _share_filename_: only available on CPU
```

This occurs when trying to pass GPU tensors through multiprocessing queues.

### Current Solution

The code has been configured to use **CPU** by default:

```yaml
# configs/config.yaml
system:
  device: "cpu"
```

### Performance Impact

- **CPU Training**: ~10-50 samples/sec (depending on CPU)
- **Potential MPS**: ~50-200 samples/sec (if working)

### Future Solutions

To enable GPU support, the architecture needs refactoring:

**Option 1: Shared Memory**
- Use `torch.multiprocessing` with shared memory tensors
- Requires changing from `mp.Queue` to shared tensors
- More complex but enables GPU

**Option 2: Single Process + DataLoader Workers**
- Keep model in single process on GPU
- Use PyTorch DataLoader with num_workers for data loading only
- Simpler, but less parallel gradient computation

**Option 3: Distributed Training**
- Use `torch.distributed` or Ray
- Each worker has own GPU copy
- Best for multi-GPU setups

### Device Detection Still Works

The `device_utils.py` module correctly detects MPS/CUDA, but the config is set to CPU to avoid multiprocessing issues. You can verify detection:

```bash
python device_utils.py
```

Output on your Mac:
```
🚀 Using MPS (Apple Silicon GPU)
```

### Testing GPU (Advanced)

If you want to test with GPU (single worker only):

1. Edit `configs/config.yaml`:
```yaml
system:
  device: "mps"  # or "cuda"
  
training:
  worker_count: 1  # IMPORTANT: Only 1 worker
```

2. This avoids multiprocessing but loses parallelism benefits

### Related Files

- `main.py` - Multiprocessing setup
- `device_utils.py` - Device detection
- `configs/config.yaml` - Device configuration
- `lookup_table.py` - Contains GPU tensors

### Contributions Welcome

If you'd like to help implement GPU support with multiprocessing, see the architecture refactor options above!

