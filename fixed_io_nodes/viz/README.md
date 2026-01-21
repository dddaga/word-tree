# Neurograph Visualization Module

This module provides web-based visualization for Neurograph networks using data captured by `ForwardPassTracer`.

## Features

- **Iteration-based navigation**: Step through forward pass iterations
- **Interactive network graph**: Visualize nodes, edges, and activations
- **Node inspection**: Click nodes to see detailed information
- **Compatible with notebooks**: Can be launched from Jupyter notebooks

## Usage

### Command Line

Run a forward pass with tracing and launch the web visualization:

```bash
python visualize_forward_pass.py --config configs/main_config3.yaml
```

Options:
- `--config`: Path to config YAML file (required)
- `--input`: Path to input tensor file (optional)
- `--save-trace`: Path to save trace data as JSON (optional)
- `--port`: Port for web server (default: 8765)
- `--host`: Host for web server (default: 0.0.0.0)

### From Jupyter Notebook

After running a forward pass with `ForwardPassTracer`:

```python
from viz.notebook_helper import launch_web_viz

# After running forward pass with tracer
with ForwardPassTracer() as tracer:
    output = model(input_tensor, tracer=tracer)
    trace_data = tracer.get_trace()

# Launch web visualization
launch_web_viz(trace_data, node_store, config)
```

## Architecture

- **`adapter.py`**: Converts `ForwardPassTracer` data to `StepResult` format
- **`utils.py`**: Helper functions for node roles, edges, and weights
- **`server.py`**: FastAPI server for serving visualization data
- **`notebook_helper.py`**: Helper for launching from notebooks
- **`static/`**: Frontend HTML/JavaScript files

## API Endpoints

- `GET /`: Main visualization page
- `GET /api/state`: Get current network state
- `POST /api/step`: Move to next iteration
- `POST /api/step_back`: Move to previous iteration
- `POST /api/set_iteration`: Jump to specific iteration
- `GET /api/iterations`: Get iteration information
- `POST /api/reset`: Reset to first iteration

## Notes

- The visualization works with trace data captured by `ForwardPassTracer`
- It's non-real-time: you navigate through iterations of a completed forward pass
- The same trace data format is used by `visualization.ipynb` for notebook-based visualization
- `ForwardPassTracer` remains unchanged and compatible with existing code
