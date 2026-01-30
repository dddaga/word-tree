"""
FastAPI server for Neurograph Visualization using ForwardPassTracer data.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
from pathlib import Path

from .adapter import TraceVisualizer, StepResult


app = FastAPI(title="Neurograph Visualization", version="2.0.0")

# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Global session storage
_session: Optional[TraceVisualizer] = None


class SetIterationRequest(BaseModel):
    iteration: int


class SetLayerRequest(BaseModel):
    layer_index: int


@app.get("/")
async def index():
    """Serve the main visualization page."""
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.post("/api/init")
async def init_network(data: Dict[str, Any] = {}):
    """
    Initialize network with trace data.
    
    Note: This endpoint is for compatibility. The server should be initialized
    using init_with_visualizer() from Python code before starting the server.
    """
    global _session
    
    if _session is None:
        raise HTTPException(
            status_code=400, 
            detail="Network not initialized. Server must be started with init_with_visualizer() from Python code."
        )
    
    # Return initial state
    _session._current_layer = getattr(_session, "_current_layer", 0)
    if _session.get_iteration_count(_session._current_layer) > 0:
        _session._current_iteration = 0
        return _session.get_state(0, _session._current_layer).__dict__
    else:
        raise HTTPException(status_code=400, detail="Trace data has no iterations")


def init_with_visualizer(visualizer: TraceVisualizer):
    """
    Initialize server with a TraceVisualizer instance.
    This should be called from Python code, not via API.
    """
    global _session
    _session = visualizer
    _session._current_layer = 0
    _session._current_iteration = 0


@app.get("/api/state")
async def get_state():
    """Get current network state."""
    if _session is None:
        raise HTTPException(status_code=400, detail="Network not initialized. Call /api/init first.")

    current_layer = getattr(_session, "_current_layer", 0)
    current_iter = getattr(_session, "_current_iteration", 0)
    layer_count = _session.get_iteration_count(current_layer)
    if current_iter >= layer_count:
        current_iter = max(0, layer_count - 1)
    state = _session.get_state(current_iter, current_layer)
    return state.__dict__


@app.post("/api/step")
async def step_forward():
    """Move to next iteration."""
    if _session is None:
        raise HTTPException(status_code=400, detail="Network not initialized")

    current_layer = getattr(_session, "_current_layer", 0)
    current_iter = getattr(_session, "_current_iteration", 0)
    max_iter = _session.get_iteration_count(current_layer) - 1
    if current_iter < max_iter:
        current_iter += 1
        _session._current_iteration = current_iter
    state = _session.get_state(current_iter, current_layer)
    return state.__dict__


@app.post("/api/step_back")
async def step_backward():
    """Move to previous iteration."""
    if _session is None:
        raise HTTPException(status_code=400, detail="Network not initialized")

    current_layer = getattr(_session, "_current_layer", 0)
    current_iter = getattr(_session, "_current_iteration", 0)
    if current_iter > 0:
        current_iter -= 1
        _session._current_iteration = current_iter
    state = _session.get_state(current_iter, current_layer)
    return state.__dict__


@app.post("/api/set_iteration")
async def set_iteration(req: SetIterationRequest):
    """Jump to a specific iteration."""
    if _session is None:
        raise HTTPException(status_code=400, detail="Network not initialized")

    current_layer = getattr(_session, "_current_layer", 0)
    max_iter = _session.get_iteration_count(current_layer) - 1
    if req.iteration < 0 or req.iteration > max_iter:
        raise HTTPException(
            status_code=400,
            detail=f"Iteration {req.iteration} out of range [0, {max_iter}]",
        )
    _session._current_iteration = req.iteration
    state = _session.get_state(req.iteration, current_layer)
    return state.__dict__


@app.get("/api/iterations")
async def get_iterations():
    """Get information about available iterations for the current layer."""
    if _session is None:
        raise HTTPException(status_code=400, detail="Network not initialized")

    current_layer = getattr(_session, "_current_layer", 0)
    count = _session.get_iteration_count(current_layer)
    current = getattr(_session, "_current_iteration", 0)
    trace_data = _session._trace_data_for_layer(current_layer)
    iterations_list = trace_data.get("iterations", [])

    return {
        "total_iterations": count,
        "current_iteration": current,
        "iterations": [
            {
                "iteration": i,
                "input_injected": iterations_list[i].get("input_injected", False),
                "active_nodes": len(iterations_list[i].get("active_nodes", [])),
            }
            for i in range(count)
        ],
    }


@app.get("/api/layers")
async def get_layers():
    """Get layer count and current layer index."""
    if _session is None:
        raise HTTPException(status_code=400, detail="Network not initialized")
    layer_count = _session.get_layer_count()
    current_layer = getattr(_session, "_current_layer", 0)
    return {
        "layer_count": layer_count,
        "current_layer": current_layer,
    }


@app.post("/api/set_layer")
async def set_layer(req: SetLayerRequest):
    """Switch to a specific layer."""
    if _session is None:
        raise HTTPException(status_code=400, detail="Network not initialized")
    layer_count = _session.get_layer_count()
    if req.layer_index < 0 or req.layer_index >= layer_count:
        raise HTTPException(
            status_code=400,
            detail=f"Layer {req.layer_index} out of range [0, {layer_count})",
        )
    _session._current_layer = req.layer_index
    _session._current_iteration = 0
    state = _session.get_state(0, req.layer_index)
    return state.__dict__


@app.post("/api/reset")
async def reset():
    """Reset to first iteration of current layer."""
    if _session is None:
        raise HTTPException(status_code=400, detail="Network not initialized")
    current_layer = getattr(_session, "_current_layer", 0)
    _session._current_iteration = 0
    state = _session.get_state(0, current_layer)
    return state.__dict__


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
