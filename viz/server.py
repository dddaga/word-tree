"""
FastAPI server for Neurograph Visualization (Vis.js version).
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os

from manager import get_session


app = FastAPI(title="Neurograph Vis.js", version="2.1.0")

# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class ConfigUpdate(BaseModel):
    num_input: Optional[int] = None
    num_input_connected: Optional[int] = None
    num_middle: Optional[int] = None
    num_output: Optional[int] = None
    
    input_cardinality: Optional[int] = None
    middle_cardinality: Optional[int] = None
    
    radiation_k: Optional[int] = None
    learning_rate: Optional[float] = None
    use_radiation: Optional[bool] = None
    
    vector_dim: Optional[int] = None
    gamma: Optional[float] = None
    beam_width: Optional[int] = None
    architecture_mode: Optional[str] = None
    
    # Energy conservation parameters
    temporal_decay: Optional[float] = None
    conductance_efficiency: Optional[float] = None
    radiation_efficiency: Optional[float] = None


class SetTargetsRequest(BaseModel):
    targets: Dict[str, float]  # {node_id: target_phase}


class InjectRequest(BaseModel):
    node_id: str
    strength: float = 1.0


@app.get("/")
async def index():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.post("/api/init")
async def init_network(config: Dict[str, Any] = {}):
    session = get_session()
    state = session.init_network(config)
    return state


@app.post("/api/step")
async def step_forward():
    session = get_session()
    return session.step_forward()


@app.post("/api/backprop")
async def step_backward():
    session = get_session()
    return session.step_backward()


@app.post("/api/config")
async def update_config(config: ConfigUpdate):
    session = get_session()
    session.update_config(**config.dict(exclude_unset=True))
    return {"status": "ok"}


@app.post("/api/inject")
async def inject(req: InjectRequest):
    session = get_session()
    session.inject(req.node_id, req.strength)
    return {"status": "ok"}


@app.post("/api/reset")
async def reset():
    session = get_session()
    session.init_network() 
    return session.get_state()


@app.post("/api/inject_sequence")
async def inject_sequence(data: dict):
    """Inject temporal sequence with positional encoding."""
    session = get_session()
    if not session.network:
        return {"error": "Network not initialized"}
    
    sequence = data.get("sequence", [])
    timestep = data.get("timestep", 0)
    
    session.network.inject_temporal_sequence(sequence, timestep)
    return {"status": "injected", "timestep": timestep}


@app.post("/api/set_targets")
async def set_targets(req: SetTargetsRequest):
    """Set target phases for output nodes."""
    session = get_session()
    if not session.network:
        return {"error": "Network not initialized"}
    
    session.set_targets(req.targets)
    return {"status": "ok", "targets_set": len(req.targets)}


@app.get("/api/state")
async def get_state():
    """Get current network state including loss."""
    session = get_session()
    if not session.network:
        return {"error": "Network not initialized"}
    return session.get_state()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
