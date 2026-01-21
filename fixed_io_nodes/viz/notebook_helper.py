"""
Helper functions for launching web visualization from Jupyter notebooks.
"""

import threading
import webbrowser
import time
from typing import Dict, Any, Optional
from .adapter import TraceVisualizer
from .server import app, init_with_visualizer
import uvicorn
from core.nodestore import NodeStore


def launch_web_viz(
    trace_data: Dict[str, Any],
    node_store: NodeStore,
    config: Optional[Dict] = None,
    port: int = 8765,
    host: str = "127.0.0.1",
    open_browser: bool = True
):
    """
    Launch web visualization from notebook.
    
    Args:
        trace_data: Trace data from ForwardPassTracer.get_trace()
        node_store: NodeStore instance
        config: Optional config dict
        port: Port for web server (default: 8765)
        host: Host for web server (default: 127.0.0.1)
        open_browser: Whether to automatically open browser (default: True)
        
    Returns:
        Thread object running the server (call .join() to keep it running)
    """
    # Create visualizer
    visualizer = TraceVisualizer(trace_data, node_store, config or {})
    
    # Initialize server
    init_with_visualizer(visualizer)
    
    # Start server in background thread
    def run_server():
        uvicorn.run(app, host=host, port=port, log_level="warning")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait a moment for server to start
    time.sleep(1)
    
    # Open browser
    if open_browser:
        url = f"http://{host}:{port}"
        print(f"Opening visualization at {url}")
        webbrowser.open(url)
    else:
        print(f"Visualization server running at http://{host}:{port}")
        print("Open this URL in your browser to view the visualization")
    
    return server_thread
