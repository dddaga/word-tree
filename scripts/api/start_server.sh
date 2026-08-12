#!/bin/bash
cd /Volumes/T9/IndraAstra/dhiraj/neuro_graph
python3 -m uvicorn scripts.api.server:app --host 0.0.0.0 --port 8000 --reload
