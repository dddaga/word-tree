#!/bin/bash
# Convenience script to run main.py with proper setup

set -e  # Exit on error

echo "🚀 Starting training setup..."

# 1. Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "📦 Activating virtual environment..."
    source "$HOME/.virtualenvs/word-tree/bin/activate"
fi

# 2. Load UV configuration (TMPDIR for ExFAT)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$SCRIPT_DIR/.uvenv" ]; then
    source "$SCRIPT_DIR/.uvenv"
    echo "✅ Loaded UV configuration (TMPDIR: $TMPDIR)"
fi

# 3. Check if Qdrant is running
echo "🔍 Checking Qdrant connection..."
if curl -s http://localhost:6333/health > /dev/null 2>&1; then
    echo "✅ Qdrant is running"
else
    echo "❌ Qdrant is not running!"
    echo ""
    echo "Start Qdrant with:"
    echo "  cd $SCRIPT_DIR && docker-compose up -d"
    echo ""
    exit 1
fi

# 4. Check if dependencies are installed
echo "📚 Checking dependencies..."
if ! python -c "import torch; import torchvision; import qdrant_client; import yaml" 2>/dev/null; then
    echo "⚠️  Missing dependencies. Installing..."
    uv pip install -r requirements.txt
    
    # Verify installation
    if ! python -c "import torch; import torchvision; import qdrant_client; import yaml" 2>&1; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi
echo "✅ All dependencies installed"

# 5. Run the script
echo ""
echo "🎯 Starting training..."
echo "   Config: configs/config.yaml"
echo "   Logs: training_logs/log0.csv"
echo ""
echo "Press Ctrl+C to stop training gracefully"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python main.py

