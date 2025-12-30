#!/bin/bash
# Quick activation script for word-tree project

# Load UV configuration (sets TMPDIR for ExFAT compatibility)
source "$(dirname "$0")/.uvenv"

# Activate virtual environment
source "$HOME/.virtualenvs/word-tree/bin/activate"

echo "✅ Activated word-tree environment"
echo "   Python: $(which python)"
echo "   TMPDIR: $TMPDIR"
echo ""
echo "You can now use 'uv pip install <package>' or 'pip install <package>'"

