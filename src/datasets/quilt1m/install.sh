#!/bin/bash
# Installation script for QUILT-1M vLLM dependencies using uv

set -e

echo "Installing QUILT-1M vLLM dependencies with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.10
fi

# Install dependencies
echo "Installing dependencies..."
uv sync

echo "Installation complete!"
echo "To activate the environment, run:"
echo "source .venv/bin/activate"