#!/usr/bin/env bash
# Start an interactive RadarAgent session.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_DIR/.venv"

if [ ! -f "$VENV/bin/python" ]; then
    echo "ERROR: venv not found at $VENV"
    echo "Run: python3.10 -m venv $VENV && $VENV/bin/pip install -r $PROJECT_DIR/requirements.txt"
    exit 1
fi

cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR" "$VENV/bin/python" main.py --interactive
