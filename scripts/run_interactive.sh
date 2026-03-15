#!/usr/bin/env bash
# Start an interactive RadarAgent session.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR" python main.py --interactive
