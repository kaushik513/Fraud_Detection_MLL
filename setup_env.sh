#!/usr/bin/env bash
# setup_env.sh — v1.0.0 (2026-10-10): create venv + install requirements (macOS / Linux)
set -e
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo
echo "Environment ready. Activate later with:  source .venv/bin/activate"
echo "Start the GUI with:                      python gui.py"
