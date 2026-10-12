@echo off
REM setup_env.bat — v1.0.0 (2026-10-10): create venv + install requirements (Windows)
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Environment ready. Activate later with:  .venv\Scripts\activate
echo Start the GUI with:                      python gui.py
