@echo off
title 30Bots Trading Terminal
cd /d "%~dp0"

echo ========================================
echo    30Bots AI Trading Terminal
echo ========================================
echo.

REM Check if venv exists, create if not
if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
    echo Installing dependencies...
    venv\Scripts\pip install PyQt6 pyqtgraph numpy pandas websocket-client alpaca-py xgboost scikit-learn pytz
)

echo Starting trading terminal...
venv\Scripts\python main.py

if errorlevel 1 (
    echo.
    echo Error occurred. Press any key to exit...
    pause >nul
)
