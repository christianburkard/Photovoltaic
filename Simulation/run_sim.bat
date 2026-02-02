@echo off
title Solar PV App Launcher
echo ==============================
echo Starting Photovoltaic Simulation
echo ==============================

REM --- Set paths (adjust if different) ---
set BACKEND_PATH=%~dp0backend
set FRONTEND_PATH=%~dp0frontend

REM --- Activate backend virtual environment ---
echo Starting backend (Flask)...
start cmd /k "cd /d %BACKEND_PATH% && call venv\Scripts\activate && python app.py"

REM --- Start frontend (React) ---
echo Starting frontend (React)...
start cmd /k "cd /d %FRONTEND_PATH% && npm start"

echo.
echo ========================================
echo Both backend and frontend are launching.
echo When finished, close their windows.
echo ========================================
pause
