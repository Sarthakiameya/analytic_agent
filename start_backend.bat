@echo off
title Analytics Agent - Backend
echo ============================================
echo  Analytics Agent Backend
echo ============================================

cd /d "%~dp0"

if not exist "backend\.venv\Scripts\python.exe" (
    echo [1/3] Creating Python venv...
    python -m venv backend\.venv
    echo [2/3] Installing dependencies...
    backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
    echo [3/3] Done!
) else (
    echo [OK] venv exists.
)

echo.
echo Starting FastAPI on http://localhost:8000 ...
echo Press Ctrl+C to stop.
echo.

:: Run from backend dir so relative imports work
cd backend
..\backend\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause
