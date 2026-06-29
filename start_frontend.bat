@echo off
title Analytics Agent - Frontend
echo ============================================
echo  Analytics Agent Frontend Setup
echo ============================================

cd /d "%~dp0\frontend"

if not exist "node_modules" (
    echo [1/2] Installing npm packages...
    npm install
    if errorlevel 1 (
        echo ERROR: npm install failed. Make sure Node.js is installed.
        pause
        exit /b 1
    )
)

echo.
echo Starting Vite dev server on http://localhost:5173 ...
echo.
npm run dev
pause
