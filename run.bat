@echo off
setlocal enabledelayedexpansion
title Explainable Multimodal Transformer - SCZ-BD Differential Analysis

echo ==============================================================================
echo   Explainable Multimodal Transformer for SCZ-BD Differential Analysis
echo   Starting Backend (FastAPI :8000) ^& Frontend (React Vite :5173)
echo ==============================================================================
echo.

:: Ensure working directory is the script location
cd /d "%~dp0"

:: 1. Verify Python availability
echo [1/4] Checking Python environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in your system PATH.
    echo Please ensure Python 3.10+ is installed and added to PATH.
    echo.
    pause
    exit /b 1
)

:: 2. Verify Node.js and npm availability
echo [2/4] Checking Node.js / npm environment...
call npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] npm was not found in your system PATH.
    echo Please install Node.js from https://nodejs.org/ and retry.
    echo.
    pause
    exit /b 1
)

:: 3. Launch FastAPI Backend
echo [3/4] Launching FastAPI Backend on http://127.0.0.1:8000 ...
start "Backend - FastAPI Server (Port 8000)" cmd /k "cd /d "%~dp0" && python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000"

:: 4. Launch React Frontend
echo [4/4] Launching React Vite Frontend on http://127.0.0.1:5173 ...
start "Frontend - Vite Dev Server (Port 5173)" cmd /k "cd /d "%~dp0frontend" && npm run dev -- --host 127.0.0.1 --port 5173"

echo.
echo ==============================================================================
echo   Services are running!
echo   ----------------------------------------------------------------------------
echo   - Web Application:     http://127.0.0.1:5173
echo   - Backend API Docs:    http://127.0.0.1:8000/docs
echo   - Backend Health:      http://127.0.0.1:8000/api/health
echo.
echo   Opening web browser in 3 seconds...
echo   (To stop the application, simply close the backend and frontend windows)
echo ==============================================================================

timeout /t 3 /nobreak >nul
start http://127.0.0.1:5173

exit /b 0
