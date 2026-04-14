@echo off
:: ══════════════════════════════════════════════════════════════════
::  WRAPPER: Ensures window NEVER closes unexpectedly.
::  The script restarts itself inside "cmd /k" which keeps the
::  window open no matter what. On success, "exit" closes it.
:: ══════════════════════════════════════════════════════════════════
if "%~1"=="__run__" goto :MAIN
cmd /k "%~f0" __run__
exit /b 0

:MAIN
setlocal enabledelayedexpansion
title CrackDetect Setup
mode con cols=80 lines=50

echo.
echo  ================================================================
echo    CrackDetect - Automatische Riss-Erkennung
echo    Powered by U-Net
echo  ================================================================
echo.

cd /d "%~dp0"
echo [INFO] Working directory: %cd%
echo.

:: ─── If already set up, skip to launch ────────────────────────────
if exist ".setup_complete" (
    echo [OK] Setup already complete - skipping installation.
    echo      Launching CrackDetect directly ...
    echo.
    if not exist "venv\Scripts\activate.bat" (
        echo  [ERROR] Virtual environment is missing!
        echo          Delete ".setup_complete" and run start.bat again.
        echo.
        echo  Type "exit" to close this window.
        goto :EOF
    )
    call venv\Scripts\activate.bat
    python -c "import customtkinter, torch, onnxruntime" >nul 2>&1
    if errorlevel 1 (
        echo  [WARNING] Packages missing in venv - repeating setup.
        del /f ".setup_complete" >nul 2>&1
        goto :STEP3
    )
    echo [OK] All packages present.
    goto :LAUNCH
)

echo  IMPORTANT: Do NOT close this window during setup!
echo  First-time setup can take 15-30 minutes depending on your
echo  internet connection.
echo.
echo  ================================================================
echo.
echo  === FIRST-TIME SETUP - this only runs once ===
echo.
echo    Step 1/4    Check Python
echo    Step 2/4    Create virtual environment
echo    Step 3/4    Install PyTorch with CUDA
echo    Step 4/4    Install base dependencies
echo.
echo  ----------------------------------------------------------------
echo   Setup starting in 5 seconds ...
echo  ----------------------------------------------------------------
ping -n 6 127.0.0.1 >nul 2>nul
echo.
echo  Setup starting now ...
echo.

:: ─── Step 1: Check Python ─────────────────────────────────────────
echo.
echo  [Step 1/4] Checking Python installation ...
echo  ----------------------------------------------------------------
python --version 2>nul
if errorlevel 1 (
    echo.
    echo  [ERROR] Python was not found!
    echo          Install Python 3.12: https://www.python.org/downloads/
    echo          Important: Check "Add Python to PATH"!
    echo.
    echo  Type "exit" to close this window.
    goto :EOF
)

for /f "tokens=2" %%v in ('python --version 2^>nul') do set PYVER=%%v
echo  [OK] Python !PYVER! found.

:: ─── Step 2: Virtual environment ──────────────────────────────────
echo.
echo  [Step 2/4] Setting up virtual environment ...
echo  ----------------------------------------------------------------
if exist "venv\Scripts\activate.bat" (
    echo  [OK] Virtual environment already exists.
) else (
    echo  [INFO] Creating virtual environment ...
    python -m venv venv
    if not exist "venv\Scripts\activate.bat" (
        echo  [ERROR] Could not create virtual environment.
        echo  Type "exit" to close this window.
        goto :EOF
    )
    echo  [OK] Virtual environment created.
)

echo  [INFO] Activating virtual environment ...
call venv\Scripts\activate.bat
echo  [OK] Virtual environment activated.

echo  [INFO] Upgrading pip ...
python -m pip install --upgrade pip >nul 2>nul
echo  [OK] pip upgraded.

:: ─── Step 3: Install PyTorch with CUDA ────────────────────────────
:STEP3
echo.
echo  [Step 3/4] Installing PyTorch 2.10 with CUDA 12.8 ...
echo  ----------------------------------------------------------------
echo  [INFO] Download size: approximately 3 GB
echo         This can take 5-20 minutes. DO NOT close this window!
echo.

cmd /c "pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128 --progress-bar on"
if errorlevel 1 (
    echo.
    echo  [WARNING] CUDA version failed. Trying CPU-only ...
    cmd /c "pip install torch torchvision --progress-bar on"
    if errorlevel 1 (
        echo  [ERROR] PyTorch installation failed completely.
        echo  Type "exit" to close this window.
        goto :EOF
    )
    echo  [OK] PyTorch CPU-only installed.
) else (
    echo  [OK] PyTorch with CUDA installed successfully.
)

:: ─── Step 4: Install base dependencies ────────────────────────────
echo.
echo  [Step 4/4] Installing base dependencies ...
echo  ----------------------------------------------------------------
cmd /c "pip install -r requirements.txt --progress-bar on"
if errorlevel 1 (
    echo  [ERROR] Could not install dependencies.
    echo  Type "exit" to close this window.
    goto :EOF
)
echo  [OK] Base dependencies installed.

:: Mark setup as complete
echo done > .setup_complete
echo  [OK] Setup complete! Next start will skip installation.
echo.

:LAUNCH
:: ─── Launch CrackDetect ──────────────────────────────────────────
title CrackDetect
echo.
echo  ================================================================
echo    All checks passed! Launching CrackDetect ...
echo  ================================================================
echo.
echo  [INFO] Das Desktop-Fenster oeffnet sich gleich.
echo.
echo  [INFO] Do NOT close this window while the app is running.
echo         To stop: press Ctrl+C
echo.

python crackdetect.py

if errorlevel 1 (
    echo.
    echo  ================================================================
    echo  [ERROR] CrackDetect exited with an error.
    echo          Check the messages above for details.
    echo  ================================================================
    echo.
    echo  Most common fix: delete ".setup_complete" and run start.bat again.
    echo.
    echo  Type "exit" to close this window.
    goto :EOF
)

echo.
echo  [OK] CrackDetect closed normally. Goodbye!
echo.

:: SUCCESS: close the window automatically
endlocal
exit
