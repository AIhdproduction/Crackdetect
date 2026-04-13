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
echo    Powered by SAM3
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
    python -c "import gradio, torch" >nul 2>&1
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
echo    Step 1/6    Check Python
echo    Step 2/6    Create virtual environment
echo    Step 3/6    Install PyTorch with CUDA
echo    Step 4/6    Install base dependencies
echo    Step 5/6    Install SAM3 from GitHub
echo    Step 6/6    Download SAM3 checkpoint
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
echo  [Step 1/6] Checking Python installation ...
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
echo  [Step 2/6] Setting up virtual environment ...
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
echo  [Step 3/6] Installing PyTorch 2.10 with CUDA 12.8 ...
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
echo  [Step 4/6] Installing base dependencies ...
echo  ----------------------------------------------------------------
cmd /c "pip install -r requirements.txt --progress-bar on"
if errorlevel 1 (
    echo  [ERROR] Could not install dependencies.
    echo  Type "exit" to close this window.
    goto :EOF
)
echo  [OK] Base dependencies installed.

:: ─── Step 5: Install SAM3 ─────────────────────────────────────────
echo.
echo  [Step 5/6] Installing SAM3 from GitHub ...
echo  ----------------------------------------------------------------
echo  [INFO] Requires Git: https://git-scm.com/downloads
echo         This may take 2-5 minutes.
echo.
cmd /c "pip install git+https://github.com/facebookresearch/sam3.git"
if errorlevel 1 (
    echo  [ERROR] SAM3 installation failed.
    echo          Make sure Git is installed: https://git-scm.com/downloads
    echo  Type "exit" to close this window.
    goto :EOF
)
echo  [OK] SAM3 installed.
echo.
echo  [INFO] Installing SAM3 extras ...
cmd /c "pip install einops pycocotools triton-windows psutil"
echo  [OK] SAM3 dependencies ready.
echo.
echo  [INFO] Fixing numpy version (SAM3 pins old version) ...
cmd /c "pip install --upgrade numpy"
echo  [OK] numpy updated.

:: ─── Step 6: HuggingFace Login and SAM3 Checkpoint ────────────────
if exist "checkpoints\sam3\model.safetensors" goto :SAM3_OK
if exist "checkpoints\sam3\config.json" goto :SAM3_OK

echo.
echo  [Step 6/6] Downloading SAM3 model checkpoint ...
echo  ----------------------------------------------------------------
echo.
echo  ================================================================
echo   HUGGINGFACE ACCESS REQUIRED
echo  ================================================================
echo.
echo   1. Create a free account at: https://huggingface.co
echo.
echo   2. Request model access - one time, usually instant:
echo      https://huggingface.co/facebook/sam3
echo      Click "Agree and access repository"
echo.
echo   3. Create an access token:
echo      https://huggingface.co/settings/tokens
echo      Click "Create new token" and enable ALL 3 checkboxes:
echo        [x] Read access to contents of all public gated repos
echo        [x] Read access to contents of all repos you can access
echo        [x] Make calls to inference providers
echo      Then click "Create token" and copy it.
echo.
echo   4. Paste the token below and press ENTER
echo.
echo  ----------------------------------------------------------------
echo   PRIVACY: Your token is stored ONLY locally on this PC at:
echo   %USERPROFILE%\.cache\huggingface\token
echo   It is NEVER sent anywhere except to huggingface.co
echo  ----------------------------------------------------------------
echo.
set /p HF_TOKEN="  Your HuggingFace Token: "
echo.

if "!HF_TOKEN!"=="" (
    echo  [ERROR] No token entered.
    echo  Type "exit" to close this window.
    goto :EOF
)

echo  [INFO] Logging in to HuggingFace ...
cmd /c "venv\Scripts\hf.exe auth login --token !HF_TOKEN!"
if errorlevel 1 (
    echo  [ERROR] HuggingFace login failed. Check your token.
    echo  Type "exit" to close this window.
    goto :EOF
)
echo  [OK] HuggingFace login successful.
echo.

mkdir checkpoints\sam3 2>nul
echo  [DOWNLOAD] Downloading SAM3 checkpoint - about 5 GB ...
echo             DO NOT close this window!
echo.
cmd /c "venv\Scripts\hf.exe download facebook/sam3 --local-dir checkpoints\sam3 --token !HF_TOKEN!"
if errorlevel 1 (
    echo  [ERROR] SAM3 download failed. Check access and token.
    echo  Type "exit" to close this window.
    goto :EOF
)
echo  [OK] SAM3 checkpoint downloaded.

:SAM3_OK
echo.
echo  [OK] SAM3 checkpoint is present.

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
echo  [INFO] The app will open in your browser at:
echo         http://127.0.0.1:7861
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
