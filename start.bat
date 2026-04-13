@echo off
:: ══════════════════════════════════════════════════════════════════
::  WRAPPER – Fenster bleibt IMMER offen (auch bei Fehlern).
::  Beim ersten Aufruf startet sich das Script neu in cmd /k.
:: ══════════════════════════════════════════════════════════════════
if "%~1"=="__run__" goto :MAIN
cmd /k "%~f0" __run__
exit /b 0

:MAIN
setlocal enabledelayedexpansion
title CrackDetect Setup
mode con cols=80 lines=55

echo.
echo  ================================================================
echo    CrackDetect – Automatische Riss-Erkennung
echo    Powered by Grounded-SAM2 (Grounding DINO + SAM2.1)
echo  ================================================================
echo.

cd /d "%~dp0"
echo [INFO] Arbeitsverzeichnis: %cd%
echo.

:: ─── Setup bereits erledigt? ────────────────────────────────────────────────
if exist ".setup_complete" (
    echo [OK] Setup bereits abgeschlossen – starte direkt ...
    echo.
    if not exist "venv\Scripts\activate.bat" (
        echo  [FEHLER] Virtual Environment fehlt!
        echo          Bitte ".setup_complete" loeschen und start.bat erneut ausfuehren.
        echo.
        echo  Druecke Enter zum Schliessen.
        pause >nul
        goto :EOF
    )
    call venv\Scripts\activate.bat
    goto :LAUNCH
)

echo  ┌─────────────────────────────────────────────────────────────┐
echo  │  ERSTES SETUP – bitte Fenster NICHT schliessen!            │
echo  │  Download-Groesse: ca. 2.5 GB (Modelle + Pakete)           │
echo  │  Dauer: 15–40 Minuten je nach Internetverbindung           │
echo  └─────────────────────────────────────────────────────────────┘
echo.
echo   Schritte:
echo   1/9   Python pruefen
echo   2/9   Virtual Environment anlegen
echo   3/9   PyTorch + CUDA installieren (~3 GB)
echo   4/9   Basis-Pakete installieren
echo   5/9   GroundingDINO installieren (braucht Git + MSVC)
echo   6/9   SAM2 installieren
echo   7/9   Grounding DINO Checkpoint laden (~700 MB)
echo   8/9   SAM2.1 Checkpoint laden (~900 MB)
echo   9/9   CrackDetect starten
echo.
echo  ── Setup startet in 5 Sekunden ────────────────────────────────
ping -n 6 127.0.0.1 >nul 2>nul
echo.

:: ────────────────────────────────────────────────────────────────────────────
::  SCHRITT 1 – Python pruefen
:: ────────────────────────────────────────────────────────────────────────────
echo  [1/9] Python pruefen ...
echo  ────────────────────────────────────────────────────────────────
python --version 2>nul
if errorlevel 1 (
    echo.
    echo  [FEHLER] Python nicht gefunden!
    echo          Python 3.10 oder 3.11 installieren:
    echo          https://www.python.org/downloads/
    echo          Wichtig: "Add Python to PATH" ankreuzen!
    echo.
    echo  Druecke Enter zum Schliessen.
    pause >nul
    goto :EOF
)
for /f "tokens=2" %%v in ('python --version 2^>nul') do set PYVER=%%v
echo  [OK] Python !PYVER! gefunden.

:: ────────────────────────────────────────────────────────────────────────────
::  SCHRITT 2 – Virtual Environment
:: ────────────────────────────────────────────────────────────────────────────
echo.
echo  [2/9] Virtual Environment ...
echo  ────────────────────────────────────────────────────────────────
if exist "venv\Scripts\activate.bat" (
    echo  [OK] Virtual Environment existiert bereits.
) else (
    echo  [INFO] Erstelle Virtual Environment ...
    python -m venv venv
    if not exist "venv\Scripts\activate.bat" (
        echo  [FEHLER] Virtual Environment konnte nicht erstellt werden.
        echo.
        echo  Druecke Enter zum Schliessen.
        pause >nul
        goto :EOF
    )
    echo  [OK] Virtual Environment erstellt.
)

echo  [INFO] Aktiviere Virtual Environment ...
call venv\Scripts\activate.bat
echo  [INFO] Aktualisiere pip ...
python -m pip install --upgrade pip setuptools wheel >nul 2>nul
echo  [OK] pip aktualisiert.

:: ────────────────────────────────────────────────────────────────────────────
::  SCHRITT 3 – PyTorch mit CUDA
:: ────────────────────────────────────────────────────────────────────────────
echo.
echo  [3/9] PyTorch + CUDA installieren ...
echo  ────────────────────────────────────────────────────────────────
echo  [INFO] Download ca. 3 GB – bitte warten ...
echo.
cmd /c "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --progress-bar on"
if errorlevel 1 (
    echo.
    echo  [WARNUNG] CUDA-Version fehlgeschlagen. Versuche CPU-only ...
    cmd /c "pip install torch torchvision --progress-bar on"
    if errorlevel 1 (
        echo  [FEHLER] PyTorch-Installation komplett fehlgeschlagen.
        echo.
        echo  Druecke Enter zum Schliessen.
        pause >nul
        goto :EOF
    )
    echo  [OK] PyTorch CPU-only installiert (GPU-Beschleunigung nicht verfuegbar).
) else (
    echo  [OK] PyTorch mit CUDA installiert.
)

:: ────────────────────────────────────────────────────────────────────────────
::  SCHRITT 4 – Basis-Pakete
:: ────────────────────────────────────────────────────────────────────────────
echo.
echo  [4/9] Basis-Pakete installieren ...
echo  ────────────────────────────────────────────────────────────────
cmd /c "pip install -r requirements.txt --progress-bar on"
if errorlevel 1 (
    echo  [FEHLER] Basis-Pakete konnten nicht installiert werden.
    echo          Pruefen Sie Ihre Internetverbindung.
    echo.
    echo  Druecke Enter zum Schliessen.
    pause >nul
    goto :EOF
)
echo  [OK] Basis-Pakete installiert.

:: ────────────────────────────────────────────────────────────────────────────
::  SCHRITT 5 – GroundingDINO
::  Braucht: Git (https://git-scm.com) + Visual Studio Build Tools
:: ────────────────────────────────────────────────────────────────────────────
echo.
echo  [5/9] GroundingDINO installieren ...
echo  ────────────────────────────────────────────────────────────────
echo  [INFO] Benoetigt Git: https://git-scm.com/downloads
echo  [INFO] Benoetigt Visual Studio Build Tools (C++ Compiler):
echo         https://visualstudio.microsoft.com/visual-cpp-build-tools/
echo         Beim Installer: "C++ Build Tools" auswaehlen
echo.
echo  [INFO] Installiere GroundingDINO ...

cmd /c "pip install git+https://github.com/IDEA-Research/GroundingDINO.git --progress-bar on"
if errorlevel 1 (
    echo.
    echo  [WARNUNG] GroundingDINO-Installation fehlgeschlagen.
    echo.
    echo  Moegliche Ursachen:
    echo    1. Git nicht installiert  →  https://git-scm.com/downloads
    echo    2. C++ Build Tools fehlen →  https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo       (Im Installer "Desktop-Entwicklung mit C++" auswaehlen)
    echo    3. CUDA-Version nicht kompatibel
    echo.
    echo  Nach der Installation: ".setup_complete" loeschen + start.bat erneut starten.
    echo.
    echo  Druecke Enter zum Schliessen.
    pause >nul
    goto :EOF
)
echo  [OK] GroundingDINO installiert.

:: ────────────────────────────────────────────────────────────────────────────
::  SCHRITT 6 – SAM2
:: ────────────────────────────────────────────────────────────────────────────
echo.
echo  [6/9] SAM2 installieren ...
echo  ────────────────────────────────────────────────────────────────
cmd /c "pip install git+https://github.com/facebookresearch/sam2.git --progress-bar on"
if errorlevel 1 (
    echo  [FEHLER] SAM2-Installation fehlgeschlagen.
    echo          Git muss installiert sein: https://git-scm.com/downloads
    echo.
    echo  Druecke Enter zum Schliessen.
    pause >nul
    goto :EOF
)
echo  [OK] SAM2 installiert.

:: ────────────────────────────────────────────────────────────────────────────
::  SCHRITT 7 – GroundingDINO Checkpoint
:: ────────────────────────────────────────────────────────────────────────────
echo.
echo  [7/9] GroundingDINO Checkpoint laden ...
echo  ────────────────────────────────────────────────────────────────

if exist "checkpoints\groundingdino_swint_ogc.pth" (
    echo  [OK] GroundingDINO Checkpoint bereits vorhanden.
    goto :GDINO_CONFIG
)

mkdir checkpoints 2>nul
echo  [DOWNLOAD] groundingdino_swint_ogc.pth (~700 MB) ...
curl -L --progress-bar -o "checkpoints\groundingdino_swint_ogc.pth" ^
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
if errorlevel 1 (
    echo  [FEHLER] Download fehlgeschlagen. Internetverbindung pruefen.
    echo.
    echo  Druecke Enter zum Schliessen.
    pause >nul
    goto :EOF
)
echo  [OK] GroundingDINO Checkpoint geladen.

:GDINO_CONFIG
if exist "checkpoints\GroundingDINO_SwinT_OGC.py" (
    echo  [OK] GroundingDINO Config bereits vorhanden.
    goto :SAM2_STEP
)
echo  [DOWNLOAD] GroundingDINO Config ...
curl -L --progress-bar -o "checkpoints\GroundingDINO_SwinT_OGC.py" ^
    "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
if errorlevel 1 (
    echo  [FEHLER] Config-Download fehlgeschlagen.
    echo.
    echo  Druecke Enter zum Schliessen.
    pause >nul
    goto :EOF
)
echo  [OK] GroundingDINO Config geladen.

:: ────────────────────────────────────────────────────────────────────────────
::  SCHRITT 8 – SAM2 Checkpoint
:: ────────────────────────────────────────────────────────────────────────────
:SAM2_STEP
echo.
echo  [8/9] SAM2.1 Checkpoint laden ...
echo  ────────────────────────────────────────────────────────────────

if exist "checkpoints\sam2.1_hiera_large.pt" (
    echo  [OK] SAM2.1 Checkpoint bereits vorhanden.
    goto :SETUP_DONE
)

echo  [DOWNLOAD] sam2.1_hiera_large.pt (~900 MB) ...
echo             Kein Login noetig – direkt von Meta CDN.
echo.
curl -L --progress-bar -o "checkpoints\sam2.1_hiera_large.pt" ^
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
if errorlevel 1 (
    echo  [FEHLER] SAM2 Checkpoint Download fehlgeschlagen.
    echo.
    echo  Druecke Enter zum Schliessen.
    pause >nul
    goto :EOF
)
echo  [OK] SAM2.1 Checkpoint geladen.

:SETUP_DONE
:: Setup-Marke setzen
echo done > .setup_complete
echo.
echo  ================================================================
echo   Setup abgeschlossen! Naechster Start ueberspringt Installation.
echo  ================================================================
echo.

:: ────────────────────────────────────────────────────────────────────────────
::  SCHRITT 9 – Starten
:: ────────────────────────────────────────────────────────────────────────────
:LAUNCH
title CrackDetect
echo  Starte CrackDetect ...
echo  ────────────────────────────────────────────────────────────────
echo.
echo  [INFO] Die App oeffnet sich in Ihrem Browser unter:
echo         http://127.0.0.1:7861
echo.
echo  [INFO] Dieses Fenster nicht schliessen waehrend die App laeuft.
echo         Zum Beenden: Strg+C druecken.
echo.

python crackdetect.py

if errorlevel 1 (
    echo.
    echo  ================================================================
    echo  [FEHLER] CrackDetect wurde mit einem Fehler beendet.
    echo          Fehlermeldung siehe oben.
    echo  ================================================================
    echo.
    echo  Haeufigstes Problem: Modelle nicht geladen.
    echo  Loesung: ".setup_complete" loeschen und start.bat erneut ausfuehren.
    echo.
    echo  Druecke Enter zum Schliessen.
    pause >nul
    goto :EOF
)

echo.
echo  [OK] CrackDetect normal beendet. Auf Wiedersehen!
echo.
endlocal
pause >nul
