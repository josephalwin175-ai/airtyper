@echo off
rem One-time setup: creates the Python environment, downloads the hand model and makes the shortcuts.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating Python 3.10 environment...
    py -3.10 -m venv .venv || (echo Python 3.10 is required: https://www.python.org/downloads/ & pause & exit /b 1)
)
".venv\Scripts\python.exe" -m pip install --upgrade pip -q
".venv\Scripts\python.exe" -m pip install -r requirements.txt -q || (pause & exit /b 1)
if not exist "models\hand_landmarker.task" (
    echo Downloading hand-tracking model...
    if not exist models mkdir models
    powershell -NoProfile -Command "Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task' -OutFile 'models\hand_landmarker.task'"
)
if not exist "assets\airtyper.ico" ".venv\Scripts\python.exe" make_icon.py
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0create_shortcuts.ps1"
echo.
echo Done. Open AirTyper from the desktop icon or the Start menu.
pause
