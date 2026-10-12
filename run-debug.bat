@echo off
rem Runs AirTyper with a console window so you can see log messages and errors.
cd /d "%~dp0"
".venv\Scripts\python.exe" airtyper.py
pause
