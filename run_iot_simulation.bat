@echo off
title Traffic Light Recognition - IoT Simulator
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

rem Use the Python launcher if present, otherwise fall back to python on PATH
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 iot_simulation.py
) else (
    python iot_simulation.py
)
pause
