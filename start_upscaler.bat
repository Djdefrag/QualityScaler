@echo off
REM This script activates the virtual environment and runs the QualityScaler script 

call .\venv\Scripts\activate.bat
.\venv\Scripts\python.exe .\QualityScaler.py
pause