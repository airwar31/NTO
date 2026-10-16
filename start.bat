@echo off
chcp 65001 > nul
echo =====================================================================
echo  🌌 AstroClassify Studio - Space Object Classifier (NTO AI Olympiad)
echo =====================================================================
echo.

if not exist ".venv" (
    echo [INFO] Virtual environment not found. Checking system Python...
)

echo [INFO] Launching AstroClassify Studio interactive web demo...
python web\app.py

pause
