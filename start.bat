@echo off
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"
echo Starting Multi-TTS server...
python -m app.main %*
pause