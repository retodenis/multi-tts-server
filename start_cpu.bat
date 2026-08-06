@echo off
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"
echo Starting Multi-TTS server (CPU)... This is much slower than GPU.
python -m app.main --device cpu %*
pause