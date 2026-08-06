@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/5] Creating virtual environment (.venv)...
    python -m venv .venv
    if errorlevel 1 goto :error
)
call ".venv\Scripts\activate.bat"

echo [2/5] Upgrading pip, installing dotenv...
python -m pip install --upgrade pip
python -m pip install python-dotenv
if errorlevel 1 goto :error

echo [3/5] Installing PyTorch (CUDA 12.6 pinned for py3.13)...
REM torch 2.6.0+cu126 is known-good on py3.13 win (CUDA build exists, no torchcodec needed).
python -m pip install "torch==2.6.0+cu126" "torchaudio==2.6.0+cu126" --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto :error

echo [4/5] Installing dependencies (qwen-tts)...
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo [5/5] Verifying CUDA...
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
if errorlevel 1 goto :error

echo.
echo Setup complete. Start with: start.bat
exit /b 0

:error
echo.
echo Setup FAILED. See messages above.
exit /b 1