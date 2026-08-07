# =============================================================================
# Multi-TTS Server - adaptive installer
# Detects Python + NVIDIA GPU and picks the right PyTorch build.
#   - RTX 50xx (Blackwell, cc>=12)  -> torch 2.7.0 + cu128
#   - RTX 30xx/40xx (Ampere/Ada)    -> torch 2.6.0 + cu126
#   - RTX 20xx (Turing)             -> torch 2.6.0 + cu118
#   - GTX 10xx/16xx (Pascal/Volta)  -> torch 2.6.0 + cu118
#   - no NVIDIA GPU                 -> torch 2.6.0 (CPU)
# =============================================================================
param(
    [switch]$Recreate,        # delete and recreate the virtualenv
    [switch]$NoGpu,           # force CPU-only PyTorch
    [switch]$SkipTorch,       # skip PyTorch install (already present)
    [switch]$SkipRequirements # skip pip install -r requirements.txt
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Write-Step([string]$msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Fail([string]$msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# -----------------------------------------------------------------------------
# 1. Pick Python (prefer 3.11-3.13, newest first)
# -----------------------------------------------------------------------------
Write-Step "[1/6] Selecting Python"
$PyExe = $null
foreach ($v in "3.13", "3.12", "3.11") {
    if (Get-Command "py" -ErrorAction SilentlyContinue) {
        $exe = (& py -$v -c "import sys; print(sys.executable)" 2>$null)
        if ($LASTEXITCODE -eq 0 -and $exe -and (Test-Path ($exe.Trim()))) { $PyExe = $exe.Trim(); break }
    }
}
if (-not $PyExe) {
    if (Get-Command "python" -ErrorAction SilentlyContinue) {
        $ver = & python -c "import sys; print(sys.version_info[:2])" 2>$null
        if ($ver -match "\((3), (1[123])\)" -and [int]$Matches[2] -le 13) {
            $PyExe = (& python -c "import sys; print(sys.executable)" 2>$null).Trim()
        }
    }
}
if (-not $PyExe) {
    if (Get-Command "py" -ErrorAction SilentlyContinue) {
        $ver = (& py -3.14 -c "import sys; print(sys.version.split()[0])" 2>$null)
        if ($LASTEXITCODE -eq 0) {
            Fail "Python $ver is too new. Install Python 3.11/3.12/3.13 (PyTorch/qwen-tts wheels missing for 3.14)."
        }
    }
    Fail "No supported Python found. Install Python 3.11/3.12/3.13 and rerun setup.bat"
}
$pyVer = & $PyExe -c "import sys; print(sys.version.split()[0])"
Write-Host "Using Python: $pyVer  ($PyExe)"

# -----------------------------------------------------------------------------
# 2. Virtualenv
# -----------------------------------------------------------------------------
Write-Step "[2/6] Virtual environment"
$Venv = Join-Path $Root ".venv"
if ($Recreate -and (Test-Path $Venv)) {
    Remove-Item -Recurse -Force $Venv
}
if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    & $PyExe -m venv $Venv
    if ($LASTEXITCODE -ne 0) { Fail "Failed to create venv" }
}
$Pip = Join-Path $Venv "Scripts\python.exe"
Write-Host "venv ready: $Venv"

# -----------------------------------------------------------------------------
# 3. pip + dotenv
# -----------------------------------------------------------------------------
Write-Step "[3/6] Upgrading pip"
& $Pip -m pip install --timeout 300 --retries 20 --upgrade pip
if ($LASTEXITCODE -ne 0) { Fail "pip upgrade failed" }
& $Pip -m pip install --timeout 300 --retries 20 python-dotenv
if ($LASTEXITCODE -ne 0) { Fail "python-dotenv install failed" }

# -----------------------------------------------------------------------------
# 4. Detect GPU and pick torch build
# -----------------------------------------------------------------------------
Write-Step "[4/6] Detecting GPU"
$GpuName = $null
$Cc = $null
if (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue) {
    try {
        $q = & nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>$null
        if ($q) {
            $parts = ($q -split ",")
            $GpuName = $parts[0].Trim()
            $Cc = $parts[1].Trim()
        }
    } catch {}
}

$torchVer = $null
$torchAudioVer = $null
$index = $null
$mode = $null

if ($SkipTorch) {
    Write-Host "Skipping torch install (-SkipTorch)."
} elseif ($NoGpu) {
    $mode = "CPU (forced)"
    $torchVer = "2.6.0"
    $torchAudioVer = "2.6.0"
    $index = "https://download.pytorch.org/whl/cpu"
} elseif ($null -eq $Cc) {
    $mode = "CPU (no NVIDIA GPU detected)"
    $torchVer = "2.6.0"
    $torchAudioVer = "2.6.0"
    $index = "https://download.pytorch.org/whl/cpu"
} else {
    $ccMajor = [int]($Cc.Split(".")[0])
    if ($ccMajor -ge 12) {
        $mode = "GPU $Cc (Blackwell, RTX 50xx)"
        $torchVer = "2.7.0+cu128"; $torchAudioVer = "2.7.0+cu128"
        $index = "https://download.pytorch.org/whl/cu128"
    } elseif ($ccMajor -ge 8) {
        $mode = "GPU $Cc (Ampere/Ada/Hopper)"
        $torchVer = "2.6.0+cu126"; $torchAudioVer = "2.6.0+cu126"
        $index = "https://download.pytorch.org/whl/cu126"
    } elseif ($ccMajor -ge 7) {
        $mode = "GPU $Cc (Turing, RTX 20xx)"
        $torchVer = "2.6.0+cu118"; $torchAudioVer = "2.6.0+cu118"
        $index = "https://download.pytorch.org/whl/cu118"
    } else {
        $mode = "GPU $Cc (older NVIDIA)"
        $torchVer = "2.6.0+cu118"; $torchAudioVer = "2.6.0+cu118"
        $index = "https://download.pytorch.org/whl/cu118"
    }
}
Write-Host "GPU: $GpuName  (compute capability: $Cc)"
if ($mode) {
    Write-Host "Torch plan: $mode -> torch==$torchVer"
}

# -----------------------------------------------------------------------------
# 5. Install torch + requirements
# -----------------------------------------------------------------------------
if (-not $SkipTorch -and $torchVer) {
    Write-Step "[5/6] Installing PyTorch ($mode)"
    $needTorch = $true
    try {
        $cur = & $Pip -c "import torch; print(torch.__version__)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $cur -eq $torchVer) {
            Write-Host "torch==$cur already installed, skipping."
            $needTorch = $false
        }
    } catch {}
    if ($needTorch) {
        & $Pip -m pip install --timeout 300 --retries 20 "torch==$torchVer" "torchaudio==$torchAudioVer" --index-url $index
        if ($LASTEXITCODE -ne 0) { Fail "PyTorch install failed" }
    }
} else {
    Write-Step "[5/6] Installing PyTorch (skipped)"
}

Write-Step "[6/6] Installing requirements"
if ($SkipRequirements) {
    Write-Host "Skipping requirements (-SkipRequirements)."
} else {
    & $Pip -m pip install --timeout 300 --retries 20 -r (Join-Path $Root "requirements.txt")
    if ($LASTEXITCODE -ne 0) { Fail "requirements install failed" }
}

# -----------------------------------------------------------------------------
# Verify
# -----------------------------------------------------------------------------
Write-Step "Verifying CUDA"
& $Pip -c "import torch; print('PyTorch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
if ($LASTEXITCODE -ne 0) { Fail "torch import failed" }

Write-Host "`nSetup complete. Start with: start.bat   (or: start_cpu.bat)" -ForegroundColor Green
