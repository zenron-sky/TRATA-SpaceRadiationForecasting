#  (TRATA) - Windows PowerShell Quick Start
# Transformer + Random Forest Hybrid Pipeline
# Run from the TRATA folder: powershell -ExecutionPolicy Bypass -File run_pipeline.ps1

Write-Host ""
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host " (TRATA) - Space Radiation Forecasting System" -ForegroundColor Cyan
Write-Host "Transformer + Random Forest Hybrid" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host ""

try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[X] ERROR: Python not found. Please install Python 3.8+ from python.org" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[1/8] Installing dependencies (torch, numpy, pandas, scipy, scikit-learn, joblib)..." -ForegroundColor Yellow
python -m pip install --quiet torch numpy pandas scipy scikit-learn joblib
if ($LASTEXITCODE -ne 0) {
    Write-Host "[X] ERROR: pip install failed" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[OK] Dependencies installed" -ForegroundColor Green

if (-not (Test-Path "outputs")) { New-Item -ItemType Directory -Path "outputs" | Out-Null }

$steps = @(
    @{n="2/8"; d="Generating synthetic GOES/Wind/Dst-Kp data"; f="data\synthetic_data_generator.py"},
    @{n="3/8"; d="Preprocessing and synchronizing data";        f="preprocessing\preprocess.py"},
    @{n="4/8"; d="Engineering physics-informed features";       f="features\feature_engineering.py"},
    @{n="5/8"; d="Training Transformer + Random Forest hybrid"; f="train.py"},
    @{n="6/8"; d="Validating model on test set";                f="validation\validate.py"},
    @{n="7/8"; d="Generating live forecast for dashboard";      f="forecast.py"}
)

foreach ($step in $steps) {
    Write-Host ""
    Write-Host "[$($step.n)] $($step.d)..." -ForegroundColor Yellow
    python $step.f
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[X] ERROR: step failed ($($step.f))" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""
Write-Host "[8/8] Pipeline complete." -ForegroundColor Green
Write-Host ""
Write-Host "=========================================================================" -ForegroundColor Green
Write-Host "PIPELINE COMPLETE" -ForegroundColor Green
Write-Host "=========================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run: start_dashboard.bat  (or: python -m http.server 8000)" -ForegroundColor White
Write-Host "  2. Open http://localhost:8000/dashboard/index.html" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to exit"
