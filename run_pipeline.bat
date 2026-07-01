@echo off
REM AdityaNetra (TRATA) - Windows Quick Start
REM Transformer + Random Forest Hybrid Pipeline
REM Run this from the adityanetra folder: run_pipeline.bat

echo.
echo =========================================================================
echo AdityaNetra (TRATA) - Space Radiation Forecasting System
echo Transformer + Random Forest Hybrid
echo =========================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo [1/8] Installing dependencies...
python -m pip install --quiet torch numpy pandas scipy scikit-learn joblib
if errorlevel 1 (
    echo ERROR: pip install failed
    pause
    exit /b 1
)

if not exist outputs mkdir outputs

echo [2/8] Generating synthetic GOES/Wind/Dst-Kp data...
python data\synthetic_data_generator.py
if errorlevel 1 goto :error

echo [3/8] Preprocessing and synchronizing data...
python preprocessing\preprocess.py
if errorlevel 1 goto :error

echo [4/8] Engineering physics-informed features...
python features\feature_engineering.py
if errorlevel 1 goto :error

echo [5/8] Training Transformer + Random Forest hybrid (this takes ~2-3 min)...
python train.py
if errorlevel 1 goto :error

echo [6/8] Validating model on test set...
python validation\validate.py
if errorlevel 1 goto :error

echo [7/8] Generating live forecast for dashboard...
python forecast.py
if errorlevel 1 goto :error

echo [8/8] Pipeline complete.
echo.
echo =========================================================================
echo PIPELINE COMPLETE
echo =========================================================================
echo.
echo Next steps:
echo   1. Run: start_dashboard.bat
echo   2. Open http://localhost:8000/dashboard/index.html
echo.
pause
exit /b 0

:error
echo.
echo ERROR: pipeline step failed. See message above.
pause
exit /b 1
