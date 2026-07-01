# TRATA on Windows — Setup Guide

## Prerequisites
- **Python 3.8+** installed (download from https://www.python.org/downloads/)
  - **IMPORTANT:** During installation, check the box "Add Python to PATH"
- A modern browser (Chrome, Edge, Firefox)

## Step-by-step

### 1. Extract the zip file
Right-click the zip → Extract All → choose a location (e.g., Desktop)

### 2. Open a terminal in the project folder
In Windows Explorer, navigate into `TRATA/` folder, then:
- **Windows 11:** right-click empty space → "Open in Terminal" or "Open in PowerShell"
- **Windows 10:** right-click empty space → "Open PowerShell window here"
- **Older:** hold Shift, right-click empty space → "Open command window here"

You should see a prompt like:
```
C:\Users\YourName\Desktop\TRATA>
```

### 3. Run the pipeline

**Option A — Batch file (recommended for beginners):**
```cmd
run_pipeline.bat

OR

.\run_pipeline.bat

```

**Option B — PowerShell:**
```powershell
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1
```

Wait 2–3 minutes. You should see:
```
======================================================================
STEP 1/6: Generating synthetic GOES/Wind/Dst-Kp data
======================================================================
Generated 25920 rows -> C:\...\outputs\synthetic_raw_data.csv
...
======================================================================
PIPELINE COMPLETE in XX.Xs
======================================================================
```

### 4. View the dashboard

**Option A — Use the dashboard batch file:**
```cmd
start_dashboard.bat
or
.\start_dashboard.bat
```

**Option B — Manual:**
```cmd
python -m http.server 8000
```

Your browser should automatically open. If not, go to:
```
http://localhost:8000/dashboard/index.html
```

To stop the server, press **Ctrl+C** in the terminal window.

## Troubleshooting

### "Python not found" or "'python' is not recognized as an internal or external command"
- **Fix:** Reinstall Python and **make sure to check "Add Python to PATH"** during installation
- Restart your terminal after reinstalling
- Verify: type `python --version` and press Enter

### "ModuleNotFoundError: No module named 'torch'"
- This means the `pip install` step failed
- Try manually running:
  ```cmd
  python -m pip install torch numpy pandas scipy scikit-learn joblib
  ```
- If that fails, try:
  ```cmd
  python -m pip install --upgrade pip
  python -m pip install torch numpy pandas scipy scikit-learn joblib
  ```

### "Address already in use" error when starting the dashboard
- Another process is using port 8000
- **Quick fix:** Use a different port:
  ```cmd
  python -m http.server 9000
  ```
  Then open `http://localhost:9000/dashboard/index.html`

### Dashboard loads but shows "File Fallback" message
- This is expected when running `index.html` locally
- Click the file picker button and select `outputs/forecast_output.json`
- Or use the `start_dashboard.bat` / `python -m http.server` approach above (which avoids this)

### Pipeline runs but model training is very slow
- This is normal on CPU — it's training a neural network
- Wait 2–3 minutes, it will complete
- If you have NVIDIA GPU + CUDA, PyTorch will automatically use it and be much faster

## What the pipeline does

| Step | Time | Output |
|---|---|---|
| 1. Generate synthetic data | 10s | `outputs/synthetic_raw_data.csv` (25k rows) |
| 2. Preprocess (clean, sync) | 5s | `outputs/clean_data.csv` |
| 3. Feature engineering | 10s | `outputs/features.csv` |
| 4. Train Transformer + Random Forest hybrid model | 60s | `outputs/model.pt` (trained weights) |
| 5. Validate on test set | 15s | `outputs/validation_metrics.json` |
| 6. Generate forecast | 10s | `outputs/forecast_output.json` (dashboard data) |

**Total:** ~2–3 minutes on CPU

## File structure after running
```
TRATA/
├── outputs/
│   ├── forecast_output.json          ← dashboard reads this
│   ├── model.pt                      ← trained neural network
│   ├── scaler.npz                    ← feature normalization params
│   ├── validation_metrics.json       ← performance stats
│   └── (intermediate CSVs deleted)
├── dashboard/
│   └── index.html                    ← open this in browser
├── run_pipeline.bat                  ← run this first
└── start_dashboard.bat               ← run this second
```

## Next steps (after the hackathon)
To use **real satellite data** instead of synthetic:
1. Replace `data/synthetic_data_generator.py` with a CDF file reader
   - Use `cdflib` package: `pip install cdflib`
   - Parse GOES, Wind, GRASP CDF files
   - Output a DataFrame with columns: `timestamp, solar_wind_speed_kms, solar_wind_density_pcc, imf_bz_nt, imf_bt_nt, dst_index_nt, kp_index, electron_flux_2mev`
2. Run `run_pipeline.py` as usual — everything else remains unchanged

## Support
If you encounter issues, check:
1. Python version: `python --version` (should be 3.8+)
2. pip works: `python -m pip --version`
3. Dependencies installed: `pip list | findstr torch`
