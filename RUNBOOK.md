# 🚀 JUDGE'S RUNBOOK (30-Second Setup)

> **⚠️ IMPORTANT WINDOWS NOTE:** If using PowerShell, always type `.\`
> before batch files (e.g., `.\run_pipeline.bat`). Just typing
> `run_pipeline.bat` will fail!

------------------------------------------------------------------------

## 📋 Prerequisites (Check once)

-   **Python 3.8+** installed. (Check with `python --version` in the
    terminal).
-   Internet connection (to download required libraries).

------------------------------------------------------------------------

## ⚡ Step-by-Step (Strict Order)

### 1. Open Terminal in this folder

``` powershell
cd C:\path\to\TRata
```

### 2. Install Dependencies (1 minute)

``` powershell
pip install -r requirements.txt
```

*(If this fails, try `python -m pip install -r requirements.txt`)*

### 3. Run the Full Pipeline (4--5 minutes)

Choose **ONE**:

**Option A (Windows)**

``` powershell
.\run_pipeline.bat
```

**Option B (Fallback)**

``` powershell
python run_pipeline.py
```

Wait until you see **Pipeline complete!**

### 4. Launch the Dashboard

``` powershell
.\start_dashboard.bat
```

Or:

``` powershell
python -m http.server 8000
```

### 5. Open your Browser

`http://localhost:8000/dashboard/index.html`

------------------------------------------------------------------------

## ❌ Troubleshooting

  ---------------------------------------------------------------------------------------------------------------
  Error                                               Cause                   Fix
  --------------------------------------------------- ----------------------- -----------------------------------
  `'run_pipeline.bat' is not recognized`              PowerShell              Use `.\run_pipeline.bat`

  `Cannot load because running scripts is disabled`   Execution Policy        Use `python run_pipeline.py`

  `ModuleNotFoundError`                               Missing packages        Run
                                                                              `pip install -r requirements.txt`

  `outputs folder missing`                            Pipeline not run        Run `run_pipeline.py` first
  ---------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## ✅ Success Check

-   3 forecast cards (45-min, 6-hr, 12-hr)
-   Solar wind & electron flux charts visible
-   Alert banner visible

**You are good to go! 🏆**
