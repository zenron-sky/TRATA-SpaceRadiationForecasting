"""
(TRATA) - End-to-End Pipeline Runner
--------------------------------------------------
Runs the full 6-step pipeline in order:
  1. Generate synthetic data
  2. Preprocess / clean
  3. Feature engineering
  4. Train Transformer + Random Forest hybrid
  5. Validate on held-out test set
  6. Generate live forecast JSON for the dashboard

Usage:
    python run_pipeline.py

NOTE: uses sys.executable (not a hardcoded "python3") so the exact same
Python interpreter running this script is also used for every subprocess.
This avoids a common Windows issue where "python3" on PATH resolves to a
different (often package-less) Python install than the one the user
actually pip-installed dependencies into.
"""

import subprocess
import sys
import time
import os

STEPS = [
    ("Generating synthetic GOES/Wind/Dst-Kp data", "data/synthetic_data_generator.py"),
    ("Preprocessing & synchronizing data", "preprocessing/preprocess.py"),
    ("Engineering physics-informed features", "features/feature_engineering.py"),
    ("Training Transformer + Random Forest hybrid", "train.py"),
    ("Validating model on test set", "validation/validate.py"),
    ("Generating live forecast for dashboard", "forecast.py"),
]


def main():
    os.makedirs("outputs", exist_ok=True)
    start = time.time()
    for i, (desc, script) in enumerate(STEPS, 1):
        print(f"\n{'='*70}\nSTEP {i}/{len(STEPS)}: {desc}\n{'='*70}")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"FAILED at step {i}: {desc}")
            sys.exit(1)
    elapsed = time.time() - start
    print(f"\n{'='*70}\nPIPELINE COMPLETE in {elapsed:.1f}s")
    print("Next: python -m http.server 8000  then open dashboard/index.html")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
