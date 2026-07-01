"""
TRATA - Preprocessing Module
-----------------------------------
Mimics Phase 1 & Phase 3 of the problem statement:
  - Reading / standardizing multi-source data (here: one synced dataframe
    standing in for parsed CDF outputs from GOES / Wind / GRASP)
  - Handling missing values, removing spikes, interpolating gaps,
    synchronizing timestamps across sources
"""

import numpy as np
import pandas as pd


def remove_outlier_spikes(series: pd.Series, window: int = 12, z_thresh: float = 5.0) -> pd.Series:
    """Rolling z-score based despiking (robust to single-point sensor glitches)."""
    rolling_med = series.rolling(window, center=True, min_periods=1).median()
    rolling_std = series.rolling(window, center=True, min_periods=1).std().replace(0, np.nan)
    z = (series - rolling_med).abs() / rolling_std
    cleaned = series.copy()
    cleaned[z > z_thresh] = np.nan
    return cleaned


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # 1. Despike sensor glitches
    for col in numeric_cols:
        df[col] = remove_outlier_spikes(df[col])

    # 2. Interpolate small data gaps (linear, time-aware)
    df[numeric_cols] = df[numeric_cols].interpolate(method="time", limit=12, limit_direction="both")

    # 3. Drop any rows that still contain NaNs (large unrecoverable gaps)
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"Preprocessing: dropped {before - after} unrecoverable rows out of {before}")

    # 4. Resample / synchronize to a strict regular cadence (handles
    #    multi-instrument timestamp misalignment)
    df = df.resample("5min").mean().interpolate(method="time", limit=3)
    df = df.dropna()

    return df


if __name__ == "__main__":
    raw = pd.read_csv("./outputs/synthetic_raw_data.csv")
    clean = preprocess(raw)
    out_path = "./outputs/clean_data.csv"
    clean.to_csv(out_path)
    print(f"Cleaned data saved -> {out_path} ({len(clean)} rows)")
