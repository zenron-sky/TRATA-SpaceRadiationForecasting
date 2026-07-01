"""
TRATA - Feature Engineering Module
------------------------------------------
Implements Phase 4: physics-informed feature extraction.

Adds:
  - Log-transformed electron flux (flux spans orders of magnitude)
  - Rolling statistics (mean/std) over multiple windows -> captures
    short-term solar wind disturbances (the "CNN sees spikes" idea)
  - Dynamic pressure proxy (~ density * speed^2)
  - Lagged electron flux features (autoregressive signal)
  - Cyclical time-of-day / day-of-year encodings
  - Target columns for each forecast horizon (45 min / 6 hr / 12 hr)
"""

import numpy as np
import pandas as pd

FREQ_MINUTES = 5
HORIZONS_MIN = {"45min": 45, "6hr": 6 * 60, "12hr": 12 * 60}
HORIZON_STEPS = {k: v // FREQ_MINUTES for k, v in HORIZONS_MIN.items()}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Log transform flux (orders-of-magnitude target -> stabilizes training)
    df["log_electron_flux"] = np.log10(df["electron_flux_2mev"].clip(lower=1))

    # Dynamic pressure proxy: P ~ n * v^2 (solar wind ram pressure driver)
    df["dynamic_pressure"] = df["solar_wind_density_pcc"] * (df["solar_wind_speed_kms"] ** 2) * 1.6726e-6

    # Rolling statistics capture short-term disturbances (CNN receptive field)
    for window, label in [(6, "30min"), (36, "3hr"), (144, "12hr")]:
        df[f"speed_roll_mean_{label}"] = df["solar_wind_speed_kms"].rolling(window, min_periods=1).mean()
        df[f"speed_roll_std_{label}"] = df["solar_wind_speed_kms"].rolling(window, min_periods=1).std().fillna(0)
        df[f"bz_roll_min_{label}"] = df["imf_bz_nt"].rolling(window, min_periods=1).min()

    # Autoregressive lag features of the target itself
    for lag_steps, label in [(12, "1hr"), (72, "6hr")]:
        df[f"flux_lag_{label}"] = df["log_electron_flux"].shift(lag_steps)

    # Cyclical time encodings
    minute_of_day = df.index.hour * 60 + df.index.minute
    df["time_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    df["time_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)

    # Forecast targets: future log flux at each horizon
    for label, steps in HORIZON_STEPS.items():
        df[f"target_{label}"] = df["log_electron_flux"].shift(-steps)

    df = df.dropna()
    return df


FEATURE_COLUMNS = [
    "solar_wind_speed_kms", "solar_wind_density_pcc", "imf_bz_nt", "imf_bt_nt",
    "dst_index_nt", "kp_index", "dynamic_pressure",
    "speed_roll_mean_30min", "speed_roll_std_30min", "bz_roll_min_30min",
    "speed_roll_mean_3hr", "speed_roll_std_3hr", "bz_roll_min_3hr",
    "speed_roll_mean_12hr", "speed_roll_std_12hr", "bz_roll_min_12hr",
    "flux_lag_1hr", "flux_lag_6hr",
    "time_sin", "time_cos",
    "log_electron_flux",
]
TARGET_COLUMNS = [f"target_{k}" for k in HORIZON_STEPS]


if __name__ == "__main__":
    clean = pd.read_csv("./outputs/clean_data.csv", index_col=0, parse_dates=True)
    feat = engineer_features(clean)
    out_path = "./outputs/features.csv"
    feat.to_csv(out_path)
    print(f"Engineered {len(FEATURE_COLUMNS)} features, {len(TARGET_COLUMNS)} targets -> {out_path}")
    print(f"Rows after feature engineering: {len(feat)}")
