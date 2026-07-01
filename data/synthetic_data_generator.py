"""
AdityaNetra - Synthetic Space Weather Data Generator
------------------------------------------------------
Generates physically-plausible synthetic time series that mimic:
  - Wind spacecraft solar wind parameters (speed, density, IMF Bz)
  - Geomagnetic indices (Kp, Dst)
  - GOES >2 MeV electron flux (target variable)

The relationship encoded:
  Solar wind speed spikes + southward IMF (negative Bz) -> geomagnetic
  storms (Dst drops, Kp rises) -> radiation belt electrons respond with a
  characteristic LAG (several hours), producing the "killer electron"
  enhancement that is the forecasting target.

This lets a forecasting model learn genuine precursor -> response structure,
even though the data is synthetic.
"""

import numpy as np
import pandas as pd


def generate_synthetic_dataset(
    days: int = 90,
    freq_minutes: int = 5,
    n_storms: int = 14,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_points = int(days * 24 * 60 / freq_minutes)
    timestamps = pd.date_range("2024-01-01", periods=n_points, freq=f"{freq_minutes}min")
    t = np.arange(n_points)

    # ---------- Baseline quiet-time solar wind ----------
    solar_wind_speed = 380 + 25 * np.sin(2 * np.pi * t / (60 / freq_minutes * 24 * 27)) \
        + rng.normal(0, 8, n_points)
    solar_wind_density = 5 + 1.5 * rng.normal(0, 1, n_points)
    solar_wind_density = np.clip(solar_wind_density, 0.5, None)
    imf_bz = rng.normal(0, 1.2, n_points)  # nT, +/- fluctuation around 0
    imf_bt = np.abs(rng.normal(5, 1.5, n_points))

    dst = -10 + rng.normal(0, 3, n_points)
    kp = np.clip(2 + rng.normal(0, 0.5, n_points), 0, 9)

    # baseline (quiet) electron flux, log-scale-ish behaviour
    electron_flux = 1e3 * np.ones(n_points) + rng.normal(0, 50, n_points)

    # ---------- Inject CME / storm events ----------
    storm_starts = rng.choice(np.arange(int(0.05 * n_points), int(0.9 * n_points)), n_storms, replace=False)
    storm_starts.sort()

    response_lag_steps = int(6 * 60 / freq_minutes)  # ~6h electron response lag (variable below)

    for start in storm_starts:
        duration = rng.integers(int(4 * 60 / freq_minutes), int(20 * 60 / freq_minutes))  # 4-20h CME passage
        end = min(start + duration, n_points - 1)
        ramp = np.linspace(0, 1, max(end - start, 1))

        # CME shock: speed jump, density spike, southward Bz turning
        peak_speed_gain = rng.uniform(250, 650)
        solar_wind_speed[start:end] += peak_speed_gain * np.sin(np.pi * ramp)
        solar_wind_density[start:end] += rng.uniform(5, 20) * np.exp(-3 * ramp)
        bz_min = -rng.uniform(8, 25)
        imf_bz[start:end] += bz_min * np.sin(np.pi * ramp)
        imf_bt[start:end] += np.abs(bz_min) * 0.6

        # Geomagnetic response (near simultaneous with solar wind driver)
        dst_min = -rng.uniform(50, 250)
        dst[start:end] += dst_min * np.sin(np.pi * ramp)
        kp[start:end] += rng.uniform(3, 6) * np.sin(np.pi * ramp)

        # Electron flux response: DELAYED relative to the driver (lag = the
        # whole point of forecasting). Belt enhancement can persist days.
        lag = int(rng.uniform(0.5, 1.5) * response_lag_steps)
        e_start = min(start + lag, n_points - 1)
        e_duration = rng.integers(int(12 * 60 / freq_minutes), int(72 * 60 / freq_minutes))
        e_end = min(e_start + e_duration, n_points)
        if e_end > e_start:
            e_ramp_up = np.linspace(0, 1, max(int((e_end - e_start) * 0.3), 1))
            e_ramp_down = np.linspace(1, 0, (e_end - e_start) - len(e_ramp_up))
            e_shape = np.concatenate([e_ramp_up, e_ramp_down])
            flux_gain = rng.uniform(8, 60)  # multiplicative enhancement factor
            electron_flux[e_start:e_end] *= (1 + flux_gain * e_shape)

    kp = np.clip(kp, 0, 9)
    solar_wind_density = np.clip(solar_wind_density, 0.3, None)
    electron_flux = np.clip(electron_flux, 10, None)

    # measurement noise / occasional small data dropouts (simulate real instruments)
    dropout_mask = rng.random(n_points) < 0.002
    df = pd.DataFrame({
        "timestamp": timestamps,
        "solar_wind_speed_kms": solar_wind_speed,
        "solar_wind_density_pcc": solar_wind_density,
        "imf_bz_nt": imf_bz,
        "imf_bt_nt": imf_bt,
        "dst_index_nt": dst,
        "kp_index": kp,
        "electron_flux_2mev": electron_flux,
    })
    df.loc[dropout_mask, ["solar_wind_speed_kms", "solar_wind_density_pcc"]] = np.nan

    return df


if __name__ == "__main__":
    df = generate_synthetic_dataset(days=90)
    out_path = "./outputs/synthetic_raw_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df.describe())
