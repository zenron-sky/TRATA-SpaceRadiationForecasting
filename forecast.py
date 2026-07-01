"""
TRATA - Forecast Generation Script (Layer 05/06: Inference + Dashboard Feed)
--------------------------------------------------------------------------------
Runs the trained Transformer -> Random Forest hybrid on the most recent
4-hour window of data and writes a single JSON file consumed directly by the
dashboard (dashboard/index.html).

Confidence: MC-DROPOUT sampled through the full hybrid pipeline. The
Transformer's dropout layers are kept ACTIVE at inference (30 stochastic
forward passes), producing 30 slightly different temporal embeddings. Each
embedding is concatenated with the current physics feature vector and run
through the (deterministic) Random Forest to get 30 point predictions per
horizon. Mean = forecast, std = confidence/uncertainty - this restores the
MC-Dropout uncertainty mechanism on top of the Transformer + Random Forest
architecture shown in the diagram.

Explainability: Random Forest feature_importances_, mapped back to named
physics variables (dynamic pressure, solar wind speed, Bz, etc). The
Transformer's embedding dimensions are reported as a single aggregate
"Temporal Pattern (Transformer)" contribution so the explainability panel
stays physically interpretable for reviewers, rather than showing 64 opaque
embedding-dimension numbers.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMNS, HORIZON_STEPS
from models.transformer_encoder import TransformerForecastEncoder
from train import SEQ_LEN
from validation.validate import classify_risk

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RISK_THRESHOLDS = {"Nominal": 0, "Elevated": 1, "Severe": 2, "Hazardous": 3}
EMBEDDING_DIM = 32  # must match TransformerForecastEncoder d_model


def build_rf_feature_names():
    """Names for the RF input vector: embedding dims (aggregated later) +
    named physics features."""
    emb_names = [f"transformer_embedding_{i}" for i in range(EMBEDDING_DIM)]
    return emb_names + list(FEATURE_COLUMNS)


def aggregate_importance(pairs):
    """Collapses all transformer_embedding_* importances into a single
    'Temporal Pattern (Transformer)' bucket; keeps named physics features
    individually. Returns a re-sorted list of (name, importance)."""
    physics_items = []
    embedding_total = 0.0
    for name, imp in pairs:
        if name.startswith("transformer_embedding_"):
            embedding_total += imp
        else:
            physics_items.append((name, imp))
    physics_items.append(("Temporal Pattern (Transformer embedding)", embedding_total))
    physics_items.sort(key=lambda kv: -kv[1])
    return physics_items


def main():
    out_dir = "./outputs"
    df = pd.read_csv(f"{out_dir}/features.csv", index_col=0, parse_dates=True)

    scaler = np.load(f"{out_dir}/scaler.npz", allow_pickle=True)
    mean, std = scaler["mean"], scaler["std"]

    transformer = TransformerForecastEncoder(
        n_features=len(FEATURE_COLUMNS), n_horizons=len(TARGET_COLUMNS)
    ).to(DEVICE)
    transformer.load_state_dict(torch.load(f"{out_dir}/transformer.pt", map_location=DEVICE))
    transformer.eval()

    rf_head = joblib.load(f"{out_dir}/random_forest_head.joblib")

    # Treat the very last SEQ_LEN rows of the dataset as the "live" window
    latest_window = df.iloc[-SEQ_LEN:]
    X_latest = ((latest_window[FEATURE_COLUMNS] - mean) / std).values.astype(np.float32)
    x_tensor = torch.tensor(X_latest, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    current_features = X_latest[-1:, :]  # current-timestep physics vector, (1, n_features)

    # ---- MC-Dropout sampling through the full hybrid pipeline ----
    N_MC_SAMPLES = 30
    mc_embeddings = transformer.get_embedding_mc_dropout(x_tensor, n_samples=N_MC_SAMPLES)
    mc_embeddings = mc_embeddings.squeeze(1).cpu().numpy()  # (n_samples, d_model)

    # For each stochastic embedding sample, run the (deterministic) Random
    # Forest to get a point prediction per horizon -> stack across samples
    mc_preds = {f"target_{h}": [] for h in HORIZON_STEPS.keys()}
    for i in range(N_MC_SAMPLES):
        X_rf_sample = np.concatenate([mc_embeddings[i:i + 1], current_features], axis=1)
        point_preds = rf_head.predict(X_rf_sample)  # {label: array([pred])}
        for label, arr in point_preds.items():
            mc_preds[label].append(arr[0])

    rf_results = {}  # {label: (mean, std)} matching the (mean_array, std_array) shape used below
    for label, vals in mc_preds.items():
        vals = np.array(vals)
        rf_results[label] = (np.array([vals.mean()]), np.array([vals.std()]))

    # Feature importances still computed from the fitted Random Forest
    # (importance is a property of the trained forest, independent of the
    # MC-Dropout sampling used here for uncertainty at inference time)
    rf_feature_names = build_rf_feature_names()
    importances = rf_head.feature_importances(rf_feature_names)

    current_row = df.iloc[-1]
    current_flux = float(10 ** current_row["log_electron_flux"])
    current_risk = classify_risk(current_row["log_electron_flux"])

    forecasts = {}
    max_risk_level = RISK_THRESHOLDS[current_risk]
    for label in HORIZON_STEPS.keys():
        target_col = f"target_{label}"
        mean_pred, std_pred = rf_results[target_col]
        log_flux_pred = float(mean_pred[0])
        uncertainty = float(std_pred[0])
        flux_pred = float(10 ** log_flux_pred)

        # confidence: inverse of normalized MC-Dropout prediction std, scaled
        # 0-100%, capped at 99% (no forecast is ever claimed to be perfectly
        # certain, even when persistence dominates a short horizon)
        confidence = float(max(5.0, min(99.0, (1.0 - uncertainty / 1.0) * 100)))

        risk = classify_risk(log_flux_pred)
        max_risk_level = max(max_risk_level, RISK_THRESHOLDS[risk])
        forecasts[label] = {
            "horizon_label": {"45min": "45 minutes", "6hr": "6 hours", "12hr": "12 hours"}[label],
            "predicted_flux_2mev": round(flux_pred, 1),
            "predicted_log10_flux": round(log_flux_pred, 3),
            "risk_level": risk,
            "confidence_pct": round(confidence, 1),
            "uncertainty_log10": round(uncertainty, 3),
        }

    overall_risk = [k for k, v in RISK_THRESHOLDS.items() if v == max_risk_level][0]
    alert = overall_risk in ("Severe", "Hazardous")

    # Explainability: use the 12-hour horizon's RF importances as the
    # representative "what's driving the forecast" panel. The 45-min horizon
    # is dominated by persistence (current flux predicts near-term flux
    # almost perfectly - a well-known space weather forecasting effect), so
    # the 12-hour horizon gives a more physically meaningful view of which
    # solar wind / geomagnetic precursors the model actually relies on.
    top_features_raw = aggregate_importance(importances["target_12hr"])[:6]

    history = df.iloc[-576:]  # last 48 hours @ 5-min cadence
    history_payload = {
        "timestamps": [ts.isoformat() for ts in history.index],
        "electron_flux_2mev": [round(float(v), 1) for v in (10 ** history["log_electron_flux"])],
        "solar_wind_speed_kms": [round(float(v), 1) for v in history["solar_wind_speed_kms"]],
        "imf_bz_nt": [round(float(v), 2) for v in history["imf_bz_nt"]],
        "kp_index": [round(float(v), 2) for v in history["kp_index"]],
        "dst_index_nt": [round(float(v), 1) for v in history["dst_index_nt"]],
    }

    metrics = {}
    metrics_path = f"{out_dir}/validation_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_architecture": "Transformer + Random Forest Hybrid (TRATA)",
        "current_conditions": {
            "timestamp": current_row.name.isoformat(),
            "electron_flux_2mev": round(current_flux, 1),
            "risk_level": current_risk,
            "solar_wind_speed_kms": round(float(current_row["solar_wind_speed_kms"]), 1),
            "solar_wind_density_pcc": round(float(current_row["solar_wind_density_pcc"]), 2),
            "imf_bz_nt": round(float(current_row["imf_bz_nt"]), 2),
            "kp_index": round(float(current_row["kp_index"]), 2),
            "dst_index_nt": round(float(current_row["dst_index_nt"]), 1),
        },
        "forecasts": forecasts,
        "overall_alert": {
            "active": alert,
            "max_risk_level": overall_risk,
            "message": (
                f"RADIATION ALERT: conditions trending toward {overall_risk} levels"
                if alert else "No alert. Conditions within nominal/elevated bounds."
            ),
        },
        "explainability": {
            "method": "Random Forest feature importance + MC-Dropout confidence (Transformer embedding dims aggregated)",
            "top_contributing_variables": [
                {"feature": f, "importance_pct": round(float(imp) * 100, 1)} for f, imp in top_features_raw
            ]
        },
        "validation_metrics": metrics,
        "history": history_payload,
    }

    out_json = f"{out_dir}/forecast_output.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Forecast written -> {out_json}")
    print(json.dumps(payload["forecasts"], indent=2))
    print("Overall alert:", payload["overall_alert"])


if __name__ == "__main__":
    main()
