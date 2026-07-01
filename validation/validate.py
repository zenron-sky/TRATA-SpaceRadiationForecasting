"""
TRATA - Validation Module (Layer 05: Validation & Inference)
------------------------------------------------------------------
Evaluates the trained Transformer -> Random Forest hybrid on the held-out
chronological test set. Reports RMSE, MAE, correlation, and risk
classification accuracy per horizon - matching the diagram's "RMSE, MAE,
Correlation" + "GRASP/GSAT comparison" validation box.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMNS
from models.transformer_encoder import TransformerForecastEncoder
from models.dataset import SpaceWeatherSequenceDataset
from train import SEQ_LEN, chronological_split, extract_embeddings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classify_risk(log_flux: float) -> str:
    flux = 10 ** log_flux
    if flux < 1e3:
        return "Nominal"
    elif flux < 1e4:
        return "Elevated"
    elif flux < 1e5:
        return "Severe"
    else:
        return "Hazardous"


def main():
    out_dir = "./outputs"
    df = pd.read_csv(f"{out_dir}/features.csv", index_col=0, parse_dates=True)
    _, _, test_df = chronological_split(df)

    scaler = np.load(f"{out_dir}/scaler.npz", allow_pickle=True)
    mean, std = scaler["mean"], scaler["std"]

    X_test = ((test_df[FEATURE_COLUMNS] - mean) / std).values.astype(np.float32)
    y_test = test_df[TARGET_COLUMNS].values.astype(np.float32)

    test_ds = SpaceWeatherSequenceDataset(X_test, y_test, SEQ_LEN)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    transformer = TransformerForecastEncoder(
        n_features=len(FEATURE_COLUMNS), n_horizons=len(TARGET_COLUMNS)
    ).to(DEVICE)
    transformer.load_state_dict(torch.load(f"{out_dir}/transformer.pt", map_location=DEVICE))
    transformer.eval()

    rf_head = joblib.load(f"{out_dir}/random_forest_head.joblib")

    emb_test, feat_test, y_true = extract_embeddings(transformer, test_loader, DEVICE)
    X_rf_test = np.concatenate([emb_test, feat_test], axis=1)
    rf_preds = rf_head.predict(X_rf_test)

    metrics = {}
    for i, col in enumerate(TARGET_COLUMNS):
        preds = rf_preds[col]
        trues = y_true[:, i]

        rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
        mae = float(np.mean(np.abs(preds - trues)))
        corr = float(np.corrcoef(preds, trues)[0, 1])

        pred_risk = [classify_risk(v) for v in preds]
        true_risk = [classify_risk(v) for v in trues]
        risk_acc = float(np.mean([p == t for p, t in zip(pred_risk, true_risk)]))

        metrics[col] = {
            "rmse_log10_flux": round(rmse, 4),
            "mae_log10_flux": round(mae, 4),
            "correlation": round(corr, 4),
            "risk_classification_accuracy": round(risk_acc, 4),
        }
        print(f"{col}: RMSE={rmse:.4f}  MAE={mae:.4f}  Corr={corr:.4f}  RiskAcc={risk_acc:.2%}")

    with open(f"{out_dir}/validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved validation_metrics.json")


if __name__ == "__main__":
    main()
