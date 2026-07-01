"""
TRATA - Hybrid Training Script (Layer 04: AI/ML Prediction Engine)
-----------------------------------------------------------------------
Two-stage training matching the architecture diagram's "Hybrid Prediction
Engine (Transformer + Random Forest)":

  STAGE A - Transformer pretraining
    Train the Transformer encoder (with a temporary linear head) via
    supervised regression on the multi-horizon targets. This teaches it to
    build a pooled embedding that captures long-range temporal dependencies
    in the solar wind / geomagnetic sequence.

  STAGE B - Random Forest fitting
    Discard the Transformer's linear head. Run the (now-frozen) Transformer
    over every training window to get pooled embeddings. Concatenate each
    embedding with the CURRENT physics feature vector (last timestep of the
    window) to keep the model grounded in interpretable variables. Fit one
    RandomForestRegressor per horizon on [embedding + physics features] ->
    target.

Chronological train/val/test split throughout (no shuffling across time -
this is a forecasting task, not iid data).
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMNS
from models.transformer_encoder import TransformerForecastEncoder
from models.random_forest_head import HybridRandomForestHead
from models.dataset import SpaceWeatherSequenceDataset

SEQ_LEN = 48  # 48 * 5min = 4 hours of history feeding the Transformer
BATCH_SIZE = 128
TRANSFORMER_EPOCHS = 6
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def chronological_split(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def extract_embeddings(transformer, loader, device):
    """Run the frozen Transformer over every window, returning:
    - embeddings: (n_samples, d_model)
    - last_features: (n_samples, n_features)  [current physics vector]
    - targets: (n_samples, n_horizons)
    """
    transformer.eval()
    all_emb, all_last_feat, all_targets = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            emb = transformer.get_embedding(xb).cpu().numpy()
            last_feat = xb[:, -1, :].cpu().numpy()  # current-timestep physics features
            all_emb.append(emb)
            all_last_feat.append(last_feat)
            all_targets.append(yb.numpy())
    return (np.concatenate(all_emb), np.concatenate(all_last_feat), np.concatenate(all_targets))


def main():
    feat_path = "./outputs/features.csv"
    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)

    train_df, val_df, test_df = chronological_split(df)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Standardize features using TRAIN stats only (avoid leakage)
    mean = train_df[FEATURE_COLUMNS].mean()
    std = train_df[FEATURE_COLUMNS].std().replace(0, 1)

    def scale(d):
        return ((d[FEATURE_COLUMNS] - mean) / std).values.astype(np.float32)

    X_train, y_train = scale(train_df), train_df[TARGET_COLUMNS].values.astype(np.float32)
    X_val, y_val = scale(val_df), val_df[TARGET_COLUMNS].values.astype(np.float32)
    X_test, y_test = scale(test_df), test_df[TARGET_COLUMNS].values.astype(np.float32)

    train_ds = SpaceWeatherSequenceDataset(X_train, y_train, SEQ_LEN)
    val_ds = SpaceWeatherSequenceDataset(X_val, y_val, SEQ_LEN)
    test_ds = SpaceWeatherSequenceDataset(X_test, y_test, SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    # non-shuffled loader for clean, order-preserving embedding extraction
    train_loader_seq = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ============================================================
    # STAGE A: Transformer pretraining (learns the temporal embedding)
    # ============================================================
    print("\n" + "=" * 70)
    print("STAGE A: Pretraining Transformer temporal encoder")
    print("=" * 70)

    transformer = TransformerForecastEncoder(
        n_features=len(FEATURE_COLUMNS), n_horizons=len(TARGET_COLUMNS)
    ).to(DEVICE)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, TRANSFORMER_EPOCHS + 1):
        transformer.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = transformer(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        transformer.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = transformer(xb)
                val_losses.append(loss_fn(pred, yb).item())

        tl, vl = float(np.mean(train_losses)), float(np.mean(val_losses))
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        print(f"Epoch {epoch:02d}/{TRANSFORMER_EPOCHS} | train_loss={tl:.5f} | val_loss={vl:.5f}")

    # ============================================================
    # STAGE B: Random Forest fitting on Transformer embeddings + physics features
    # ============================================================
    print("\n" + "=" * 70)
    print("STAGE B: Extracting embeddings & fitting Random Forest heads")
    print("=" * 70)

    emb_train, feat_train, y_rf_train = extract_embeddings(transformer, train_loader_seq, DEVICE)
    emb_val, feat_val, y_rf_val = extract_embeddings(transformer, val_loader, DEVICE)
    emb_test, feat_test, y_rf_test = extract_embeddings(transformer, test_loader, DEVICE)

    # Hybrid input = [Transformer embedding | current physics feature vector]
    X_rf_train = np.concatenate([emb_train, feat_train], axis=1)
    X_rf_val = np.concatenate([emb_val, feat_val], axis=1)
    X_rf_test = np.concatenate([emb_test, feat_test], axis=1)

    print(f"Random Forest input dimensionality: {X_rf_train.shape[1]} "
          f"({emb_train.shape[1]} embedding dims + {feat_train.shape[1]} physics features)")

    rf_head = HybridRandomForestHead(horizon_labels=list(TARGET_COLUMNS))
    rf_head.fit(X_rf_train, y_rf_train)

    rf_preds_val = rf_head.predict(X_rf_val)
    for i, label in enumerate(TARGET_COLUMNS):
        rmse = float(np.sqrt(np.mean((rf_preds_val[label] - y_rf_val[:, i]) ** 2)))
        print(f"  Val RMSE ({label}): {rmse:.4f}")

    # ============================================================
    # Save all artifacts
    # ============================================================
    out_dir = "./outputs"
    torch.save(transformer.state_dict(), f"{out_dir}/transformer.pt")
    joblib.dump(rf_head, f"{out_dir}/random_forest_head.joblib")
    np.savez(f"{out_dir}/scaler.npz", mean=mean.values, std=std.values, columns=FEATURE_COLUMNS)
    with open(f"{out_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nSaved transformer.pt, random_forest_head.joblib, scaler.npz, history.json")

    # ---------- Test-set quick summary ----------
    rf_preds_test = rf_head.predict(X_rf_test)
    print("\nTest set performance:")
    for i, label in enumerate(TARGET_COLUMNS):
        rmse = float(np.sqrt(np.mean((rf_preds_test[label] - y_rf_test[:, i]) ** 2)))
        print(f"Test RMSE ({label}, log10 flux units): {rmse:.4f}")


if __name__ == "__main__":
    main()
