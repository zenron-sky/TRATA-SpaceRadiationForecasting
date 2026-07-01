# TRATA— Code Reference & API

## Module overview

### 1. `data/synthetic_data_generator.py`
Generates physically-plausible synthetic space weather data.

**Main function:**
```python
def generate_synthetic_dataset(days: int = 90, freq_minutes: int = 5, 
                               n_storms: int = 14, seed: int = 42) -> pd.DataFrame
```

**Returns:** DataFrame with columns:
- `timestamp`: datetime index
- `solar_wind_speed_kms`: solar wind speed (km/s)
- `solar_wind_density_pcc`: number density (particles/cm³)
- `imf_bz_nt`: southward IMF component (nanoTesla)
- `imf_bt_nt`: total IMF magnitude (nanoTesla)
- `dst_index_nt`: Dst geomagnetic index (nanoTesla)
- `kp_index`: Kp geomagnetic index (0–9)
- `electron_flux_2mev`: >2 MeV electron flux (particles/cm²/s)

**Physics encoded:**
- CME events: solar wind speed + density spikes + southward Bz turning
- Geomagnetic response: Dst dips, Kp rises (near-simultaneous)
- Delayed electron response: flux enhancement 6±3 hours after the driver (realistic lag)
- Storm recovery: flux decays over 12–72 hours

---

### 2. `preprocessing/preprocess.py`
Cleans and standardizes multi-source data.

**Main function:**
```python
def preprocess(df: pd.DataFrame) -> pd.DataFrame
```

**Operations:**
- Despike sensor glitches (rolling z-score, threshold=5.0)
- Interpolate time-aware gaps (linear, max 12 steps)
- Drop unrecoverable rows (still NaN after interpolation)
- Resample to strict 5-minute cadence (handles instrument misalignment)

---

### 3. `features/feature_engineering.py`
Constructs physics-informed features for the model.

**Key constants:**
```python
FEATURE_COLUMNS = [
    "solar_wind_speed_kms",
    "solar_wind_density_pcc",
    "imf_bz_nt",
    "imf_bt_nt",
    "dst_index_nt",
    "kp_index",
    "dynamic_pressure",                   # ∝ ρ·v²
    "speed_roll_mean_30min",             # rolling 30-min mean
    "speed_roll_std_30min",              # rolling std (capture variability)
    "bz_roll_min_30min",                 # minimum Bz in window (southward turn)
    # ... (3hr and 12hr windows)
    "flux_lag_1hr",                      # past 1h electron flux
    "flux_lag_6hr",                      # past 6h electron flux
    "time_sin", "time_cos",              # cyclical time encoding
    "log_electron_flux",                 # log(current flux)
]
TARGET_COLUMNS = ["target_45min", "target_6hr", "target_12hr"]
```

**Main function:**
```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame
```

**Transformations:**
- Log transform electron flux (stabilizes for neural networks)
- Dynamic pressure (ram pressure proxy for storm intensity)
- Rolling statistics (CNN receptive field analog)
- Autoregressive lag features (LSTM memory input)
- Cyclical encodings (handles day-of-year periodicity)
- Shift targets ahead by each horizon (for supervised learning)

---

### 4. `models/cnn_lstm_model.py`
PyTorch model architecture.

**Class:**
```python
class CNNLSTMForecaster(nn.Module):
    def __init__(self, n_features: int = 21, n_horizons: int = 3, 
                 cnn_channels: int = 32, lstm_hidden: int = 64, 
                 lstm_layers: int = 2, dropout: float = 0.2)
```

**Architecture:**
```
Input (batch, seq_len=48, n_features=21)
  ↓
Conv1d (1D convolutions, kernel 5→3)  [CNN: extracts spike patterns]
  ↓
LSTM (bidirectional concept through layers)  [LSTM: captures temporal lag]
  ↓
Linear layers (32→n_horizons)  [output head: 3 forecasts]
  ↓
Output (batch, 3)  [log10(flux) for each horizon]
```

**Methods:**
```python
def forward(x: Tensor) -> Tensor
    # Standard forward pass
    # x: (batch, seq_len, n_features)
    # returns: (batch, n_horizons)

def predict_with_uncertainty(x: Tensor, n_samples: int = 20) 
    -> Tuple[Tensor, Tensor]
    # MC-Dropout inference sampled through the Transformer+RF hybrid
    # returns: (mean_pred, std_pred) for confidence estimation
```

**Dropout usage:**
- Training: normal dropout regularization
- Inference: keep dropout ON, sample multiple forward passes
  → mean = forecast, std = epistemic uncertainty proxy

---

### 5. `models/dataset.py`
Sliding-window sequence dataset for PyTorch DataLoader.

**Class:**
```python
class SpaceWeatherSequenceDataset(Dataset):
    def __init__(self, feature_array: np.ndarray, 
                 target_array: np.ndarray, seq_len: int = 48)
```

**Behavior:**
- Takes flat arrays (n_time, n_features) and (n_time, n_targets)
- Yields (x_window, y_target) tuples
- Window size: 48 timesteps = 4 hours @ 5-min cadence
- Sample i covers times [t_i, t_i+47], target at t_i+47

**Example:**
```python
ds = SpaceWeatherSequenceDataset(X, y, seq_len=48)
x, y = ds[100]
# x.shape = (48, 21), y.shape = (3,)
```

---

### 6. `train.py`
Training orchestration.

**Main flow:**
```python
def main():
    # Load features
    # Chronological train/val/test split (70/15/15)
    # Standardize using TRAIN stats only (no leakage)
    # Create datasets + dataloaders
    # Initialize model
    # Training loop (15 epochs)
    # Save model.pt, scaler.npz, history.json
```

**Constants:**
```python
SEQ_LEN = 48          # 4 hours
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
DEVICE = "cuda" if available else "cpu"
```

**Outputs:**
- `model.pt`: PyTorch state_dict
- `scaler.npz`: feature mean/std for inference normalization
- `history.json`: train/val loss per epoch

---

### 7. `validation/validate.py`
Evaluation on chronological test set.

**Metrics computed per horizon:**
- **RMSE**: root mean squared error (log flux units)
- **MAE**: mean absolute error
- **Correlation**: Pearson coefficient (pred vs. actual)
- **Risk accuracy**: % of predictions where risk category matches actual

**Risk thresholds:**
```python
log_flux < 3.0        → Nominal
3.0 ≤ log_flux < 4.0  → Elevated
4.0 ≤ log_flux < 5.0  → Severe
log_flux ≥ 5.0        → Hazardous
```

**Output:**
```json
{
  "target_45min": {
    "rmse_log10_flux": 0.1446,
    "mae_log10_flux": 0.0541,
    "correlation": 0.9559,
    "risk_classification_accuracy": 0.5402
  }
}
```

---

### 8. `forecast.py`
Live inference and dashboard JSON generation.

**Main flow:**
```python
def main():
    # Load trained model + scaler
    # Take last 48-timestep window (last 4 hours)
    # Run model.predict_with_uncertainty() [MC-Dropout, 30 samples]
    # Compute feature saliency (gradient-based)
    # Classify risk per horizon
    # Check for alert conditions (Severe/Hazardous)
    # Gather last 48h history for charts
    # Write forecast_output.json
```

**Output JSON structure:**
```json
{
  "generated_at": "2024-01-15T10:30:00Z",
  "current_conditions": {
    "timestamp": "2024-01-15T10:30:00Z",
    "electron_flux_2mev": 1547.3,
    "risk_level": "Elevated",
    "solar_wind_speed_kms": 450.2,
    "kp_index": 5.2,
    "dst_index_nt": -85.0
  },
  "forecasts": {
    "45min": {
      "horizon_label": "45 minutes",
      "predicted_flux_2mev": 1554.2,
      "predicted_log10_flux": 3.192,
      "risk_level": "Elevated",
      "confidence_pct": 84.6,
      "uncertainty_log10": 0.23
    },
    "6hr": { ... },
    "12hr": { ... }
  },
  "overall_alert": {
    "active": false,
    "max_risk_level": "Elevated",
    "message": "No alert. Conditions within nominal/elevated bounds."
  },
  "explainability": {
    "top_contributing_variables": [
      { "feature": "imf_bz_nt", "importance_pct": 18.3 },
      { "feature": "dst_index_nt", "importance_pct": 15.7 },
      ...
    ]
  },
  "validation_metrics": { ... },
  "history": {
    "timestamps": [ "2024-01-15T08:30Z", ... ],
    "electron_flux_2mev": [ 1000.1, 1010.2, ... ],
    "solar_wind_speed_kms": [ 400.1, 405.3, ... ],
    ...
  }
}
```

---

## Extending the system

### Swap real data
Replace `data/synthetic_data_generator.py` with:
```python
import cdflib

def read_real_goes_wind_data(goes_file, wind_file, grasp_file):
    # Parse CDF files
    goes_cdf = cdflib.CDF(goes_file)
    wind_cdf = cdflib.CDF(wind_file)
    
    # Extract variables
    timestamps = goes_cdf.varget("Epoch")
    electron_flux = goes_cdf.varget("E_2MeV_Electrons")
    
    solar_wind_speed = wind_cdf.varget("proton_speed")
    # ... etc
    
    # Align times, create DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "solar_wind_speed_kms": solar_wind_speed,
        # ... etc
    })
    return df
```

### Add new model architectures
Create `models/transformer_forecaster.py`:
```python
import torch.nn as nn

class TransformerForecaster(nn.Module):
    def __init__(self, n_features, n_horizons, d_model=64, n_heads=4):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True),
            num_layers=3
        )
        self.head = nn.Linear(d_model, n_horizons)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # last timestep
        return self.head(x)
```

Update `train.py` to import and use it.

### Add longer horizons
In `feature_engineering.py`:
```python
HORIZONS_MIN = {
    "45min": 45,
    "6hr": 360,
    "12hr": 720,
    "24hr": 1440,  # new
    "48hr": 2880,  # new
}
```

Retrain the model (it will auto-adjust output size).

---

## Debugging

### Check data quality
```python
import pandas as pd
df = pd.read_csv("outputs/features.csv", index_col=0, parse_dates=True)
print(df.describe())
print(df[["electron_flux_2mev", "target_45min"]].head(20))
```

### Inspect model
```python
import torch
from models.cnn_lstm_model import CNNLSTMForecaster

model = CNNLSTMForecaster()
print(model)  # see architecture
for name, param in model.named_parameters():
    print(f"{name:30} {param.shape}")
```

### Watch training live
Modify `train.py` to log to TensorBoard:
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer.add_scalar("loss/train", tl, epoch)
writer.add_scalar("loss/val", vl, epoch)
```

Then: `tensorboard --logdir=runs`

---

## Performance tuning

| Parameter | Effect |
|---|---|
| `SEQ_LEN` | Longer = more context (slower), shorter = faster. 48 (4h) is good for space weather |
| `BATCH_SIZE` | Larger = faster training but more memory. 64 is typical. Try 32 or 128. |
| `cnn_channels` | More channels = more capacity. 32 is modest, 64 is mid-range. |
| `lstm_hidden` | LSTM state width. Larger = more expressive but slower. 64 is balanced. |
| `EPOCHS` | More = potentially better but also overfitting. 15 is good. Watch val loss. |
| `LR` | Learning rate. 1e-3 works well for Adam. Try 1e-2 or 1e-4 if unstable. |
| `dropout` | Regularization (prevent overfitting). 0.2 is gentle, 0.5 is strong. |

---

**End-to-End Space Radiation Forecasting · Implemented in PyTorch · Physics-Informed Features**
