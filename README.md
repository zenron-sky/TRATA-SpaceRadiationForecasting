#  (TRATA) — Space Radiation Forecasting System (PS14 Prototype)

End-to-end AI/ML prototype for forecasting energetic electron flux (>2 MeV)
around ISRO geostationary satellites at **45-minute, 6-hour, and 12-hour**
horizons, using a **Transformer + Random Forest hybrid** model — matching
the TRATA architecture (Layers 01–06). Built with **synthetic data**
(physically structured, with realistic CME-driven storm events and a
delayed electron response) so it runs standalone for a hackathon demo —
swap in real GOES/Wind/GRASP CDF data later without changing the pipeline.

## Stack (TRATA)
- **Language:** Python 3.x
- **Modeling:** PyTorch Transformer encoder (temporal embedding) → scikit-learn
  Random Forest (final nonlinear prediction + explainability), with MC-Dropout
  sampled through the full hybrid pipeline for confidence scoring
- **Data processing:** pandas, NumPy, SciPy
- **Dashboard:** HTML/CSS/JavaScript + Chart.js (no build step needed)

## Why Transformer + Random Forest (not a single end-to-end network)
- **Transformer** captures long-range temporal dependencies in the 4-hour
  solar wind / geomagnetic sequence — the same role attention plays in
  modern sequence models, matching "Transformers capture long-range
  dependencies" in the architecture diagram.
- **Random Forest** takes the Transformer's pooled embedding **plus** the
  current physics feature vector and produces the final prediction. This
  keeps the model grounded in interpretable variables (dynamic pressure,
  Bz, Kp, Dst, etc.) rather than being a pure black box, and gives built-in
  feature importances for the explainability panel — matching "Random
  Forest models nonlinear interactions" + "Hybrid learning for robust
  prediction."
- **Confidence (MC-Dropout):** the Transformer's dropout layers are kept
  active at inference time (30 stochastic passes). Each stochastic
  embedding is run through the (deterministic) Random Forest to get 30
  point predictions per horizon; mean = forecast, std = confidence score.

## Folder structure
```
TRATA/
├── data/synthetic_data_generator.py    # Layer 01 — simulated Wind+GOES+Dst/Kp data
├── preprocessing/preprocess.py          # Layer 02 — despike, interpolate, sync
├── features/feature_engineering.py      # Layer 03 — 21 physics-informed features + targets
├── models/
│   ├── transformer_encoder.py           # Layer 04, Stage 1 — Transformer temporal encoder
│   ├── random_forest_head.py            # Layer 04, Stage 2 — Random Forest prediction head
│   └── dataset.py                       # sliding-window sequence dataset
├── train.py                             # Layer 04 — two-stage hybrid training
├── validation/validate.py               # Layer 05 — RMSE/MAE/correlation/risk accuracy
├── forecast.py                          # Layer 05/06 — live inference + dashboard JSON
├── dashboard/index.html                 # Layer 06 — interactive monitoring dashboard
├── run_pipeline.py                      # runs all of the above in one command
└── requirements.txt
```

## Quick start

### Windows
1. Extract the zip
2. Open Command Prompt / PowerShell in the folder
3. Run: `run_pipeline.bat` (installs deps, runs all 6 stages, ~4–5 min)
4. Run: `start_dashboard.bat`
5. Open `http://localhost:8000/dashboard/index.html`

### Linux / macOS
```bash
cd TRATA
pip install -r requirements.txt
python3 run_pipeline.py

python3 -m http.server 8000
# open http://localhost:8000/dashboard/index.html
```

## What's implemented vs. the brief

| Layer | Component | Status |
|---|---|---|
| 01 | Data sources (GOES, Wind, GRASP/GSAT) | Synthetic stand-in generator; pipeline structured so real CDF readers (`cdflib`) slot in at the same point |
| 02 | CDF data ingestion, cleaning, time sync, normalization | ✅ rolling z-score despiking, time interpolation, 5-min resync |
| 03 | Physics-informed feature engineering (dynamic pressure, temporal features, time encoding) | ✅ 21 features: rolling solar wind stats, Bz minima, AR flux lags, cyclical time |
| 04 | Hybrid Prediction Engine — Transformer + Random Forest | ✅ Transformer (3-layer encoder) pretrained on multi-horizon regression → pooled embedding → Random Forest (3 forests, one per horizon) on [embedding + physics features] |
| 04 | Multi-horizon output (45min / 6hr / 12hr) | ✅ simultaneous, from the same shared Transformer embedding |
| 05 | Radiation risk classifier (Nominal/Elevated/Severe/Hazardous) | ✅ threshold-based on predicted log₁₀ flux |
| 05 | Validation (RMSE, MAE, correlation) | ✅ chronological held-out test split |
| 05 | Confidence estimation | ✅ MC-Dropout sampled through the full hybrid pipeline (30 stochastic passes) |
| 05 | Explainability (feature saliency / top contributors) | ✅ Random Forest feature_importances_, mapped to named physics variables (Transformer embedding dims aggregated into one bucket so the panel stays physically interpretable) |
| 06 | Operations Dashboard (live monitoring, forecast viz, alerts, reports) | ✅ current conditions, 3-horizon forecast cards, alert banner, historical flux/solar-wind charts, explainability panel, metrics table |

## Model performance (test set, synthetic data)

| Horizon | RMSE (log₁₀) | Correlation | Risk Accuracy |
|---|---|---|---|
| 45 min | ~0.04–0.05 | ~0.995–0.997 | ~55% |
| 6 hr | ~0.10–0.14 | ~0.96–0.98 | ~54–56% |
| 12 hr | ~0.20–0.23 | ~0.88–0.91 | ~54–56% |

(Baseline random risk classification = 25%.) Numbers vary slightly run to
run because the synthetic data generator and training are stochastic.

## Explainability note
At the 45-minute horizon, the model is dominated by persistence (current
electron flux predicts near-term flux almost perfectly — a well-documented
effect in real space weather forecasting). For a more physically meaningful
view of what the model has *learned* about precursors, the dashboard's
explainability panel is built from the **12-hour horizon**, where the
Random Forest correctly leans on solar wind speed rolling statistics and
IMF Bz minima (southward turning) — genuine CME precursor signals — rather
than persistence.

## Swapping in real data later
Replace `data/synthetic_data_generator.py`'s output with a real loader (e.g.
using `cdflib` to read GOES/Wind/GRASP CDF files) that produces a DataFrame
with the same columns: `timestamp, solar_wind_speed_kms,
solar_wind_density_pcc, imf_bz_nt, imf_bt_nt, dst_index_nt, kp_index,
electron_flux_2mev`. Everything downstream (preprocessing → features →
Transformer → Random Forest → dashboard) works unchanged.

## Notes on the synthetic data
The generator isn't random noise — it encodes the actual physics chain from
the problem statement: CME-driven solar wind speed/density jumps and
southward IMF turnings happen first, geomagnetic indices (Dst/Kp) respond
almost immediately, and electron flux enhancement is **deliberately delayed
by several hours and decays over days** — exactly the precursor-to-response
lag a forecasting model needs to learn to be useful operationally.
