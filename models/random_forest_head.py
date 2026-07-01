"""
TRATA - Random Forest Prediction Head (Layer 04, Stage 2 of Hybrid Engine)
------------------------------------------------------------------------------
Matches the architecture diagram: "Random Forest models nonlinear
interactions" -> "Hybrid learning for robust prediction".

Takes as input the concatenation of:
  1. The Transformer's pooled temporal embedding (captures long-range /
     sequential solar-wind-to-radiation-belt dynamics)
  2. The current physics-engineered feature vector (21 features - dynamic
     pressure, rolling stats, Bz minima, flux lags, etc. - keeps the model
     grounded in interpretable physics quantities, not just a black-box
     embedding)

Produces:
  - Multi-horizon predictions (45min / 6hr / 12hr log10 flux) - one
    RandomForestRegressor per horizon, trained jointly via a wrapper class.
  - Per-horizon confidence via TREE VARIANCE (std of predictions across all
    trees in the forest) rather than MC-Dropout - this is the "simpler,
    no dropout" uncertainty approach.
  - Feature importances (built into RandomForestRegressor) mapped back to
    the named physics features -> feeds the Validation & Explainability
    module (Layer 05) "feature saliency / top contributors" panel.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class HybridRandomForestHead:
    """One RandomForestRegressor per forecast horizon (45min / 6hr / 12hr).

    Using separate forests per horizon (rather than one multi-output forest)
    lets each horizon have importances and tree-variance uncertainty that
    are specific to its own prediction difficulty (12hr is harder / noisier
    than 45min).
    """

    def __init__(self, horizon_labels, n_estimators: int = 120, max_depth: int = 10,
                 min_samples_leaf: int = 5, random_state: int = 42, n_jobs: int = -1):
        self.horizon_labels = horizon_labels
        self.models = {
            label: RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            for label in horizon_labels
        }

    def fit(self, X: np.ndarray, y: np.ndarray):
        """X: (n_samples, n_embedding_dims + n_physics_features)
        y: (n_samples, n_horizons) - column order matches self.horizon_labels
        """
        for i, label in enumerate(self.horizon_labels):
            self.models[label].fit(X, y[:, i])
        return self

    def predict(self, X: np.ndarray) -> dict:
        """Returns {horizon_label: point_prediction_array}"""
        return {label: self.models[label].predict(X) for label in self.horizon_labels}

    def predict_with_uncertainty(self, X: np.ndarray) -> dict:
        """Tree-variance uncertainty: for each horizon, gather every tree's
        individual prediction and compute mean + std across the forest.
        This replaces MC-Dropout as the confidence mechanism.

        Returns {horizon_label: (mean_array, std_array)}
        """
        results = {}
        for label in self.horizon_labels:
            forest = self.models[label]
            # shape: (n_estimators, n_samples)
            tree_preds = np.stack([tree.predict(X) for tree in forest.estimators_], axis=0)
            mean = tree_preds.mean(axis=0)
            std = tree_preds.std(axis=0)
            results[label] = (mean, std)
        return results

    def feature_importances(self, feature_names) -> dict:
        """Returns {horizon_label: {feature_name: importance}} sorted desc."""
        out = {}
        for label in self.horizon_labels:
            importances = self.models[label].feature_importances_
            pairs = sorted(zip(feature_names, importances), key=lambda kv: -kv[1])
            out[label] = pairs
        return out
