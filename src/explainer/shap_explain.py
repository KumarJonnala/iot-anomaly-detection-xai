import numpy as np
import shap

from src.config import SENSOR_COLS


class SHAPExplainer:
    """SHAP TreeExplainer wrapper for the IsolationForest detector.

    Initialize once from the fitted IsolationForest and background DataFrame.
    Thread-safe for concurrent explain() calls after init.
    """

    def __init__(self, if_model, background_df) -> None:
        bg = background_df[SENSOR_COLS].values.astype('float32')
        self._explainer = shap.TreeExplainer(
            if_model,
            data=shap.sample(bg, min(100, len(bg))),
            feature_perturbation='interventional',
        )
        self._features = SENSOR_COLS

    def explain(self, row_values: np.ndarray) -> dict[str, float]:
        """row_values: shape (5,) normalized sensor values.

        Returns {sensor_col: shap_value} where positive = pushes toward anomaly.
        """
        x = row_values.reshape(1, -1).astype('float32')
        sv = self._explainer.shap_values(x)
        if isinstance(sv, list):
            sv = sv[1]   # binary IF: index 1 = anomaly class
        sv = np.asarray(sv).flatten()
        return {feat: float(sv[i]) for i, feat in enumerate(self._features)}
