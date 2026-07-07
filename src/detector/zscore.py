import numpy as np
import pandas as pd

from .constants import ROLLING_WINDOW, ZSCORE_THRESHOLD


def compute_zscores(
    df: pd.DataFrame,
    sensor_cols: list[str],
    rolling_window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Add global + dynamic Z-score columns for each sensor, plus aggregate flags.

    Extracted from test_notebooks/anomaly_detection.ipynb — logic unchanged.

    Adds columns:
        {col}_zscore_global, {col}_zscore_dynamic  (per sensor)
        zscore_global_max, zscore_dynamic_max, zscore_max
        zscore_flag  (zscore_max > ZSCORE_THRESHOLD)
    """
    result = df.copy()
    for col in sensor_cols:
        g_mean = df[col].mean()
        g_std  = df[col].std()
        result[f'{col}_zscore_global'] = (df[col] - g_mean) / (g_std + 1e-9)

        r_mean = df[col].rolling(rolling_window, min_periods=5).mean()
        r_std  = df[col].rolling(rolling_window, min_periods=5).std()
        result[f'{col}_zscore_dynamic'] = (df[col] - r_mean) / (r_std + 1e-9)

    global_cols  = [f'{c}_zscore_global'  for c in sensor_cols]
    dynamic_cols = [f'{c}_zscore_dynamic' for c in sensor_cols]
    result['zscore_global_max']  = result[global_cols].abs().max(axis=1)
    result['zscore_dynamic_max'] = result[dynamic_cols].abs().fillna(0).max(axis=1)
    result['zscore_max']         = result[['zscore_global_max', 'zscore_dynamic_max']].max(axis=1)
    result['zscore_flag']        = result['zscore_max'] > ZSCORE_THRESHOLD
    return result
