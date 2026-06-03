import json
import numpy as np
import pandas as pd

from .constants import FUSION_THRESHOLD, FUSION_WEIGHTS, SENSOR_WINDOW


def fuse_scores(
    df: pd.DataFrame,
    weights: tuple = FUSION_WEIGHTS,
    threshold: float = FUSION_THRESHOLD,
) -> pd.DataFrame:
    """Combine Z-score, IF, and AE scores into a single combined_score.

    Extracted from test_notebooks/anomaly_detection.ipynb — logic unchanged.

    Adds columns:
        zscore_norm, combined_score, anomaly, agreement
    """
    w_z, w_if, w_ae = weights
    zscore_norm = np.clip(df['zscore_max'] / 5.0, 0, 1)
    combined    = w_z * zscore_norm + w_if * df['if_score'] + w_ae * df['ae_score']

    df = df.copy()
    df['zscore_norm']    = zscore_norm
    df['combined_score'] = combined

    rule_any  = df['rule_hdf'] | df['rule_twf'] | df['rule_osf']
    df['anomaly'] = (combined > threshold) | rule_any

    n_agree = (df['zscore_flag'].astype(int) +
               df['if_flag'].astype(int) +
               df['ae_flag'].astype(int))
    df['agreement'] = n_agree.map(
        {0: 'none', 1: 'one_only', 2: 'two_of_three', 3: 'all_three'})
    return df


def build_anomaly_records(
    df: pd.DataFrame,
    sensor_cols: list[str],
    ranges: dict,
    window: int = SENSOR_WINDOW,
) -> list[dict]:
    """Build the canonical anomaly record list from a fully-scored DataFrame.

    Extracted from test_notebooks/anomaly_detection.ipynb — logic unchanged.
    Only rows where df['anomaly'] == True are included.
    """
    records = []
    for row_idx in df.index[df['anomaly']].tolist():
        row = df.iloc[row_idx]

        zs    = {c: abs(row.get(f'{c}_zscore_global', 0)) for c in sensor_cols}
        worst = max(zs, key=zs.get)

        start    = max(0, row_idx - window)
        end      = min(len(df), row_idx + window + 1)
        win_vals = df[worst].iloc[start:end].round(4).tolist()

        record = {
            'dataset':       'ai4i',
            'row_idx':       int(row_idx),
            'worst_sensor':  worst,
            'flagged_value': round(float(row[worst]), 4),

            'window':        win_vals,
            'window_median': round(float(np.median(win_vals)), 4),
            'window_std':    round(float(np.std(win_vals)), 4),

            'global_mean':   ranges[worst]['mean'],
            'global_std':    ranges[worst]['std'],
            'global_p5':     round(float(df[worst].quantile(0.05)), 4),
            'global_p95':    round(float(df[worst].quantile(0.95)), 4),

            'zscore_max':          round(float(row['zscore_max']), 3),
            'zscore_flag':         bool(row['zscore_flag']),
            'if_score':            round(float(row['if_score']), 3),
            'if_flag':             bool(row['if_flag']),
            'ae_error_total':      round(float(row['ae_error']), 6),
            'ae_error_per_sensor': {
                c: round(float(row[f'ae_error_{c}']), 6) for c in sensor_cols
            },
            'ae_flag':             bool(row['ae_flag']),

            'rule_hdf':            bool(row['rule_hdf']),
            'rule_twf':            bool(row['rule_twf']),
            'rule_osf':            bool(row['rule_osf']),

            'combined_score':      round(float(row['combined_score']), 3),
            'agreement':           row['agreement'],

            'true_label':   int(row['machine_failure']),
            'failure_type': row.get('failure_type', None),
        }
        records.append(record)
    return records
