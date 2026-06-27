import json
import numpy as np
import pandas as pd

from .constants import (
    FUSION_THRESHOLD, FUSION_WEIGHTS, ML_FUSION_THRESHOLD,
    SENSOR_WINDOW, TWF_IF_THRESHOLD,
)


def fuse_scores(
    df: pd.DataFrame,
    weights: tuple = FUSION_WEIGHTS,
    threshold: float = FUSION_THRESHOLD,
    twf_if_threshold: float = TWF_IF_THRESHOLD,
) -> pd.DataFrame:
    """Combine Z-score, IF, AE scores with physics rules into a final flag.

    Decision logic (three pathways combined with OR):
      - strict_rules   : HDF, PWF, OSF always trigger — ~100% precise by construction
      - twf_confirmed  : TWF rule + IF score > threshold (cuts ~750 false positives)
      - ml_fusion      : combined > ML_FUSION_THRESHOLD (safety net for what rules miss)

    Changes vs previous version
    ---------------------------
    - rule_pwf is now included in strict_rules (was missing)
    - TWF only flags when IF score agrees (was firing on all 800+ wear-window rows)
    - rule_count column added for diagnostics
    """
    w_z, w_if, w_ae = weights
    zscore_norm = np.clip(df['zscore_max'] / 5.0, 0, 1)
    combined    = w_z * zscore_norm + w_if * df['if_score'] + w_ae * df['ae_score']

    df = df.copy()
    df['zscore_norm']    = zscore_norm
    df['combined_score'] = combined

    # Pathway 1: physics rules that are always trustworthy
    strict_rules  = df['rule_hdf'] | df['rule_pwf'] | df['rule_osf']

    # Pathway 2 (twf_confirmed):  TWF rule + IF score > 0.50 — cuts ~750 false alarms
    twf_confirmed = df['rule_twf'] & (df['if_score'] > twf_if_threshold)

    # Pathway 3: ML safety net for patterns the rules can't see
    ml_fusion     = combined > threshold

    df['anomaly'] = strict_rules | twf_confirmed | ml_fusion

    # Diagnostics
    n_agree = (df['zscore_flag'].astype(int) +
               df['if_flag'].astype(int) +
               df['ae_flag'].astype(int))
    df['agreement'] = n_agree.map(
        {0: 'none', 1: 'one_only', 2: 'two_of_three', 3: 'all_three'})

    df['rule_count'] = (
        df['rule_hdf'].astype(int) +
        df['rule_pwf'].astype(int) +
        df['rule_osf'].astype(int) +
        df['rule_twf'].astype(int)
    )
    return df


def build_anomaly_records(
    df: pd.DataFrame,
    sensor_cols: list[str],
    ranges: dict,
    window: int = SENSOR_WINDOW,
) -> list[dict]:
    """Build the canonical anomaly record list from a fully-scored DataFrame.

    Only rows where df['anomaly'] == True are included.
    Change: rule_pwf is now included in the record dict.
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

            'rule_hdf':  bool(row['rule_hdf']),
            'rule_pwf':  bool(row.get('rule_pwf', False)),   # NEW
            'rule_twf':  bool(row['rule_twf']),
            'rule_osf':  bool(row['rule_osf']),

            'combined_score': round(float(row['combined_score']), 3),
            'agreement':      row['agreement'],

            'true_label':   int(row['machine_failure']),
            'failure_type': row.get('failure_type', None),
        }
        records.append(record)
    return records
