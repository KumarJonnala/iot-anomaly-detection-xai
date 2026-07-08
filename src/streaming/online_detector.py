from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.detector.autoencoder import DEVICE, load_autoencoder
from src.detector.constants import (
    AE_PERCENTILE,
    ML_FUSION_THRESHOLD,
    FUSION_WEIGHTS,
    ROLLING_WINDOW,
    SENSOR_WINDOW,
    ZSCORE_THRESHOLD,
)
from src.detector.isolation_forest import fit_isolation_forest
from src.explainer.constants import SENSOR_COLS

_AE_PATH = Path('data/output/ae_ai4i.pt')


class OnlineDetector:
    """Per-row anomaly detector using pre-fitted IF, loaded AE, and rolling Z-score.

    All three models are initialised once from the full dataset, then used for
    inference-only on each incoming row — no retraining during the stream.
    The anomaly record format is identical to build_anomaly_records() output so
    enrich_node can consume records from either the batch or streaming path.
    """

    def __init__(self, df, ranges: dict, ae_path: Path = _AE_PATH) -> None:
        self._sensor_cols = SENSOR_COLS
        self._ranges = ranges
        self._df = df
        self._rolling: dict[str, deque] = {
            c: deque(maxlen=ROLLING_WINDOW) for c in SENSOR_COLS
        }

        X = df[SENSOR_COLS].values.astype(np.float32)

        # Isolation Forest — fit once, store score range for normalisation
        self._clf_if = fit_isolation_forest(X)
        raw_if = self._clf_if.score_samples(X)
        self._if_min = float(raw_if.min())
        self._if_max = float(raw_if.max())

        # Autoencoder — load saved weights, compute threshold from normal rows
        self._ae = load_autoencoder(ae_path, n_features=len(SENSOR_COLS))
        X_normal = df.loc[df['machine_failure'] == 0, SENSOR_COLS].values.astype(np.float32)
        with torch.no_grad():
            recon = self._ae(torch.FloatTensor(X_normal).to(DEVICE)).cpu().numpy()
        normal_errors = ((X_normal - recon) ** 2).mean(axis=1)
        self._ae_threshold = float(np.percentile(normal_errors, AE_PERCENTILE))

        # Global stats for Z-score baseline
        self._g_mean = {c: float(df[c].mean()) for c in SENSOR_COLS}
        self._g_std  = {c: float(df[c].std())  for c in SENSOR_COLS}

    def score_row(self, row_idx: int, row) -> dict | None:
        """Score one incoming row. Returns an anomaly record dict or None."""
        vals = np.array([float(row[c]) for c in self._sensor_cols], dtype=np.float32)

        for c in self._sensor_cols:
            self._rolling[c].append(float(row[c]))

        # Z-score (global)
        zs_global = {
            c: abs((float(row[c]) - self._g_mean[c]) / (self._g_std[c] + 1e-9))
            for c in self._sensor_cols
        }
        zscore_global_max = max(zs_global.values())

        # Z-score (dynamic rolling)
        zs_dynamic = {}
        for c in self._sensor_cols:
            buf = list(self._rolling[c])
            if len(buf) >= 5:
                zs_dynamic[c] = abs((float(row[c]) - np.mean(buf)) / (np.std(buf) + 1e-9))
            else:
                zs_dynamic[c] = 0.0
        zscore_dynamic_max = max(zs_dynamic.values())

        zscore_max  = max(zscore_global_max, zscore_dynamic_max)
        zscore_flag = zscore_max > ZSCORE_THRESHOLD

        # Isolation Forest
        raw_if   = float(self._clf_if.score_samples(vals.reshape(1, -1))[0])
        if_score = float(np.clip(
            1 - (raw_if - self._if_min) / (self._if_max - self._if_min + 1e-9), 0, 1))
        if_flag  = bool(self._clf_if.predict(vals.reshape(1, -1))[0] == -1)

        # Autoencoder
        with torch.no_grad():
            recon = self._ae(
                torch.FloatTensor(vals.reshape(1, -1)).to(DEVICE)
            ).cpu().numpy()[0]
        ae_per_sensor = (vals - recon) ** 2
        ae_error  = float(ae_per_sensor.mean())
        ae_score  = float(np.clip(ae_error / (self._ae_threshold * 2), 0, 1))
        ae_flag   = ae_error > self._ae_threshold

        # Fusion
        w_z, w_if, w_ae = FUSION_WEIGHTS
        zscore_norm    = float(np.clip(zscore_max / 5.0, 0, 1))
        combined_score = w_z * zscore_norm + w_if * if_score + w_ae * ae_score

        rule_hdf = bool(row.get('rule_hdf', False))
        rule_twf = bool(row.get('rule_twf', False))
        rule_osf = bool(row.get('rule_osf', False))
        rule_pwf = bool(row.get('rule_pwf', False))

        if not ((combined_score > ML_FUSION_THRESHOLD) or rule_hdf or rule_twf or rule_osf or rule_pwf):
            return None

        n_agree   = int(zscore_flag) + int(if_flag) + int(ae_flag)
        agreement = {0: 'none', 1: 'one_only', 2: 'two_of_three', 3: 'all_three'}[n_agree]

        worst  = max(zs_global, key=zs_global.get)
        start  = max(0, row_idx - SENSOR_WINDOW)
        end    = min(len(self._df), row_idx + SENSOR_WINDOW + 1)
        win_vals = self._df[worst].iloc[start:end].round(4).tolist()

        return {
            'dataset':       'ai4i',
            'row_idx':       int(row_idx),
            'worst_sensor':  worst,
            'flagged_value': round(float(row[worst]), 4),

            'window':        win_vals,
            'window_median': round(float(np.median(win_vals)), 4),
            'window_std':    round(float(np.std(win_vals)), 4),

            'global_mean':   self._ranges[worst]['mean'],
            'global_std':    self._ranges[worst]['std'],
            'global_p5':     round(float(self._df[worst].quantile(0.05)), 4),
            'global_p95':    round(float(self._df[worst].quantile(0.95)), 4),

            'zscore_max':          round(zscore_max, 3),
            'zscore_flag':         zscore_flag,
            'if_score':            round(if_score, 3),
            'if_flag':             if_flag,
            'ae_error_total':      round(ae_error, 6),
            'ae_error_per_sensor': {
                c: round(float(ae_per_sensor[i]), 6)
                for i, c in enumerate(self._sensor_cols)
            },
            'ae_flag':             ae_flag,

            'rule_hdf':      rule_hdf,
            'rule_twf':      rule_twf,
            'rule_osf':      rule_osf,
            'rule_pwf':      rule_pwf,

            'combined_score': round(combined_score, 3),
            'agreement':      agreement,

            'true_label':   int(row['machine_failure']),
            'failure_type': str(row.get('failure_type', 'NORMAL')),
        }
