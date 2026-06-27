import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from .constants import (
    FAILURE_COLS, RENAME_MAP, RULE_THRESHOLDS, SENSOR_COLS, TYPE_MAP,
)


def load_raw(path: Path) -> pd.DataFrame:
    """Read original AI4I CSV."""
    return pd.read_csv(path)


def _get_failure_type(row) -> str:
    for col in FAILURE_COLS:
        if row[col] == 1:
            return col
    return 'NORMAL'


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop UDI/Product ID, encode Type, rename columns, derive failure_type."""
    out = df.drop(columns=['UDI', 'Product ID']).copy()
    # Keep the letter (L/M/H) before mapping to int so OSF rule can use it
    out['_type_letter'] = out['Type']
    out['Type'] = out['Type'].map(TYPE_MAP)
    out = out.rename(columns=RENAME_MAP)
    out['failure_type'] = out.apply(_get_failure_type, axis=1)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain features and the FOUR physics rule flags.

    Must be called BEFORE normalise() — thresholds use original units.

    Changes vs previous version
    ---------------------------
    rule_pwf  : ADDED    — was completely missing (Power Failure, 95 rows)
    rule_osf  : FIXED    — now uses type-aware threshold (L/M/H) instead of flat 11000
    rule_twf  : TIGHTENED — now 200-240 min range instead of >= 200
    """
    out = df.copy()

    # Engineered numeric features (original-scale)
    out['temp_diff_k'] = out['process_temp_k'] - out['air_temp_k']
    out['power_w']     = out['torque_nm'] * out['rot_speed_rpm'] * (2 * np.pi / 60)
    out['wear_torque'] = out['tool_wear_min'] * out['torque_nm']

    t = RULE_THRESHOLDS

    # HDF: low temp differential AND low speed (UNCHANGED)
    out['rule_hdf'] = (
        (out['temp_diff_k']   < t['hdf']['max_temp_diff']) &
        (out['rot_speed_rpm'] < t['hdf']['max_rot_speed'])
    )

    # PWF: power outside safe window 3500-9000 W (NEW)
    out['rule_pwf'] = (
        (out['power_w'] < t['pwf']['min_power_w']) |
        (out['power_w'] > t['pwf']['max_power_w'])
    )

    # OSF: wear*torque exceeds type-specific threshold (FIXED)
    if '_type_letter' in out.columns:
        osf_threshold = out['_type_letter'].map(t['osf'])
    else:
        # Fallback when type is already int-encoded
        inv = {v: k for k, v in TYPE_MAP.items()}
        osf_threshold = out['type'].map(inv).map(t['osf'])
    out['rule_osf'] = out['wear_torque'] > osf_threshold

    # TWF: tool wear inside the 200-240 min failure window (TIGHTENED)
    out['rule_twf'] = (
        (out['tool_wear_min'] >= t['twf']['min_tool_wear']) &
        (out['tool_wear_min'] <= t['twf']['max_tool_wear'])
    )

    # Drop helper column before saving
    out = out.drop(columns=['_type_letter'], errors='ignore')
    return out


def normalise(df: pd.DataFrame) -> tuple:
    """Min-max scale SENSOR_COLS. Returns (scaled_df, ranges_dict)."""
    out = df.copy()
    ranges = {
        c: {
            'min':  float(out[c].min()),
            'max':  float(out[c].max()),
            'mean': round(float(out[c].mean()), 2),
            'std':  round(float(out[c].std()), 2),
        }
        for c in SENSOR_COLS
    }
    scaler = MinMaxScaler()
    out[SENSOR_COLS] = scaler.fit_transform(out[SENSOR_COLS])
    return out, ranges


def export(df: pd.DataFrame, ranges: dict, data_dir: Path) -> None:
    """Write ai4i_clean.csv and ai4i_ranges.json to data_dir."""
    df.to_csv(data_dir / 'ai4i_clean.csv', index=False)
    with open(data_dir / 'ai4i_ranges.json', 'w') as f:
        json.dump(ranges, f, indent=2)
