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
    out['Type'] = out['Type'].map(TYPE_MAP)
    out = out.rename(columns=RENAME_MAP)
    out['failure_type'] = out.apply(_get_failure_type, axis=1)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temp_diff_k, power_w, wear_torque and rule flag columns.

    Must be called before normalise() — thresholds are calibrated to original units.
    """
    out = df.copy()
    out['temp_diff_k'] = out['process_temp_k'] - out['air_temp_k']
    out['power_w']     = out['torque_nm'] * out['rot_speed_rpm'] * (2 * np.pi / 60)
    out['wear_torque'] = out['tool_wear_min'] * out['torque_nm']

    t = RULE_THRESHOLDS
    out['rule_hdf'] = (out['temp_diff_k'] < t['hdf']['max_temp_diff']) & \
                      (out['rot_speed_rpm'] < t['hdf']['max_rot_speed'])
    out['rule_twf'] = out['tool_wear_min'] >= t['twf']['min_tool_wear']
    out['rule_osf'] = out['wear_torque'] > t['osf']['min_wear_torque']
    return out


def normalise(df: pd.DataFrame) -> tuple:
    """Min-max scale SENSOR_COLS. Returns (scaled_df, ranges_dict).

    Ranges are computed on the pre-scale values so they can be used for
    denormalisation in later stages.
    """
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
