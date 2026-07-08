import numpy as np
import pandas as pd

from src.config import PRE_WINDOW_SIZE, RULE_THRESHOLDS, SENSOR_COLS, SENSOR_LABELS, SENSOR_UNITS


def denormalise(value: float, sensor: str, ranges: dict) -> float:
    """Convert a normalised [0,1] value back to original engineering units."""
    r = ranges[sensor]
    return r['min'] + value * (r['max'] - r['min'])


def compute_sensor_context(
    df: pd.DataFrame,
    idx: int,
    sensor: str,
    ranges: dict,
    pre_window: int = PRE_WINDOW_SIZE,
) -> dict:
    """Rolling and global statistics for one sensor at one row index."""
    col_vals = df[sensor].values
    current_norm = float(col_vals[idx])
    current_orig = round(denormalise(current_norm, sensor, ranges), 2)

    # Global percentile rank
    global_percentile = round(float(np.mean(col_vals <= current_norm)) * 100, 1)

    # Pre-window (readings before this point)
    start = max(0, idx - pre_window)
    pre_vals = col_vals[start:idx]
    if len(pre_vals) > 0:
        pre_median_norm = round(float(np.median(pre_vals)), 4)
        pre_std_norm    = round(float(np.std(pre_vals)),    4)
        pre_min_norm    = round(float(np.min(pre_vals)),    4)
        pre_max_norm    = round(float(np.max(pre_vals)),    4)
    else:
        pre_median_norm = pre_std_norm = pre_min_norm = pre_max_norm = current_norm

    # Global percentile bands
    global_p5_norm  = round(float(np.percentile(col_vals, 5)),  4)
    global_p50_norm = round(float(np.percentile(col_vals, 50)), 4)
    global_p95_norm = round(float(np.percentile(col_vals, 95)), 4)

    return {
        'current_norm':      round(current_norm, 4),
        'current_orig':      current_orig,
        'unit':              SENSOR_UNITS[sensor],
        'global_percentile': global_percentile,
        'pre_median_norm':   pre_median_norm,
        'pre_std_norm':      pre_std_norm,
        'pre_min_norm':      pre_min_norm,
        'pre_max_norm':      pre_max_norm,
        'global_p5_norm':    global_p5_norm,
        'global_p50_norm':   global_p50_norm,
        'global_p95_norm':   global_p95_norm,
    }


def build_ae_attribution(ae_error_per_sensor: dict) -> list[dict]:
    """Rank sensors by AE reconstruction error with percentage contribution."""
    total = sum(ae_error_per_sensor.values()) or 1.0
    ranked = sorted(ae_error_per_sensor.items(), key=lambda x: x[1], reverse=True)
    return [
        {
            'sensor': sensor,
            'label':  SENSOR_LABELS[sensor],
            'error':  round(error, 6),
            'pct':    round(100 * error / total, 1),
        }
        for sensor, error in ranked
    ]


def build_rule_explanation(record: dict, df_row: pd.Series, ranges: dict) -> str:
    """Plain-English explanation of triggered domain rules."""
    triggered = []
    t = RULE_THRESHOLDS

    if record.get('rule_hdf'):
        temp_diff  = round(float(df_row['temp_diff_k']), 2)
        rot_speed  = round(denormalise(float(df_row['rot_speed_rpm']), 'rot_speed_rpm', ranges), 0)
        triggered.append(
            f'Heat Dissipation Failure rule fired: temperature differential is {temp_diff} K '
            f'(threshold < {t["hdf"]["max_temp_diff"]} K) and rotational speed is {int(rot_speed)} rpm '
            f'(threshold < {t["hdf"]["max_rot_speed"]} rpm). Insufficient cooling is indicated.'
        )

    if record.get('rule_twf'):
        wear_min = round(denormalise(float(df_row['tool_wear_min']), 'tool_wear_min', ranges), 0)
        triggered.append(
            f'Tool Wear Failure rule fired: tool wear is {int(wear_min)} min, '
            f'within the scheduled replacement window ({t["twf"]["min_tool_wear"]}–{t["twf"]["max_tool_wear"]} min). '
            f'Immediate tool inspection is recommended.'
        )

    if record.get('rule_osf'):
        wear_torque  = round(float(df_row['wear_torque']), 1)
        type_label   = {0: 'L', 1: 'M', 2: 'H'}.get(int(df_row['type']), 'L')
        osf_threshold = t['osf'][type_label]
        triggered.append(
            f'Overstrain Failure rule fired: wear-torque product is {wear_torque} Nm·min '
            f'(threshold > {osf_threshold} Nm·min for type-{type_label} machine). '
            f'Spindle overload risk — consider reducing feed rate or replacing the tool.'
        )

    if record.get('rule_pwf'):
        power_w = round(float(df_row['power_w']), 1)
        triggered.append(
            f'Power Failure rule fired: spindle power is {power_w} W, '
            f'outside the safe operating envelope ({t["pwf"]["min_power_w"]}–{t["pwf"]["max_power_w"]} W). '
            f'Check for motor overload or tool binding.'
        )

    if not triggered:
        return 'No domain rules triggered.'
    return ' | '.join(triggered)


def _detector_summary(record: dict) -> str:
    """Human-readable summary of detector votes and agreement."""
    votes = []
    if record.get('zscore_flag'):
        votes.append(f'Z-score (max={record["zscore_max"]:.2f}σ)')
    if record.get('if_flag'):
        votes.append(f'Isolation Forest (score={record["if_score"]:.3f})')
    if record.get('ae_flag'):
        votes.append(f'Autoencoder (error={record["ae_error_total"]:.5f})')

    rule_votes = [k.replace('rule_', '').upper()
                  for k in ('rule_hdf', 'rule_twf', 'rule_osf', 'rule_pwf')
                  if record.get(k)]

    parts = []
    if votes:
        parts.append('Statistical detectors: ' + ', '.join(votes))
    if rule_votes:
        parts.append('Domain rules: ' + ', '.join(rule_votes))

    agreement = record.get('agreement', 'none')
    agreement_map = {
        'none':         'rule-only (no statistical detectors agree)',
        'one_only':     'one of three statistical detectors',
        'two_of_three': 'two of three statistical detectors',
        'all_three':    'all three statistical detectors',
    }
    parts.append(f'Detector agreement: {agreement_map.get(agreement, agreement)}')
    parts.append(f'Combined score: {record["combined_score"]:.3f}')
    return '. '.join(parts) + '.'


def enrich_record(
    record: dict,
    df: pd.DataFrame,
    ranges: dict,
    pre_window: int = PRE_WINDOW_SIZE,
) -> dict:
    """Enrich one anomaly record into a full context payload."""
    idx = record['row_idx']
    df_row = df.iloc[idx]

    sensor_context = {
        sensor: compute_sensor_context(df, idx, sensor, ranges, pre_window)
        for sensor in SENSOR_COLS
    }

    return {
        'sensor_context':   sensor_context,
        'ae_attribution':   build_ae_attribution(record['ae_error_per_sensor']),
        'rule_explanation': build_rule_explanation(record, df_row, ranges),
        'detector_summary': _detector_summary(record),
        'anomaly_metadata': {
            'row_idx':        idx,
            'worst_sensor':   record['worst_sensor'],
            'zscore_max':     record.get('zscore_max'),
            'combined_score': record['combined_score'],
            'agreement':      record.get('agreement'),
            'failure_type':   record['failure_type'],
            'true_label':     record.get('true_label'),
        },
    }


def enrich_all(
    records: list[dict],
    df: pd.DataFrame,
    ranges: dict,
    pre_window: int = PRE_WINDOW_SIZE,
) -> list[dict]:
    """Enrich all anomaly records. Returns list same length as input."""
    return [enrich_record(r, df, ranges, pre_window) for r in records]
