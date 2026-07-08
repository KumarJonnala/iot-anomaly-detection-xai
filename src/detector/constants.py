from pathlib import Path

SENSOR_COLS = ['air_temp_k', 'process_temp_k', 'rot_speed_rpm', 'torque_nm', 'tool_wear_min']

ORIG_NAMES = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
]

RENAME_MAP = {
    'Type':                    'type',
    'Air temperature [K]':     'air_temp_k',
    'Process temperature [K]': 'process_temp_k',
    'Rotational speed [rpm]':  'rot_speed_rpm',
    'Torque [Nm]':             'torque_nm',
    'Tool wear [min]':         'tool_wear_min',
    'Machine failure':         'machine_failure',
}

TYPE_MAP = {'L': 0, 'M': 1, 'H': 2}

FAILURE_COLS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# ── Physics-based rule thresholds (AI4I 2020 documentation) ──────────────────
RULE_THRESHOLDS = {
    'hdf': {                        # Heat Dissipation Failure
        'max_temp_diff': 8.6,       # K  (process_temp - air_temp)
        'max_rot_speed': 1380,      # rpm
    },
    'pwf': {                        # Power Failure  ← NEW (was missing entirely)
        'min_power_w': 3500,        # W
        'max_power_w': 9000,        # W
    },
    'osf': {                        # Overstrain Failure  ← FIXED (type-aware)
        'L': 11000,                 # min·Nm
        'M': 12000,
        'H': 13000,
    },
    'twf': {                        # Tool Wear Failure  ← TIGHTENED (range, not floor)
        'min_tool_wear': 200,       # min
        'max_tool_wear': 240,       # min
    },
}

# ── Detector parameters ──────────────────────────────────────────────────────
ZSCORE_THRESHOLD = 3.0
ROLLING_WINDOW   = 50
IF_CONTAMINATION = 0.034
AE_PERCENTILE    = 95
AE_LATENT_DIM    = 4
FUSION_THRESHOLD    = 0.5    # kept — used by ML safety net base
ML_FUSION_THRESHOLD = 0.70   # raised from 0.5 — cuts false alarms
FUSION_WEIGHTS   = (1/3, 1/3, 1/3)
SENSOR_WINDOW    = 10
WINDOW_SIZE      = 10

# Isolation Forest score threshold for TWF confirmation
TWF_IF_THRESHOLD    = 0.30   # set to 0.30 — looser TWF confirmation (better TWF recall, reasonable balance)
PALETTE = {
    'NORMAL': '#90CAF9',
    'HDF':    '#EF5350',
    'PWF':    '#FFA726',
    'OSF':    '#66BB6A',
    'TWF':    '#AB47BC',
    'RNF':    '#78909C',
}
