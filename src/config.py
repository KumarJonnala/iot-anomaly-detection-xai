import os

# ── Sensor columns and schema ─────────────────────────────────────────────────
SENSOR_COLS   = ['air_temp_k', 'process_temp_k', 'rot_speed_rpm', 'torque_nm', 'tool_wear_min']
FAILURE_COLS  = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
TYPE_MAP      = {'L': 0, 'M': 1, 'H': 2}

SENSOR_LABELS = {
    'air_temp_k':      'Air Temperature',
    'process_temp_k':  'Process Temperature',
    'rot_speed_rpm':   'Rotational Speed',
    'torque_nm':       'Torque',
    'tool_wear_min':   'Tool Wear',
}

SENSOR_UNITS = {
    'air_temp_k':      'K',
    'process_temp_k':  'K',
    'rot_speed_rpm':   'rpm',
    'torque_nm':       'Nm',
    'tool_wear_min':   'min',
}

PALETTE = {
    'NORMAL': '#90CAF9',
    'HDF':    '#EF5350',
    'PWF':    '#FFA726',
    'OSF':    '#66BB6A',
    'TWF':    '#AB47BC',
    'RNF':    '#78909C',
}

# ── Domain rule thresholds ────────────────────────────────────────────────────
RULE_THRESHOLDS = {
    'hdf': {'max_temp_diff': 8.6,  'max_rot_speed': 1380},
    'pwf': {'min_power_w':   3500, 'max_power_w':   9000},
    'osf': {'L': 11000, 'M': 12000, 'H': 13000},
    'twf': {'min_tool_wear': 200,  'max_tool_wear': 240},
}

# ── Window / rolling sizes ────────────────────────────────────────────────────
SENSOR_WINDOW   = 10   # ±readings stored in anomaly record for spike chart
WINDOW_SIZE     = 10   # alias used in preprocessing
ROLLING_WINDOW  = 50   # online Z-score deque length
PRE_WINDOW_SIZE = 50   # readings before anomaly for LLM context assembly

# ── Z-score detector ──────────────────────────────────────────────────────────
ZSCORE_THRESHOLD = 3.0

# ── Isolation Forest ──────────────────────────────────────────────────────────
IF_CONTAMINATION = 0.034
TWF_IF_THRESHOLD = 0.30

# ── Autoencoder ───────────────────────────────────────────────────────────────
AE_PERCENTILE = 95
AE_LATENT_DIM = 4

# ── Score fusion ──────────────────────────────────────────────────────────────
FUSION_WEIGHTS      = (1/3, 1/3, 1/3)  # (z-score weight, IF weight, AE weight)
ML_FUSION_THRESHOLD = 0.70             # ML path gate

# ── LLM / embedding models ───────────────────────────────────────────────────
EXPLAINER_MODEL = os.getenv('EXPLAINER_MODEL', 'gemma3:4b')
EMBED_MODEL     = os.getenv('EMBED_MODEL',     'nomic-embed-text:latest')
