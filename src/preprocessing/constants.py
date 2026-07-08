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

RULE_THRESHOLDS = {
    'hdf': {'max_temp_diff': 8.6, 'max_rot_speed': 1380},
    'pwf': {'min_power_w': 3500, 'max_power_w': 9000},
    'osf': {'L': 11000, 'M': 12000, 'H': 13000},
    'twf': {'min_tool_wear': 200, 'max_tool_wear': 240},
}

SENSOR_WINDOW = 10
WINDOW_SIZE   = 10

PALETTE = {
    'NORMAL': '#90CAF9',
    'HDF':    '#EF5350',
    'PWF':    '#FFA726',
    'OSF':    '#66BB6A',
    'TWF':    '#AB47BC',
    'RNF':    '#78909C',
}
