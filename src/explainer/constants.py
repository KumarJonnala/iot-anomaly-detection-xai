KB_ENTRIES: list[dict] = [
    {
        'id': 'hdf_mechanism',
        'failure_type': 'HDF',
        'title': 'Heat Dissipation Failure — Mechanism',
        'text': (
            'Heat Dissipation Failure (HDF) occurs when the machine cannot cool itself adequately. '
            'The temperature difference between process and air falls below 8.6 K while rotational '
            'speed drops below 1380 rpm. Low rotational speed reduces convective airflow, trapping '
            'heat in the spindle and bearings. If sustained, this leads to thermal expansion of '
            'mechanical components and eventual seizure.'
        ),
    },
    {
        'id': 'hdf_thresholds',
        'failure_type': 'HDF',
        'title': 'Heat Dissipation Failure — Threshold Rationale',
        'text': (
            'The HDF rule fires when BOTH conditions hold: (1) process_temp minus air_temp < 8.6 K '
            'and (2) rotational speed < 1380 rpm. These thresholds were derived from the AI4I 2020 '
            'dataset. A temperature differential below 8.6 K indicates insufficient heat being '
            'carried away. The speed threshold of 1380 rpm corresponds to the lower operating limit '
            'for adequate forced-air cooling in this machine class.'
        ),
    },
    {
        'id': 'twf_mechanism',
        'failure_type': 'TWF',
        'title': 'Tool Wear Failure — Mechanism',
        'text': (
            'Tool Wear Failure (TWF) occurs when the cutting tool has degraded beyond its designed '
            'service life. Tool wear accumulates linearly with operation time. Once wear exceeds '
            '200 minutes of cutting time, the tool edge is too blunt to cut accurately, causing '
            'increased vibration, surface finish defects, and elevated torque demand. Continued '
            'operation past this threshold risks catastrophic tool fracture.'
        ),
    },
    {
        'id': 'twf_thresholds',
        'failure_type': 'TWF',
        'title': 'Tool Wear Failure — Threshold and Inspection',
        'text': (
            'The TWF rule fires when tool_wear_min is in the range [200, 240] minutes. The lower '
            'bound marks the recommended replacement interval; the upper bound caps the rule to '
            'avoid flagging extreme wear values that are artefacts rather than imminent failures. '
            'When this flag triggers, operators should inspect the tool immediately. Torque readings '
            'above normal for a given speed are a secondary indicator of advanced wear.'
        ),
    },
    {
        'id': 'osf_mechanism',
        'failure_type': 'OSF',
        'title': 'Overstrain Failure — Mechanism',
        'text': (
            'Overstrain Failure (OSF) occurs when the cumulative mechanical stress on the tool '
            'and spindle exceeds structural limits. It is quantified as the product of tool wear '
            'and torque (wear_torque = tool_wear_min × torque_nm). As the tool wears, it requires '
            'greater torque to remove material, amplifying mechanical stress. Beyond 11,000 Nm·min, '
            'the risk of spindle overload or tool breakage rises sharply.'
        ),
    },
    {
        'id': 'osf_thresholds',
        'failure_type': 'OSF',
        'title': 'Overstrain Failure — Threshold and Mitigation',
        'text': (
            'The OSF rule fires when tool_wear_min × torque_nm exceeds a type-aware threshold: '
            '11,000 Nm·min for type-L machines, 12,000 for type-M, and 13,000 for type-H. '
            'Heavier-duty machine types tolerate higher cumulative stress before overstrain risk. '
            'Mitigation options include reducing feed rate (lowering torque demand), replacing '
            'the tool (resetting wear to 0), or reducing material hardness. OSF often co-occurs '
            'with TWF when the tool is both old and working under high load.'
        ),
    },
    {
        'id': 'pwf_mechanism',
        'failure_type': 'PWF',
        'title': 'Power Failure — Mechanism',
        'text': (
            'Power Failure (PWF) occurs when the spindle power consumption falls outside the safe '
            'operating envelope. Power is computed as torque × rotational speed × (2π/60) in watts. '
            'Extremely high power indicates the motor is overloaded, likely due to heavy cutting '
            'conditions or tool binding. Extremely low power during machining may indicate tool '
            'disengagement or drive belt slip.'
        ),
    },
    {
        'id': 'pwf_context',
        'failure_type': 'PWF',
        'title': 'Power Failure — Operating Context',
        'text': (
            'PWF is detected by a deterministic rule: rule_pwf fires when spindle power is below '
            '3,500 W or above 9,000 W. Power below 3,500 W during machining suggests tool '
            'disengagement or drive belt slip; power above 9,000 W indicates motor overload from '
            'heavy cutting or tool binding. Operators should check spindle load meters, verify '
            'cutting programme parameters, and inspect for tool binding or workpiece clamping '
            'issues when PWF is flagged.'
        ),
    },
    {
        'id': 'rnf_mechanism',
        'failure_type': 'RNF',
        'title': 'Random Failure — Mechanism',
        'text': (
            'Random Failure (RNF) has no identifiable physical root cause correlated with sensor '
            'readings. It occurs due to stochastic processes such as material inclusions, coolant '
            'contamination, or electrical transients. RNF cannot be predicted from the five sensor '
            'channels alone. When an anomaly is flagged but no domain rule fires and sensor '
            'attribution is ambiguous, RNF is the most likely failure mode.'
        ),
    },
    {
        'id': 'ae_reconstruction',
        'failure_type': None,
        'title': 'Autoencoder Reconstruction Error — Interpretation',
        'text': (
            'The autoencoder was trained exclusively on normal operating data. High reconstruction '
            'error for a sensor means that sensor\'s reading pattern deviates significantly from '
            'its normal relationships with other sensors. The sensor with the highest error is the '
            'primary driver of the anomaly signal. Low reconstruction error despite an anomaly flag '
            'means the anomaly was detected by statistical (Z-score) or ensemble (Isolation Forest) '
            'methods rather than the neural reconstruction.'
        ),
    },
    {
        'id': 'general_interpretation',
        'failure_type': None,
        'title': 'Anomaly Interpretation — General Guidance',
        'text': (
            'Anomaly detection uses two independent paths. The ML path flags a row when the '
            'combined score (equal-weighted average of Z-score, Isolation Forest, and Autoencoder) '
            'exceeds 0.70. The rule path flags independently when any domain rule fires '
            '(rule_hdf, rule_twf, rule_osf, or rule_pwf); these have near-zero false positive '
            'rates and should be treated as actionable immediately. Statistical-only flags '
            '(agreement=none or one_only) carry higher uncertainty and warrant monitoring rather '
            'than immediate action.'
        ),
    },
    {
        'id': 'sensor_correlations',
        'failure_type': None,
        'title': 'Sensor Correlation Patterns',
        'text': (
            'In normal operation, air temperature and process temperature are strongly correlated '
            '(r≈0.88), both rising slowly over a shift. Rotational speed and torque are negatively '
            'correlated (r≈-0.88) — higher speeds require lower torque for the same power. Tool '
            'wear increases monotonically and independently of other sensors. Deviations from these '
            'expected relationships appear as high autoencoder reconstruction error and are a '
            'reliable early indicator of emerging failures.'
        ),
    },
]
