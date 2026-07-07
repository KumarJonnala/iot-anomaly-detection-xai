from .constants import SENSOR_COLS, SENSOR_LABELS, SENSOR_UNITS

_CLOSING = (
    'In 2-3 sentences, explain what this anomaly likely means, '
    'which sensors are most involved, and what an operator should investigate next.'
)


def _format_sensor_line(col: str, ctx: dict) -> str:
    label = SENSOR_LABELS[col]
    unit  = SENSOR_UNITS[col]
    pct   = int(ctx['global_percentile'])
    orig  = ctx['current_orig']
    norm  = ctx['current_norm']
    pre_m = ctx['pre_median_norm']
    pre_s = ctx['pre_std_norm']
    return (
        f'  {label}: {orig} {unit} (norm {norm:.4f}, {pct}th pct globally); '
        f'prior-50-reading median={pre_m:.4f}, std={pre_s:.4f}'
    )


def build_zero_shot(record: dict) -> str:
    """Minimal prompt with only the flagged sensor and detector summary."""
    worst  = SENSOR_LABELS.get(record['worst_sensor'], record['worst_sensor'])
    val    = record['flagged_value']
    zscore = record.get('zscore_max', 'N/A')
    agree  = record.get('agreement', 'unknown')

    lines = [
        'You are an IoT maintenance analyst.',
        '',
        '=== ANOMALY ALERT ===',
        f'Worst sensor   : {worst}',
        f'Flagged value  : {val:.4f} (normalised)',
        f'Max Z-score    : {zscore}',
        f'Detector agree : {agree}',
        '',
        _CLOSING,
    ]
    return '\n'.join(lines)


def build_contextualised(record: dict, context_payload: dict) -> str:
    """Full context prompt: all sensors + AE attribution + rules + detector summary."""
    sc    = context_payload['sensor_context']
    ae    = context_payload['ae_attribution']
    rules = context_payload['rule_explanation']
    det   = context_payload['detector_summary']
    meta  = context_payload['anomaly_metadata']

    sensor_lines = '\n'.join(_format_sensor_line(col, sc[col]) for col in SENSOR_COLS)

    top3_ae = ae[:3]
    ae_lines = '\n'.join(
        f'  {i+1}. {a["label"]}: error={a["error"]:.6f} ({a["pct"]:.1f}% of total)'
        for i, a in enumerate(top3_ae)
    )

    lines = [
        'You are an IoT maintenance analyst.',
        '',
        '=== ANOMALY ALERT ===',
        f'Row index      : {meta["row_idx"]}',
        f'Worst sensor   : {SENSOR_LABELS.get(meta["worst_sensor"], meta["worst_sensor"])}',
        '',
        '--- Current sensor readings (normalised [0,1] + original units) ---',
        sensor_lines,
        '',
        '--- Autoencoder reconstruction error (top 3 by contribution) ---',
        ae_lines,
        '',
        '--- Domain rule status ---',
        rules,
        '',
        '--- Detector summary ---',
        det,
        '',
        _CLOSING,
    ]
    return '\n'.join(lines)


def build_rag(record: dict, context_payload: dict, retrieved_docs: list[dict]) -> str:
    """Contextualised prompt extended with retrieved knowledge base passages."""
    base = build_contextualised(record, context_payload)

    doc_lines = []
    for doc in retrieved_docs:
        doc_lines.append(f'[{doc["id"]}] {doc["title"]}')
        doc_lines.append(doc['text'])
        doc_lines.append('')

    rag_section = '\n'.join([
        '',
        '=== DOMAIN KNOWLEDGE ===',
        *doc_lines,
    ])

    # Remove the closing from the base and re-append after the KB section
    base_without_closing = base[:base.rfind(_CLOSING)].rstrip()
    return base_without_closing + rag_section + '\n\n' + _CLOSING
