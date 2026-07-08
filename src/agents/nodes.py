from pathlib import Path

import numpy as np
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from src.detector import (
    Autoencoder,
    build_anomaly_records,
    compute_zscores,
    fit_isolation_forest,
    fuse_scores,
    load_autoencoder,
    save_autoencoder,
    score_autoencoder,
    score_isolation_forest,
    train_autoencoder,
)
from src.detector.autoencoder import DEVICE
from src.explainer.constants import SENSOR_COLS
from src.explainer.context import enrich_record
from src.explainer.export import save_results
from src.explainer.llm import generate_explanation
from src.explainer.prompts import build_contextualised, build_rag, build_zero_shot
from src.preprocessing.pipeline import clean, engineer_features, load_raw, normalise

from .state import PipelineState
from .store import get_resource, set_resource

_AE_PATH = Path('data/output/ae_ai4i.pt')


def _thread(config: RunnableConfig) -> str:
    return config['configurable']['thread_id']


# ── Stage 1: Preprocessing ────────────────────────────────────────────────────

def preprocess_node(state: PipelineState, config: RunnableConfig) -> dict:
    raw        = load_raw(Path(state['data_path']))
    df         = engineer_features(clean(raw))
    df, ranges = normalise(df)
    set_resource(_thread(config), 'df', df)   # DataFrame → registry (not serializable)
    return {'ranges': ranges}                  # ranges dict → state (serializable)


# ── Stage 2: Anomaly Detection ────────────────────────────────────────────────

def detect_node(state: PipelineState, config: RunnableConfig) -> dict:
    tid    = _thread(config)
    df     = get_resource(tid, 'df').copy()
    ranges = state['ranges']

    # Z-score (global + dynamic)
    df = compute_zscores(df, SENSOR_COLS)

    # Isolation Forest
    X               = df[SENSOR_COLS].values
    clf_if          = fit_isolation_forest(X)
    if_scores, if_flags = score_isolation_forest(clf_if, X)
    df['if_score']  = if_scores
    df['if_flag']   = if_flags

    # Autoencoder — load saved weights if available, otherwise train fresh
    X_all    = df[SENSOR_COLS].values.astype(np.float32)
    X_normal = df.loc[df['machine_failure'] == 0, SENSOR_COLS].values.astype(np.float32)

    if _AE_PATH.exists():
        ae_model = load_autoencoder(_AE_PATH, n_features=len(SENSOR_COLS))
    else:
        print('Training autoencoder (no saved model found)...')
        ae_model, _, _ = train_autoencoder(X_normal, n_features=len(SENSOR_COLS))
        save_autoencoder(ae_model, _AE_PATH)

    ae_errors, ae_per_sensor, ae_scores, ae_flags = score_autoencoder(
        ae_model, X_all, X_normal, device=DEVICE)

    df['ae_error'] = ae_errors
    df['ae_score'] = ae_scores
    df['ae_flag']  = ae_flags
    for i, col in enumerate(SENSOR_COLS):
        df[f'ae_error_{col}'] = ae_per_sensor[:, i]

    df = fuse_scores(df)

    set_resource(tid, 'df_detected', df)   # DataFrame → registry

    records = build_anomaly_records(df, SENSOR_COLS, ranges)
    return {
        'anomaly_records':    records,
        'current_record_idx': 0,
    }


# ── Stage 3: Context Enrichment ───────────────────────────────────────────────

def enrich_node(state: PipelineState, config: RunnableConfig) -> dict:
    tid         = _thread(config)
    df_detected = get_resource(tid, 'df_detected')
    idx         = state['current_record_idx']
    record      = state['anomaly_records'][idx]
    ctx         = enrich_record(record, df_detected, state['ranges'])
    return {
        'current_record':    record,
        'context_payload':   ctx,
        'explanations':      {},   # empty dict (not None) so _merge reducer has a base
        'prompts_used':      {},
        'operator_decision': None,
    }


def retrieve_docs_node(state: PipelineState, config: RunnableConfig) -> dict:
    kb   = get_resource(_thread(config), 'kb')
    docs = kb.retrieve_for_record(
        state['current_record'], state['context_payload'], k=2)
    return {'retrieved_docs': docs}


# ── Stage 4: Parallel LLM Explanation Agents ─────────────────────────────────

def zero_shot_node(state: PipelineState) -> dict:
    prompt = build_zero_shot(state['current_record'])
    text   = generate_explanation(prompt, model=state['model_name'])
    return {
        'explanations': {'zero_shot': text},
        'prompts_used': {'zero_shot': prompt},
    }


def contextualised_node(state: PipelineState) -> dict:
    prompt = build_contextualised(state['current_record'], state['context_payload'])
    text   = generate_explanation(prompt, model=state['model_name'])
    return {
        'explanations': {'contextualised': text},
        'prompts_used': {'contextualised': prompt},
    }


def rag_node(state: PipelineState) -> dict:
    prompt = build_rag(
        state['current_record'], state['context_payload'], state['retrieved_docs'])
    text = generate_explanation(prompt, model=state['model_name'])
    return {
        'explanations': {'rag': text},
        'prompts_used': {'rag': prompt},
    }


# ── Stage 5: HITL Operator Review ────────────────────────────────────────────

def operator_review_node(state: PipelineState) -> dict:
    """Pause graph execution for operator confirm / reject / snooze.

    The dict passed to interrupt() is surfaced to the caller as GraphInterrupt.value.
    Execution resumes when graph.stream(Command(resume=decision), config) is called.
    """
    decision = interrupt({
        'record':       state['current_record'],
        'context':      state['context_payload'],
        'explanations': state['explanations'],
        'prompts':      state['prompts_used'],
        'rag_docs':     state['retrieved_docs'],
    })
    return {'operator_decision': decision}


# ── Post-review: package result and advance loop counter ─────────────────────

def advance_record_node(state: PipelineState) -> dict:
    record = state['current_record']
    result = {
        'row_idx':           record['row_idx'],
        'failure_type':      record['failure_type'],
        'agreement':         record.get('agreement'),
        'combined_score':    record['combined_score'],
        'context_payload':   state['context_payload'],
        'prompts':           state['prompts_used'],
        'explanations':      state['explanations'],
        'rag_docs_retrieved': [
            {'id': d['id'], 'title': d['title'], 'score': d['score']}
            for d in (state['retrieved_docs'] or [])
        ],
        'operator_decision': state['operator_decision'],
        'confidence_path':   'high',
    }
    return {
        'completed_results':  [result],
        'current_record_idx': state['current_record_idx'] + 1,
    }


# ── Low-confidence path: record without LLM or HITL ─────────────────────────

def monitor_only_node(state: PipelineState, config: RunnableConfig) -> dict:
    """Skip LLM explanation and operator review for low-confidence flags."""
    tid         = _thread(config)
    df_detected = get_resource(tid, 'df_detected')
    idx         = state['current_record_idx']
    record      = state['anomaly_records'][idx]
    ctx         = enrich_record(record, df_detected, state['ranges'])
    result = {
        'row_idx':           record['row_idx'],
        'failure_type':      record['failure_type'],
        'agreement':         record.get('agreement'),
        'combined_score':    record['combined_score'],
        'context_payload':   ctx,
        'prompts':           {},
        'explanations':      {'status': 'low_confidence_monitor_only'},
        'rag_docs_retrieved': [],
        'operator_decision': 'auto_monitor',
        'confidence_path':   'low',
    }
    return {
        'completed_results':  [result],
        'current_record_idx': idx + 1,
        'current_record':     record,
        'context_payload':    ctx,
    }


# ── Export ────────────────────────────────────────────────────────────────────

def export_node(state: PipelineState) -> dict:
    out = save_results(state['completed_results'], Path('data/output'))
    print(f'Exported {len(state["completed_results"])} records → {out}')
    return {}
