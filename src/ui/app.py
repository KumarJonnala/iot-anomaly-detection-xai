"""Streamlit dashboard — IoT Anomaly XAI.

Run from project root:
    streamlit run src/ui/app.py
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src.*' imports resolve regardless
# of the working directory Streamlit uses when launching the script.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import time
import urllib.request
import uuid

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langgraph.types import Command

from src.agents.graph import build_graph
from src.agents.store import set_resource
from src.config import EMBED_MODEL, EXPLAINER_MODEL, SENSOR_LABELS
from src.explainer.llm import GROQ_MODELS, groq_available, stream_explanation
from src.explainer.rag import build_kb
from src.streaming.background import StreamingWorker

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='IoT Anomaly XAI',
    page_icon='🔧',
    layout='wide',
)

# ── Session state bootstrap — batch mode ──────────────────────────────────────
if 'graph' not in st.session_state:
    graph, _ = build_graph()
    st.session_state.graph        = graph
    st.session_state.thread_id    = str(uuid.uuid4())
    st.session_state.config       = {'configurable': {'thread_id': st.session_state.thread_id}}
    st.session_state.interrupt    = None
    st.session_state.started      = False
    st.session_state.finished     = False
    st.session_state.history      = []
    st.session_state.record_count = 0

# ── Session state bootstrap — streaming mode ──────────────────────────────────
if 'stream_started' not in st.session_state:
    st.session_state.stream_started    = False
    st.session_state.stream_done       = False
    st.session_state.stream_worker     = None
    st.session_state.stream_start_time = None
    st.session_state.stream_explained  = []   # list of {record, context, explanation}
    st.session_state.stream_history    = []   # lightweight anomaly dicts for sidebar


# ── Sidebar helpers ───────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _ollama_models() -> list[str]:
    """Fetch model names from local Ollama; fall back to defaults."""
    try:
        with urllib.request.urlopen('http://localhost:11434/api/tags', timeout=3) as r:
            data = json.loads(r.read())
        names = [m['name'] for m in data.get('models', [])]
        return [n for n in names if 'embed' not in n.lower()] or [EXPLAINER_MODEL]
    except Exception:
        return [EXPLAINER_MODEL]


@st.cache_data(ttl=300)
def _available_datasets(data_dir: str = 'data') -> list[str]:
    """List CSV files under data/input/ and data/output/."""
    csvs = sorted(Path(data_dir).glob('**/*.csv'))
    return [str(p) for p in csvs] or ['data/input/ai4i2020.csv']


# ── Shared sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('Pipeline Control')
    st.markdown('---')

    _datasets = _available_datasets()
    _default_ds = 'data/input/ai4i_clean.csv'
    data_path = st.selectbox(
        'Dataset',
        _datasets,
        index=_datasets.index(_default_ds) if _default_ds in _datasets else 0,
    )

    _providers = ['Groq (cloud)', 'Ollama (local)'] if groq_available() else ['Ollama (local)']
    provider = st.selectbox('LLM provider', _providers)

    if provider == 'Groq (cloud)':
        _models = GROQ_MODELS
        _default_llm = 'llama-3.3-70b-versatile'
    else:
        _models = _ollama_models()
        _default_llm = EXPLAINER_MODEL if EXPLAINER_MODEL in _models else _models[0]

    model_name = st.selectbox(
        'LLM model',
        _models,
        index=_models.index(_default_llm) if _default_llm in _models else 0,
    )

    st.markdown('---')
    st.caption('**Batch mode**')
    st.metric('Records reviewed', st.session_state.record_count)

    if st.session_state.history:
        decisions = [r.get('operator_decision') for r in st.session_state.history]
        st.caption(
            f"Confirmed: {decisions.count('confirm')} | "
            f"Rejected: {decisions.count('reject')} | "
            f"Snoozed: {decisions.count('snooze')} | "
            f"Auto: {decisions.count('auto_monitor')}"
        )

    st.markdown('---')
    st.caption('**Stream mode**')
    st.metric('Stream anomalies detected', len(st.session_state.stream_history))
    st.metric('Explained', len(st.session_state.stream_explained))


# ── Helpers — batch ───────────────────────────────────────────────────────────

def _advance_graph(input_or_command) -> None:
    graph  = st.session_state.graph
    config = st.session_state.config
    for chunk in graph.stream(input_or_command, config, stream_mode='values'):
        results = chunk.get('completed_results', [])
        if results:
            st.session_state.history      = results
            st.session_state.record_count = len(results)

    snapshot = graph.get_state(config)
    if snapshot.next:
        interrupts = snapshot.tasks[0].interrupts if snapshot.tasks else ()
        st.session_state.interrupt = interrupts[0].value if interrupts else {}
    else:
        st.session_state.interrupt = None
        st.session_state.finished  = True
    st.rerun()


# ── Helpers — streaming ───────────────────────────────────────────────────────

_SPEED_OPTIONS = {
    '10 rows/sec (fast demo)': 0.1,
    '2 rows/sec':              0.5,
    '1 row/sec (default)':     1.0,
    '1 row/2 sec':             2.0,
    '1 row/5 sec (slow)':      5.0,
}


def _start_streaming(path: str, llm_model: str, row_interval: float) -> None:
    with st.spinner('Loading dataset and initialising detectors…'):
        worker = StreamingWorker(Path(path), llm_model, row_interval)
    try:
        with st.spinner('Building knowledge base for RAG explanations…'):
            worker._kb = build_kb(model=EMBED_MODEL)
    except Exception:
        worker._kb = None   # RAG unavailable; other explanation types still work
    worker.start()
    st.session_state.stream_worker     = worker
    st.session_state.stream_started    = True
    st.session_state.stream_start_time = time.time()
    st.session_state.stream_explained  = []
    st.session_state.stream_history    = []
    st.session_state.stream_done       = False


# ── Tab rendering ─────────────────────────────────────────────────────────────

def render_batch_tab() -> None:
    if st.session_state.finished:
        st.success('All anomaly records processed.')
        st.balloons()
        return

    if not st.session_state.started:
        st.title('IoT Anomaly Detection — XAI Dashboard')
        st.info('Configure the dataset path and LLM model in the sidebar, then click **Start Pipeline**.')

        if st.button('Start Pipeline', type='primary'):
            with st.spinner('Building knowledge base...'):
                kb = build_kb(model=EMBED_MODEL)
            set_resource(st.session_state.thread_id, 'kb', kb)
            st.session_state.started = True
            _advance_graph({
                'data_path':          data_path,
                'model_name':         model_name,
                'completed_results':  [],
                'current_record_idx': 0,
            })
        return

    interrupt = st.session_state.interrupt

    if interrupt is None:
        st.title('Processing...')
        st.info('Running preprocessing, detection, and enrichment.')
        return

    record       = interrupt['record']
    context      = interrupt['context']
    explanations = interrupt['explanations']
    prompts      = interrupt['prompts']
    rag_docs     = interrupt['rag_docs'] or []

    st.title(f'Anomaly Review — Row {record["row_idx"]}')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Combined Score',      f'{record["combined_score"]:.3f}')
    col2.metric('Detector Agreement',  record['agreement'].replace('_', ' ').title())
    col3.metric('Failure Type',        record['failure_type'])
    col4.metric('True Label',          'Failure' if record['true_label'] == 1 else 'Normal')

    st.markdown('---')
    left, right = st.columns([1, 2])

    with left:
        st.subheader('Detector Summary')
        st.caption(context.get('detector_summary', ''))

        st.subheader('Domain Rules')
        st.caption(context.get('rule_explanation', 'No domain rules triggered.'))

        if rag_docs:
            st.subheader('Retrieved KB Entries')
            for doc in rag_docs:
                st.caption(f"**[{doc['id']}]** {doc['title']} *(score: {doc['score']:.3f})*")

        st.subheader('AE Attribution')
        for a in context.get('ae_attribution', [])[:3]:
            st.caption(f"{a['label']}: {a['pct']:.1f}% of reconstruction error")

    with right:
        st.subheader('LLM Explanations')
        tab1, tab2, tab3 = st.tabs(['Zero-Shot', 'Contextualised', 'RAG'])
        with tab1:
            text = explanations.get('zero_shot', '')
            st.markdown(text) if text else st.info('No explanation generated.')
            if st.button('Re-stream Zero-Shot', key='b_zs'):
                st.write_stream(stream_explanation(prompts.get('zero_shot', ''), model_name))
        with tab2:
            text = explanations.get('contextualised', '')
            st.markdown(text) if text else st.info('No explanation generated.')
            if st.button('Re-stream Contextualised', key='b_ctx'):
                st.write_stream(stream_explanation(prompts.get('contextualised', ''), model_name))
        with tab3:
            text = explanations.get('rag', '')
            st.markdown(text) if text else st.info('No explanation generated.')
            if st.button('Re-stream RAG', key='b_rag'):
                st.write_stream(stream_explanation(prompts.get('rag', ''), model_name))

    st.markdown('---')
    st.subheader('Operator Decision')
    st.caption('Your decision is logged and the pipeline advances to the next anomaly.')
    btn1, btn2, btn3 = st.columns(3)
    if btn1.button('Confirm — anomaly is real',      type='primary', use_container_width=True):
        _advance_graph(Command(resume='confirm'))
    if btn2.button('Reject — false positive',                        use_container_width=True):
        _advance_graph(Command(resume='reject'))
    if btn3.button('Snooze — monitor, revisit later',                use_container_width=True):
        _advance_graph(Command(resume='snooze'))

    if st.session_state.history:
        with st.expander(f'Review history ({len(st.session_state.history)} records)'):
            for r in reversed(st.session_state.history):
                decision = r.get('operator_decision', '—')
                icon = {'confirm': '✅', 'reject': '❌', 'snooze': '⏸️',
                        'auto_monitor': '👁️'}.get(decision, '—')
                st.caption(
                    f"{icon} Row {r['row_idx']} | {r['failure_type']} | "
                    f"score={r['combined_score']:.3f} | {r['agreement']} | {decision}"
                )


def render_stream_tab() -> None:
    st.title('Stream Monitor')

    # ── Pre-start ─────────────────────────────────────────────────────────────
    if not st.session_state.stream_started:
        st.info(
            'Simulates a live sensor feed at a configurable rate. '
            'Anomalies are detected instantly; LLM explanations generate in the background.'
        )
        speed_label = st.selectbox(
            'Simulation speed',
            list(_SPEED_OPTIONS.keys()),
            index=2,  # default: 1 row/sec
        )
        row_interval = _SPEED_OPTIONS[speed_label]
        st.caption(f'Each sensor reading arrives every **{row_interval}s** — LLM runs asynchronously.')

        if st.button('Start Streaming', type='primary'):
            _start_streaming(data_path, model_name, row_interval)
            st.rerun()
        return

    worker: StreamingWorker = st.session_state.stream_worker

    stream_live_tab, stream_anomaly_tab = st.tabs(['Live Feed', 'Anomaly Explanations'])

    # ── Tab 1: Live Feed — auto-reruns every second ───────────────────────────
    with stream_live_tab:
        @st.fragment(run_every=1)
        def _monitor_fragment() -> None:
            w = st.session_state.stream_worker
            if w is None:
                return

            # Drain again inside the fragment so the live tab stays current
            while not w.results_queue.empty():
                result = w.results_queue.get_nowait()
                st.session_state.stream_explained.append(result)
                st.session_state.stream_history.append(result['record'])
            if w.done and not st.session_state.stream_done:
                st.session_state.stream_done = True

            elapsed  = time.time() - st.session_state.stream_start_time
            rows_sec = w.rows_processed / max(elapsed, 1e-3)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Rows processed',      f'{w.rows_processed:,} / {w.total_rows:,}')
            c2.metric('Anomalies detected',   w.anomalies_found)
            c3.metric('Awaiting explanation', w.pending_count)
            c4.metric('Rows/sec',             f'{rows_sec:.2f}')

            st.progress(min(w.rows_processed / max(w.total_rows, 1), 1.0))

            # Pause / resume button
            if not st.session_state.stream_done:
                if getattr(w, 'paused', False):
                    if st.button('▶ Resume Stream', type='primary'):
                        w.resume()
                else:
                    if st.button('⏸ Pause Stream'):
                        w.pause()

            feed = list(w.live_feed)
            if feed:
                for item in reversed(feed[-15:]):
                    icon  = '⚠️' if item['is_anomaly'] else '✓'
                    label = f"ANOMALY ({item['failure_type']})" if item['is_anomaly'] else 'normal'
                    st.caption(
                        f"{icon}  Row {item['row_idx']:05d}  —  {label}  "
                        f"(score={item['combined_score']:.3f})"
                    )
            else:
                st.caption('Waiting for first row…')

            if st.session_state.stream_done:
                st.success(
                    f'Stream complete — {w.rows_processed:,} rows processed, '
                    f'{w.anomalies_found} anomalies detected, '
                    f'{len(st.session_state.stream_explained)} explained.'
                )

        _monitor_fragment()

        if st.session_state.stream_done:
            if st.button('Reset Stream'):
                worker.stop()
                for k in [k for k in st.session_state if k.startswith('stream_')]:
                    del st.session_state[k]
                st.rerun()

    # ── Tab 2: Anomaly Explanations — auto-refreshes every 2 s ──────────────
    with stream_anomaly_tab:
        @st.fragment(run_every=2)
        def _anomaly_fragment() -> None:
            w = st.session_state.stream_worker
            if w is None:
                return

            # Drain queue here too so this tab is self-sufficient
            # (safe: queue.get_nowait() is atomic; no race with Live Feed fragment
            #  because only one fragment runs at a time in a given session)
            while not w.results_queue.empty():
                result = w.results_queue.get_nowait()
                st.session_state.stream_explained.append(result)
                st.session_state.stream_history.append(result['record'])

            explained = st.session_state.stream_explained
            pending   = w.pending_count

            st.caption(
                f'{len(explained)} explained'
                + (f' · {pending} generating…' if pending > 0 else '')
            )

            if not explained and w.anomalies_found == 0:
                st.info('No anomalies detected yet — check back as the stream runs.')
            elif not explained:
                st.info(
                    f'{w.anomalies_found} anomaly/anomalies detected — '
                    'LLM explanations generating in background…'
                )

            for result in reversed(explained):
                rec   = result['record']
                agree = rec.get('agreement', '').replace('_', ' ')
                with st.container(border=True):
                    h1, h2, h3, h4 = st.columns([2, 2, 2, 2])
                    h1.markdown(f"**Row {rec['row_idx']:05d}**")
                    h2.markdown(f"🔴 {rec['failure_type']}")
                    h3.markdown(f"Score: `{rec['combined_score']:.3f}`")
                    h4.markdown(f"Agreement: {agree}")
                    explanations = result.get('explanations', {})
                    ez, ec, er = st.tabs(['Zero-Shot', 'Contextualised', 'RAG'])
                    with ez:
                        st.markdown(explanations.get('zero_shot', '—'))
                    with ec:
                        st.markdown(explanations.get('contextualised', '—'))
                    with er:
                        rdocs = result.get('rag_docs', [])
                        if rdocs:
                            st.caption(' · '.join(
                                f"[{d['id']}] {d['title']} ({d['score']:.2f})" for d in rdocs
                            ))
                        st.markdown(explanations.get('rag', '—'))

                    shap_vals = result.get('shap_values', {})
                    ae_errors = rec.get('ae_error_per_sensor', {})
                    if shap_vals or ae_errors:
                        import pandas as pd
                        chart_left, chart_right = st.columns(2)
                        if shap_vals:
                            with chart_left:
                                st.caption('**IF SHAP — sensor contribution to anomaly score**')
                                df_shap = pd.DataFrame({
                                    'Sensor': [SENSOR_LABELS.get(k, k) for k in shap_vals],
                                    'SHAP value': list(shap_vals.values()),
                                }).set_index('Sensor').sort_values('SHAP value')
                                st.bar_chart(df_shap)
                        if ae_errors:
                            with chart_right:
                                total = sum(ae_errors.values()) or 1.0
                                st.caption('**AE reconstruction error — % per sensor**')
                                df_ae = pd.DataFrame({
                                    'Sensor': [SENSOR_LABELS.get(k, k) for k in ae_errors],
                                    '% error': [100 * v / total for v in ae_errors.values()],
                                }).set_index('Sensor').sort_values('% error')
                                st.bar_chart(df_ae)

        _anomaly_fragment()


# ── Main layout ───────────────────────────────────────────────────────────────
tab_batch, tab_stream = st.tabs(['Batch Mode', 'Stream Simulation'])

with tab_batch:
    render_batch_tab()

with tab_stream:
    render_stream_tab()
