"""Streamlit dashboard — IoT Anomaly XAI (Stream mode).

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

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

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

# ── Session state bootstrap ───────────────────────────────────────────────────
if 'stream_started' not in st.session_state:
    st.session_state.stream_started    = False
    st.session_state.stream_done       = False
    st.session_state.stream_worker     = None
    st.session_state.stream_start_time = None
    st.session_state.stream_explained  = []
    st.session_state.stream_history    = []


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
    st.metric('Anomalies detected', len(st.session_state.stream_history))
    st.metric('Explained', len(st.session_state.stream_explained))


# ── Helpers ───────────────────────────────────────────────────────────────────

_SPEED_OPTIONS = {
    '5 rows/sec (fast)':   0.2,
    '1 row/sec (default)': 1.0,
    '1 row/5 sec (slow)':  5.0,
}


def _start_streaming(path: str, llm_model: str, row_interval: float) -> None:
    with st.spinner('Loading dataset and initialising detectors…'):
        worker = StreamingWorker(Path(path), llm_model, row_interval)
    try:
        with st.spinner('Building knowledge base for RAG explanations…'):
            worker._kb = build_kb(model=EMBED_MODEL)
    except Exception:
        worker._kb = None
    worker.start()
    st.session_state.stream_worker     = worker
    st.session_state.stream_started    = True
    st.session_state.stream_start_time = time.time()
    st.session_state.stream_explained  = []
    st.session_state.stream_history    = []
    st.session_state.stream_done       = False


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title('IoT Anomaly Detection — Stream Monitor')

if not st.session_state.stream_started:
    st.info(
        'Simulates a live sensor feed at a configurable rate. '
        'Anomalies are detected instantly; LLM explanations generate in the background.'
    )
    speed_label = st.selectbox(
        'Simulation speed',
        list(_SPEED_OPTIONS.keys()),
        index=1,  # default: 1 row/sec
    )
    row_interval = _SPEED_OPTIONS[speed_label]
    st.caption(f'Each sensor reading arrives every **{row_interval}s** — LLM runs asynchronously.')

    if st.button('Start Streaming', type='primary'):
        _start_streaming(data_path, model_name, row_interval)
        st.rerun()

else:
    worker: StreamingWorker = st.session_state.stream_worker

    stream_live_tab, stream_anomaly_tab = st.tabs(['Live Feed', 'Anomaly Explanations'])

    # ── Tab 1: Live Feed — auto-reruns every second ───────────────────────────
    with stream_live_tab:
        @st.fragment(run_every=1)
        def _monitor_fragment() -> None:
            w = st.session_state.stream_worker
            if w is None:
                return

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
                    if shap_vals:
                        import pandas as pd
                        st.caption('**SHAP — sensor contribution to anomaly score (Isolation Forest)**')
                        df_shap = pd.DataFrame({
                            'Sensor': [SENSOR_LABELS.get(k, k) for k in shap_vals],
                            'SHAP value': list(shap_vals.values()),
                        }).set_index('Sensor').sort_values('SHAP value')
                        st.bar_chart(df_shap)

        _anomaly_fragment()
