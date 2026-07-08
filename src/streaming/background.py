import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.explainer.context import enrich_record
from src.explainer.llm import generate_explanation
from src.explainer.prompts import build_zero_shot, build_contextualised, build_rag
from src.explainer.rag import KnowledgeBase
from src.explainer.shap_explain import SHAPExplainer
from src.config import SENSOR_COLS

from .online_detector import OnlineDetector
from .simulator import CSVStreamSimulator


class StreamingWorker:
    """Streaming detection + async LLM explanation in daemon threads.

    The background thread owns the CSV replay loop, sleeping `row_interval`
    seconds between rows to simulate a realistic sensor feed.  When an anomaly
    is detected, an LLM explanation job is submitted to a ThreadPoolExecutor
    so detection continues without waiting for the (slow) LLM call to finish.
    Completed explanations are placed in `results_queue` for the UI to consume.
    """

    def __init__(self, path: Path, model_name: str, row_interval: float = 1.0, kb: KnowledgeBase | None = None) -> None:
        self._sim      = CSVStreamSimulator(path)
        self._detector = OnlineDetector(self._sim.df, self._sim.ranges)
        self._model    = model_name
        self._interval = row_interval
        self._kb       = kb

        self._shap     = SHAPExplainer(self._detector.clf_if, self._sim.df)
        self._sensor_cols = SENSOR_COLS

        self._stop     = threading.Event()
        self._pause    = threading.Event()   # set = paused, clear = running
        self._lock     = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=3)

        # Public state — read by UI thread; only updated under _lock
        self.rows_processed  = 0
        self.anomalies_found = 0
        self.live_feed: list[dict] = []   # last 20 row status dicts
        self.pending_count   = 0
        self.done            = False
        self.paused          = False
        self.total_rows      = self._sim.total_rows

        # Thread-safe result channel: {'record', 'context', 'explanations', 'prompts', 'rag_docs', 'shap_values'}
        self.results_queue: queue.Queue = queue.Queue()

        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name='StreamWorker')
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._pause.clear()          # unblock thread so it can see the stop signal
        self._executor.shutdown(wait=False, cancel_futures=True)

    def pause(self) -> None:
        self._pause.set()
        with self._lock:
            self.paused = True

    def resume(self) -> None:
        self._pause.clear()
        with self._lock:
            self.paused = False

    # ── internal ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        for row_idx, row in self._sim.stream():
            # Block here while paused; wake up immediately if stopped
            while self._pause.is_set():
                if self._stop.is_set():
                    return
                time.sleep(0.1)

            if self._stop.is_set():
                break

            record = self._detector.score_row(row_idx, row)
            status = {
                'row_idx':        row_idx,
                'is_anomaly':     record is not None,
                'combined_score': record['combined_score'] if record else 0.0,
                'failure_type':   record['failure_type']   if record else 'NORMAL',
            }

            with self._lock:
                self.rows_processed += 1
                self.live_feed.append(status)
                if len(self.live_feed) > 20:
                    self.live_feed.pop(0)
                if record is not None:
                    self.anomalies_found += 1
                    self.pending_count   += 1

            if record is not None:
                row_vals = row[self._sensor_cols].values.astype('float32')
                self._executor.submit(self._explain, record, row_vals)

            time.sleep(self._interval)

        with self._lock:
            self.done = True

    def _explain(self, record: dict, row_values: np.ndarray) -> None:
        row = record['row_idx']
        print(f'[worker] explaining row {row} with model={self._model}', flush=True)
        try:
            context  = enrich_record(record, self._sim.df, self._sim.ranges)
            rag_docs = self._kb.retrieve_for_record(record, context) if self._kb else []
            prompts  = {
                'zero_shot':      build_zero_shot(record),
                'contextualised': build_contextualised(record, context),
                'rag':            build_rag(record, context, rag_docs),
            }
            explanations = {k: generate_explanation(v, model=self._model) for k, v in prompts.items()}
            shap_values  = self._shap.explain(row_values)
            print(f'[worker] explanation done for row {row}', flush=True)
            self.results_queue.put({
                'record':       record,
                'context':      context,
                'explanations': explanations,
                'prompts':      prompts,
                'rag_docs':     [{'id': d['id'], 'title': d['title'], 'score': d['score']} for d in rag_docs],
                'shap_values':  shap_values,
            })
        except Exception as exc:
            print(f'[worker] explanation error for row {row}: {exc}', flush=True)
            self.results_queue.put({
                'record':       record,
                'context':      {},
                'explanations': {k: f'[Error: {exc}]' for k in ('zero_shot', 'contextualised', 'rag')},
                'prompts':      {},
                'rag_docs':     [],
                'shap_values':  {},
            })
        finally:
            with self._lock:
                self.pending_count -= 1
            print(f'[worker] pending now {self.pending_count}', flush=True)
