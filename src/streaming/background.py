import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.explainer.context import enrich_record
from src.explainer.llm import generate_explanation
from src.explainer.prompts import build_contextualised

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

    def __init__(self, path: Path, model_name: str, row_interval: float = 1.0) -> None:
        self._sim      = CSVStreamSimulator(path)
        self._detector = OnlineDetector(self._sim.df, self._sim.ranges)
        self._model    = model_name
        self._interval = row_interval

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

        # Thread-safe result channel: {'record', 'context', 'explanation'}
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
                self._executor.submit(self._explain, record)

            time.sleep(self._interval)

        with self._lock:
            self.done = True

    def _explain(self, record: dict) -> None:
        row = record['row_idx']
        print(f'[worker] explaining row {row} with model={self._model}', flush=True)
        try:
            context     = enrich_record(record, self._sim.df, self._sim.ranges)
            prompt      = build_contextualised(record, context)
            explanation = generate_explanation(prompt, model=self._model)
            print(f'[worker] explanation done for row {row}', flush=True)
            self.results_queue.put({
                'record':      record,
                'context':     context,
                'explanation': explanation,
            })
        except Exception as exc:
            print(f'[worker] explanation error for row {row}: {exc}', flush=True)
            self.results_queue.put({
                'record':      record,
                'context':     {},
                'explanation': f'[Error: {exc}]',
            })
        finally:
            with self._lock:
                self.pending_count -= 1
            print(f'[worker] pending now {self.pending_count}', flush=True)
