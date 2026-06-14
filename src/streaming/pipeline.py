from pathlib import Path
from typing import Generator

from .online_detector import OnlineDetector
from .simulator import CSVStreamSimulator


def stream_rows(path: Path) -> Generator[dict, None, None]:
    """Replay a CSV row by row, yielding a status dict for every row.

    The caller controls pacing — no sleep is applied here.

    Yields:
        {
            'row_idx':       int,
            'is_anomaly':    bool,
            'record':        dict | None,   # full anomaly record when is_anomaly
            'combined_score': float,
        }
    """
    sim      = CSVStreamSimulator(path)
    detector = OnlineDetector(sim.df, sim.ranges)

    for row_idx, row in sim.stream():
        record = detector.score_row(row_idx, row)
        yield {
            'row_idx':        row_idx,
            'is_anomaly':     record is not None,
            'record':         record,
            'combined_score': record['combined_score'] if record else 0.0,
        }
