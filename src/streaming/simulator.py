import json
from pathlib import Path
from typing import Generator

import pandas as pd

from src.preprocessing.pipeline import clean, engineer_features, load_raw, normalise


class CSVStreamSimulator:
    """Simulates a live sensor feed by replaying a preprocessed CSV row by row.

    Accepts either the raw AI4I CSV (has 'UDI'/'Product ID' columns) or the
    already-processed ai4i_clean.csv.  For the latter, ranges are loaded from
    the companion ai4i_ranges.json so denormalisation stays correct.
    """

    def __init__(self, path: Path) -> None:
        raw = load_raw(path)
        if 'UDI' in raw.columns or 'Product ID' in raw.columns:
            # Raw CSV — run full preprocessing pipeline
            df_clean = engineer_features(clean(raw))
            self._df, self._ranges = normalise(df_clean)
        else:
            # Already-processed CSV — load companion ranges file
            ranges_path = path.parent / 'ai4i_ranges.json'
            if not ranges_path.exists():
                # Fall back: re-run engineer + normalise (ranges will be [0,1])
                self._df, self._ranges = normalise(engineer_features(raw))
            else:
                with open(ranges_path) as fh:
                    self._ranges = json.load(fh)
                self._df = raw

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def ranges(self) -> dict:
        return self._ranges

    @property
    def total_rows(self) -> int:
        return len(self._df)

    def stream(self) -> Generator[tuple[int, pd.Series], None, None]:
        """Yield (row_idx, row_series) for every row in the dataset."""
        for idx in range(len(self._df)):
            yield idx, self._df.iloc[idx]
