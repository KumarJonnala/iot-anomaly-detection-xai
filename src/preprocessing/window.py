import pandas as pd

from .constants import WINDOW_SIZE


def get_window(df: pd.DataFrame, idx: int, col: str, window: int = WINDOW_SIZE) -> list:
    """Return list of values in a ±window band around idx for col."""
    start = max(0, idx - window)
    end   = min(len(df), idx + window + 1)
    return df[col].iloc[start:end].round(4).tolist()
