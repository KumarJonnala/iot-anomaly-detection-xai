import numpy as np
from sklearn.ensemble import IsolationForest

from .constants import IF_CONTAMINATION


def fit_isolation_forest(
    X: np.ndarray,
    contamination: float = IF_CONTAMINATION,
    n_estimators: int = 200,
    random_state: int = 42,
) -> IsolationForest:
    """Fit and return an IsolationForest on the full sensor matrix."""
    clf = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    clf.fit(X)
    return clf


def score_isolation_forest(
    clf: IsolationForest,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Score rows with a fitted IsolationForest.

    Returns:
        if_scores: float array [0, 1], higher = more anomalous
        if_flags:  bool array, True where clf predicts anomaly
    """
    raw_scores = clf.score_samples(X)
    denom      = raw_scores.max() - raw_scores.min() + 1e-9
    if_scores  = np.clip(1 - (raw_scores - raw_scores.min()) / denom, 0, 1)
    if_flags   = clf.predict(X) == -1
    return if_scores, if_flags
