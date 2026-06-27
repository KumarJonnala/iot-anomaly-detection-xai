from typing import Annotated, Any, Optional
from typing_extensions import TypedDict


# ── Reducers ──────────────────────────────────────────────────────────────────

def _last(current: Any, update: Any) -> Any:
    """Last-write-wins: standard sequential node behaviour."""
    return update


def _merge(current: Optional[dict], update: dict) -> dict:
    """Merge two dicts — used for parallel fan-out nodes writing partial results.

    Each of the three LLM explanation nodes writes one key to 'explanations'
    and 'prompts_used'. This reducer combines them as they complete.
    """
    return {**(current or {}), **update}


def _append(current: Optional[list], update: list) -> list:
    """Accumulate completed anomaly results across loop iterations."""
    return (current or []) + update


# ── State ─────────────────────────────────────────────────────────────────────
# Only msgpack-serializable types live here.
# Non-serializable objects (pd.DataFrame, KnowledgeBase) are held in
# src/agents/store.py keyed by thread_id and accessed via config['configurable'].

class PipelineState(TypedDict):
    # Stage 1 — Preprocessing
    data_path:           Annotated[Optional[str],  _last]
    ranges:              Annotated[Optional[dict], _last]   # min/max/mean/std per sensor

    # Stage 2 — Detection
    anomaly_records:     Annotated[Optional[list], _last]   # list of serializable dicts
    current_record_idx:  Annotated[int,            _last]

    # Per-anomaly working fields (reset each loop iteration in enrich_node)
    current_record:      Annotated[Optional[dict], _last]
    context_payload:     Annotated[Optional[dict], _last]
    retrieved_docs:      Annotated[Optional[list], _last]

    # Stage 4 — Parallel LLM results
    # _merge lets zero_shot/contextualised/rag nodes each write one key
    explanations:        Annotated[Optional[dict], _merge]
    prompts_used:        Annotated[Optional[dict], _merge]

    # Stage 5 — HITL operator decision
    operator_decision:   Annotated[Optional[str],  _last]

    # Accumulated across all loop iterations
    completed_results:   Annotated[list,           _append]

    # Shared serializable config
    model_name:          Annotated[str,            _last]
