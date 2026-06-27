from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from .nodes import (
    advance_record_node,
    contextualised_node,
    detect_node,
    enrich_node,
    export_node,
    monitor_only_node,
    operator_review_node,
    preprocess_node,
    rag_node,
    retrieve_docs_node,
    zero_shot_node,
)
from .state import PipelineState


# ── Routing ───────────────────────────────────────────────────────────────────

def route_by_confidence(state: PipelineState) -> str:
    """Route based on detector agreement level, or to export when all records done."""
    idx     = state['current_record_idx']
    records = state.get('anomaly_records') or []

    if idx >= len(records):
        return 'export'

    agreement = records[idx].get('agreement', 'none')
    return 'high_confidence' if agreement in ('two_of_three', 'all_three') else 'low_confidence'


def _fan_out_explanations(state: PipelineState) -> list[Send]:
    """Dispatch three concurrent ChatOllama explanation agents via Send API."""
    return [
        Send('zero_shot',     state),
        Send('contextualised', state),
        Send('rag',           state),
    ]


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> tuple:
    """Build and compile the LangGraph pipeline. Returns (graph, checkpointer)."""
    builder = StateGraph(PipelineState)

    # Register all nodes
    builder.add_node('preprocess',      preprocess_node)
    builder.add_node('detect',          detect_node)
    builder.add_node('enrich',          enrich_node)
    builder.add_node('retrieve_docs',   retrieve_docs_node)
    builder.add_node('monitor_only',    monitor_only_node)
    builder.add_node('zero_shot',       zero_shot_node)
    builder.add_node('contextualised',  contextualised_node)
    builder.add_node('rag',             rag_node)
    builder.add_node('aggregate',       lambda s: {})   # sync barrier after fan-out
    builder.add_node('operator_review', operator_review_node)
    builder.add_node('advance_record',  advance_record_node)
    builder.add_node('export',          export_node)

    # Linear backbone
    builder.set_entry_point('preprocess')
    builder.add_edge('preprocess', 'detect')
    builder.add_edge('detect',     'enrich')

    # Confidence routing — shared across three loop-back sources
    for source in ('enrich', 'monitor_only', 'advance_record'):
        builder.add_conditional_edges(
            source,
            route_by_confidence,
            {
                'high_confidence': 'retrieve_docs',
                'low_confidence':  'monitor_only',
                'export':          'export',
            },
        )

    # Parallel Send fan-out: retrieve_docs → [zero_shot, contextualised, rag]
    builder.add_conditional_edges(
        'retrieve_docs',
        _fan_out_explanations,
        ['zero_shot', 'contextualised', 'rag'],
    )

    # All three explanation agents converge to the aggregate sync barrier
    builder.add_edge('zero_shot',      'aggregate')
    builder.add_edge('contextualised', 'aggregate')
    builder.add_edge('rag',            'aggregate')

    # HITL → advance → loop back (via route_by_confidence on advance_record)
    builder.add_edge('aggregate',       'operator_review')
    builder.add_edge('operator_review', 'advance_record')

    builder.add_edge('export', END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer), checkpointer
