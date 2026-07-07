from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from .nodes import (
    contextualised_node,
    enrich_node,
    operator_review_node,
    rag_node,
    retrieve_docs_node,
    zero_shot_node,
)
from .state import PipelineState


def _fan_out_explanations(state: PipelineState) -> list[Send]:
    return [
        Send('zero_shot',      state),
        Send('contextualised', state),
        Send('rag',            state),
    ]


def collect_result_node(state: PipelineState) -> dict:
    """Package the completed record after operator decision. No loop — single record only."""
    record = state['current_record']
    result = {
        'row_idx':            record['row_idx'],
        'failure_type':       record['failure_type'],
        'agreement':          record.get('agreement'),
        'combined_score':     record['combined_score'],
        'context_payload':    state['context_payload'],
        'prompts':            state['prompts_used'],
        'explanations':       state['explanations'],
        'rag_docs_retrieved': [
            {'id': d['id'], 'title': d['title'], 'score': d['score']}
            for d in (state['retrieved_docs'] or [])
        ],
        'operator_decision':  state['operator_decision'],
        'confidence_path':    'stream',
    }
    return {'completed_results': [result]}


def build_stream_graph() -> tuple:
    """Single-record explain+HITL graph for streaming mode.

    Entry: enrich (takes anomaly_records=[record], current_record_idx=0)
    Exit:  collect_result → END

    Use a fresh graph instance + fresh thread_id per anomaly so there is no
    state carryover between records.
    """
    builder = StateGraph(PipelineState)

    builder.add_node('enrich',          enrich_node)
    builder.add_node('retrieve_docs',   retrieve_docs_node)
    builder.add_node('zero_shot',       zero_shot_node)
    builder.add_node('contextualised',  contextualised_node)
    builder.add_node('rag',             rag_node)
    builder.add_node('aggregate',       lambda s: {})
    builder.add_node('operator_review', operator_review_node)
    builder.add_node('collect_result',  collect_result_node)

    builder.set_entry_point('enrich')
    builder.add_edge('enrich', 'retrieve_docs')

    builder.add_conditional_edges(
        'retrieve_docs',
        _fan_out_explanations,
        ['zero_shot', 'contextualised', 'rag'],
    )

    builder.add_edge('zero_shot',      'aggregate')
    builder.add_edge('contextualised', 'aggregate')
    builder.add_edge('rag',            'aggregate')

    builder.add_edge('aggregate',       'operator_review')
    builder.add_edge('operator_review', 'collect_result')
    builder.add_edge('collect_result',  END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer), checkpointer
