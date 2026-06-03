"""Thread-keyed in-memory store for non-msgpack-serializable objects.

LangGraph's MemorySaver checkpoints state via msgpack, so pd.DataFrame and
KnowledgeBase (which holds numpy arrays) cannot live in PipelineState.
Nodes retrieve them here using the thread_id from config['configurable'].
"""
from typing import Any

_registry: dict[str, dict[str, Any]] = {}


def set_resource(thread_id: str, key: str, value: Any) -> None:
    if thread_id not in _registry:
        _registry[thread_id] = {}
    _registry[thread_id][key] = value


def get_resource(thread_id: str, key: str, default: Any = None) -> Any:
    return _registry.get(thread_id, {}).get(key, default)


def clear_thread(thread_id: str) -> None:
    _registry.pop(thread_id, None)
