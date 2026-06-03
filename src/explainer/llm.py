from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from .constants import EXPLAINER_MODEL

_SYSTEM_PROMPT = (
    'You are an experienced IoT maintenance analyst. '
    'You interpret sensor anomalies for machine operators who have no ML background. '
    'Be precise, concise, and actionable. Use engineering units when available.'
)


def check_ollama(base_url: str = 'http://localhost:11434') -> bool:
    """Return True if Ollama is reachable at base_url."""
    try:
        import urllib.request
        with urllib.request.urlopen(base_url, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def make_llm(model: str = EXPLAINER_MODEL, temperature: float = 0.3) -> ChatOllama:
    """Construct a ChatOllama instance."""
    return ChatOllama(model=model, temperature=temperature)


def generate_explanation(
    prompt: str,
    model: str = EXPLAINER_MODEL,
    options: dict | None = None,
) -> str:
    """Generate a natural language explanation via ChatOllama (blocking).

    Signature unchanged — existing pipeline.py and notebooks work without modification.
    """
    temperature = (options or {}).get('temperature', 0.3)
    llm = make_llm(model=model, temperature=temperature)
    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    return response.content.strip()


def stream_explanation(
    prompt: str,
    model: str = EXPLAINER_MODEL,
    options: dict | None = None,
):
    """Generator: yields string tokens as they arrive from Ollama.

    Use with Streamlit's st.write_stream() for real-time display.
    """
    temperature = (options or {}).get('temperature', 0.3)
    llm = make_llm(model=model, temperature=temperature)
    for chunk in llm.stream([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]):
        yield chunk.content
