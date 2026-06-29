import os

from langchain_core.messages import HumanMessage, SystemMessage

from .constants import EXPLAINER_MODEL

_OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
_GROQ_API_KEY    = os.getenv('GROQ_API_KEY')

_groq_models_env = os.getenv('GROQ_MODELS', 'llama-3.3-70b-versatile,llama-3.1-8b-instant,gemma2-9b-it,mixtral-8x7b-32768')
GROQ_MODELS: list[str] = [m.strip() for m in _groq_models_env.split(',') if m.strip()]

_SYSTEM_PROMPT = (
    'You are an experienced IoT maintenance analyst. '
    'You interpret sensor anomalies for machine operators who have no ML background. '
    'Be precise, concise, and actionable. Use engineering units when available.'
)


def _is_groq(model: str) -> bool:
    """Groq model names have no ':' tag; Ollama names do (e.g. gemma3:4b)."""
    return ':' not in model


def groq_available() -> bool:
    return bool(_GROQ_API_KEY)


def check_ollama(base_url: str = _OLLAMA_BASE_URL) -> bool:
    """Return True if Ollama is reachable."""
    try:
        import urllib.request
        with urllib.request.urlopen(base_url, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def make_llm(model: str = EXPLAINER_MODEL, temperature: float = 0.3):
    """Return a LangChain chat model — Groq or Ollama based on model name."""
    if _is_groq(model):
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature, api_key=_GROQ_API_KEY)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, temperature=temperature, base_url=_OLLAMA_BASE_URL)


def generate_explanation(
    prompt: str,
    model: str = EXPLAINER_MODEL,
    options: dict | None = None,
) -> str:
    """Generate a natural language explanation (blocking)."""
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
    """Generator: yield string tokens as they arrive. Use with st.write_stream()."""
    temperature = (options or {}).get('temperature', 0.3)
    llm = make_llm(model=model, temperature=temperature)
    for chunk in llm.stream([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]):
        yield chunk.content
