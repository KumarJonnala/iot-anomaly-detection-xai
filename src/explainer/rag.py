import os

from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

from .constants import EMBED_MODEL, KB_ENTRIES, SENSOR_LABELS

_OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')


_DEFAULT_EMBED_MODEL = 'nomic-embed-text:latest'


class KnowledgeBase:
    """Vector store over KB_ENTRIES, embedded once at construction via OllamaEmbeddings."""

    def __init__(self, model: str = None) -> None:
        model    = model or EMBED_MODEL or _DEFAULT_EMBED_MODEL
        embedder = OllamaEmbeddings(model=model, base_url=_OLLAMA_BASE_URL)
        self._store  = InMemoryVectorStore(embedder)
        docs = [
            Document(
                page_content=entry['text'],
                metadata={'id': entry['id'], 'title': entry['title']},
            )
            for entry in KB_ENTRIES
        ]
        self._store.add_documents(docs)

    def retrieve(self, query: str, k: int = 2) -> list[dict]:
        """Top-k entries by cosine similarity to query."""
        results = self._store.similarity_search_with_score(query, k=k)
        return [
            {
                'id':    doc.metadata['id'],
                'title': doc.metadata['title'],
                'text':  doc.page_content,
                'score': float(score),
            }
            for doc, score in results
        ]

    def retrieve_for_record(self, record: dict, context_payload: dict, k: int = 2) -> list[dict]:
        """Build a query from anomaly context and retrieve top-k KB entries."""
        worst            = SENSOR_LABELS.get(record['worst_sensor'], record['worst_sensor'])
        failure          = record.get('failure_type', '')
        rules            = ' '.join(
            name for name in ('HDF', 'TWF', 'OSF')
            if record.get(f'rule_{name.lower()}')
        )
        ae_top           = context_payload.get('ae_attribution', [{}])
        top_sensor_label = ae_top[0].get('label', '') if ae_top else ''
        query = f'{worst} {top_sensor_label} anomaly {failure} {rules}'.strip()
        return self.retrieve(query, k=k)


def build_kb(model: str = None) -> KnowledgeBase:
    """Construct and return an embedded KnowledgeBase."""
    return KnowledgeBase(model=model)
