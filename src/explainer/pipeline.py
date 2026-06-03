import pandas as pd

try:
    from .constants import EXPLAINER_MODEL, PRE_WINDOW_SIZE
    from .context import enrich_record
    from .llm import generate_explanation
    from .prompts import build_contextualised, build_rag, build_zero_shot
    from .rag import KnowledgeBase
except ImportError:
    # Running as a script — add project root to sys.path
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.explainer.constants import EXPLAINER_MODEL, PRE_WINDOW_SIZE
    from src.explainer.context import enrich_record
    from src.explainer.llm import generate_explanation
    from src.explainer.prompts import build_contextualised, build_rag, build_zero_shot
    from src.explainer.rag import KnowledgeBase

_DEFAULT_OPTIONS = {'temperature': 0.3}


def run_single(
    record: dict,
    context_payload: dict,
    kb: KnowledgeBase,
    model: str = EXPLAINER_MODEL,
    rag_k: int = 2,
) -> dict:
    """Run all three prompting strategies for one anomaly record."""
    rag_docs = kb.retrieve_for_record(record, context_payload, k=rag_k)

    prompts = {
        'zero_shot':      build_zero_shot(record),
        'contextualised': build_contextualised(record, context_payload),
        'rag':            build_rag(record, context_payload, rag_docs),
    }

    explanations = {
        strategy: generate_explanation(prompt, model=model, options=_DEFAULT_OPTIONS)
        for strategy, prompt in prompts.items()
    }

    return {
        'row_idx':          record['row_idx'],
        'failure_type':     record['failure_type'],
        'agreement':        record.get('agreement'),
        'combined_score':   record['combined_score'],
        'context_payload':  context_payload,
        'prompts':          prompts,
        'explanations':     explanations,
        'rag_docs_retrieved': [
            {'id': d['id'], 'title': d['title'], 'score': d['score']}
            for d in rag_docs
        ],
    }


def run_pipeline(
    records: list[dict],
    df: pd.DataFrame,
    ranges: dict,
    kb: KnowledgeBase,
    model: str = EXPLAINER_MODEL,
    pre_window: int = PRE_WINDOW_SIZE,
    rag_k: int = 2,
    verbose: bool = True,
) -> list[dict]:
    """Enrich and explain every record in the list. Returns results in same order."""
    results = []
    n = len(records)
    for i, record in enumerate(records):
        if verbose:
            print(f'[{i+1:3d}/{n}] row={record["row_idx"]:5d}  type={record["failure_type"]:6s}  '
                  f'agreement={record.get("agreement",""):12s}  score={record["combined_score"]:.3f}')
        ctx = enrich_record(record, df, ranges, pre_window)
        results.append(run_single(record, ctx, kb, model=model, rag_k=rag_k))
    return results


if __name__ == '__main__':
    import json
    import sys
    from pathlib import Path

    # Optional: python3 src/explainer/pipeline.py 5   (runs on first N records)
    n_records = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    data_dir = Path(__file__).resolve().parent.parent.parent / 'data'

    print(f'Loading data from {data_dir}...')
    df_main = pd.read_csv(data_dir / 'ai4i_clean.csv')
    with open(data_dir / 'ai4i_ranges.json') as f:
        ranges_main = json.load(f)
    with open(data_dir / 'ai4i_anomaly_records.json') as f:
        records_main = json.load(f)

    print(f'Building knowledge base...')
    from src.explainer.rag import build_kb
    kb_main = build_kb()

    print(f'Running pipeline on first {n_records} record(s)...\n')
    results_main = run_pipeline(records_main[:n_records], df_main, ranges_main, kb_main)

    print()
    for r in results_main:
        print(f"{'='*60}")
        print(f"Row {r['row_idx']} | {r['failure_type']} | {r['agreement']} | score={r['combined_score']:.3f}")
        for strategy in ('zero_shot', 'contextualised', 'rag'):
            print(f"\n  [{strategy}]")
            print(f"  {r['explanations'][strategy]}")
