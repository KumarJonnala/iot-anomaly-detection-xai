import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import SENSOR_LABELS


def save_results(results: list[dict], data_dir: Path) -> Path:
    """Serialise explanation results to data/ai4i_explanations.json."""
    out = data_dir / 'ai4i_explanations.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    return out


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Flatten results to one row per (record × strategy) for RQ2 analysis."""
    rows = []
    for r in results:
        base = {
            'row_idx':        r['row_idx'],
            'failure_type':   r['failure_type'],
            'agreement':      r.get('agreement'),
            'combined_score': r['combined_score'],
        }
        for strategy, explanation in r['explanations'].items():
            rows.append({
                **base,
                'strategy':        strategy,
                'explanation':     explanation,
                'explanation_len': len(explanation),
            })
    return pd.DataFrame(rows)


def plot_strategy_comparison(df_comparison: pd.DataFrame, data_dir: Path) -> None:
    """Bar chart of mean explanation length per strategy (qualitative proxy for richness)."""
    summary = (
        df_comparison
        .groupby('strategy')['explanation_len']
        .agg(['mean', 'std'])
        .reindex(['zero_shot', 'contextualised', 'rag'])
    )

    colours = ['#90CAF9', '#66BB6A', '#FFA726']
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(summary.index, summary['mean'], color=colours,
                  yerr=summary['std'], capsize=5, edgecolor='white', linewidth=0.8)

    for bar, val in zip(bars, summary['mean']):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                f'{int(val)} chars',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title('Mean Explanation Length by Prompting Strategy', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('Characters')
    ax.set_xlabel('Strategy')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(data_dir / 'ai4i_strategy_comparison.png', dpi=120, bbox_inches='tight')
    plt.show()


def plot_explanation_sample(results: list[dict], n: int = 3, data_dir: Path | None = None) -> None:
    """Print explanations for one record of each target failure type for visual comparison."""
    target_types = ['HDF', 'TWF', 'NORMAL']
    shown = 0
    for target in target_types:
        for r in results:
            if r['failure_type'] == target:
                print(f"\n{'='*70}")
                print(f" Record {r['row_idx']} | failure={r['failure_type']} | "
                      f"agreement={r['agreement']} | score={r['combined_score']:.3f}")
                print('='*70)
                for strategy in ('zero_shot', 'contextualised', 'rag'):
                    print(f'\n[{strategy.upper()}]')
                    print(r['explanations'][strategy])
                shown += 1
                break
        if shown >= n:
            break
