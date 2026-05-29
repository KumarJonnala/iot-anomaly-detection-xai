import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .constants import FAILURE_COLS, ORIG_NAMES, PALETTE


def _derive_failure_type(df):
    ft = df[FAILURE_COLS].idxmax(axis=1)
    return ft.where(df[FAILURE_COLS].sum(axis=1) > 0, 'NORMAL')


def plot_label_distribution(df, data_dir: Path) -> None:
    """Bar chart of failure type counts. Saves ai4i_label_distribution.png."""
    failure_type = _derive_failure_type(df)
    counts = failure_type.value_counts()
    total  = counts.sum()
    labels = counts.index.tolist()
    colors = [PALETTE.get(lbl, '#BDBDBD') for lbl in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor='white', linewidth=0.8)

    for bar, count in zip(bars, counts.values):
        pct = 100 * count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f'{count:,}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=9, fontweight='bold',
        )

    ax.set_title('AI4I 2020 — Target Class Label Distribution', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('Sample Count')
    ax.set_xlabel('Failure Type')
    ax.set_ylim(0, counts.max() * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(data_dir / 'ai4i_label_distribution.png', dpi=120, bbox_inches='tight')
    plt.show()


def plot_scatter_matrix(df, data_dir: Path) -> None:
    """Pairplot of sensor features coloured by failure type. Saves ai4i_scatter_matrix.png."""
    plot_df = df[ORIG_NAMES].copy()
    plot_df['failure_type'] = _derive_failure_type(df)
    sample_df = plot_df.sample(n=min(1500, len(plot_df)), random_state=42)

    sns.pairplot(
        sample_df,
        vars=ORIG_NAMES,
        hue='failure_type',
        corner=True,
        plot_kws={'alpha': 0.45, 's': 16, 'edgecolor': 'none'},
        diag_kind='hist',
        palette='tab10',
    )
    plt.suptitle('AI4I 2020 — Scatter Plot Matrix', fontsize=13, fontweight='bold', y=1.02)
    plt.savefig(data_dir / 'ai4i_scatter_matrix.png', dpi=120, bbox_inches='tight')
    plt.show()


def plot_correlations(df, data_dir: Path) -> None:
    """Lower-triangle correlation heatmap of sensor features. Saves ai4i_correlations.png."""
    corr = df[ORIG_NAMES].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        corr,
        annot=True, fmt='.2f', cmap='RdYlGn', center=0,
        square=True, ax=ax, mask=mask, linewidths=0.5,
        cbar_kws={'shrink': 0.8},
    )
    ax.set_title('AI4I — Sensor Feature Correlations', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(data_dir / 'ai4i_correlations.png', dpi=120, bbox_inches='tight')
    plt.show()
