import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_parser import RAGDataParser
from visualization_utils import RAGVisualizer

def _extract_metrics(run_data, dataset_key):
    m = run_data['datasets'][dataset_key]
    return {
        'Accuracy': m['accuracy'] * 100,
        'Context Precision': m['context_precision'] * 100,
        'Context Recall': m['context_recall'] * 100,
    }

def _plot_tradeoff(output_dir: str, dataset_key: str, dataset_name: str):
    parser = RAGDataParser()
    viz = RAGVisualizer(output_dir)

    r4 = parser.load_run_summary('result_4_chunk+question+keyword')
    r5 = parser.load_run_summary('result_5_full')

    r4_m = _extract_metrics(r4, dataset_key)
    r5_m = _extract_metrics(r5, dataset_key)

    metrics_order = ['Accuracy', 'Context Precision', 'Context Recall']
    x = np.arange(len(metrics_order))
    width = 0.35
    c4, c5 = viz.color_schemes['comparison'][0], viz.color_schemes['comparison'][1]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Filter Trade-off on {dataset_name}: R4 vs R5', fontsize=16, fontweight='bold')

    b1 = ax.bar(x - width/2, [r4_m[m] for m in metrics_order], width, label='R4 (No Filter)', color=c4, alpha=0.85)
    b2 = ax.bar(x + width/2, [r5_m[m] for m in metrics_order], width, label='R5 (With Filter)', color=c5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{m} (%)' for m in metrics_order], rotation=0)
    ax.set_ylim(0, max(list(r4_m.values()) + list(r5_m.values())) + 10)
    ax.set_ylabel('Value (%)')
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.legend(loc='upper right')

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'figure_4_3_{dataset_key}_r4_vs_r5.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path

def main():
    output_dir = '/home/pushihao/RAG/Reports/docs/pics'
    p1 = _plot_tradeoff(output_dir, 'hotpotqa', 'HotpotQA')
    print(p1)
    p2 = _plot_tradeoff(output_dir, 'triviaqa', 'TriviaQA')
    print(p2)

if __name__ == '__main__':
    main()