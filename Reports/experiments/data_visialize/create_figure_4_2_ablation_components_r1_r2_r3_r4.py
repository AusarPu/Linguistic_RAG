import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_parser import RAGDataParser
from visualization_utils import RAGVisualizer

def create_figure(output_dir: str):
    parser = RAGDataParser()
    viz = RAGVisualizer(output_dir)

    r1 = parser.load_run_summary('result_1_chunk_only')
    r2 = parser.load_run_summary('result_2_chunk+question')
    r3 = parser.load_run_summary('result_3_chunk+keyword')
    r4 = parser.load_run_summary('result_4_chunk+question+keyword')

    datasets_keys = ['hotpotqa', 'triviaqa', 'natural_questions', 'ms_marco']
    labels = parser.get_dataset_labels()
    names = [labels[d] for d in datasets_keys]

    series = {
        'R1 (P1)': [r1['datasets'][d]['accuracy'] * 100 for d in datasets_keys],
        'R2 (P1+P2)': [r2['datasets'][d]['accuracy'] * 100 for d in datasets_keys],
        'R3 (P1+P3)': [r3['datasets'][d]['accuracy'] * 100 for d in datasets_keys],
        'R4 (P1+P2+P3)': [r4['datasets'][d]['accuracy'] * 100 for d in datasets_keys],
    }

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Ablation Study: P1 vs P1+P2 vs P1+P3 vs P1+P2+P3', fontsize=16, fontweight='bold')

    x = np.arange(len(names))
    width = 0.2
    colors = viz.color_schemes['ablation'][:len(series)]
    keys = list(series.keys())

    for i, k in enumerate(keys):
        vals = series[k]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=k, color=colors[i], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title('Accuracy (%) by Dataset', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, max(sum(series.values(), [])) + 10)
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right')

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'figure_4_2_ablation_r1_r2_r3_r4.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path

def main():
    output_dir = '/home/pushihao/RAG/Reports/docs/pics'
    p = create_figure(output_dir)
    print(p)

if __name__ == '__main__':
    main()