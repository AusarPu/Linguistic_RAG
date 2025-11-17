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
    r6 = parser.load_run_summary('result_6_bm25')

    datasets_keys = ['hotpotqa', 'triviaqa', 'natural_questions', 'ms_marco']
    labels = parser.get_dataset_labels()
    names = [labels[d] for d in datasets_keys]

    r1_acc = [r1['datasets'][d]['accuracy'] * 100 for d in datasets_keys]
    r6_acc = [r6['datasets'][d]['accuracy'] * 100 for d in datasets_keys]
    r1_rec = [r1['datasets'][d]['context_recall'] * 100 for d in datasets_keys]
    r6_rec = [r6['datasets'][d]['context_recall'] * 100 for d in datasets_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Baseline Systems: R1 (Naive RAG) vs R6 (BM25)', fontsize=16, fontweight='bold')

    x = np.arange(len(names))
    width = 0.35
    c1, c2 = viz.color_schemes['comparison'][0], viz.color_schemes['comparison'][1]

    b1 = ax1.bar(x - width/2, r1_acc, width, label='R1: Naive RAG', color=c1, alpha=0.85)
    b2 = ax1.bar(x + width/2, r6_acc, width, label='R6: BM25', color=c2, alpha=0.85)
    ax1.set_title('Accuracy (%)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15)
    ax1.set_ylim(0, max(r1_acc + r6_acc) + 10)
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, alpha=0.3, axis='y')
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    b3 = ax2.bar(x - width/2, r1_rec, width, label='R1: Naive RAG', color=c1, alpha=0.85)
    b4 = ax2.bar(x + width/2, r6_rec, width, label='R6: BM25', color=c2, alpha=0.85)
    ax2.set_title('Context Recall (%)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15)
    ax2.set_ylim(0, max(r1_rec + r6_rec) + 10)
    ax2.set_ylabel('Context Recall (%)')
    ax2.grid(True, alpha=0.3, axis='y')
    for bars in [b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    handles, labels_leg = ax1.get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='upper right')

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'figure_4_1_baseline_r1_vs_r6.png')
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