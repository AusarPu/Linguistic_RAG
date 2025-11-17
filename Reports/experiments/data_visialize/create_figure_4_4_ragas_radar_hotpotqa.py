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
    r4 = parser.load_run_summary('result_4_chunk+question+keyword')
    r5 = parser.load_run_summary('result_5_full')

    metrics = ['Accuracy', 'Faithfulness', 'Answer Relevancy', 'Context Precision', 'Context Recall']

    def _vals(run, key):
        m = run['datasets'][key]
        return [
            m['accuracy'] * 100,
            m['faithfulness'] * 100,
            m['answer_relevancy'] * 100,
            m['context_precision'] * 100,
            m['context_recall'] * 100,
        ]

    r1_vals = _vals(r1, 'hotpotqa')
    r4_vals = _vals(r4, 'hotpotqa')
    r5_vals = _vals(r5, 'hotpotqa')

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    r1_plot = r1_vals + [r1_vals[0]]
    r4_plot = r4_vals + [r4_vals[0]]
    r5_plot = r5_vals + [r5_vals[0]]
    angles_plot = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    fig.suptitle('RAGAS Metrics on HotpotQA: R1 vs R4 vs R5', fontsize=16, fontweight='bold')

    ax.plot(angles_plot, r1_plot, label='R1 (Naive RAG)', color=viz.color_schemes['comparison'][0])
    ax.fill(angles_plot, r1_plot, alpha=0.15, color=viz.color_schemes['comparison'][0])
    ax.plot(angles_plot, r4_plot, label='R4 (No Filter)', color=viz.color_schemes['ablation'][3])
    ax.fill(angles_plot, r4_plot, alpha=0.15, color=viz.color_schemes['ablation'][3])
    ax.plot(angles_plot, r5_plot, label='R5 (With Filter)', color=viz.color_schemes['ablation'][0])
    ax.fill(angles_plot, r5_plot, alpha=0.15, color=viz.color_schemes['ablation'][0])

    ax.set_thetagrids(angles * 180/np.pi, metrics)
    ax.set_ylim(30, 90)
    ax.set_yticks([50, 60, 70, 80, 90])
    ax.set_yticklabels(['50%', '60%', '70%', '80%', '90%'])
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'figure_4_4_ragas_radar_hotpotqa.png')
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