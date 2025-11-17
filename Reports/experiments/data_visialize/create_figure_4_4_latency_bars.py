import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "/home/pushihao/RAG/Reports/docs/pics"
OVERALL_CSV = "/home/pushihao/RAG/Reports/experiments/datasets/runs/20251115-152214-enh=pipeline-bs=50/advanced_evaluation_results/rag_latency_summary_overall.csv"

def load_overall(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def make_latency_bars(df: pd.DataFrame, out_path: str) -> plt.Figure:
    # Focus on R1 and R5; include R6 as supplementary bars
    subset = df[df['run_label'].isin(['R1','R4','R5'])].copy()
    subset['P50_total_ms'] = subset['total_ms_p50'].astype(float)
    subset['P95_total_ms'] = subset['total_ms_p95'].astype(float)
    subset['P50_gen_ms'] = subset['generation_ms_p50'].astype(float)
    subset['P95_gen_ms'] = subset['generation_ms_p95'].astype(float)
    subset['run_label'] = pd.Categorical(subset['run_label'], categories=['R1','R4','R5'], ordered=True)
    subset = subset.sort_values('run_label')

    labels = subset['run_label'].tolist()
    x = np.arange(len(labels))
    width = 0.35

    sns.set_palette("husl")
    fig, ax = plt.subplots(figsize=(10, 6))

    bars_p50 = ax.bar(x - width/2, subset['P50_total_ms'], width, label='P50 Total (ms)', color='#2E86AB', alpha=0.85)
    bars_p95 = ax.bar(x + width/2, subset['P95_total_ms'], width, label='P95 Total (ms)', color='#A23B72', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('End-to-End Latency Comparison (P50 vs P95)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()

    # Annotate values on top of bars
    def annotate(bars):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{int(h)}", xy=(b.get_x()+b.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    annotate(bars_p50)
    annotate(bars_p95)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    return fig

def main():
    df = load_overall(OVERALL_CSV)
    out = os.path.join(OUTPUT_DIR, 'figure_4_4_latency_bars.png')
    fig = make_latency_bars(df, out)
    plt.close(fig)

if __name__ == '__main__':
    main()