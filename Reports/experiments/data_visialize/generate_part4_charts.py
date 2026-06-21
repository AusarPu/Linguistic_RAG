import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

FIG_SIZE = (6.0, 3.75)

def gen_threshold(df, dataset, outdir):
    d = df[(df['final_topk'] == 10) & (df['dataset'] == dataset)]
    d_r1 = d[d['run'] == 'r1'].sort_values('threshold')
    d_r4 = d[d['run'] == 'r4'].sort_values('threshold')
    x1 = d_r1['threshold'].tolist()
    y1 = d_r1['context_recall'].tolist()
    x2 = d_r4['threshold'].tolist()
    y2 = d_r4['context_recall'].tolist()
    plt.figure(figsize=FIG_SIZE)
    plt.plot(x1, y1, label='Baseline (r1)', marker='o')
    plt.plot(x2, y2, label='MARS (r4)', marker='o')
    plt.xlabel('Threshold')
    plt.ylabel('Context Recall')
    plt.title(f'Threshold Sensitivity – {dataset}')
    plt.ylim(0.28, 0.92)
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = os.path.join(outdir, f"threshold_sensitivity_{dataset}.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def gen_depth(df, dataset, outdir, thr=0.65):
    d = df[(df['threshold'] == thr) & (df['dataset'] == dataset)]
    d_r1 = d[d['run'] == 'r1'].sort_values('final_topk')
    d_r4 = d[d['run'] == 'r4'].sort_values('final_topk')
    x1 = d_r1['final_topk'].tolist()
    y1 = d_r1['answer_correctness'].tolist()
    x2 = d_r4['final_topk'].tolist()
    y2 = d_r4['answer_correctness'].tolist()
    plt.figure(figsize=FIG_SIZE)
    plt.plot(x1, y1, label='Baseline (r1)', marker='o')
    plt.plot(x2, y2, label='MARS (r4)', marker='o')
    plt.xlabel('Top-K')
    plt.ylabel('Answer Correctness')
    plt.title(f'Depth Efficiency – {dataset} (thr={thr})')
    plt.ylim(0.22, 0.55)
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = os.path.join(outdir, f"depth_efficiency_{dataset}_correctness_thr{thr}.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    csv_path = "/home/pushihao/RAG/Reports/experiments/data_visialize/aggregated_ragas_summary_structured.csv"
    outdir = "/home/pushihao/RAG/Reports/docs/pics"
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path)
    datasets = ['hotpotqa', 'ms_marco', 'natural_questions', 'triviaqa']
    for ds in datasets:
        gen_threshold(df, ds, outdir)
        gen_depth(df, ds, outdir, thr=0.65)
        gen_depth(df, ds, outdir, thr=0.7)

    dsets4 = ['hotpotqa', 'ms_marco', 'natural_questions', 'triviaqa']
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.0), sharey=True)
    for i, ds in enumerate(dsets4):
        ax = axes[i // 2][i % 2]
        d = df[(df['final_topk'] == 10) & (df['dataset'] == ds)]
        d_r1 = d[d['run'] == 'r1'].sort_values('threshold')
        d_r4 = d[d['run'] == 'r4'].sort_values('threshold')
        x1 = d_r1['threshold'].tolist()
        y1 = d_r1['context_recall'].tolist()
        x2 = d_r4['threshold'].tolist()
        y2 = d_r4['context_recall'].tolist()
        ax.plot(x1, y1, label='Baseline (r1)', marker='o')
        ax.plot(x2, y2, label='MARS (r4)', marker='o')
        ax.set_title(ds)
        ax.set_xlabel('Threshold')
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))
    axes[0][0].set_ylim(0.28, 0.92)
    path4 = os.path.join(outdir, "threshold_sensitivity_4x1.png")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path4)
    plt.close(fig)

    dsets3 = ['hotpotqa', 'natural_questions', 'triviaqa']
    fig2, axes2 = plt.subplots(3, 1, figsize=(6.0, 5.0), sharex=True)
    thr = 0.65
    for i, ds in enumerate(dsets3):
        d = df[(df['threshold'] == thr) & (df['dataset'] == ds)]
        d_r1 = d[d['run'] == 'r1'].sort_values('final_topk')
        d_r4 = d[d['run'] == 'r4'].sort_values('final_topk')
        x1 = d_r1['final_topk'].tolist()
        y1 = d_r1['answer_correctness'].tolist()
        x2 = d_r4['final_topk'].tolist()
        y2 = d_r4['answer_correctness'].tolist()
        ax = axes2[i]
        ax.plot(x1, y1, label='Baseline (r1)', marker='o')
        ax.plot(x2, y2, label='MARS (r4)', marker='o')
        ax.set_title(ds)
        ax.set_ylabel('Answer Correctness')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.22, 0.55)
    axes2[-1].set_xlabel('Top-K')
    handles2, labels2 = axes2[0].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))
    path3 = os.path.join(outdir, "depth_efficiency_correctness_thr0.65_3x1.png")
    fig2.tight_layout(rect=(0, 0.10, 1, 1))
    fig2.savefig(path3)
    plt.close(fig2)

    d = df[(df['threshold'] == 0.7) & (df['final_topk'] == 5) & (df['dataset'] == 'natural_questions')]
    d_r1 = d[d['run'] == 'r1']
    d_r4 = d[d['run'] == 'r4']
    x_labels = ['Context Recall', 'Faithfulness']
    r1_vals = [float(d_r1['context_recall'].iloc[0]), float(d_r1['faithfulness'].iloc[0])]
    r4_vals = [float(d_r4['context_recall'].iloc[0]), float(d_r4['faithfulness'].iloc[0])]
    import numpy as np
    x = np.arange(len(x_labels))
    w = 0.35
    plt.figure(figsize=(4.5, 3.0))
    plt.bar(x - w/2, r1_vals, width=w, label='Baseline (r1)')
    plt.bar(x + w/2, r4_vals, width=w, label='MARS (r4)')
    plt.xticks(x, x_labels)
    plt.ylim(0.35, 0.80)
    plt.ylabel('Score')
    plt.title('Faithfulness vs. Recall – NQ (K=5, τ=0.7)')
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='lower right')
    path4bar = os.path.join(outdir, "figure_4_faithfulness_vs_recall_nq_k5_tau0.7.png")
    plt.tight_layout()
    plt.savefig(path4bar)
    plt.close()

main()
