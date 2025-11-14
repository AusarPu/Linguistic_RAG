import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from visualization_utils import RAGVisualizer
from data_parser import RAGDataParser

def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def extract_metrics(df: pd.DataFrame, datasets: list) -> tuple:
    acc = []
    recall = []
    for d in datasets:
        row = df[df["dataset"] == d]
        acc.append(float(row["accuracy"].values[0]) * 100)
        recall.append(float(row["context_recall"].values[0]) * 100)
    return acc, recall

def plot_baseline(r1_csv: str, r6_csv: str, output_dir: str, output_name: str) -> str:
    labels = {
        "hotpotqa": "HotpotQA",
        "ms_marco": "MS MARCO",
        "natural_questions": "Natural Questions",
        "triviaqa": "TriviaQA",
    }
    visualizer = RAGVisualizer(output_dir)
    sns.set_palette("husl")

    df_r1 = load_summary(r1_csv)
    df_r6 = load_summary(r6_csv)

    datasets = ["hotpotqa", "ms_marco", "natural_questions", "triviaqa"]
    x_names = [labels.get(d, d) for d in datasets]

    r1_acc, r1_recall = extract_metrics(df_r1, datasets)
    r6_acc, r6_recall = extract_metrics(df_r6, datasets)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Baseline Performance Comparison: R1 vs R6", fontsize=16, fontweight="bold")

    x = np.arange(len(x_names))
    width = 0.35
    c1, c2 = visualizer.color_schemes["comparison"][0], visualizer.color_schemes["comparison"][1]

    b1 = ax1.bar(x - width/2, r1_acc, width, label="R1: Naive RAG (P1)", color=c1, alpha=0.85)
    b2 = ax1.bar(x + width/2, r6_acc, width, label="R6: BM25", color=c2, alpha=0.85)
    ax1.set_title("Accuracy (%)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_names, rotation=15)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis="y")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    b3 = ax2.bar(x - width/2, r1_recall, width, label="R1: Naive RAG (P1)", color=c1, alpha=0.85)
    b4 = ax2.bar(x + width/2, r6_recall, width, label="R6: BM25", color=c2, alpha=0.85)
    ax2.set_title("Context Recall (%)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_names, rotation=15)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis="y")
    for bars in [b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    handles, labels_leg = ax1.get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper right", fontsize=11)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=False)
    ap.add_argument("--r6", required=False)
    ap.add_argument("--r1-run", required=False, help="run name, e.g., result_1_chunk_only")
    ap.add_argument("--r6-run", required=False, help="run name, e.g., result_6_bm25")
    ap.add_argument("--output-dir", default="/home/pushihao/RAG/Reports/docs/pics")
    ap.add_argument("--output-name", default="baseline_performance_comparison.png")
    args = ap.parse_args()

    if args.r1_run and args.r6_run:
        base = "/home/pushihao/RAG/Reports/experiments/datasets/runs"
        r1_csv = os.path.join(base, args.r1_run, "advanced_evaluation_results", "ragas_summary.csv")
        r6_csv = os.path.join(base, args.r6_run, "advanced_evaluation_results", "ragas_summary.csv")
        p = plot_baseline(r1_csv, r6_csv, args.output_dir, args.output_name)
        print(p)
        return

    if args.r1 and args.r6:
        p = plot_baseline(args.r1, args.r6, args.output_dir, args.output_name)
        print(p)
        return

    parser = RAGDataParser()
    runs = parser.load_all_runs()
    if "result_1_chunk_only" in runs and "result_6_bm25" in runs:
        base = "/home/pushihao/RAG/Reports/experiments/datasets/runs"
        r1_csv = os.path.join(base, "result_1_chunk_only", "advanced_evaluation_results", "ragas_summary.csv")
        r6_csv = os.path.join(base, "result_6_bm25", "advanced_evaluation_results", "ragas_summary.csv")
        p = plot_baseline(r1_csv, r6_csv, args.output_dir, args.output_name)
        print(p)
        return

    raise FileNotFoundError("请通过 --r1/--r6 指定 CSV，或通过 --r1-run/--r6-run 指定 run 名")

if __name__ == "__main__":
    main()