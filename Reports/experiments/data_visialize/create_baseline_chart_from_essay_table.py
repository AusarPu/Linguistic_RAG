import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from visualization_utils import RAGVisualizer

def parse_baseline_table(essay_path: str):
    lines = open(essay_path, "r", encoding="utf-8").read().splitlines()
    start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("| 数据集 |"):
            start = i
            break
    rows = []
    i = start + 1
    num_pat = re.compile(r"^[0-9]+(\.[0-9]+)?$")
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("|"):
            break
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 6 and num_pat.match(parts[2]) and num_pat.match(parts[3]) and num_pat.match(parts[4]) and num_pat.match(parts[5]):
            name = parts[1]
            r1_acc = float(parts[2]) * 100.0
            r1_rec = float(parts[3]) * 100.0
            r6_acc = float(parts[4]) * 100.0
            r6_rec = float(parts[5]) * 100.0
            rows.append((name, r1_acc, r1_rec, r6_acc, r6_rec))
        i += 1
    return rows

def plot_baseline(rows, output_dir: str, output_name: str):
    visualizer = RAGVisualizer(output_dir)
    sns.set_palette("husl")
    names = [r[0] for r in rows]
    r1_acc = [r[1] for r in rows]
    r1_rec = [r[2] for r in rows]
    r6_acc = [r[3] for r in rows]
    r6_rec = [r[4] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Baseline Performance Comparison: R1 vs R6", fontsize=16, fontweight="bold")
    x = np.arange(len(names))
    width = 0.35
    c1, c2 = visualizer.color_schemes["comparison"][0], visualizer.color_schemes["comparison"][1]
    b1 = ax1.bar(x - width/2, r1_acc, width, label="R1: Naive RAG (P1)", color=c1, alpha=0.85)
    b2 = ax1.bar(x + width/2, r6_acc, width, label="R6: BM25", color=c2, alpha=0.85)
    ax1.set_title("Accuracy (%)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis="y")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    b3 = ax2.bar(x - width/2, r1_rec, width, label="R1: Naive RAG (P1)", color=c1, alpha=0.85)
    b4 = ax2.bar(x + width/2, r6_rec, width, label="R6: BM25", color=c2, alpha=0.85)
    ax2.set_title("Context Recall (%)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis="y")
    for bars in [b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=11)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path

def main():
    essay_path = "/home/pushihao/RAG/Reports/docs/essay.md"
    output_dir = "/home/pushihao/RAG/Reports/docs/pics"
    output_name = "baseline_performance_comparison.png"
    rows = parse_baseline_table(essay_path)
    p = plot_baseline(rows, output_dir, output_name)
    print(p)

if __name__ == "__main__":
    main()
