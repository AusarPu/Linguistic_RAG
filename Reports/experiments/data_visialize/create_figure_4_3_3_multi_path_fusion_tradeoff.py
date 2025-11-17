import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from visualization_utils import RAGVisualizer

def main():
    output_dir = "/home/pushihao/RAG/Reports/docs/pics"
    os.makedirs(output_dir, exist_ok=True)
    viz = RAGVisualizer(output_dir)
    sns.set_palette("husl")

    datasets = ["HotpotQA", "TriviaQA", "Natural Questions", "MS MARCO"]
    # Accuracy
    r2_acc = [0.75 * 100, 0.77 * 100, 0.62 * 100, 0.63 * 100]
    r3_acc = [0.75 * 100, 0.81 * 100, 0.69 * 100, 0.62 * 100]
    r4_acc = [0.75 * 100, 0.80 * 100, 0.75 * 100, 0.64 * 100]
    # Context Recall
    r2_rec = [0.88 * 100, 0.84 * 100, 0.865 * 100, 0.519 * 100]
    r3_rec = [0.89 * 100, 0.816 * 100, 0.909 * 100, 0.534 * 100]
    r4_rec = [0.90 * 100, 0.83 * 100, 0.887 * 100, 0.554 * 100]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Ablation: Multi-Path Fusion — R2 vs R3 vs R4", fontsize=16, fontweight="bold")

    x = np.arange(len(datasets))
    width = 0.25
    colors = viz.color_schemes["ablation"]

    b1 = ax1.bar(x - width, r2_acc, width, label="R2 (P1+P2)", color=colors[1], alpha=0.85)
    b2 = ax1.bar(x, r3_acc, width, label="R3 (P1+P3)", color=colors[2], alpha=0.85)
    b3 = ax1.bar(x + width, r4_acc, width, label="R4 (P1+P2+P3)", color=colors[3], alpha=0.85)
    ax1.set_title("Accuracy (%)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=15)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(True, alpha=0.3, axis="y")
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.legend(loc="upper left")

    c1 = ax2.bar(x - width, r2_rec, width, label="R2 (P1+P2)", color=colors[1], alpha=0.85)
    c2 = ax2.bar(x, r3_rec, width, label="R3 (P1+P3)", color=colors[2], alpha=0.85)
    c3 = ax2.bar(x + width, r4_rec, width, label="R4 (P1+P2+P3)", color=colors[3], alpha=0.85)
    ax2.set_title("Context Recall (%)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=15)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Context Recall (%)")
    ax2.grid(True, alpha=0.3, axis="y")
    for bars in [c1, c2, c3]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "figure_4_3_3_multi_path_fusion_tradeoff.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out_path)

if __name__ == "__main__":
    main()