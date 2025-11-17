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
    r1_acc = [0.69 * 100, 0.77 * 100, 0.71 * 100, 0.58 * 100]
    r2_acc = [0.75 * 100, 0.77 * 100, 0.62 * 100, 0.63 * 100]
    r1_rec = [0.88 * 100, 0.79 * 100, 0.885 * 100, 0.52 * 100]
    r2_rec = [0.88 * 100, 0.84 * 100, 0.865 * 100, 0.519 * 100]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Ablation: Intent Alignment (P2) — R1 vs R2", fontsize=16, fontweight="bold")

    x = np.arange(len(datasets))
    width = 0.35
    c1, c2 = viz.color_schemes["comparison"][0], viz.color_schemes["comparison"][1]

    b1 = ax1.bar(x - width/2, r1_acc, width, label="R1: Naive RAG (P1)", color=c1, alpha=0.85)
    b2 = ax1.bar(x + width/2, r2_acc, width, label="R2: P1+P2", color=c2, alpha=0.85)
    ax1.set_title("Accuracy (%)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=15)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(True, alpha=0.3, axis="y")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.legend(loc="upper left")

    b3 = ax2.bar(x - width/2, r1_rec, width, label="R1: Naive RAG (P1)", color=c1, alpha=0.85)
    b4 = ax2.bar(x + width/2, r2_rec, width, label="R2: P1+P2", color=c2, alpha=0.85)
    ax2.set_title("Context Recall (%)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=15)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Context Recall (%)")
    ax2.grid(True, alpha=0.3, axis="y")
    for bars in [b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "figure_4_3_1_p2_effectiveness_bars.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out_path)

if __name__ == "__main__":
    main()