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
    r3_acc = [0.75 * 100, 0.81 * 100, 0.69 * 100, 0.62 * 100]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle("Ablation: Keyword Enhancement (P3) — R1 vs R3", fontsize=16, fontweight="bold")

    x = np.arange(len(datasets))
    width = 0.35
    c1, c2 = viz.color_schemes["comparison"][0], viz.color_schemes["comparison"][1]

    b1 = ax.bar(x - width/2, r1_acc, width, label="R1: Naive RAG (P1)", color=c1, alpha=0.85)
    b2 = ax.bar(x + width/2, r3_acc, width, label="R3: P1+P3", color=c2, alpha=0.85)
    ax.set_title("Accuracy (%)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.3, axis="y")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.legend(loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "figure_4_3_2_p3_dual_effect_bars.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out_path)

if __name__ == "__main__":
    main()