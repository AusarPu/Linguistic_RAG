import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from visualization_utils import RAGVisualizer

def plot_p3_dual_effect(output_dir: str, output_name: str) -> str:
    datasets = ["HotpotQA", "TriviaQA"]
    r1_acc = [0.71 * 100, 0.73 * 100]
    r3_acc = [0.68 * 100, 0.79 * 100]

    visualizer = RAGVisualizer(output_dir)
    sns.set_palette("husl")

    x = np.arange(len(datasets))
    width = 0.35
    c1, c2 = visualizer.color_schemes["comparison"][0], visualizer.color_schemes["comparison"][1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_title("Ablation: Dual Effect of Keyword Enhancement (P3)", fontweight="bold")

    b1 = ax.bar(x - width/2, r1_acc, width, label="R1: Naive RAG (P1)", color=c1, alpha=0.85)
    b2 = ax.bar(x + width/2, r3_acc, width, label="R3: P1 + P3", color=c2, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=0)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", fontsize=10)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path

def main():
    output_dir = "/home/pushihao/RAG/Reports/docs/pics"
    output_name = "ablation_p3_dual_effect.png"
    p = plot_p3_dual_effect(output_dir, output_name)
    print(p)

if __name__ == "__main__":
    main()