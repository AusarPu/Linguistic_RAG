import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from visualization_utils import RAGVisualizer

def plot_threshold_sensitivity(output_dir: str, output_name: str) -> str:
    datasets = ["HotpotQA", "TriviaQA"]
    r5_acc = [0.58 * 100, 0.74 * 100]
    r5_1_acc = [0.62 * 100, 0.69 * 100]

    viz = RAGVisualizer(output_dir)
    sns.set_palette("husl")
    x = np.arange(len(datasets))
    width = 0.35
    c1, c2 = viz.color_schemes["comparison"][0], viz.color_schemes["comparison"][1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_title("Threshold Sensitivity: R5 (r=0.3) vs R5_1 (r=0.5)", fontweight="bold")
    b1 = ax.bar(x - width/2, r5_acc, width, label="R5: r=0.3", color=c1, alpha=0.85)
    b2 = ax.bar(x + width/2, r5_1_acc, width, label="R5_1: r=0.5", color=c2, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom")
    ax.legend(loc="upper left")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path

def main():
    output_dir = "/home/pushihao/RAG/Reports/docs/pics"
    output_name = "threshold_sensitivity_r5_r5_1.png"
    p = plot_threshold_sensitivity(output_dir, output_name)
    print(p)

if __name__ == "__main__":
    main()