import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

CSV_PATH = "/home/pushihao/RAG/Reports/experiments/data_visialize/aggregated_ragas_summary_structured.csv"
OUTPUT_DIR = "/home/pushihao/RAG/Reports/docs/pics"
METRIC = "context_precision"
TOPK_ORDER = [3, 5, 7, 10]

def main():
    df = pd.read_csv(CSV_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    datasets = sorted(df["dataset"].unique())
    for dataset in datasets:
        sub = df[(df["dataset"] == dataset) & (df["run"].isin(["r1", "r4"]))]
        r1 = sub[sub["run"] == "r1"].pivot(index="threshold", columns="final_topk", values=METRIC)
        r4 = sub[sub["run"] == "r4"].pivot(index="threshold", columns="final_topk", values=METRIC)
        delta = r4 - r1
        tau_order = sorted(delta.index.tolist())
        topk_order = [k for k in TOPK_ORDER if k in delta.columns]
        delta = delta.reindex(index=tau_order, columns=topk_order)
        mat = delta.values
        v = np.nanmax(np.abs(mat))
        fig, ax = plt.subplots(figsize=(8, 6), dpi=480)
        norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
        im = ax.imshow(np.ma.masked_invalid(mat), cmap="seismic", norm=norm, aspect="auto")
        ax.set_xticks(np.arange(len(topk_order)))
        ax.set_xticklabels([str(k) for k in topk_order])
        ax.set_xlabel("Top-K")
        ax.set_yticks(np.arange(len(tau_order)))
        ax.set_yticklabels([str(t) for t in tau_order])
        ax.set_ylabel("Threshold")
        ax.invert_yaxis()
        ax.set_title(f"{dataset} - Delta (MARS - Baseline) on {METRIC}")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Delta")
        out = os.path.join(OUTPUT_DIR, f"delta_heatmap_{dataset}_{METRIC}.png")
        plt.tight_layout()
        plt.savefig(out)
        plt.close(fig)

if __name__ == "__main__":
    main()
