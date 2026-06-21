import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

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

FIG_SIZE = (6.0, 5.0)

CSV_PATH = "/home/pushihao/RAG/Reports/experiments/data_visialize/aggregated_ragas_summary_structured.csv"
OUTPUT_PATH = "/home/pushihao/RAG/Reports/docs/pics/delta_heatmap_answer_correctness_1x3.png"
METRIC = "answer_correctness"
TOPK_ORDER = [3, 5, 7, 10]
DATASETS = ["hotpotqa", "natural_questions", "triviaqa"]

def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"dataset", "run", "threshold", "final_topk", METRIC}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return df

def compute_delta_for_dataset(df: pd.DataFrame, dataset: str, metric: str):
    sub = df[(df["dataset"] == dataset) & (df["run"].isin(["r1", "r4"]))]
    if sub.empty:
        raise ValueError(f"no data for dataset {dataset}")
    r1 = sub[sub["run"] == "r1"].pivot(index="threshold", columns="final_topk", values=metric)
    r4 = sub[sub["run"] == "r4"].pivot(index="threshold", columns="final_topk", values=metric)
    tau_all = sorted(set(r1.index.tolist()) | set(r4.index.tolist()))
    cols_all = sorted(set(r1.columns.tolist()) | set(r4.columns.tolist()))
    missing_cols = [k for k in TOPK_ORDER if k not in cols_all]
    if missing_cols:
        raise ValueError(f"dataset {dataset} missing topk {missing_cols}")
    r1 = r1.reindex(index=tau_all, columns=cols_all)
    r4 = r4.reindex(index=tau_all, columns=cols_all)
    delta = r4 - r1
    if delta.isna().any().any():
        raise ValueError(f"delta contains NaN for dataset {dataset}")
    delta = delta.reindex(index=tau_all, columns=TOPK_ORDER)
    return delta.values, tau_all, TOPK_ORDER

def build_global_norm(deltas):
    v = 0.0
    for mat in deltas:
        v = max(v, float(np.max(np.abs(mat))))
    norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
    return norm

def main():
    df = load_csv(CSV_PATH)
    mats = []
    tau_orders = []
    for ds in DATASETS:
        mat, tau_order, topk_order = compute_delta_for_dataset(df, ds, METRIC)
        mats.append(mat)
        tau_orders.append(tau_order)
    norm = build_global_norm(mats)
    fig = plt.figure(figsize=FIG_SIZE, constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.06])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0])]
    for i, ds in enumerate(DATASETS):
        ax = axs[i]
        mat = mats[i]
        tau_order = tau_orders[i]
        im = ax.imshow(mat, cmap="seismic", norm=norm, aspect="auto")
        ax.set_xticks(np.arange(len(TOPK_ORDER)))
        ax.set_xticklabels([str(k) for k in TOPK_ORDER])
        ax.set_xlabel("Top-K")
        ax.set_yticks(np.arange(len(tau_order)))
        ax.set_yticklabels([str(t) for t in tau_order])
        ax.invert_yaxis()
        ax.set_title(ds)
        if i == 0:
            ax.set_ylabel("Threshold")
    cax = fig.add_subplot(gs[3, 0])
    sm = plt.cm.ScalarMappable(cmap="seismic", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Delta")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH)
    plt.close(fig)

if __name__ == "__main__":
    main()
