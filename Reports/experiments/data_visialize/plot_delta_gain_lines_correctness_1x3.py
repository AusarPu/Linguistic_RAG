import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, PROJECT_ROOT.as_posix())

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
CSV_PATH = PROJECT_ROOT / "Reports" / "experiments" / "data_visialize" / "aggregated_ragas_summary_structured.csv"
OUTPUT_PATH = PROJECT_ROOT / "paper_figures" / "delta_gain_answer_correctness_1x3.png"
METRIC = "answer_correctness"
RUN_BASELINE = "r1"
RUN_MARS = "r4"
TOPK_ORDER = [3, 5, 7, 10]
DATASET_ORDER = ["hotpotqa", "natural_questions", "triviaqa"]
DATASET_LABELS = {
    "hotpotqa": "HotpotQA",
    "natural_questions": "Natural Questions",
    "triviaqa": "TriviaQA",
}
LINE_STYLES = {
    3: {"color": "#1f77b4", "marker": "o"},
    5: {"color": "#ff7f0e", "marker": "s"},
    7: {"color": "#2ca02c", "marker": "^"},
    10: {"color": "#d62728", "marker": "D"},
}


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"dataset", "run", "threshold", "final_topk", METRIC}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return df


def compute_dataset_delta(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    sub = df[(df["dataset"] == dataset) & (df["run"].isin([RUN_BASELINE, RUN_MARS]))]
    baseline = sub[sub["run"] == RUN_BASELINE].pivot(index="threshold", columns="final_topk", values=METRIC)
    mars = sub[sub["run"] == RUN_MARS].pivot(index="threshold", columns="final_topk", values=METRIC)
    thresholds = sorted(set(baseline.index.tolist()) | set(mars.index.tolist()))
    delta = (mars - baseline).reindex(index=thresholds, columns=TOPK_ORDER)
    if delta.isna().any().any():
        raise ValueError(f"delta contains NaN for dataset {dataset}")
    return delta * 100.0


def compute_global_ylim(deltas: list[pd.DataFrame]) -> tuple[float, float]:
    min_value = min(float(delta.min().min()) for delta in deltas)
    max_value = max(float(delta.max().max()) for delta in deltas)
    bound = max(abs(min_value), abs(max_value))
    pad = 0.5
    return -bound - pad, bound + pad


def main() -> None:
    df = load_csv(CSV_PATH)
    deltas = [compute_dataset_delta(df, dataset) for dataset in DATASET_ORDER]
    y_min, y_max = compute_global_ylim(deltas)
    fig, axes = plt.subplots(3, 1, figsize=FIG_SIZE, sharex=True, sharey=True)

    for index, dataset in enumerate(DATASET_ORDER):
        ax = axes[index]
        delta = deltas[index]
        x_values = delta.index.tolist()
        for topk in TOPK_ORDER:
            style = LINE_STYLES[topk]
            y_values = delta[topk].tolist()
            ax.plot(
                x_values,
                y_values,
                label=f"K={topk}",
                color=style["color"],
                marker=style["marker"],
                linewidth=1.7,
                markersize=4.8,
            )
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylim(y_min, y_max)
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_ylabel("Delta Correctness (pp)")

    axes[-1].set_xlabel("Threshold (tau)")
    axes[-1].set_xticks(deltas[0].index.tolist())
    axes[-1].set_xticklabels([f"{value:g}" for value in deltas[0].index.tolist()])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    fig.savefig(OUTPUT_PATH.as_posix(), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
