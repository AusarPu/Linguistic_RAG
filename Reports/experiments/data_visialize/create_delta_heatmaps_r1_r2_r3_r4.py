import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from visualization_utils import RAGVisualizer

def _build_delta_df(metric_name):
    datasets = ["HotpotQA", "TriviaQA", "Natural Questions", "MS MARCO"]
    runs = ["R1 (P1)", "R2 (P1+P2)", "R3 (P1+P3)", "R4 (P1+P2+P3)"]
    values = {
        "Accuracy": {
            "HotpotQA": {"r1": 0.69, "r2": 0.75, "r3": 0.75, "r4": 0.75},
            "Natural Questions": {"r1": 0.71, "r2": 0.62, "r3": 0.69, "r4": 0.75},
            "TriviaQA": {"r1": 0.77, "r2": 0.77, "r3": 0.81, "r4": 0.80},
            "MS MARCO": {"r1": 0.58, "r2": 0.63, "r3": 0.62, "r4": 0.64},
        },
        "Faithfulness": {
            "HotpotQA": {"r1": 0.78, "r2": 0.75, "r3": 0.80, "r4": 0.82},
            "Natural Questions": {"r1": 0.804, "r2": 0.779, "r3": 0.820, "r4": 0.806},
            "TriviaQA": {"r1": 0.750, "r2": 0.779, "r3": 0.779, "r4": 0.849},
            "MS MARCO": {"r1": 0.679, "r2": 0.724, "r3": 0.722, "r4": 0.662},
        },
        "Context Recall": {
            "HotpotQA": {"r1": 0.88, "r2": 0.88, "r3": 0.89, "r4": 0.90},
            "Natural Questions": {"r1": 0.885, "r2": 0.865, "r3": 0.909, "r4": 0.887},
            "TriviaQA": {"r1": 0.79, "r2": 0.84, "r3": 0.816, "r4": 0.83},
            "MS MARCO": {"r1": 0.52, "r2": 0.519, "r3": 0.534, "r4": 0.554},
        },
        "Context Precision": {
            "HotpotQA": {"r1": 0.713, "r2": 0.701, "r3": 0.708, "r4": 0.690},
            "Natural Questions": {"r1": 0.707, "r2": 0.718, "r3": 0.701, "r4": 0.684},
            "TriviaQA": {"r1": 0.813, "r2": 0.815, "r3": 0.810, "r4": 0.772},
            "MS MARCO": {"r1": 0.655, "r2": 0.638, "r3": 0.675, "r4": 0.638},
        },
    }
    scale_pp = 100.0 if metric_name in ["Accuracy", "Context Recall", "Context Precision"] else 1.0
    rows = []
    for run in runs:
        if run.startswith("R1"):
            rows.append([0.0 for _ in datasets])
        else:
            deltas = []
            for ds in datasets:
                base = values[metric_name][ds]["r1"]
                key = "r2" if run.startswith("R2") else ("r3" if run.startswith("R3") else "r4")
                delta = (values[metric_name][ds][key] - base) * scale_pp
                deltas.append(delta)
            rows.append(deltas)
    df = pd.DataFrame(rows, index=runs, columns=datasets)
    return df

def _heatmap(df, title, cbar_label, fmt, output_file):
    viz = RAGVisualizer()
    data = df.values
    abs_max = float(np.max(np.abs(data)))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt=fmt, cmap=viz.color_schemes["heatmap"], vmin=-abs_max, vmax=abs_max, center=0.0, square=False, ax=ax, cbar_kws={"shrink": 0.8, "label": cbar_label})
    ax.set_title(title, fontweight="bold", pad=20)
    ax.set_xlabel("Datasets")
    ax.set_ylabel("Runs")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    viz.save_figure(output_file, fig)
    plt.close(fig)

def main():
    os.makedirs("/home/pushihao/RAG/Reports/docs/pics", exist_ok=True)
    df_acc = _build_delta_df("Accuracy")
    df_fai = _build_delta_df("Faithfulness")
    df_rec = _build_delta_df("Context Recall")
    df_pre = _build_delta_df("Context Precision")
    _heatmap(df_acc, "Delta vs R1 — Accuracy (Runs × Datasets)", "Delta (pp)", ".1f", "delta_heatmap_accuracy_r1_r2_r3_r4.png")
    _heatmap(df_fai, "Delta vs R1 — Faithfulness (Runs × Datasets)", "Delta", ".2f", "delta_heatmap_faithfulness_r1_r2_r3_r4.png")
    _heatmap(df_rec, "Delta vs R1 — Context Recall (Runs × Datasets)", "Delta (pp)", ".1f", "delta_heatmap_context_recall_r1_r2_r3_r4.png")
    _heatmap(df_pre, "Delta vs R1 — Context Precision (Runs × Datasets)", "Delta (pp)", ".1f", "delta_heatmap_context_precision_r1_r2_r3_r4.png")

if __name__ == "__main__":
    main()