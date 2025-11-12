#!/usr/bin/env python3
"""
扩展聚合脚本：在不修改现有评估代码的前提下，读取各数据集的 ragas_metrics.csv，
计算更丰富的聚合指标并输出到单独的汇总 CSV（ragas_summary_extended.csv）。

功能要点：
- 扫描指定目录下各数据集子目录（hotpotqa、ms_marco、natural_questions、triviaqa 等），
  对存在 ragas_metrics.csv 的数据集进行统计。
- 指标列：answer_relevancy、faithfulness、context_precision、context_recall。
- 输出聚合：
  * 均值 mean
  * 中位数 median
  * 标准差 std
  * 四分位距 IQR（Q3 - Q1）
  * 分位数 p10/p25/p75/p90
  * 截尾均值 trimmed_mean（默认 10%）
  * 通过率 pass_rate（默认阈值 0.7）
  * 有效样本率 valid_rate（四项指标均非 NaN 的样本比例）
  * NaN 统计（各列 NaN 数量与任一列 NaN 的行数）
  * 检索 F1（micro：逐行计算后求均值；macro：先分别取均值再用调和平均）
  * 可选：若存在 ragas_metrics_*.csv（多次重复评估输出），计算跨运行“列均值”的方差（variance）

使用方式：
python Reports/experiments/evaluation/aggregate_ragas_metrics.py \
  --input-dir /home/pushihao/RAG/Reports/experiments/datasets/final_result_0_full_compnent \
  --output-file /home/pushihao/RAG/Reports/experiments/datasets/final_result_0_full_compnent/ragas_summary_extended.csv \
  --pass-threshold 0.7 \
  --trim-fraction 0.1
"""

import os
import math
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd


METRICS_COLS = [
    "answer_relevancy",
    "faithfulness",
    "context_precision",
    "context_recall",
    "accuracy",
]


def trimmed_mean(series: pd.Series, fraction: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return float("nan")
    arr = np.sort(s.values)
    n = arr.size
    k = int(math.floor(n * fraction))
    if k * 2 >= n:
        return float(np.mean(arr))
    return float(np.mean(arr[k:n - k]))


def compute_basic_stats(series: pd.Series, pass_threshold: float, trim_fraction: float) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "iqr": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "trimmed_mean": float("nan"),
            "pass_rate": float("nan"),
        }
    q25 = s.quantile(0.25)
    q75 = s.quantile(0.75)
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "iqr": float(q75 - q25),
        "p10": float(s.quantile(0.10)),
        "p25": float(q25),
        "p75": float(q75),
        "p90": float(s.quantile(0.90)),
        "trimmed_mean": trimmed_mean(s, trim_fraction),
        "pass_rate": float((s >= pass_threshold).mean()),
    }


def compute_retrieval_f1(numeric_df: pd.DataFrame) -> Dict[str, float]:
    p = pd.to_numeric(numeric_df["context_precision"], errors="coerce").fillna(0.0).values
    r = pd.to_numeric(numeric_df["context_recall"], errors="coerce").fillna(0.0).values
    denom = p + r
    f1_per_row = np.where(denom > 0.0, (2.0 * p * r) / denom, 0.0)
    micro = float(np.mean(f1_per_row)) if f1_per_row.size > 0 else float("nan")

    p_mean = float(pd.to_numeric(numeric_df["context_precision"], errors="coerce").mean())
    r_mean = float(pd.to_numeric(numeric_df["context_recall"], errors="coerce").mean())
    denom_macro = p_mean + r_mean
    macro = (2.0 * p_mean * r_mean) / denom_macro if denom_macro > 0.0 else 0.0
    return {"retrieval_f1_micro": micro, "retrieval_f1_macro": macro}


def compute_repeat_variances(dataset_dir: str) -> Dict[str, float]:
    # 聚合多次评估（ragas_metrics_*.csv）的“列均值”方差
    files = [f for f in os.listdir(dataset_dir) if f.startswith("ragas_metrics_") and f.endswith(".csv")]
    if len(files) < 2:
        return {f"var_means_{col}": 0.0 for col in METRICS_COLS}
    files.sort()
    means_per_run: List[Dict[str, float]] = []
    for fname in files:
        df = pd.read_csv(os.path.join(dataset_dir, fname))
        # 使用 reindex 以容忍缺失列（例如旧CSV无 accuracy）
        numeric_df = df.reindex(columns=METRICS_COLS).apply(pd.to_numeric, errors="coerce")
        means_per_run.append({col: float(numeric_df[col].mean()) for col in METRICS_COLS})
    variances: Dict[str, float] = {}
    for col in METRICS_COLS:
        values = np.array([m[col] for m in means_per_run], dtype=float)
        variances[f"var_means_{col}"] = float(np.var(values, ddof=1)) if values.size >= 2 else 0.0
    return variances


def summarize_dataset(dataset: str, dataset_dir: str, pass_threshold: float, trim_fraction: float) -> Dict[str, float]:
    csv_path = os.path.join(dataset_dir, "ragas_metrics.csv")
    df = pd.read_csv(csv_path)
    # 容忍缺失列（例如 accuracy 尚未附加时），缺失列将以 NaN 填充
    numeric_df = df.reindex(columns=METRICS_COLS).apply(pd.to_numeric, errors="coerce")

    nan_counts = {col: int(numeric_df[col].isna().sum()) for col in METRICS_COLS}
    nan_rows_total = int(numeric_df.isna().any(axis=1).sum())
    valid_rows = int(numeric_df.notna().all(axis=1).sum())
    total_questions = int(len(df))
    valid_rate = float(valid_rows / total_questions) if total_questions > 0 else float("nan")

    stats: Dict[str, float] = {
        "dataset": dataset,
        "total_questions": total_questions,
        "valid_rate": valid_rate,
        "nan_rows_total": float(nan_rows_total),
    }
    for col in METRICS_COLS:
        col_stats = compute_basic_stats(numeric_df[col], pass_threshold, trim_fraction)
        for k, v in col_stats.items():
            stats[f"{col}_{k}"] = v
        stats[f"nan_count_{col}"] = float(nan_counts[col])

    f1_stats = compute_retrieval_f1(numeric_df)
    stats.update(f1_stats)

    repeat_vars = compute_repeat_variances(dataset_dir)
    stats.update(repeat_vars)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Ragas 指标扩展聚合（不修改原有评估代码）")
    parser.add_argument("--input-dir", required=True, help="包含各数据集 ragas_metrics.csv 的目录（如 final_result_0_full_compnent）")
    parser.add_argument("--output-file", required=False, help="输出的汇总CSV路径，默认写入 input-dir/ragas_summary_extended.csv")
    parser.add_argument("--pass-threshold", type=float, default=0.7, help="通过率阈值（默认0.7）")
    parser.add_argument("--trim-fraction", type=float, default=0.1, help="截尾均值比例（默认10%）")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_file = args.output_file or os.path.join(input_dir, "ragas_summary_extended.csv")
    pass_threshold = args.pass_threshold
    trim_fraction = args.trim_fraction

    datasets = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    rows: List[Dict[str, float]] = []
    for dataset in datasets:
        dataset_dir = os.path.join(input_dir, dataset)
        candidate = os.path.join(dataset_dir, "ragas_metrics.csv")
        if not os.path.exists(candidate):
            continue
        rows.append(summarize_dataset(dataset, dataset_dir, pass_threshold, trim_fraction))

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()