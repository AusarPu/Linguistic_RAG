#!/usr/bin/env python3
"""
NaN 审计脚本：在所有评估执行完毕后统一检查每数据集的 ragas_metrics.csv，
统计四项指标的 NaN 情况并输出一个总的审计汇总 CSV。
"""

import os
import sys
import argparse
import logging
from typing import List
import pandas as pd


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - nan_audit - %(message)s",
    )


def audit_datasets(output_dir: str, datasets: List[str]) -> str:
    metrics_cols = [
        "answer_relevancy",
        "faithfulness",
        "context_precision",
        "context_recall",
    ]

    summary_rows = []
    for dataset in datasets:
        dataset_dir = os.path.join(output_dir, dataset)
        csv_path = os.path.join(dataset_dir, "ragas_metrics.csv")
        if not os.path.exists(csv_path):
            logging.warning(f"数据集 {dataset} 未找到 ragas_metrics.csv: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        numeric_df = df[metrics_cols].apply(pd.to_numeric, errors="coerce")
        nan_counts = {col: int(numeric_df[col].isna().sum()) for col in metrics_cols}
        nan_rows_total = int(numeric_df.isna().any(axis=1).sum())

        row = {
            "dataset": dataset,
            "total_questions": len(df),
            **{f"nan_count_{col}": nan_counts[col] for col in metrics_cols},
            "nan_rows_total": nan_rows_total,
        }
        summary_rows.append(row)
        logging.info(
            f"[NaN审计] {dataset}: nan_rows_total={nan_rows_total}, "
            + ", ".join([f"nan_count_{c}={nan_counts[c]}" for c in metrics_cols])
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "nan_audit_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"NaN审计汇总CSV写入: {summary_path}")
    return summary_path


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="NaN 审计脚本")
    parser.add_argument("--output-dir", required=True, help="高级评估结果输出目录")
    parser.add_argument("--datasets", nargs='*', help="数据集名称列表（可选）")
    args = parser.parse_args()

    output_dir = args.output_dir
    datasets = args.datasets

    # 如果未指定数据集，则自动从输出目录扫描
    if not datasets:
        datasets = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

    audit_datasets(output_dir, datasets)


if __name__ == "__main__":
    main()