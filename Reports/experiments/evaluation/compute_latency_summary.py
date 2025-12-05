#!/usr/bin/env python3
import os
import sys
import csv
import glob
from pathlib import Path
from typing import List, Dict, Tuple

def percentile(values: List[float], p: float) -> float:
    if not values:
        return float('nan')
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[int(k)]
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return d0 + d1

def load_latency_csvs(root_dir: str) -> List[Tuple[str, str, Dict[str, float]]]:
    out: List[Tuple[str, str, Dict[str, float]]] = []
    for csv_path in glob.glob(os.path.join(root_dir, "rag_evaluation_results", "*", "latency_records_*.csv")):
        dataset = Path(csv_path).parent.name
        run_label = Path(csv_path).stem.replace("latency_records_", "")
        with open(csv_path, 'r', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    rec = {
                        "retrieval_ms": float(row.get("retrieval_ms") or "nan"),
                        "usefulness_ms": float(row.get("usefulness_ms") or "nan"),
                        "generation_ms": float(row.get("generation_ms") or "nan"),
                        "total_ms": float(row.get("total_ms") or "nan"),
                    }
                    out.append((dataset, run_label, rec))
                except Exception:
                    continue
    return out

def aggregate(records: List[Tuple[str, str, Dict[str, float]]]) -> List[Dict[str, str]]:
    buckets: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for dataset, run_label, rec in records:
        key = (dataset, run_label)
        b = buckets.setdefault(key, {"retrieval_ms": [], "usefulness_ms": [], "generation_ms": [], "total_ms": []})
        for k in b.keys():
            v = rec.get(k)
            if v is not None and v == v:  # filter nan
                b[k].append(v)
    rows: List[Dict[str, str]] = []
    for (dataset, run_label), vals in buckets.items():
        row = {
            "dataset_name": dataset,
            "run_label": run_label,
        }
        for k, arr in vals.items():
            row[f"{k}_p50"] = f"{percentile(arr, 0.5):.2f}"
            row[f"{k}_p95"] = f"{percentile(arr, 0.95):.2f}"
        rows.append(row)
    return rows

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate latency records into P50/P95 summary")
    parser.add_argument("--runs-dir", type=str, required=True, help="Path to runs/<run_id> directory")
    parser.add_argument("--overall", action="store_true", help="Also write overall summary across datasets by run_label")
    args = parser.parse_args()

    records = load_latency_csvs(args.runs_dir)
    rows = aggregate(records)
    out_dir = os.path.join(args.runs_dir, "advanced_evaluation_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "rag_latency_summary.csv")
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset_name","run_label",
            "retrieval_ms_p50","retrieval_ms_p95",
            "usefulness_ms_p50","usefulness_ms_p95",
            "generation_ms_p50","generation_ms_p95",
            "total_ms_p50","total_ms_p95",
        ])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(out_path)

    if args.overall:
        # Aggregate across datasets by run_label
        buckets: Dict[str, Dict[str, List[float]]] = {}
        for _, run_label, rec in records:
            b = buckets.setdefault(run_label, {"retrieval_ms": [], "usefulness_ms": [], "generation_ms": [], "total_ms": []})
            for k in b.keys():
                v = rec.get(k)
                if v is not None and v == v:
                    b[k].append(v)
        overall_rows: List[Dict[str, str]] = []
        for run_label, vals in buckets.items():
            row = {"run_label": run_label}
            for k, arr in vals.items():
                row[f"{k}_p50"] = f"{percentile(arr, 0.5):.2f}"
                row[f"{k}_p95"] = f"{percentile(arr, 0.95):.2f}"
            overall_rows.append(row)
        out_overall = os.path.join(out_dir, "rag_latency_summary_overall.csv")
        with open(out_overall, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=[
                "run_label",
                "retrieval_ms_p50","retrieval_ms_p95",
                "usefulness_ms_p50","usefulness_ms_p95",
                "generation_ms_p50","generation_ms_p95",
                "total_ms_p50","total_ms_p95",
            ])
            w.writeheader()
            for row in overall_rows:
                w.writerow(row)
        print(out_overall)

if __name__ == "__main__":
    main()