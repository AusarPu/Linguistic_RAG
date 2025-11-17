import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd

RUNS_BASE = Path("/home/pushihao/RAG/Reports/experiments/datasets/runs")

RUN_ALIAS = {
    "r1": "result_1_chunk_only",
    "r2": "result_2_chunk+question",
    "r3": "result_3_chunk+keyword",
    "r4": "result_4_chunk+question+keyword",
    "r5": "result_5_full",
    "r5_1": "result_5_1",
    "r6": "result_6_bm25",
}

DATASETS = ["hotpotqa", "triviaqa"]
CASES_MD_PATH = Path("/home/pushihao/RAG/Reports/docs/cases.md")

def normalize_question(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def resolve_run_name(name: str) -> str:
    if name in RUN_ALIAS:
        name = RUN_ALIAS[name]
    return name

def ragas_csv_path(run: str, dataset: str) -> Path:
    return RUNS_BASE / run / "advanced_evaluation_results" / dataset / "ragas_metrics.csv"

def eval_json_path(run: str, dataset: str) -> Path:
    return RUNS_BASE / run / "rag_evaluation_results" / dataset / "evaluation_results.json"

def kb_meta_path(run: str, dataset: str) -> Path:
    return RUNS_BASE / run / "knowledge_bases" / dataset / "indexed_chunks_metadata.json"

def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return candidates[0]

def load_ragas_df(csv_file: Path) -> pd.DataFrame:
    return pd.read_csv(csv_file)

def build_map(df: pd.DataFrame):
    qcol = pick_col(df, ["question", "user_input"])
    gtcol = pick_col(df, ["ground_truth", "reference"])
    metrics = {}
    for _, row in df.iterrows():
        q = str(row[qcol])
        k = normalize_question(q)
        metrics[k] = {
            "question": q,
            "ground_truth": row.get(gtcol, ""),
            "faithfulness": float(row.get("faithfulness", 0) or 0),
            "answer_relevancy": float(row.get("answer_relevancy", 0) or 0),
            "context_recall": float(row.get("context_recall", 0) or 0),
            "context_precision": float(row.get("context_precision", 0) or 0),
            "accuracy": float(row.get("accuracy", 0) or 0),
        }
    return metrics

def load_eval_json(json_file: Path):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    m = {}
    for item in data:
        q = str(item.get("question", ""))
        k = normalize_question(q)
        m[k] = {
            "record": item,
        }
    return m

def strip_eval_item_fields(item: dict):
    x = dict(item)
    if "pipeline_end_reason" in x:
        x.pop("pipeline_end_reason")
    if "has_reasoning" in x:
        x.pop("has_reasoning")
    return x

def load_chunk_texts(meta_file: Path):
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    out = {}
    for it in meta:
        cid = it.get("chunk_id")
        txt = it.get("text") or it.get("text_chunk_content") or ""
        if cid is not None:
            out[cid] = txt
    return out

def find_pairs(map_a, map_b, direction):
    pairs = []
    ks = set(map_a.keys()) & set(map_b.keys())
    for k in ks:
        a = map_a[k]
        b = map_b[k]
        cond = (a["accuracy"] >= 0.5 and b["accuracy"] < 0.5) if direction == "a_success_b_fail" else (a["accuracy"] < 0.5 and b["accuracy"] >= 0.5)
        if cond:
            pairs.append({
                "key": k,
                "question": a["question"],
                "ground_truth": a.get("ground_truth", ""),
                "run_a": a,
                "run_b": b,
                "diff_faithfulness": abs((a.get("faithfulness", 0) or 0) - (b.get("faithfulness", 0) or 0)),
                "diff_context_recall": abs((a.get("context_recall", 0) or 0) - (b.get("context_recall", 0) or 0)),
                "diff_answer_relevancy": abs((a.get("answer_relevancy", 0) or 0) - (b.get("answer_relevancy", 0) or 0)),
            })
    pairs.sort(key=lambda x: (-x["diff_context_recall"], -x["diff_answer_relevancy"]))
    return pairs

def compute_chunk_diffs(eval_a, eval_b, case):
    ka = eval_a.get(case["key"], {})
    kb = eval_b.get(case["key"], {})
    ra = (ka.get("record", {}) or {}).get("retrieved_chunk_ids", []) or []
    rb = (kb.get("record", {}) or {}).get("retrieved_chunk_ids", []) or []
    sa = set(ra)
    sb = set(rb)
    return {
        "a_minus_b": sorted(list(sa - sb)),
        "b_minus_a": sorted(list(sb - sa)),
    }

def load_kb_text_map(run: str, dataset: str):
    p1 = kb_meta_path(run, dataset)
    if p1.exists():
        return load_chunk_texts(p1)
    p2 = Path("/home/pushihao/RAG/Reports/experiments/datasets/knowledge_bases") / dataset / "indexed_chunks_metadata.json"
    if p2.exists():
        return load_chunk_texts(p2)
    return {}

def get_texts_for_ids(text_map: dict, ids: list):
    out = []
    for cid in ids:
        out.append({"chunk_id": cid, "text": text_map.get(cid, "")})
    return out

def render_markdown(run_a, run_b, dataset, cases, with_chunks, eval_a, eval_b, chunk_texts_a, chunk_texts_b, top_k):
    lines = []
    lines.append(f"### {dataset}: {run_a} vs {run_b}")
    for i, c in enumerate(cases[:top_k], 1):
        lines.append(f"- Case {i}")
        lines.append(f"  - Question: {c['question']}")
        lines.append(f"  - Ground Truth: {c.get('ground_truth','')}")
        lines.append(f"  - {run_a} accuracy: {c['run_a']['accuracy']:.3f}, recall: {c['run_a']['context_recall']:.3f}, precision: {c['run_a']['context_precision']:.3f}, faithfulness: {c['run_a']['faithfulness']:.3f}")
        lines.append(f"  - {run_b} accuracy: {c['run_b']['accuracy']:.3f}, recall: {c['run_b']['context_recall']:.3f}, precision: {c['run_b']['context_precision']:.3f}, faithfulness: {c['run_b']['faithfulness']:.3f}")
        ra = strip_eval_item_fields((eval_a.get(c["key"], {}) or {}).get("record", {}) or {})
        rb = strip_eval_item_fields((eval_b.get(c["key"], {}) or {}).get("record", {}) or {})
        lines.append("```json")
        lines.append(json.dumps({"record_a": ra}, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("```json")
        lines.append(json.dumps({"record_b": rb}, ensure_ascii=False, indent=2))
        lines.append("```")
        if with_chunks:
            ids_a = ra.get("retrieved_chunk_ids", []) or []
            ids_b = rb.get("retrieved_chunk_ids", []) or []
            txts_a = get_texts_for_ids(chunk_texts_a, ids_a)
            txts_b = get_texts_for_ids(chunk_texts_b, ids_b)
            lines.append(f"  - Retrieved texts in {run_a}:")
            for t in txts_a:
                lines.append(f"    - {t['chunk_id']}: {t['text']}")
            lines.append(f"  - Retrieved texts in {run_b}:")
            for t in txts_b:
                lines.append(f"    - {t['chunk_id']}: {t['text']}")
    return "\n".join(lines)

def write_cases_md(md_text: str):
    CASES_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CASES_MD_PATH, "a", encoding="utf-8") as f:
        f.write(md_text + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--datasets", default="hotpotqa,triviaqa")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--emit-markdown", action="store_true")
    ap.add_argument("--output", default="")
    ap.add_argument("--show-chunks", action="store_true")
    ap.add_argument("--no-write-md", action="store_true")
    args = ap.parse_args()

    run_a = resolve_run_name(args.run_a)
    run_b = resolve_run_name(args.run_b)
    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    direction = "a_success_b_fail"

    results = {}
    for ds in datasets:
        p_a = ragas_csv_path(run_a, ds)
        p_b = ragas_csv_path(run_b, ds)
        if not (p_a.exists() and p_b.exists()):
            continue
        df_a = load_ragas_df(p_a)
        df_b = load_ragas_df(p_b)
        m_a = build_map(df_a)
        m_b = build_map(df_b)
        pairs = find_pairs(m_a, m_b, direction)

        eval_a = {}
        eval_b = {}
        chunk_texts_a = {}
        chunk_texts_b = {}
        ej_a = eval_json_path(run_a, ds)
        ej_b = eval_json_path(run_b, ds)
        if ej_a.exists():
            eval_a = load_eval_json(ej_a)
        if ej_b.exists():
            eval_b = load_eval_json(ej_b)
        if args.show_chunks:
            chunk_texts_a = load_kb_text_map(run_a, ds)
            chunk_texts_b = load_kb_text_map(run_b, ds)

        results[ds] = {
            "run_a": run_a,
            "run_b": run_b,
            "cases": pairs[:args.top_k],
        }

        md = render_markdown(run_a, run_b, ds, pairs, args.show_chunks, eval_a, eval_b, chunk_texts_a, chunk_texts_b, args.top_k)
        if args.emit_markdown:
            print(md)
        if not args.no_write_md:
            write_cases_md(md)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()