#!/usr/bin/env python3
import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path

# Ensure project root in path
PROJECT_ROOT = "/home/pushihao/RAG"
sys.path.insert(0, PROJECT_ROOT)

from preprocess.build_core_indexes import tokenize_for_bm25

KB_BASE = "/home/pushihao/RAG/Reports/experiments/datasets/knowledge_bases_test/natural_questions"
BM25_PATH = os.path.join(KB_BASE, "chunk_bm25_index.pkl")
META_PATH = os.path.join(KB_BASE, "indexed_chunks_metadata.json")

def main():
    assert Path(BM25_PATH).is_file(), f"BM25索引不存在: {BM25_PATH}"
    assert Path(META_PATH).is_file(), f"元数据不存在: {META_PATH}"

    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
        bm25 = data["bm25_model"]
        chunk_texts = data.get("chunk_texts", [])

    with open(META_PATH, "r", encoding="utf-8") as f:
        metas = json.load(f)

    queries = [
        "The Walking Dead Season 8",
        "AMC network episodes",
        "Frank Darabont producer",
    ]

    for q in queries:
        toks = tokenize_for_bm25(q)
        scores = bm25.get_scores(toks)
        ranked = list(np.argsort(scores)[::-1][:5])
        print("\nQuery:", q)
        for r in ranked:
            s = float(scores[r])
            meta = metas[r] if 0 <= r < len(metas) else {"chunk_id": f"idx_{r}"}
            print(f"  rank={len(ranked)-ranked.index(r)} idx={r} score={s:.4f} id={meta.get('chunk_id')} text_preview={(chunk_texts[r] if r < len(chunk_texts) else '')[:80]}")

if __name__ == "__main__":
    main()
