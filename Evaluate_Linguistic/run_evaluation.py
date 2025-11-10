#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate_Linguistic 评估封装脚本

目的：
- 读取 Generate_Linguistic_Question/datasets/evaluation_clean.csv；
- 加载 Evaluate_Linguistic/indexes 下的所有检索索引；
- 并发运行 RAG 检索与生成，输出评估结果到 Evaluate_Linguistic/evaluation/。

约束：
- 不修改底层模块逻辑与日志策略；
- 不使用 try/except；必要的分支仅用于流程控制；
- 索引路径通过覆盖 script/config_rag.py 的常量实现，并在覆盖后重新加载 script.knowledge_base。
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
import asyncio
import time
import requests

# 项目根目录：.../RAG
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from script import config_rag as config
from script.rag_pipeline import execute_rag_flow


def _eval_paths() -> dict:
    root = PROJECT_ROOT / "Evaluate_Linguistic"
    return {
        "root": root,
        "indexes": root / "indexes",
        "evaluation": root / "evaluation",
        "eval_csv": PROJECT_ROOT / "Generate_Linguistic_Question" / "datasets" / "evaluation_clean.csv",
    }


def ensure_directories() -> None:
    paths = _eval_paths()
    os.makedirs(paths["indexes"], exist_ok=True)
    os.makedirs(paths["evaluation"], exist_ok=True)

def preflight_check_services() -> None:
    """
    评估前健康检查：
    - 确认 vLLM 生成端 (8001) 的 /v1/models 可访问并返回模型 id；
    - 确认 Embedding 端 (8850) 的 /v1/embeddings 可用并返回向量。

    注意：不使用 try/except。若服务不可用，程序会在此处直接失败并退出，避免进入耗费 GPU 的评估阶段。
    """
    # 生成端健康检查
    models_url = config.GENERATOR_API_URL.replace('/v1/chat/completions', '/v1/models')
    headers = {"Accept": "application/json"}
    resp = requests.get(models_url, headers=headers, timeout=10)
    result = resp.json()
    model_id = result["data"][0]["id"]
    print(f"[健康检查] Generator OK: {models_url} -> model={model_id}")

    # Embedding 端健康检查
    payload = {
        "model": config.EMBEDDING_MODEL_NAME_FOR_API,
        "input": ["健康检查"],
        "encoding_format": "float"
    }
    eh = {"Content-Type": "application/json"}
    resp2 = requests.post(config.EMBEDDING_API_URL, json=payload, headers=eh, timeout=10)
    result2 = resp2.json()
    emb = result2["data"][0]["embedding"]
    assert isinstance(emb, list) and len(emb) > 0, "Embedding 返回为空"
    print(f"[健康检查] Embedding OK: {config.EMBEDDING_API_URL} -> dims={len(emb)}")


def override_config_paths_for_eval(index_dir: Path) -> None:
    """将 config_rag 的索引相关路径覆盖为 Evaluate_Linguistic/indexes 下的文件名。"""
    index_dir_str = str(index_dir)
    config.PROCESSED_DATA_DIR = index_dir_str
    # 块文本稠密检索
    config.FAISS_INDEX_CHUNKS_SAVE_PATH = os.path.join(index_dir_str, "faiss_index_chunks_ip.idx")
    config.INDEXED_CHUNKS_METADATA_SAVE_PATH = os.path.join(index_dir_str, "indexed_chunks_metadata.json")
    # 关键词短语
    config.PHRASE_DENSE_EMBEDDINGS_MAP_SAVE_PATH = os.path.join(index_dir_str, "phrase_dense_embeddings_map.pkl")
    # BM25：短语BM25与块BM25分别配置
    config.BM25_INDEX_SAVE_PATH = os.path.join(index_dir_str, "phrase_bm25_index.pkl")
    config.CHUNK_BM25_INDEX_SAVE_PATH = os.path.join(index_dir_str, "chunk_bm25_index.pkl")
    # 预生成问题
    config.FAISS_INDEX_QUESTIONS_SAVE_PATH = os.path.join(index_dir_str, "faiss_index_questions_ip.idx")
    config.QUESTION_INDEX_TO_CHUNK_ID_MAP_SAVE_PATH = os.path.join(index_dir_str, "question_index_to_chunk_id_map.json")
    config.ALL_QUESTION_TEXTS_SAVE_PATH = os.path.join(index_dir_str, "all_question_texts.json")


def load_questions_from_csv(csv_path: Path, max_questions: int | None) -> list:
    """读取评估 CSV，返回问题列表（包含必要字段）。"""
    assert csv_path.is_file(), f"评估数据集不存在：{csv_path}"
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # 必要列校验
        required = ["id", "doc_name", "page_id", "chunk_id", "question", "answer", "type"]
        for col in required:
            assert col in reader.fieldnames, f"CSV 缺少必要列：{col}"
        count = 0
        for rec in reader:
            q = rec.get("question", "").strip()
            a = rec.get("answer", "").strip()
            if len(q) == 0 or len(a) == 0:
                # 结构不完整的记录直接跳过（流程控制，不是规避错误）
                continue
            rows.append({
                "id": rec.get("id", ""),
                "question": q,
                "answer": a,
            })
            count += 1
            if isinstance(max_questions, int) and max_questions > 0 and count >= max_questions:
                break
    return rows


async def process_single_question(question: str,
                                  kb_instance,
                                  question_id: str,
                                  use_query_rewriter: bool,
                                  use_dense_chunks: bool,
                                  use_dense_keywords: bool,
                                  use_dense_questions: bool,
                                  use_usefulness_judger: bool) -> dict:
    retrieved_chunk_ids = []
    system_answer = ""
    reasoning_text = ""
    pipeline_end_reason = ""

    async for event in execute_rag_flow(
        user_query=question,
        chat_history_openai=[],
        kb_instance=kb_instance,
        use_query_rewriter=use_query_rewriter,
        use_dense_chunks=use_dense_chunks,
        use_dense_keywords=use_dense_keywords,
        use_dense_questions=use_dense_questions,
        use_usefulness_judger=use_usefulness_judger,
    ):
        et = event.get("type", "")
        if et == "useful_chunks_preview":
            retrieved_chunk_ids = [c.get("chunk_id") for c in event.get("preview", [])]
        elif et == "retrieved_chunks_preview":
            retrieved_chunk_ids = [c.get("id") for c in event.get("preview", [])]
        elif et == "content_delta":
            system_answer += event.get("text", "")
        elif et == "reasoning_delta":
            reasoning_text += event.get("text", "")
        elif et == "final_answer_complete":
            full_text = event.get("full_text", "")
            if len(full_text) > 0:
                system_answer = full_text
        elif et == "pipeline_end":
            pipeline_end_reason = event.get("reason", "completed")
            break

    if len(system_answer.strip()) == 0:
        if pipeline_end_reason == "no_context_found_after_retrieval":
            system_answer = "抱歉，我没有找到与您问题相关的直接信息。"
        elif pipeline_end_reason == "no_context_found_after_usefulness":
            system_answer = "抱歉，我没有找到与您问题直接相关的有用信息。"
        elif pipeline_end_reason == "error":
            system_answer = "处理过程中发生错误，无法生成回答。"
        else:
            system_answer = "未能生成有效回答。"

    return {
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "system_answer": system_answer.strip(),
        "has_reasoning": len(reasoning_text.strip()) > 0,
        "pipeline_end_reason": pipeline_end_reason,
    }


async def evaluate_questions(questions: list,
                             kb_instance,
                             batch_size: int,
                             use_query_rewriter: bool,
                             use_dense_chunks: bool,
                             use_dense_keywords: bool,
                             use_dense_questions: bool,
                             use_usefulness_judger: bool) -> list:
    """按批次并发评估问题列表。"""
    results = []
    total = len(questions)
    batches = [(i, questions[i:i+batch_size]) for i in range(0, total, batch_size)]
    for batch_id, batch in batches:
        tasks = []
        for j, item in enumerate(batch):
            qid = f"batch_{batch_id}_q_{j+1}"
            tasks.append(
                process_single_question(
                    item["question"], kb_instance, qid,
                    use_query_rewriter, use_dense_chunks,
                    use_dense_keywords, use_dense_questions,
                    use_usefulness_judger
                )
            )
        batch_outputs = await asyncio.gather(*tasks)
        for item, out in zip(batch, batch_outputs):
            rec = {
                "id": item["id"],
                "question": item["question"],
                "ground_truth_answer": item["answer"],
            }
            rec.update(out)
            results.append(rec)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate_Linguistic 评估脚本")
    parser.add_argument("--mode", choices=["sample", "full"], default="sample", help="运行模式：sample=示例；full=全量")
    parser.add_argument("--batch-size", type=int, default=5, help="并发批大小")
    parser.add_argument("--max-questions", type=int, default=30, help="示例模式下最大问题数；full 模式忽略")
    parser.add_argument("--use-query-rewriter", action="store_true", default=True, help="启用查询重写")
    parser.add_argument("--no-query-rewriter", action="store_false", dest="use_query_rewriter")
    parser.add_argument("--use-dense-chunks", action="store_true", default=True, help="启用文本稠密检索")
    parser.add_argument("--no-dense-chunks", action="store_false", dest="use_dense_chunks")
    parser.add_argument("--use-dense-keywords", action="store_true", default=True, help="启用关键词稠密检索")
    parser.add_argument("--no-dense-keywords", action="store_false", dest="use_dense_keywords")
    parser.add_argument("--use-dense-questions", action="store_true", default=True, help="启用问题稠密检索")
    parser.add_argument("--no-dense-questions", action="store_false", dest="use_dense_questions")
    parser.add_argument("--use-usefulness-judger", action="store_true", default=True, help="启用有用性判断")
    parser.add_argument("--no-usefulness-judger", action="store_false", dest="use_usefulness_judger")
    args = parser.parse_args()

    paths = _eval_paths()
    # 在任何重计算前进行服务健康检查，避免浪费 GPU 资源
    preflight_check_services()
    ensure_directories()

    # 覆盖配置路径并重新加载知识库模块
    override_config_paths_for_eval(paths["indexes"])
    import importlib
    import script.knowledge_base as kb_mod
    importlib.reload(kb_mod)
    from script.knowledge_base import KnowledgeBase

    kb_instance = KnowledgeBase()

    # 加载问题数据
    max_q = args.max_questions if args.mode == "sample" else None
    questions = load_questions_from_csv(paths["eval_csv"], max_q)
    print(f"评估问题加载完成：{len(questions)} 条。模式={args.mode}，批大小={args.batch_size}")

    # 运行评估
    t0 = time.time()
    results = asyncio.run(
        evaluate_questions(
            questions,
            kb_instance,
            args.batch_size,
            args.use_query_rewriter,
            args.use_dense_chunks,
            args.use_dense_keywords,
            args.use_dense_questions,
            args.use_usefulness_judger,
        )
    )
    dt = time.time() - t0
    print(f"评估完成，耗时 {dt:.2f}s，成功处理 {len(results)} 条问题。")

    # 写出结果
    out_dir = paths["evaluation"]
    out_file = out_dir / "evaluation_results.json"
    payload = {
        "dataset": "Evaluate_Linguistic",
        "mode": args.mode,
        "total_questions": len(questions),
        "processed": len(results),
        "results": results,
        "timestamp": int(time.time()),
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"评估结果已写出：{out_file}")

    # 额外提示：如果生成器端口不可用（例如 8001 未启动），结果中的 system_answer 可能为空或为占位文本。
    # 建议在启动 vLLM 生成服务后，使用默认参数重新运行以获得完整答案。


if __name__ == "__main__":
    main()