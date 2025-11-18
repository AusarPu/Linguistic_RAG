#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ragas 调试脚本：完整输出传入 Ragas 的数据集内容、评估配置、以及评估结果。

使用方式：
python Reports/experiments/evaluation/ragas_debug_dump.py \
  --input-file /home/pushihao/RAG/Reports/experiments/datasets/rag_evaluation_results/hotpotqa/evaluation_results.json \
  --output-file /home/pushihao/RAG/002.txt \
  --dataset hotpotqa

说明：
- 读取评估结果 JSON（包含 question、system_answer、ground_truth_answer、retrieved_chunk_ids）。
- 根据数据集名称初始化 KnowledgeBase，映射 chunk_id 到文本，构造 Ragas 的 Dataset。
- 打印 Ragas 评估配置（LLM/Embeddings/指标/并发配置）。
- 调用 Ragas 进行评估，输出每条样本的指标与总体均值。
- 所有中间过程与最终结果写入指定输出文件。
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# 添加项目根目录到路径，确保能导入 script 包
PROJECT_ROOT = \
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)

# Ragas & 客户端包装
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from datasets import Dataset

# LangChain OpenAI 兼容客户端（指向 vLLM/OpenAI 风格接口）
from langchain_openai import OpenAIEmbeddings

# 项目内模块
from script import config_rag as config
from script.llm_wrappers import RagasOpenAICompatLLMWrapper, RagasOpenAICompatEmbeddings
from preprocess.vllm_tokenizer import fast_token_length


def _get_dataset_name_from_path(input_file: str) -> str:
    """从输入文件路径推断数据集名称（例如 hotpotqa/ms_marco/...）。"""
    # 预期格式：.../rag_evaluation_results/<dataset>/evaluation_results.json
    parts = input_file.split("/rag_evaluation_results/")
    if len(parts) > 1:
        tail = parts[1].strip("/")
        dataset = tail.split("/")[0]
        if dataset:
            return dataset
    # 兜底：从路径的上一级目录名取
    parent = Path(input_file).parent.name
    return parent or "hotpotqa"


def _init_kb_for_dataset(dataset_name: str):
    """为指定数据集初始化 KnowledgeBase（通过动态设置 config 路径）。"""
    index_dir = f"/home/pushihao/RAG/Reports/experiments/datasets/knowledge_bases/{dataset_name}"
    import importlib
    from script import config_rag as config_module

    # 动态指向该数据集的索引目录
    config_module.FAISS_INDEX_CHUNKS_SAVE_PATH = os.path.join(index_dir, "faiss_index_chunks_ip.idx")
    config_module.INDEXED_CHUNKS_METADATA_SAVE_PATH = os.path.join(index_dir, "indexed_chunks_metadata.json")
    config_module.PHRASE_DENSE_EMBEDDINGS_MAP_SAVE_PATH = os.path.join(index_dir, "phrase_dense_embeddings_map.pkl")
    config_module.BM25_INDEX_SAVE_PATH = os.path.join(index_dir, "phrase_bm25_index.pkl")
    config_module.FAISS_INDEX_QUESTIONS_SAVE_PATH = os.path.join(index_dir, "faiss_index_questions_ip.idx")
    config_module.QUESTION_INDEX_TO_CHUNK_ID_MAP_SAVE_PATH = os.path.join(index_dir, "question_index_to_chunk_id_map.json")
    config_module.ALL_QUESTION_TEXTS_SAVE_PATH = os.path.join(index_dir, "all_question_texts.json")
    config_module.CHUNK_BM25_INDEX_SAVE_PATH = os.path.join(index_dir, "chunk_bm25_index.pkl")

    # 重新导入知识库以应用新的路径设置
    import script.knowledge_base
    importlib.reload(script.knowledge_base)
    from script.knowledge_base import KnowledgeBase
    return KnowledgeBase()


def _prepare_ragas_dataset(results: List[Dict[str, Any]], kb) -> Dataset:
    """将评估结果转换为 Ragas 数据集（问题、上下文、答案、参考答案）。

    上下文裁剪策略：按检索顺序前缀保留，直到达到配置的 token 总量与最大块数上限。
    """
    questions: List[str] = []
    contexts: List[List[str]] = []
    answers: List[str] = []
    ground_truths: List[str] = []

    # 使用 chunk_id 映射到文本
    chunk_map = getattr(kb, "chunk_id_to_metadata_map", {})

    for item in results:
        q = item.get("question", "")
        a = item.get("system_answer", "")
        gt = item.get("ground_truth_answer", "")
        chunk_ids = item.get("retrieved_chunk_ids", []) or []

        # 收集原始上下文文本（过滤缺失/空文本）
        raw_ctx_texts: List[str] = []
        for cid in chunk_ids:
            meta = chunk_map.get(cid)
            if meta:
                text_val = meta.get("text") or meta.get("text_chunk_content") or ""
                if text_val:
                    raw_ctx_texts.append(text_val)

        # 基于配置进行上下文裁剪：按顺序保留，直到达到 token 总量与最大块数的上限
        from script import config_rag as config
        max_tokens = getattr(config, "EVALUATION_CONTEXTS_MAX_INPUT_TOKENS", None)
        max_chunks = getattr(config, "EVALUATION_CONTEXTS_MAX_CHUNKS", None)
        ctx_texts: List[str] = []
        total_tokens = 0
        for t in raw_ctx_texts:
            # 块数上限
            if isinstance(max_chunks, int) and max_chunks > 0 and len(ctx_texts) >= max_chunks:
                break
            # token 总量上限
            if isinstance(max_tokens, int) and max_tokens > 0:
                tlen = fast_token_length(t)
                if total_tokens + tlen > max_tokens:
                    break
                total_tokens += tlen
            ctx_texts.append(t)

        questions.append(q)
        answers.append(a)
        ground_truths.append(gt)
        contexts.append(ctx_texts)

    return Dataset.from_dict({
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "ground_truth": ground_truths,
    })


def _get_ragas_clients() -> Tuple:
    """初始化 Ragas 评估所需的 LLM 与 Embeddings 客户端（统一为 OpenAI 兼容封装）。"""
    from script.config_rag import (
        GENERATOR_API_URL,
        GENERATOR_MODEL_NAME_FOR_API,
        EMBEDDING_API_URL,
        EMBEDDING_MODEL_NAME_FOR_API,
    )

    generator_base_url = config.get_ragas_llm_base_url()
    embedding_base_url = EMBEDDING_API_URL.rsplit("/embeddings", 1)[0]

    llm = RagasOpenAICompatLLMWrapper(
        base_url=generator_base_url,
        model=config.get_ragas_llm_model(),
        api_key=config.get_ragas_llm_api_key(),
        temperature=0.2,
        top_p=0.9,
    )
    embeddings = RagasOpenAICompatEmbeddings(
        base_url=embedding_base_url,
        api_key="-",
        model=EMBEDDING_MODEL_NAME_FOR_API,
    )

    return llm, embeddings


def _format_ragas_inputs(dataset: Dataset) -> str:
    """将传入 Ragas 的 Dataset 格式化为可读文本。"""
    lines: List[str] = []
    lines.append("=== Ragas 输入数据集（question/answer/ground_truth/contexts）===")
    lines.append(f"样本数: {len(dataset)}")
    for i in range(len(dataset)):
        row = dataset[i]
        q = row.get("question", "")
        a = row.get("answer", "")
        gt = row.get("ground_truth", "")
        ctxs = row.get("contexts", []) or []
        lines.append("")
        lines.append(f"[样本 {i}] question: {q}")
        lines.append(f"[样本 {i}] answer: {a}")
        lines.append(f"[样本 {i}] ground_truth: {gt}")
        lines.append(f"[样本 {i}] contexts_count: {len(ctxs)}")
        for j, ctx in enumerate(ctxs):
            lines.append(f"[样本 {i}] context[{j}]: {ctx}")
    return "\n".join(lines)


def _format_ragas_config(metrics_list, run_config: RunConfig) -> str:
    names = [m.name if hasattr(m, "name") else str(m) for m in metrics_list]
    lines = [
        "=== Ragas 评估配置 ===",
        f"metrics: {', '.join(names)}",
        f"timeout: {run_config.timeout}",
        f"max_workers: {run_config.max_workers}",
        f"max_retries: {run_config.max_retries}",
        f"max_wait: {run_config.max_wait}",
    ]
    return "\n".join(lines)


def _format_clients_info() -> str:
    from script.config_rag import (
        GENERATOR_API_URL,
        GENERATOR_MODEL_NAME_FOR_API,
        EMBEDDING_API_URL,
        EMBEDDING_MODEL_NAME_FOR_API,
    )
    lines = [
        "=== LLM / Embeddings 客户端信息 ===",
        f"LLM base_url: {GENERATOR_API_URL.rsplit('/chat/completions', 1)[0]}",
        f"LLM model: {GENERATOR_MODEL_NAME_FOR_API}",
        f"Embeddings base_url: {EMBEDDING_API_URL.rsplit('/embeddings', 1)[0]}",
        f"Embeddings model: {EMBEDDING_MODEL_NAME_FOR_API}",
    ]
    return "\n".join(lines)


def _format_ragas_results(df) -> str:
    """格式化 Ragas 的 DataFrame 结果为文本，包含逐样本与总体均值。"""
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        return "=== Ragas 评估结果 ===\n结果不是 DataFrame，无法格式化。"

    lines: List[str] = []
    lines.append("=== Ragas 评估结果（逐样本）===")
    # 打印完整表格（包含所有列）
    lines.append(df.to_csv(index=False))

    # 打印总体均值（仅针对指标列）
    metric_cols = [
        col for col in df.columns
        if col in {"answer_relevancy", "faithfulness", "context_precision", "context_recall"}
    ]
    if metric_cols:
        means = df[metric_cols].mean().to_dict()
        lines.append("\n=== 指标均值 ===")
        for k, v in means.items():
            lines.append(f"{k}: {v:.6f}")
    else:
        lines.append("\n未发现指标列，可能评估失败或返回格式变化。")
    return "\n".join(lines)


def debug_ragas(input_file: str, output_file: str, dataset_name: Optional[str] = None,
                limit: Optional[int] = None,
                timeout: int = 60, max_workers: int = 8,
                max_retries: int = 2, max_wait: int = 10) -> None:
    """执行调试流程，写入完整输入与输出到指定文件。"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 读取输入 JSON
    with open(input_file, "r", encoding="utf-8") as f:
        results: List[Dict[str, Any]] = json.load(f)
    if limit and limit > 0:
        results = results[:limit]

    # 推断/使用数据集名称，初始化 KB
    ds_name = dataset_name or _get_dataset_name_from_path(input_file)
    kb = _init_kb_for_dataset(ds_name)

    # 构建 Ragas Dataset（含上下文裁剪）
    hf_dataset = _prepare_ragas_dataset(results, kb)

    # 客户端与配置
    evaluator_llm, evaluator_embeddings = _get_ragas_clients()
    metrics_list = [faithfulness, answer_relevancy, context_recall, context_precision]
    run_config = RunConfig(
        timeout=timeout, max_workers=max_workers, max_retries=max_retries, max_wait=max_wait
    )

    # 将评估并发限制与 CLI 选项保持一致，并显式应用到客户端
    from script import config_rag as config
    config.EVALUATION_CONCURRENCY_LIMIT = max_workers
    evaluator_llm.run_config = run_config
    evaluator_embeddings.set_run_config(run_config)

    # 组织输出文本
    sections: List[str] = []
    sections.append(f"=== 输入文件 ===\n{input_file}")
    sections.append(f"=== 数据集名称 ===\n{ds_name}")
    sections.append(_format_clients_info())
    sections.append(_format_ragas_config(metrics_list, run_config))
    sections.append(_format_ragas_inputs(hf_dataset))

    # 调用 Ragas 评估
    df_text = None
    try:
        ragas_result = ragas_evaluate(
            dataset=hf_dataset,
            metrics=metrics_list,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config,
        )
        df = ragas_result.to_pandas()
        df_text = _format_ragas_results(df)
        sections.append(df_text)
    except Exception as e:
        sections.append("=== Ragas 评估执行失败 ===")
        sections.append(str(e))

    # 写入输出文件
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("\n\n".join(sections))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ragas I/O 调试脚本")
    parser.add_argument("--input-file", type=str,
                        default="/home/pushihao/RAG/Reports/experiments/datasets/rag_evaluation_results/hotpotqa/evaluation_results.json",
                        help="评估结果 JSON 文件路径")
    parser.add_argument("--output-file", type=str,
                        default="/home/pushihao/RAG/002.txt",
                        help="写出完整调试输出的文件路径")
    parser.add_argument("--dataset", type=str, default=None, help="数据集名称（如 hotpotqa）")
    parser.add_argument("--limit", type=int, default=None, help="可选：限制样本数以便快速查看")
    parser.add_argument("--timeout", type=int, default=60, help="评估请求超时")
    parser.add_argument("--max-workers", type=int, default=8, help="并发 worker 数量")
    parser.add_argument("--max-retries", type=int, default=2, help="请求失败重试次数")
    parser.add_argument("--max-wait", type=int, default=10, help="限流最大等待")
    args = parser.parse_args()

    debug_ragas(
        input_file=args.input_file,
        output_file=args.output_file,
        dataset_name=args.dataset,
        limit=args.limit,
        timeout=args.timeout,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        max_wait=args.max_wait,
    )


if __name__ == "__main__":
    main()