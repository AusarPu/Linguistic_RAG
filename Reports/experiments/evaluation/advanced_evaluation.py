#!/usr/bin/env python3
"""
高级评估脚本
仅使用Ragas评估RAG系统的答案正确性和检索准确性
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import importlib

# Ragas & wrappers
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
from ragas.run_config import RunConfig
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from script import config_rag as config
import script.knowledge_base as knowledge_base_module
from script.llm_wrappers import RagasOpenAICompatLLMWrapper, RagasOpenAICompatEmbeddings

# 设置日志
config.setup_logging()
logger = logging.getLogger(__name__)

# === Ragas 集成辅助逻辑 ===
def _get_dataset_name_from_path(input_file: str) -> str:
    """从输入文件路径中推断数据集名称"""
    p = Path(input_file)
    # 期望结构: .../rag_evaluation_results/<dataset>/<file>.json
    return p.parent.name


def _init_kb_for_dataset(dataset_name: str):
    """为指定数据集初始化 KnowledgeBase（通过动态设置 config 路径）"""
    # 动态指向该数据集的索引目录
    index_dir = f"/home/pushihao/RAG/Reports/experiments/datasets/knowledge_bases/{dataset_name}"
    config.FAISS_INDEX_CHUNKS_SAVE_PATH = os.path.join(index_dir, "faiss_index_chunks_ip.idx")
    config.INDEXED_CHUNKS_METADATA_SAVE_PATH = os.path.join(index_dir, "indexed_chunks_metadata.json")
    config.PHRASE_DENSE_EMBEDDINGS_MAP_SAVE_PATH = os.path.join(index_dir, "phrase_dense_embeddings_map.pkl")
    config.BM25_INDEX_SAVE_PATH = os.path.join(index_dir, "phrase_bm25_index.pkl")
    config.FAISS_INDEX_QUESTIONS_SAVE_PATH = os.path.join(index_dir, "faiss_index_questions_ip.idx")
    config.QUESTION_INDEX_TO_CHUNK_ID_MAP_SAVE_PATH = os.path.join(index_dir, "question_index_to_chunk_id_map.json")
    config.ALL_QUESTION_TEXTS_SAVE_PATH = os.path.join(index_dir, "all_question_texts.json")
    config.CHUNK_BM25_INDEX_SAVE_PATH = os.path.join(index_dir, "chunk_bm25_index.pkl")

    # 重新导入知识库以应用新的路径设置
    importlib.reload(knowledge_base_module)
    return knowledge_base_module.KnowledgeBase()


def _prepare_ragas_dataset(results: List[Dict[str, Any]], kb) -> Dataset:
    """将评估结果转换为 Ragas 数据集（问题、上下文、答案、参考答案）"""
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
        chunk_ids = item.get("retrieved_chunk_ids", [])

        ctx_texts: List[str] = []
        for cid in chunk_ids:
            meta = chunk_map[cid]
            text_val = meta.get("text") or meta.get("text_chunk_content")
            ctx_texts.append(text_val)

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


def _get_ragas_clients():
    """初始化 Ragas 评估所需的 LLM 与 Embeddings 客户端"""
    # 从项目配置读取 API 地址与模型名
    generator_base_url = config.GENERATOR_API_URL.rsplit("/chat/completions", 1)[0]
    embedding_base_url = config.EMBEDDING_API_URL.rsplit("/embeddings", 1)[0]

    # 使用自定义封装，直接走 OpenAI 兼容接口，支持 n>1
    llm = RagasOpenAICompatLLMWrapper(
        base_url=generator_base_url,
        model=config.GENERATOR_MODEL_NAME_FOR_API,
        api_key="-",
        temperature=0.2,
        top_p=0.9,
    )
    embeddings = RagasOpenAICompatEmbeddings(
        base_url=embedding_base_url,
        api_key="-",
        model=config.EMBEDDING_MODEL_NAME_FOR_API,
    )

    return llm, embeddings


def evaluate_with_ragas(input_file: str, csv_output_file: Optional[str] = None,
                        summary_csv_path: Optional[str] = None, limit: Optional[int] = None,
                        max_workers: int = 8, timeout: int = 120, max_retries: int = 2, max_wait: int = 10) -> Optional[pd.DataFrame]:
    """运行 Ragas 评估并写出 CSV（以及可选的汇总 CSV）"""
    with open(input_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not isinstance(results, list):
        logger.error("输入文件格式错误：应为JSON数组")
        return None

    if limit and limit > 0:
        results = results[:limit]

    dataset_name = _get_dataset_name_from_path(input_file)
    kb = _init_kb_for_dataset(dataset_name)

    # 构建 ragas 数据集
    hf_dataset = _prepare_ragas_dataset(results, kb)

    # 初始化评估客户端
    evaluator_llm, evaluator_embeddings = _get_ragas_clients()

    # 构建指标列表（直接导入的指标）
    metrics_list = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ]

    # 配置并发和重试运行参数
    run_config = RunConfig(
        timeout=timeout,
        max_workers=max_workers,
        max_retries=max_retries,
        max_wait=max_wait,
    )
    logger.info(
        f"Ragas运行配置: max_workers={max_workers}, timeout={timeout}s, max_retries={max_retries}, max_wait={max_wait}s"
    )
    logger.info(
        "启用指标: " + ", ".join([m.name if hasattr(m, 'name') else str(m) for m in metrics_list])
    )

    # 运行评估
    ragas_result = ragas_evaluate(
        dataset=hf_dataset,
        metrics=metrics_list,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )

    df = ragas_result.to_pandas()

    # 写 per-dataset CSV
    # 默认将输出文件名设置为同目录下的 ragas_metrics.csv
    if not csv_output_file:
        csv_output_file = os.path.join(
            os.path.dirname(input_file),
            "ragas_metrics.csv",
        )
    os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)
    df.to_csv(csv_output_file, index=False)
    logger.info(f"Ragas评估CSV已保存: {csv_output_file}")

    # 追加/生成根汇总 CSV
    if summary_csv_path:
        summary_cols = [
            "answer_relevancy",
            "faithfulness",
            "context_precision",
            "context_recall",
        ]
        means = {col: float(pd.to_numeric(df[col], errors="coerce").mean()) for col in summary_cols}
        summary_row = {
            "dataset": dataset_name,
            **means,
            "total_questions": len(df),
        }

        os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
        # 如果文件不存在，写入头；否则追加
        write_header = not os.path.exists(summary_csv_path)
        with open(summary_csv_path, "a", encoding="utf-8") as f:
            if write_header:
                # 动态写入表头
                header = ["dataset"] + summary_cols + ["total_questions"]
                f.write(",".join(header) + "\n")
            f.write(
                ",".join([
                    summary_row["dataset"],
                    *[f"{summary_row[col]:.6f}" for col in summary_cols],
                    str(summary_row["total_questions"])]) + "\n"
            )
        logger.info(f"Ragas汇总CSV已更新: {summary_csv_path}")

    # 关闭客户端会话（避免未关闭的 aiohttp ClientSession 警告）
    if hasattr(evaluator_llm, "close"):
        evaluator_llm.close()
    if hasattr(evaluator_embeddings, "close"):
        evaluator_embeddings.close()

    return df


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="高级RAG评估脚本（仅Ragas）")
    parser.add_argument("input_file", help="输入的评估结果文件路径")
    parser.add_argument("--csv-output-file", type=str, help="Ragas评估结果CSV输出路径")
    parser.add_argument("--summary-csv", type=str, help="Ragas汇总CSV输出路径（写在原txt目录）")
    parser.add_argument("--limit", type=int, help="限制处理的结果数量（用于测试）")
    # Ragas加速相关参数
    parser.add_argument("--max-workers", type=int, default=200, help="Ragas并发工作数")
    parser.add_argument("--timeout", type=int, default=60, help="Ragas评判请求超时（秒）")
    parser.add_argument("--max-retries", type=int, default=5, help="Ragas请求失败重试次数")
    parser.add_argument("--max-wait", type=int, default=5, help="Ragas遇到限流时的最大等待（秒）")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_file):
        logger.error(f"输入文件不存在: {args.input_file}")
        return
    
    # 运行Ragas评估，并导出CSV（以及可选的根汇总CSV）
    evaluate_with_ragas(
        input_file=args.input_file,
        csv_output_file=args.csv_output_file,
        summary_csv_path=args.summary_csv,
        limit=args.limit,
        max_workers=args.max_workers,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_wait=args.max_wait,
    )

if __name__ == "__main__":
    main()
# 诊断增强：降低冗余告警，提升关键日志
logging.getLogger("ragas.executor").setLevel(logging.INFO)
logging.getLogger("ragas.prompt.pydantic_prompt").setLevel(logging.ERROR)