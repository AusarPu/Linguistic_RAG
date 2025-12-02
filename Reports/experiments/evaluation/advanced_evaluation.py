#!/usr/bin/env python3
"""
高级评估脚本
仅使用Ragas评估RAG系统的答案正确性和检索准确性
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import importlib
import statistics
import urllib.request
from tqdm import tqdm

# Ragas & wrappers
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from datasets import Dataset
from ragas.run_config import RunConfig

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from script import config_rag as config
import script.knowledge_base as knowledge_base_module
from script.llm_wrappers import RagasOpenAICompatLLMWrapper, RagasOpenAICompatEmbeddings
from preprocess.vllm_tokenizer import fast_token_length

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
    """为指定数据集初始化 KnowledgeBase（通过动态设置 config 路径）
    支持通过环境变量 KB_BASE_DIR 覆盖默认知识库基目录，以适配 runs/<run_id>。
    """
    # 动态指向该数据集的索引目录
    kb_base = os.environ.get("KB_BASE_DIR", "/home/pushihao/RAG/Reports/experiments/datasets/knowledge_bases")
    index_dir = os.path.join(kb_base, dataset_name)
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

        # 收集原始上下文文本
        raw_ctx_texts: List[str] = []
        for cid in chunk_ids:
            meta = chunk_map.get(cid)
            if not meta:
                continue
            text_val = meta.get("text") or meta.get("text_chunk_content") or ""
            if text_val:
                raw_ctx_texts.append(text_val)

        ctx_texts: List[str] = raw_ctx_texts

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
    generator_base_url = config.EVALUATION_LLM_API_URL.rsplit("/chat/completions", 1)[0]
    embedding_base_url = config.EMBEDDING_API_URL.rsplit("/embeddings", 1)[0]
    llm = RagasOpenAICompatLLMWrapper(
        base_url=generator_base_url,
        model=config.EVALUATION_LLM_MODEL_LOCAL_PATH,
        api_key="-",
        temperature=0,
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

    # 并发控制：采用 config.EVALUATION_CONCURRENCY_LIMIT（不再用 CLI 覆盖），保持评估阶段统一并发策略

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
        answer_correctness,
    ]

    # 配置并发和重试运行参数
    run_config = RunConfig(
        timeout=timeout,
        max_workers=max_workers,
        max_retries=max_retries,
        max_wait=max_wait,
    )
    # 统一超时：显式将 RunConfig 应用于 LLM 与 Embeddings 客户端，使两者使用相同的 timeout 设置
    evaluator_llm.run_config = run_config
    evaluator_embeddings.set_run_config(run_config)
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

    # 使用 Ragas 的 answer_correctness 指标作为 accuracy 值
    if "answer_correctness" in df.columns:
        df["accuracy"] = pd.to_numeric(df["answer_correctness"], errors="coerce")
    else:
        df["accuracy"] = float("nan")

    df.to_csv(csv_output_file, index=False)
    logger.info(f"Ragas评估CSV已保存: {csv_output_file}")

    # === NaN 检查：统计每个指标中的 NaN 数量，并列出含 NaN 的行 ===
    metrics_cols = [
        "answer_relevancy",
        "faithfulness",
        "context_precision",
        "context_recall",
        "accuracy",
    ]
    numeric_df = df[metrics_cols].apply(pd.to_numeric, errors="coerce")
    nan_counts = {col: int(numeric_df[col].isna().sum()) for col in metrics_cols}
    nan_row_mask = numeric_df.isna().any(axis=1)
    nan_rows_total = int(nan_row_mask.sum())
    nan_row_indices = df.index[nan_row_mask].tolist()

    # 输出日志，帮助快速定位问题
    if nan_rows_total > 0:
        logger.warning(f"数据集 {dataset_name} 含有 NaN 行: {nan_rows_total}，行索引: {nan_row_indices}")
    for col in metrics_cols:
        if nan_counts[col] > 0:
            logger.warning(f"数据集 {dataset_name} 指标 {col} 存在 NaN 数量: {nan_counts[col]}")
    if nan_rows_total == 0 and all(v == 0 for v in nan_counts.values()):
        logger.info(f"数据集 {dataset_name} 指标无 NaN")

    # 生成每数据集的 NaN 报告 CSV，仅包含出现 NaN 的行
    nan_report_dir = os.path.dirname(csv_output_file)
    # 根据输出CSV文件名自动派生 NaN 报告文件名（如 ragas_metrics_0.csv -> nan_report_0.csv）
    nan_report_suffix = ""
    csv_stem = Path(csv_output_file).stem
    last_seg = csv_stem.split("_")[-1]
    if last_seg.isdigit():
        nan_report_suffix = f"_{last_seg}"
    nan_report_file = os.path.join(nan_report_dir, f"nan_report{nan_report_suffix}.csv")
    nan_flags = {f"nan_{col}": numeric_df[col].isna() for col in metrics_cols}
    nan_any = pd.DataFrame(nan_flags)
    nan_any["row_index"] = numeric_df.index
    nan_any["is_nan_any"] = nan_any[[f"nan_{c}" for c in metrics_cols]].any(axis=1)
    nan_any_rows = nan_any[nan_any["is_nan_any"]]
    nan_any_rows.to_csv(nan_report_file, index=False)
    logger.info(f"NaN检查报告已保存: {nan_report_file}")

    # 追加/生成根汇总 CSV
    if summary_csv_path:
        # 使用与 NaN 检查一致的指标列，保证统计一致性
        metrics_cols = [
            "answer_relevancy",
            "faithfulness",
            "context_precision",
            "context_recall",
            "accuracy",
        ]
        means = {col: float(pd.to_numeric(df[col], errors="coerce").mean()) for col in metrics_cols}
        # 已在上文计算 numeric_df / nan_counts / nan_rows_total
        summary_row = {
            "dataset": dataset_name,
            **means,
            "total_questions": len(df),
            "nan_rows_total": nan_rows_total,
            **{f"nan_count_{col}": nan_counts[col] for col in metrics_cols},
        }

        os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
        write_header = not os.path.exists(summary_csv_path)
        with open(summary_csv_path, "a", encoding="utf-8") as f:
            if write_header:
                header = [
                    "dataset",
                    *metrics_cols,
                    "total_questions",
                    "nan_rows_total",
                    *[f"nan_count_{col}" for col in metrics_cols],
                ]
                f.write(",".join(header) + "\n")
            f.write(
                ",".join([
                    summary_row["dataset"],
                    *[f"{summary_row[col]:.6f}" for col in metrics_cols],
                    str(summary_row["total_questions"]),
                    str(summary_row["nan_rows_total"]),
                    *[str(summary_row[f"nan_count_{col}"]) for col in metrics_cols],
                ]) + "\n"
            )
        logger.info(f"Ragas汇总CSV已更新: {summary_csv_path}")

    # 关闭客户端会话（避免未关闭的 aiohttp ClientSession 警告）
    if hasattr(evaluator_llm, "close"):
        evaluator_llm.close()
    if hasattr(evaluator_embeddings, "close"):
        evaluator_embeddings.close()

    return df


def _compute_means_from_df(df: pd.DataFrame) -> Dict[str, float]:
    """从评估结果 DataFrame 计算指标均值（与汇总写入逻辑一致）。"""
    metrics_cols = [
        "answer_relevancy",
        "faithfulness",
        "context_precision",
        "context_recall",
    ]
    means = {col: float(pd.to_numeric(df[col], errors="coerce").mean()) for col in metrics_cols}
    return means


def run_repeated_evaluation(input_file: str,
                            csv_output_file_base: Optional[str],
                            summary_csv_path: Optional[str],
                            limit: Optional[int],
                            repeat: int,
                            max_workers: int,
                            timeout: int,
                            max_retries: int,
                            max_wait: int) -> None:
    """按指定次数重复运行评估，并输出跨重复的方差（txt或csv）。

    输出命名规则：
    - 若提供了 csv_output_file_base，例如 /path/ragas_metrics.csv，则各次输出为 /path/ragas_metrics_0.csv, /path/ragas_metrics_1.csv, ...
    - 未提供 csv_output_file_base 时，默认在数据集目录下生成 ragas_metrics_0.csv, ragas_metrics_1.csv, ...
    - NaN 报告文件同样按 _i 后缀命名（nan_report_0.csv 等）。
    - 方差文件默认输出到相同目录下，命名为 ragas_variance.txt 或 ragas_variance.csv。
    """
    dataset_name = _get_dataset_name_from_path(input_file)
    base_dir = os.path.dirname(csv_output_file_base) if csv_output_file_base else os.path.dirname(input_file)
    base_stem = Path(csv_output_file_base).stem if csv_output_file_base else "ragas_metrics"
    base_ext = Path(csv_output_file_base).suffix if csv_output_file_base else ".csv"

    means_per_run: List[Dict[str, float]] = []

    for i in range(repeat):
        csv_out = os.path.join(base_dir, f"{base_stem}_{i}{base_ext}")
        df = evaluate_with_ragas(
            input_file=input_file,
            csv_output_file=csv_out,
            summary_csv_path=summary_csv_path,
            limit=limit,
            max_workers=max_workers,
            timeout=timeout,
            max_retries=max_retries,
            max_wait=max_wait,
        )
        if df is None:
            logger.error("评估返回空结果，停止重复执行")
            return
        means_per_run.append(_compute_means_from_df(df))

    metrics_cols = [
        "answer_relevancy",
        "faithfulness",
        "context_precision",
        "context_recall",
    ]
    variances: Dict[str, float] = {}
    for col in metrics_cols:
        values = [m[col] for m in means_per_run]
        if len(values) >= 2:
            variances[col] = float(statistics.variance(values))
        else:
            variances[col] = 0.0

    var_file = os.path.join(base_dir, "ragas_variance.csv")
    write_header = not os.path.exists(var_file)
    with open(var_file, "a", encoding="utf-8") as f:
        if write_header:
            f.write("dataset,repeats,metric,variance\n")
        for col in metrics_cols:
            f.write(f"{dataset_name},{repeat},{col},{variances[col]:.6f}\n")
    logger.info(f"重复实验方差已输出: {var_file}")


 



def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="高级RAG评估脚本（仅Ragas）")
    parser.add_argument("input_file", nargs="?", help="输入的评估结果文件路径")
    parser.add_argument("--csv-output-file", type=str, help="Ragas评估结果CSV输出路径")
    parser.add_argument("--summary-csv", type=str, help="Ragas汇总CSV输出路径（写在原txt目录）")
    parser.add_argument("--limit", type=int, help="限制处理的结果数量（用于测试）")
    # Ragas加速相关参数
    parser.add_argument("--max-workers", type=int, default=config.EVALUATION_CONCURRENCY_LIMIT, help="Ragas并发工作数")
    parser.add_argument("--timeout", type=int, default=1200, help="Ragas评判请求超时（秒）")
    parser.add_argument("--max-retries", type=int, default=10, help="Ragas请求失败重试次数")
    parser.add_argument("--max-wait", type=int, default=1200, help="Ragas遇到限流时的最大等待（秒）")
    # 重复实验控制
    parser.add_argument("--repeat", type=int, default=1, help="重复实验次数（默认3）")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if args.input_file and not os.path.exists(args.input_file):
        logger.error(f"输入文件不存在: {args.input_file}")
        return
    
 

    # 根据 repeat 控制单次或重复评估
    if args.repeat and args.repeat > 1:
        if not args.input_file:
            logger.error("未提供 input_file，在重复评估模式下无法运行。")
            return
        run_repeated_evaluation(
            input_file=args.input_file,
            csv_output_file_base=args.csv_output_file,
            summary_csv_path=args.summary_csv,
            limit=args.limit,
            repeat=args.repeat,
            max_workers=args.max_workers,
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_wait=args.max_wait,
        )
    else:
        # 运行Ragas评估，并导出CSV（以及可选的根汇总CSV）
        if not args.input_file:
            logger.error("未提供 --input-file，无法执行评估模式。")
            return
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
# 统一日志策略：只输出警告（含重试），屏蔽普通 info
logging.getLogger("ragas.executor").setLevel(logging.WARNING)
logging.getLogger("ragas.prompt.pydantic_prompt").setLevel(logging.ERROR)
