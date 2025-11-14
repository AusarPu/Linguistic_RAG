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
import aiohttp
from tqdm import tqdm

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

# === LLM返回内容日志开关与辅助 ===
_EVAL_LOG_ENABLED = str(os.getenv("EVAL_LOG_LLM", "")).strip().lower() in ("1","true","yes","on")
_EVAL_LOG_FILE = os.getenv("EVAL_LOG_FILE", os.path.join(project_root, "Reports", "experiments", "evaluation", "llm_return_log.txt"))

def _append_llm_log(gt: str, ans: str, content: str, raw_obj: any, tag: str) -> None:
    if not _EVAL_LOG_ENABLED:
        return
    os.makedirs(os.path.dirname(_EVAL_LOG_FILE), exist_ok=True)
    record = {
        "tag": tag,
        "ground_truth": gt,
        "system_answer": ans,
        "llm_content": content,
        "raw": raw_obj,
    }
    with open(_EVAL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

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

        # 基于配置进行上下文裁剪：按顺序保留，直到达到 token 总量与最大块数的上限
        max_tokens = getattr(config, "EVALUATION_CONTEXTS_MAX_INPUT_TOKENS", None)
        max_chunks = getattr(config, "EVALUATION_CONTEXTS_MAX_CHUNKS", None)
        ctx_texts: List[str] = []
        total_tokens = 0
        for t in raw_ctx_texts:
            # 如果达到最大块数限制则停止
            if isinstance(max_chunks, int) and max_chunks > 0 and len(ctx_texts) >= max_chunks:
                break
            # 如果达到最大token限制则停止
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

# === LLM 正确性判断（新增） ===
_CORRECTNESS_PROMPT = (
    "你是一位严格的正确性判别器。\n"
    "给定标准答案（ground truth）和系统回答（system answer），请基于事实一致性进行判断。\n"
    "只输出 `correct` 或 `incorrect`，不要输出其他内容。\n"
    "判断要求：\n"
    "- 忽略措辞差异与同义表达；\n"
    "- 若系统回答与标准答案语义等价或包含完整核心事实，判为 `correct`；\n"
    "- 若系统回答与标准答案矛盾、缺失关键事实或给出错误信息，判为 `incorrect`。\n"
)

def _gt_is_no_answer(gt: str) -> bool:
    s = (gt or "").strip().lower()
    # 包含常见“无答案”表达（含用户明确提到的拼写：no answer presnted）
    no_answer_set = {
        "no answer presented",
        "no answer presnted",
        "no answer",
        "no-answer",
        "noanswer",
        "none",
        "n/a",
        "not provided",
        "unknown",
        "未提供答案",
        "没有答案",
        "无答案",
        "未知",
        "不详",
    }
    return s in no_answer_set

def _ans_is_insufficient(ans: str) -> bool:
    s = (ans or "").strip().lower()
    patterns = [
        "信息不足",
        "无法回答",
        "无法确定",
        "依据不足",
        "缺少信息",
        "无法提供",
        "不知道",
        "不确定",
        "无法判断",
        "insufficient information",
        "not enough information",
        "cannot determine",
        "cannot answer",
        "unknown",
        "insufficient context",
        "lack of information",
        "not provided in context",
    ]
    return any(p in s for p in patterns)

def _llm_classify_correctness(gt: str, ans: str) -> str:
    base_url = config.GENERATOR_API_URL.rsplit("/chat/completions", 1)[0]
    url = f"{base_url}/chat/completions"
    payload = {
        "model": config.GENERATOR_MODEL_NAME_FOR_API,
        "messages": [
            {"role": "system", "content": _CORRECTNESS_PROMPT},
            {"role": "user", "content": f"[Ground Truth]\n{gt}\n[System Answer]\n{ans}"},
        ],
        "temperature": 0.0,
        "top_p": 0.95,
        "max_tokens": getattr(config, "EVALUATION_MAX_TOKENS", 10240),
        # 引导分类，仅返回 `correct` 或 `incorrect`
        "guided_choice": ["correct", "incorrect"],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        resp_json = json.loads(resp.read().decode("utf-8"))
    content = resp_json["choices"][0]["message"]["content"]
    _append_llm_log(gt, ans, content, resp_json, "sync")
    return (content or "").strip().lower()

async def _async_llm_classify_correctness(gt: str, ans: str, session: aiohttp.ClientSession, url: str) -> str:
    """异步版本的正确性判别，返回 'correct' 或 'incorrect'。"""
    payload = {
        "model": config.GENERATOR_MODEL_NAME_FOR_API,
        "messages": [
            {"role": "system", "content": _CORRECTNESS_PROMPT},
            {"role": "user", "content": f"[Ground Truth]\n{gt}\n[System Answer]\n{ans}"},
        ],
        "temperature": 0.0,
        "top_p": 0.95,
        "max_tokens": getattr(config, "EVALUATION_MAX_TOKENS", 10240),
        "guided_choice": ["correct", "incorrect"],
    }
    headers = {"Content-Type": "application/json"}
    async with session.post(url, json=payload, headers=headers) as resp:
        result = await resp.json()
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    _append_llm_log(gt, ans, content, result, "async")
    return (content or "").strip().lower()

def _judge_correctness(gt: str, ans: str) -> str:
    # 特殊规则：ground truth 标记“无答案”，系统回答为“信息不足”类表述，判为正确
    if _gt_is_no_answer(gt) and _ans_is_insufficient(ans):
        return "correct"
    return _llm_classify_correctness(gt, ans)

async def _compute_llm_accuracy_async(results: List[Dict[str, Any]], concurrency_limit: Optional[int] = None, max_retries: int = 3) -> List[int]:
    """使用异步并发与信号量计算逐条准确率。

    - 并发量由 config.EVALUATION_CONCURRENCY_LIMIT 控制，或传入覆盖。
    - 默认重试 3 次（用户要求）。
    """
    base_url = config.GENERATOR_API_URL.rsplit("/chat/completions", 1)[0]
    url = f"{base_url}/chat/completions"
    limit = concurrency_limit or getattr(config, "EVALUATION_CONCURRENCY_LIMIT", 100)
    sem = asyncio.Semaphore(limit)
    timeout_cfg = aiohttp.ClientTimeout(total=getattr(config, "VLLM_REQUEST_TIMEOUT", 60*20),
                                        connect=getattr(config, "VLLM_REQUEST_TIMEOUT", 60*20),
                                        sock_read=getattr(config, "VLLM_REQUEST_TIMEOUT", 60*20))
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        acc: List[int] = [0] * len(results)

        async def run_one(idx: int) -> None:
            item = results[idx]
            gt = item.get("ground_truth_answer", "")
            ans = item.get("system_answer", "")
            # 特殊规则优先
            if _gt_is_no_answer(gt) and _ans_is_insufficient(ans):
                acc[idx] = 1
                return
            attempt = 0
            while attempt < max_retries:
                attempt += 1
                try:
                    async with sem:
                        verdict = await _async_llm_classify_correctness(gt, ans, session, url)
                    acc[idx] = 1 if verdict == "correct" else 0
                    return
                except Exception:
                    # 显式重试（需求：默认3次），简单退避
                    if attempt >= max_retries:
                        acc[idx] = 0
                    else:
                        await asyncio.sleep(min(0.2 * attempt, 2.0))

        tasks = [asyncio.create_task(run_one(i)) for i in range(len(results))]
        progress = tqdm(total=len(results), desc="Computing accuracy (LLM)", unit="item")
        for t in asyncio.as_completed(tasks):
            await t
            progress.update(1)
        progress.close()
        return acc


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
                        max_workers: int = 8, timeout: int = 120, max_retries: int = 2, max_wait: int = 10,
                        llm_max_retries: int = 3) -> Optional[pd.DataFrame]:
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

    # === 新增：LLM 正确率列（accuracy） — 异步并发 + 重试 ===
    # 并发量：config.EVALUATION_CONCURRENCY_LIMIT；重试：默认3次
    llm_accuracy = asyncio.run(_compute_llm_accuracy_async(results, getattr(config, "EVALUATION_CONCURRENCY_LIMIT", 100), llm_max_retries))
    df["accuracy"] = llm_accuracy

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
            llm_max_retries=3,
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


def _attach_accuracy_to_existing_csv(dataset_dir: str, summary_csv_path: Optional[str] = None,
                                     llm_max_retries: int = 3) -> None:
    """在指定数据集目录下，读取 ragas_metrics.csv 并附加 accuracy 列（异步并发 + 重试）。

    若提供 summary_csv_path，则追加该数据集的均值到 ragas_summary（含 accuracy）。
    """
    csv_path = os.path.join(dataset_dir, "ragas_metrics.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    gt_col = "reference" if "reference" in cols else ("ground_truth" if "ground_truth" in cols else None)
    ans_col = "response" if "response" in cols else ("answer" if "answer" in cols else None)
    if gt_col is None or ans_col is None:
        return
    gt_list = df[gt_col].astype(str).tolist()
    ans_list = df[ans_col].astype(str).tolist()
    results = [{"ground_truth_answer": g, "system_answer": a} for g, a in zip(gt_list, ans_list)]
    llm_accuracy = asyncio.run(_compute_llm_accuracy_async(results, getattr(config, "EVALUATION_CONCURRENCY_LIMIT", 100), llm_max_retries))
    df["accuracy"] = llm_accuracy
    df.to_csv(csv_path, index=False)

    if summary_csv_path:
        dataset_name = os.path.basename(dataset_dir.rstrip("/"))
        metrics_cols = [
            "answer_relevancy",
            "faithfulness",
            "context_precision",
            "context_recall",
            "accuracy",
        ]
        means = {col: float(pd.to_numeric(df[col], errors="coerce").mean()) if col in df.columns else float("nan") for col in metrics_cols}
        numeric_df = df[[c for c in metrics_cols if c in df.columns]].apply(pd.to_numeric, errors="coerce")
        nan_counts = {col: int(numeric_df[col].isna().sum()) if col in numeric_df.columns else 0 for col in metrics_cols}
        nan_rows_total = int(numeric_df.isna().any(axis=1).sum()) if not numeric_df.empty else 0
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
                    *[f"{summary_row[col]:.6f}" if isinstance(summary_row[col], float) else str(summary_row[col]) for col in metrics_cols],
                    str(summary_row["total_questions"]),
                    str(summary_row["nan_rows_total"]),
                    *[str(summary_row[f"nan_count_{col}"]) for col in metrics_cols],
                ]) + "\n"
            )



def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="高级RAG评估脚本（仅Ragas）")
    parser.add_argument("input_file", nargs="?", help="输入的评估结果文件路径")
    parser.add_argument("--csv-output-file", type=str, help="Ragas评估结果CSV输出路径")
    parser.add_argument("--summary-csv", type=str, help="Ragas汇总CSV输出路径（写在原txt目录）")
    parser.add_argument("--limit", type=int, help="限制处理的结果数量（用于测试）")
    # Ragas加速相关参数
    parser.add_argument("--max-workers", type=int, default=20, help="Ragas并发工作数")
    parser.add_argument("--timeout", type=int, default=1200, help="Ragas评判请求超时（秒）")
    parser.add_argument("--max-retries", type=int, default=10, help="Ragas请求失败重试次数")
    parser.add_argument("--max-wait", type=int, default=1200, help="Ragas遇到限流时的最大等待（秒）")
    # 重复实验控制
    parser.add_argument("--repeat", type=int, default=1, help="重复实验次数（默认3）")
    # 独立运行：指定数据集目录，对现有 ragas_metrics.csv 附加 accuracy
    parser.add_argument("--attach-accuracy-dir", type=str, help="指定数据集父目录（包含各子数据集），对其中的 ragas_metrics.csv 附加 accuracy 列")
    parser.add_argument("--llm-max-retries", type=int, default=3, help="LLM 正确性判别的重试次数（默认3）")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if args.input_file and not os.path.exists(args.input_file):
        logger.error(f"输入文件不存在: {args.input_file}")
        return
    
    # 独立运行：附加 accuracy 到指定目录的现有 CSV（优先执行）
    if args.attach_accuracy_dir:
        base_dir = args.attach_accuracy_dir
        if os.path.isdir(base_dir):
            for d in os.listdir(base_dir):
                dataset_dir = os.path.join(base_dir, d)
                if os.path.isdir(dataset_dir):
                    candidate = os.path.join(dataset_dir, "ragas_metrics.csv")
                    if os.path.exists(candidate):
                        _attach_accuracy_to_existing_csv(dataset_dir, summary_csv_path=args.summary_csv, llm_max_retries=args.llm_max_retries)
        else:
            logger.error(f"attach_accuracy_dir 非目录: {base_dir}")
        return

    # 根据 repeat 控制单次或重复评估
    if args.repeat and args.repeat > 1:
        if not args.input_file:
            logger.error("未提供 input_file，在重复评估模式下无法运行。若仅需为现有CSV附加accuracy，请使用 --attach-accuracy-dir")
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
            logger.error("未提供 --input-file，无法执行评估模式。若仅需为现有CSV附加accuracy，请使用 --attach-accuracy-dir")
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
            llm_max_retries=args.llm_max_retries,
        )

if __name__ == "__main__":
    main()
# 统一日志策略：只输出警告（含重试），屏蔽普通 info
logging.getLogger("ragas.executor").setLevel(logging.WARNING)
logging.getLogger("ragas.prompt.pydantic_prompt").setLevel(logging.ERROR)
