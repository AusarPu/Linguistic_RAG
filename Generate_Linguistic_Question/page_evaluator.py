#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
页面级评估与重试脚本（基于 vLLM Generator）

功能概述：
- 针对已“切分并优化”的 JSON（每条含 doc_name、page_number、chunk_id、text 等），按“页”聚合评估；
- 默认先以当前页（单页上下文）发送给 LLM；若返回“信息不足”，则以前后页拼接（共三页上下文）重试一次；
- 仍“信息不足”则跳过该页（不写入 CSV）；
- 当返回“正常结果”时，将 JSON 中的条目扁平化写入 CSV。

配置来源：直接使用 script.config_rag 中的 Generator 配置（GENERATOR_API_URL、GENERATOR_MODEL_NAME_FOR_API、GENERATION_CONFIG、VLLM_REQUEST_TIMEOUT_GENERATION）。

注意事项（遵循项目规则）：
- 不使用 try/except；
- 不加入为避免出错而设计的 if/else，分支仅用于业务逻辑；
- 该脚本假定输入 JSON 结构正确且 vLLM 服务可用。
"""

import os
import sys
import json
import csv
import asyncio
import time
from typing import List, Dict, Any, Optional, Literal

import aiohttp
from tqdm import tqdm
from pydantic import BaseModel, Field
import pydantic_core

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from script import config_rag as config
from preprocess.llm_chunk_processor import extract_content_from_vllm_response


class EvaluationItem(BaseModel):
    question: str
    answer: Optional[str] = None


class PageEvaluationOutput(BaseModel):
    decision_type: Literal["NO_ISSUE", "INSUFFICIENT_INFO", "VALID_RESULT"]
    reason: Optional[str] = None
    items: List[EvaluationItem] = Field(default_factory=list)


def compose_eval_prompt(doc_name: str,
                        page_number: int,
                        page_text: str,
                        prev_text: Optional[str] = None,
                        next_text: Optional[str] = None,
                        use_triple_context: bool = False,
                        max_items: int = 10) -> str:
    """构造中文评估提示词（对齐 workguide3.md 的质量规则）。"""
    header = (
        "你是评估助手。请基于给定的页面内容，判断该页是否存在可评估的题目或问答。注意，题目是页面中已经存在的题目，禁止自己生成题目。\n"
        "请严格按以下三种结果之一返回：\n"
        "1）NO_ISSUE：这一页没有问题（没有问题或者问题都是略、或不适合评估）。\n"
        "2）INSUFFICIENT_INFO：这页包含应该纳入评估的问题，但上下文不足，需要更多上下文。\n"
        "3）VALID_RESULT：正常返回，items 为题目条目列表，每条包含 question、answer（可选）。\n"
        "要求：\n"
        "- 问题完整化：每个问题必须是完整自然的中文疑问句，并以“？”结尾；禁止半截句与“包括：”“例如：”“主要有：”式尾部。\n"
        "- 对于答案部分，如果是选择题，不能在答案中包含选项（A、B、C、D），只能包含选项的解释。\n"
        "- 对于答案部分，如果答案是略或者 ``参见本章复习笔记相关内容等等``，不直接说明答案的，则废弃此问题，不要输出，\n"
        "- 自动纠错：对轻微错别字、繁简混用与标点错误进行纠正，不改变事实与术语指称。\n"
        "- 枚举整合与去重（同页/同chunk）：遇到“包括/分为/由……组成”等结构，仅生成一题，在答案中列出完整枚举（顺序与原文一致，用“、”分隔）；语义相近或同义问题只保留信息最完整的一条。\n"
        "- 输出上限：本页最多生成" + str(max_items) + "条问答。\n\n"
        "返回结构为严格 JSON，字段遵循提供的 JSON Schema，不得输出任何解释文本或 Markdown 代码块。\n"
    )

    if use_triple_context:
        ctx = (
            f"文档：{doc_name}，页码：{page_number}\n"
            "【前页内容】\n" + (prev_text or "N/A") + "\n\n"
            "【当前页内容】\n" + page_text + "\n\n"
            "【后页内容】\n" + (next_text or "N/A") + "\n"
        )
    else:
        ctx = (
            f"文档：{doc_name}，页码：{page_number}\n"
            "【当前页内容】\n" + page_text + "\n"
        )

    tail = (
        "严格要求：所有字段值必须使用英文双引号\"，字符串内部出现的双引号必须转义为 \\\"；只返回 JSON，不要返回额外文本或注释。\n"
        "请返回如下 JSON Schema 对象：\n"
        "{\n"
        "  \"decision_type\": \"NO_ISSUE|INSUFFICIENT_INFO|VALID_RESULT\",\n"
        "  \"items\": [\n"
        "    {\n"
        "      \"question\": string,\n"
        "      \"answer\": string 可选\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )

    return header + ctx + "\n" + tail


async def call_llm_for_page(prompt: str) -> PageEvaluationOutput:
    """调用 vLLM Generator，使用非 strict 的 JSON Schema 响应；优先使用 message.parsed；
    在 parsed 缺失时，基于内容提取进行 JSON 解析的业务回退。"""
    payload = {
        "model": config.GENERATOR_MODEL_NAME_FOR_API,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "page_evaluation_output",
                "schema": PageEvaluationOutput.model_json_schema()
            }
        },
        **{k: v for k, v in config.GENERATION_CONFIG.items() if v is not None},
        "chat_template_kwargs": {"enable_thinking": True},
        "stream": False,
    }

    timeout = aiohttp.ClientTimeout(total=config.VLLM_REQUEST_TIMEOUT_GENERATION)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(config.GENERATOR_API_URL, json=payload) as resp:
            data = await resp.json()
            message = data["choices"][0]["message"]

            # 优先使用 OpenAI/vLLM 的严格 JSON Schema 解析结果（message.parsed）
            parsed = message.get("parsed")
            if isinstance(parsed, dict):
                # 构造输出对象（使用 model_construct 避免校验异常）
                items_raw = parsed.get("items", [])
                built_items: List[EvaluationItem] = []
                for it in items_raw:
                    built_items.append(
                        EvaluationItem.model_construct(
                            question=it.get("question", ""),
                            answer=it.get("answer")
                        )
                    )
                dt = parsed.get("decision_type") or "VALID_RESULT"
                rsn = parsed.get("reason")
                return PageEvaluationOutput.model_construct(
                    decision_type=dt,
                    reason=rsn,
                    items=built_items
                )

            # 若无 parsed 字段，执行业务回退：尝试基于内容进行 JSON 解析
            cleaned_content = extract_content_from_vllm_response(message, config.GENERATION_CONFIG)
            left = cleaned_content.find("{")
            right = cleaned_content.rfind("}")
            if left != -1 and right != -1 and right >= left:
                json_str = cleaned_content[left:right + 1]
                parsed2 = json.loads(json_str)
                items_raw = parsed2.get("items", [])
                built_items2: List[EvaluationItem] = []
                for it in items_raw:
                    built_items2.append(
                        EvaluationItem.model_construct(
                            question=it.get("question", ""),
                            answer=it.get("answer")
                        )
                    )
                dt2 = parsed2.get("decision_type") or "VALID_RESULT"
                rsn2 = parsed2.get("reason")
                return PageEvaluationOutput.model_construct(
                    decision_type=dt2,
                    reason=rsn2,
                    items=built_items2
                )

            # 内容无法解析为 JSON 时，返回信息不足以便上层业务逻辑重试
            return PageEvaluationOutput.model_construct(
                decision_type="INSUFFICIENT_INFO",
                reason="LLM未提供可解析的JSON内容，需重试",
                items=[]
            )


async def call_llm_for_page_with_retry(prompt: str,
                                        max_retries: int = 3,
                                        wait_seconds: float = 0.5) -> PageEvaluationOutput:
    """基于业务状态的有限重试（不使用异常捕获）。"""
    attempt = 0
    last_result: PageEvaluationOutput = PageEvaluationOutput.model_construct(
        decision_type="INSUFFICIENT_INFO",
        reason="初始化占位",
        items=[]
    )
    while attempt < max_retries:
        result = await call_llm_for_page(prompt)
        # 业务分支：仅在信息不足时重试；其他结果直接返回
        if result.decision_type != "INSUFFICIENT_INFO":
            return result
        last_result = result
        attempt += 1
        if attempt < max_retries:
            await asyncio.sleep(wait_seconds)
    return last_result


def group_pages_by_doc(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """将输入块按 doc_name 和 page_number 分组合并为页级结构。"""
    pages: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for ch in chunks:
        doc = ch["doc_name"]
        page = int(ch["page_number"]) if isinstance(ch["page_number"], int) else int(ch["page_number"])
        pages.setdefault(doc, {})
        if page not in pages[doc]:
            pages[doc][page] = {
                "doc_name": doc,
                "page_number": page,
                "chunk_ids": [],
                "text_list": []
            }
        pages[doc][page]["chunk_ids"].append(ch["chunk_id"])
        pages[doc][page]["text_list"].append(ch["text"])
    return pages


def build_unique_id_chunk_index(chunk_id: str, item_idx: int) -> str:
    return f"{chunk_id}_i{item_idx}"




async def _evaluate_one_page(doc_name: str,
                             page_number: int,
                             chunk_id_main: str,
                             page_text: str,
                             prev_text: str,
                             next_text: str,
                             start_ts: float,
                             semaphore: asyncio.Semaphore,
                             max_items: int = 10) -> List[List[Any]]:
    """评估单页（含必要的三页重试），返回需写入 CSV 的行列表。"""
    rows: List[List[Any]] = []
    async with semaphore:
        prompt_single = compose_eval_prompt(doc_name, page_number, page_text, use_triple_context=False, max_items=max_items)
        result_single = await call_llm_for_page_with_retry(prompt_single)

        if result_single.decision_type == "INSUFFICIENT_INFO":
            prompt_triple = compose_eval_prompt(doc_name, page_number, page_text, prev_text, next_text, use_triple_context=True, max_items=max_items)
            result_triple = await call_llm_for_page_with_retry(prompt_triple)
            if result_triple.decision_type == "INSUFFICIENT_INFO":
                return rows
            if result_triple.decision_type == "NO_ISSUE":
                return rows
            if result_triple.decision_type == "VALID_RESULT":
                for i, item in enumerate(result_triple.items):
                    uid = build_unique_id_chunk_index(chunk_id_main, i)

                    rows.append([
                        uid,
                        doc_name,
                        page_number,
                        chunk_id_main,
                        "VALID_RESULT",
                        item.question,
                        item.answer or "",
                        "triple"
                    ])
            return rows

        if result_single.decision_type == "NO_ISSUE":
            return rows

        if result_single.decision_type == "VALID_RESULT":
            for i, item in enumerate(result_single.items):
                uid = build_unique_id_chunk_index(chunk_id_main, i)

                rows.append([
                    uid,
                    doc_name,
                    page_number,
                    chunk_id_main,
                    "VALID_RESULT",
                    item.question,
                    item.answer or "",
                    "single"
                ])
    return rows


async def evaluate_pages_and_write_csv(input_json_path: str,
                                       output_csv_path: str,
                                       concurrency: int = 100,
                                       max_items_per_page: int = 10) -> None:
    start_ts = time.time()
    with open(input_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    pages_by_doc = group_pages_by_doc(chunks)
    doc_names = list(pages_by_doc.keys())
    sem = asyncio.Semaphore(concurrency)
    total_pages = sum(len(pages_by_doc[doc]) for doc in doc_names)
    pbar = tqdm(total=total_pages, desc="页级评估进度", unit="页")
    # 提前打开输出文件并写入表头；后续按页完成即写入并flush
    with open(output_csv_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "id", "doc_name", "page_number", "chunk_id",
            "decision_type", "question", "answer", "source_context"
        ])

        for doc in doc_names:
            pages_sorted = sorted(pages_by_doc[doc].keys())
            tasks: List[asyncio.Task] = []
            for idx, page in enumerate(pages_sorted):
                page_info = pages_by_doc[doc][page]
                doc_name = page_info["doc_name"]
                page_number = page_info["page_number"]
                chunk_id_main = page_info["chunk_ids"][0]
                page_text = "\n\n".join(page_info["text_list"]).strip()

                prev_text = "\n\n".join(pages_by_doc[doc][pages_sorted[idx - 1]]["text_list"]).strip() if idx - 1 >= 0 else "N/A"
                next_text = "\n\n".join(pages_by_doc[doc][pages_sorted[idx + 1]]["text_list"]).strip() if idx + 1 < len(pages_sorted) else "N/A"

                tasks.append(asyncio.create_task(
                    _evaluate_one_page(
                        doc_name,
                        page_number,
                        chunk_id_main,
                        page_text,
                        prev_text,
                        next_text,
                        start_ts,
                        sem,
                        max_items=max_items_per_page
                    )
                ))

            for coro in asyncio.as_completed(tasks):
                rows = await coro
                if rows:
                    for r in rows:
                        writer.writerow(r)
                    csvfile.flush()
                pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="对已优化的页面进行评估并导出CSV")
    parser.add_argument("--input", default=os.path.join("Generate_Linguistic_Question", "datasets", "chunked_question.json"))
    parser.add_argument("--output", default=os.path.join("Generate_Linguistic_Question", "datasets", "evaluation.csv"))
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--max-items-per-page", type=int, default=10)
    args = parser.parse_args()

    asyncio.run(evaluate_pages_and_write_csv(
        args.input,
        args.output,
        args.concurrency,
        max_items_per_page=args.max_items_per_page
    ))