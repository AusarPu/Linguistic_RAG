#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
清洗并校验 Generate_Linguistic_Question/datasets/evaluation.csv，生成报告与可选的清洗结果。

特性（对应 workguide.md 要求）：
- 审核模式（audit）：仅统计问题，不写出清洗后的 CSV。
- 清洗模式（fix）：应用规则修复与去重，可选调用本机 vLLM (/v1/chat/completions) 进行轻量改写。
- 去重范围：page/doc 两种；枚举题支持成员并集合并。
- 报告输出：clean_report.json（问题计数与示例采样）。
- 可选输出：invalid_rows.csv（被剔除的条目）。

注意：
- 按用户要求，代码不使用 try/except；必要的前置条件通过 assert 保证。
- 分支语句仅用于流程与规则实现，而非“避免错误”的用途。
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import asyncio
from typing import Dict, List, Tuple, Optional
from tqdm.auto import tqdm


# --------------------------- 项目配置导入 ---------------------------
# 为了使用 /home/pushihao/RAG/script/config_rag.py 中的 vLLM 生成器配置，这里显式加入项目根路径
_THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_SCRIPT_DIR))
if _PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_DIR)
from script import config_rag as rag_config  # 使用集中配置中的生成器参数

# --------------------------- 常量与正则 ---------------------------
MEANINGLESS_ANSWER_PATTERNS = [
    r"^见上文$", r"^如下$", r"^略$", r"^不确定$", r"^无法确定$", r"^参考原文$",
    r"^无$", r"^N\/?A$", r"^Unknown$", r"^参见本章复习笔记相关内容$"
]

PRONOUN_AMBIGUOUS_WORDS = [
    "它", "其", "这", "那", "这些", "那些", "该", "此", "上述"
]

ENUM_QUESTION_HINTS = [
    "包括", "分为", "由", "组成", "哪些", "分别是哪些", "有哪些"
]

QUESTION_REQUIRED_FIELDS = ["id", "doc_name", "page_id", "chunk_id", "question", "answer", "type"]

# 允许的别名映射（用于读取外部数据时的字段规范化）
ALIASES = {
    "id": ["id"],
    "doc_name": ["doc_name"],
    "page_id": ["page_id", "page_number"],
    "chunk_id": ["chunk_id"],
    "question": ["question"],
    "answer": ["answer"],
    "type": ["type", "decision_type"],
    # 可选字段：若存在则透传
    "source_context": ["source_context"],
}

def _aliases_ok(fieldnames: List[str]) -> bool:
    """检查输入 CSV 的列是否至少包含每个必需字段的一个别名。"""
    return all(any(alias in fieldnames for alias in ALIASES[k]) for k in QUESTION_REQUIRED_FIELDS)

def normalize_row_with_alias(raw_row: Dict[str, str], fieldnames: List[str]) -> Dict[str, str]:
    """将原始行映射为规范字段命名；若存在可选字段 source_context 则透传。"""
    out: Dict[str, str] = {}
    for k in QUESTION_REQUIRED_FIELDS:
        aliases = ALIASES[k]
        # 取第一个在行中出现的别名
        val = ""
        for ak in aliases:
            if ak in raw_row:
                val = raw_row.get(ak, "")
                break
        out[k] = val
    # 透传可选字段
    if any(a in fieldnames for a in ALIASES.get("source_context", [])):
        for ak in ALIASES["source_context"]:
            if ak in raw_row:
                out["source_context"] = raw_row.get(ak, "")
                break
    return out


# --------------------------- 文本规范化 ---------------------------
def _fullwidth_to_halfwidth(s: str) -> str:
    """将常见全角标点转换为半角或统一形式；保留中文问号。"""
    mapping = {
        ord("，"): ",",
        ord("；"): ";",
        ord("："): ":",
        ord("．"): ".",
        ord("！"): "!",
        ord("（"): "(",
        ord("）"): ")",
        ord("【"): "[",
        ord("】"): "]",
        ord("、"): "、",  # 枚举分隔维持中文顿号
        ord("？"): "？",  # 问号维持中文
    }
    return s.translate(mapping)


def normalize_question_for_key(q: str) -> str:
    """构造用于去重判断的规范化键：去空格、统一标点、归一部分疑问词。"""
    s = _fullwidth_to_halfwidth(q.strip())
    s = s.replace("?", "？")
    # 疑问词归一（仅用于键，不改变真实文本）
    s = re.sub(r"分别是哪些|有哪些|是什么|哪一类", "哪些", s)
    # 去多余尾部标点
    s = re.sub(r"[、：；]+$", "", s)
    # 统一内部空白
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_answer_text(a: str) -> str:
    """统一答案标点与分隔，去除尾部多余分隔符。"""
    s = _fullwidth_to_halfwidth(a.strip())
    # 将逗号/分号统一为中文顿号分隔
    s = s.replace(",", "、").replace(";", "、")
    # 去除尾部分隔
    s = re.sub(r"[、：；]+$", "", s)
    # 统一连续分隔符
    s = re.sub(r"(、)+", "、", s)
    return s


def ensure_question_mark(q: str) -> str:
    s = q.strip().replace("?", "？")
    return s if s.endswith("？") else (s + "？")


def detect_enum_question(q: str) -> bool:
    s = q.strip()
    return any(h in s for h in ENUM_QUESTION_HINTS)


def split_enums(a: str) -> List[str]:
    s = normalize_answer_text(a)
    parts = re.split(r"[、，;,]\s*", s)
    items = [p.strip() for p in parts if len(p.strip()) > 0]
    return items


def is_meaningless_answer(a: str) -> bool:
    txt = a.strip()
    return any(re.match(pat, txt, flags=re.IGNORECASE) for pat in MEANINGLESS_ANSWER_PATTERNS)


def has_ambiguous_pronoun(q: str) -> bool:
    s = q.strip()
    return any(w in s for w in PRONOUN_AMBIGUOUS_WORDS)


def is_incomplete_question(q: str, a: str) -> bool:
    """题面不完整：未以问号结尾，或存在“包括/例如/主要有/如：”未闭合结构且答案为空或占位。"""
    s = q.strip()
    no_qmark = not s.endswith("？") and not s.endswith("?")
    skeleton = ("包括：" in s) or ("例如：" in s) or ("主要有：" in s) or ("如：" in s)
    incomplete = skeleton and (len(a.strip()) == 0 or is_meaningless_answer(a))
    return no_qmark or incomplete


def fix_question_templates(q: str) -> str:
    """将模板化片段改写为完整疑问句。"""
    s = _fullwidth_to_halfwidth(q.strip()).replace("?", "？")
    s = re.sub(r"：\s*$", "：", s)
    s = re.sub(r"(.+?)包括：\s*$", r"\1包括哪些？", s)
    s = re.sub(r"(.+?)例如：\s*$", r"\1有哪些例子？", s)
    s = re.sub(r"(.+?)主要有：\s*$", r"\1主要有哪些？", s)
    s = re.sub(r"(.+?)如：\s*$", r"\1有哪些？", s)
    return ensure_question_mark(s)


def merge_enum_answers(a1: str, a2: str) -> str:
    items = split_enums(a1) + split_enums(a2)
    # 稳定去重：保持首次出现的顺序
    seen = set()
    ordered = []
    for it in items:
        if it not in seen:
            seen.add(it)
            ordered.append(it)
    return "、".join(ordered)


# --------------------------- vLLM 客户端 ---------------------------
async def _fetch_model_id(llm_url: str) -> str:
    url = llm_url.rstrip("/") + "/v1/models"
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as resp:
            text = await resp.text()
            data = json.loads(text)
            model_id = data["data"][0]["id"]
            return model_id


async def _call_vllm_chat(llm_url: str, endpoint: str, model: str, system_prompt: str, user_content: str,
                          max_tokens: int, temperature: float, top_p: float) -> str:
    url = llm_url.rstrip("/") + endpoint
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            body = await resp.text()
            data = json.loads(body)
            # 兼容 OpenAI 格式
            content = data["choices"][0]["message"]["content"]
            return content


def _extract_first_json_object(s: str) -> str:
    """从文本中提取首个完整的 JSON 对象子串（基于括号计数）。"""
    start = s.find("{")
    assert start != -1, f"LLM 返回内容不含 JSON 起始符: {s[:200]}"
    depth = 0
    end = start
    for i, ch in enumerate(s[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    assert depth == 0, "JSON 对象括号未闭合"
    return s[start:end + 1]


def _sanitize_json_object(s: str) -> str:
    """对 JSON 字符串进行常见问题的修复：
    - 统一为双引号；
    - 为未加引号的键补充引号；
    - 删除 } 或 ] 前的尾随逗号；
    - 规范 actions 字段为字符串数组。
    """
    t = s
    # 统一中文引号为英文双引号
    t = t.replace("“", '"').replace("”", '"')
    # 将用单引号包裹的字符串统一成双引号（值域）
    t = re.sub(r"(?<=[:\s])'([^']*)'", r'"\1"', t)
    # 为未加引号的键补加双引号
    t = re.sub(r'([\{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', t)
    # 删除 } 或 ] 前的尾随逗号
    t = re.sub(r',\s*([}\]])', r'\1', t)
    # 若 actions 为字符串，转换为数组
    t = re.sub(r'"actions"\s*:\s*"([^"]*)"', r'"actions": ["\1"]', t)
    return t


def _extract_json_fields(s: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """从包含 JSON 的文本中抽取 question/answer/actions 字段，尽量容错。"""
    obj = _extract_first_json_object(s)
    obj = _sanitize_json_object(obj)
    mq = re.search(r'"question"\s*:\s*"([^"]*)"', obj)
    ma = re.search(r'"answer"\s*:\s*"([^"]*)"', obj)
    mact = re.search(r'"actions"\s*:\s*\[(.*?)\]', obj, re.S)
    arr_text = (mact and mact.group(1)) or ""
    actions = re.findall(r'"([^"]*?)"', arr_text)
    q = (mq and mq.group(1)) or None
    a = (ma and ma.group(1)) or None
    return q, a, actions


def llm_fix_record(
    llm_url: str,
    endpoint: str,
    model: str,
    doc_name: str,
    page_id: str,
    question: str,
    answer: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, str, List[str]]:
    """使用 vLLM 轻量修复。约定返回 JSON：{"question":..., "answer":..., "actions": [...]}"""
    sys_prompt = (
        "你是数据清洗助手。仅对输入中的 question 和 answer 进行轻量修复，要求："
        "1) 补全不完整的疑问句，确保以“？”结尾；"
        "2) 若指代不明且可依据上下文主题进行替换，则将这些/它们等替换为给定主题；"
        "3) 枚举型问题的答案使用“、”分隔，若明显缺漏可补全；"
        "4) 统一标点与轻微拼写；不要改变事实。"
        "仅输出一个 JSON 对象，包含字段：question, answer, actions（字符串数组）。"
    )
    theme = f"主题={doc_name}；页码={page_id}。"
    user_content = json.dumps({
        "theme": theme,
        "question": question,
        "answer": answer,
    }, ensure_ascii=False)
    content = asyncio.run(_call_vllm_chat(
        llm_url, endpoint, model, sys_prompt, user_content, max_tokens, temperature, top_p
    ))
    res_q, res_a, actions = _extract_json_fields(content)
    fixed_q = ensure_question_mark(res_q or question)
    fixed_a = normalize_answer_text(res_a or answer)
    return fixed_q, fixed_a, actions


async def llm_fix_record_async(
    llm_url: str,
    endpoint: str,
    model: str,
    doc_name: str,
    page_id: str,
    question: str,
    answer: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, str, List[str]]:
    """异步版本：使用 vLLM 轻量修复。约定返回 JSON：{"question":..., "answer":..., "actions": [...]}"""
    sys_prompt = (
        "你是数据清洗助手。仅对输入中的 question 和 answer 进行轻量修复，要求："
        "1) 补全不完整的疑问句，确保以“？”结尾；"
        "2) 若指代不明且可依据上下文主题进行替换，则将这些/它们等替换为给定主题；"
        "3) 枚举型问题的答案使用“、”分隔，若明显缺漏可补全；"
        "4) 统一标点与轻微拼写；不要改变事实。"
        "仅输出一个 JSON 对象，包含字段：question, answer, actions（字符串数组）。"
    )
    theme = f"主题={doc_name}；页码={page_id}。"
    user_content = json.dumps({
        "theme": theme,
        "question": question,
        "answer": answer,
    }, ensure_ascii=False)
    content = await _call_vllm_chat(
        llm_url, endpoint, model, sys_prompt, user_content, max_tokens, temperature, top_p
    )
    res_q, res_a, actions = _extract_json_fields(content)
    fixed_q = ensure_question_mark(res_q or question)
    fixed_a = normalize_answer_text(res_a or answer)
    return fixed_q, fixed_a, actions


async def llm_fix_many(
    rows: List[Dict[str, str]],
    llm_url: str,
    endpoint: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    concurrency: int,
) -> List[Dict[str, str]]:
    """并发修复：对 rows 中的 question/answer 进行异步轻量修复，遵循并发信号量限制。"""
    sem = asyncio.Semaphore(concurrency)

    async def _worker(idx: int, rec: Dict[str, str]) -> Tuple[int, Dict[str, str]]:
        async with sem:
            fixed_q, fixed_a, _acts = await llm_fix_record_async(
                llm_url, endpoint, model,
                rec.get("doc_name", ""), rec.get("page_id", ""),
                rec.get("question", ""), rec.get("answer", ""),
                max_tokens, temperature, top_p
            )
            rec["question"] = fixed_q
            rec["answer"] = fixed_a
            return idx, rec

    tasks = [
        _worker(i, dict(r)) for i, r in enumerate(rows)
    ]
    results: List[Dict[str, str]] = [None] * len(rows)
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM修复", ncols=100):
        i, rec = await fut
        results[i] = rec
    return results


# --------------------------- 主流程 ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("clean_evaluation")
    # 依据脚本位置推导默认路径（满足工作指南的目录约定）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Generate_Linguistic_Question
    datasets_dir = os.path.join(base_dir, "datasets")
    default_input = os.path.join(datasets_dir, "evaluation.csv")
    default_output = os.path.join(datasets_dir, "evaluation_clean.csv")
    default_report = os.path.join(datasets_dir, "clean_report.json")
    default_invalid = os.path.join(datasets_dir, "invalid_rows.csv")

    p.add_argument("--input", default=default_input, help="输入 CSV 文件路径")
    p.add_argument("--output", default=default_output, help="清洗后输出 CSV 文件路径（fix 模式下使用）")
    p.add_argument("--report", default=default_report, help="报告 JSON 输出路径")
    p.add_argument("--invalid", default=default_invalid, help="可选：输出被剔除条目 CSV 的路径")
    p.add_argument("--scope", choices=["page", "doc"], default="doc", help="去重范围")
    p.add_argument("--mode", choices=["audit", "fix"], default="fix", help="运行模式")
    p.add_argument("--max-answer-chars", type=int, default=2048, help="答案长度上限")
    p.add_argument("--min-enum-members", type=int, default=2, help="枚举成员最小数量")
    p.add_argument("--allowed-types", type=str, default="single", help="允许的 type 集合，逗号分隔")
    p.add_argument("--use-llm", type=str, default="true", help="是否使用 vLLM 进行轻量修复 true/false")
    # vLLM 参数默认读取 script/config_rag.py 的生成器配置
    default_llm_base = f"http://{rag_config.VLLM_GENERATOR_HOST}:{rag_config.VLLM_GENERATOR_PORT}"
    p.add_argument("--llm-url", type=str, default=default_llm_base, help="vLLM 服务基础 URL")
    p.add_argument("--llm-endpoint", type=str, default="/v1/chat/completions", help="vLLM 生成端点路径")
    p.add_argument("--llm-model", type=str, default=rag_config.GENERATOR_MODEL_NAME_FOR_API, help="模型名或 auto 使用 /v1/models")
    p.add_argument("--llm-max-tokens", type=int, default=rag_config.GENERATION_CONFIG.get("max_tokens", 20480), help="LLM 修复最大生成长度")
    p.add_argument("--llm-temperature", type=float, default=rag_config.GENERATION_CONFIG.get("temperature", 0.2), help="LLM 温度")
    p.add_argument("--llm-top-p", type=float, default=rag_config.GENERATION_CONFIG.get("top_p", 0.95), help="LLM top_p")
    p.add_argument("--llm-concurrency", type=int, default=100, help="LLM 并发信号量上限（默认100）")
    p.add_argument("--keep-ambiguous", action="store_true", help="是否保留指代不明条目（默认剔除）")
    p.add_argument("--crop-overlong", action="store_true", help="是否裁剪超长答案（默认仅标记不裁剪）")
    return p.parse_args()


def _parse_bool(s: str) -> bool:
    return str(s).lower() in {"true", "1", "yes", "y"}


def read_csv(path: str) -> List[Dict[str, str]]:
    assert os.path.exists(path), f"输入文件不存在: {path}"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        assert _aliases_ok(fieldnames), (
            f"CSV 字段缺失或命名不一致，需要至少包含每个必需字段的一个别名。必需字段: {QUESTION_REQUIRED_FIELDS}；当前列: {fieldnames}"
        )
        rows = [normalize_row_with_alias(row, fieldnames) for row in reader]
        return rows


def write_csv(path: str, rows: List[Dict[str, str]], field_order: List[str]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if len(out_dir) > 0:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in field_order})


def sample_examples(examples: List[Dict[str, str]], limit: int = 5) -> List[Dict[str, str]]:
    # 简单采样：前 N 条
    return examples[:limit]


def clean_and_report(args: argparse.Namespace) -> None:
    rows = read_csv(args.input)
    use_llm = _parse_bool(args.use_llm)
    model = args.llm_model
    allowed_types = {t.strip() for t in args.allowed_types.split(",") if len(t.strip()) > 0}

    # 若用户显式指定 auto，则从 vLLM 服务获取模型 ID
    if model == "auto":
        model = asyncio.run(_fetch_model_id(args.llm_url))

    total = len(rows)
    removed_rows: List[Dict[str, str]] = []
    cleaned_rows: List[Dict[str, str]] = []

    counts = {
        "total": total,
        "structural_missing": 0,
        "meaningless_answer": 0,
        "incomplete_question": 0,
        "ambiguous_pronoun": 0,
        "enum_incomplete": 0,
        "answer_overlong": 0,
        "duplicates": 0,
        "type_illegal": 0,
    }

    examples = {
        "structural_missing": [],
        "meaningless_answer": [],
        "incomplete_question": [],
        "ambiguous_pronoun": [],
        "enum_incomplete": [],
        "answer_overlong": [],
        "duplicates": [],
        "type_illegal": [],
    }

    # 第一步：规则处理（除去重与并发 LLM 修复），收集候选记录
    candidates: List[Dict[str, str]] = []
    for row in tqdm(rows, desc="规则处理", ncols=100):
        id_ = row.get("id", "").strip()
        doc_name = row.get("doc_name", "").strip()
        page_id = row.get("page_id", "").strip()
        chunk_id = row.get("chunk_id", "").strip()
        q = row.get("question", "").strip()
        a = row.get("answer", "").strip()
        typ = row.get("type", "").strip()
        source_ctx = row.get("source_context", "").strip()

        missing_field = (len(id_) == 0 or len(doc_name) == 0 or len(page_id) == 0 or
                         len(chunk_id) == 0 or len(q) == 0 or len(a) == 0 or len(typ) == 0)
        if missing_field:
            counts["structural_missing"] += 1
            examples["structural_missing"].append({"id": id_, "question": q, "answer": a})
            if args.mode == "fix":
                removed = dict(row)
                removed["reason"] = "structural_missing"
                removed_rows.append(removed)
            continue

        if len(allowed_types) > 0 and (typ not in allowed_types):
            counts["type_illegal"] += 1
            examples["type_illegal"].append({"id": id_, "question": q, "answer": a, "type": typ})
            if args.mode == "fix":
                removed = dict(row)
                removed["reason"] = "type_illegal"
                removed_rows.append(removed)
                continue

        if not q.endswith("？") and not q.endswith("?"):
            counts["incomplete_question"] += 1
            examples["incomplete_question"].append({"id": id_, "question": q, "answer": a})
            if args.mode == "fix":
                q = ensure_question_mark(q)

        if is_meaningless_answer(a):
            counts["meaningless_answer"] += 1
            examples["meaningless_answer"].append({"id": id_, "question": q, "answer": a})
            if args.mode == "fix":
                removed = dict(row)
                removed["reason"] = "meaningless_answer"
                removed_rows.append(removed)
                continue

        if is_incomplete_question(q, a):
            counts["incomplete_question"] += 1
            examples["incomplete_question"].append({"id": id_, "question": q, "answer": a})
            if args.mode == "fix":
                q = fix_question_templates(q)

        if has_ambiguous_pronoun(q):
            counts["ambiguous_pronoun"] += 1
            examples["ambiguous_pronoun"].append({"id": id_, "question": q, "answer": a})
            if args.mode == "fix" and not args.keep_ambiguous:
                removed = dict(row)
                removed["reason"] = "ambiguous_pronoun"
                removed_rows.append(removed)
                continue

        if detect_enum_question(q):
            members = split_enums(a)
            if len(members) < args.min_enum_members:
                counts["enum_incomplete"] += 1
                examples["enum_incomplete"].append({"id": id_, "question": q, "answer": a})
                if args.mode == "fix":
                    a = normalize_answer_text(a)

        if len(a) > args.max_answer_chars:
            counts["answer_overlong"] += 1
            examples["answer_overlong"].append({"id": id_, "question": q, "answer": a[:80] + "..."})
            if args.mode == "fix" and args.crop_overlong:
                a = a[:args.max_answer_chars]

        candidates.append({
            "id": id_,
            "doc_name": doc_name,
            "page_id": page_id,
            "chunk_id": chunk_id,
            "question": q,
            "answer": a,
            "type": typ,
            **({"source_context": source_ctx} if len(source_ctx) > 0 else {})
        })

    # 第二步：并发 LLM 轻量修复（仅在 fix 模式且使用 LLM 时）
    if args.mode == "fix" and use_llm:
        candidates = asyncio.run(llm_fix_many(
            candidates,
            args.llm_url, args.llm_endpoint, model,
            args.llm_max_tokens, args.llm_temperature, args.llm_top_p,
            args.llm_concurrency
        ))

    # 第三步：去重与写入 cleaned_rows
    key_to_index: Dict[str, int] = {}
    for rec in candidates:
        id_ = rec["id"]
        doc_name = rec["doc_name"]
        page_id = rec["page_id"]
        chunk_id = rec["chunk_id"]
        q = rec["question"]
        a = rec["answer"]
        typ = rec["type"]

        scope_key = f"{doc_name}::{page_id}" if args.scope == "page" else doc_name
        dedup_key = scope_key + "::" + normalize_question_for_key(q)

        if dedup_key in key_to_index:
            counts["duplicates"] += 1
            examples["duplicates"].append({"id": id_, "question": q, "answer": a})
            idx = key_to_index[dedup_key]
            prev = cleaned_rows[idx]
            if detect_enum_question(q):
                merged = merge_enum_answers(prev["answer"], a)
                prev["answer"] = normalize_answer_text(merged)
            else:
                prev_len = len(prev["answer"].strip())
                cur_len = len(a.strip())
                if cur_len > prev_len:
                    prev["question"] = q
                    prev["answer"] = a
            continue

        cleaned = {
            "id": id_,
            "doc_name": doc_name,
            "page_id": page_id,
            "chunk_id": chunk_id,
            "question": ensure_question_mark(q),
            "answer": normalize_answer_text(a),
            "type": typ,
        }
        if "source_context" in rec:
            cleaned["source_context"] = rec["source_context"]
        key_to_index[dedup_key] = len(cleaned_rows)
        cleaned_rows.append(cleaned)

    report = {
        "scope": args.scope,
        "mode": args.mode,
        "counts": counts,
        "examples": {k: sample_examples(v) for k, v in examples.items()},
        "total_cleaned": len(cleaned_rows),
        "total_removed": len(removed_rows),
    }
    out_dir = os.path.dirname(os.path.abspath(args.report))
    if len(out_dir) > 0:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as rf:
        rf.write(json.dumps(report, ensure_ascii=False, indent=2))

    if args.mode == "fix" and len(args.output) > 0:
        field_order_clean = QUESTION_REQUIRED_FIELDS + (["source_context"] if any("source_context" in r for r in cleaned_rows) else [])
        write_csv(args.output, cleaned_rows, field_order_clean)
    if args.mode == "fix" and len(args.invalid) > 0 and len(removed_rows) > 0:
        field_order = QUESTION_REQUIRED_FIELDS + ["reason"] + (["source_context"] if any("source_context" in r for r in removed_rows) else [])
        write_csv(args.invalid, removed_rows, field_order)


def main():
    args = parse_args()
    clean_and_report(args)


if __name__ == "__main__":
    # 延迟导入 aiohttp（与不支持的环境兼容）；requirements 已包含 aiohttp
    import aiohttp  # noqa: F401
    main()