#!/usr/bin/env python3
"""
通用工具：将统一格式数据集({id, question, answer, context})构造成知识库块
- 读取 NDJSON 或 JSON 列表文件
- 规范化 context
- 使用 LangChain 的 RecursiveCharacterTextSplitter 切分
- 输出与 preprocess/preprocess_documents.py 相同结构: [{doc_name, page_number, chunk_id, text}]
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Iterable, Dict, Any, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 与现有 preprocess_documents.py 中默认分隔符保持一致（并稍作扩展）
DEFAULT_SEPARATORS = ["\n\n", "。", "！", "？", "，", "、", ". ", "! ", "? ", ", ", " ", ""]


def normalize_context_to_text(context: Any) -> str:
    """
    将 context 统一为字符串。
    支持 str / list / dict / 其它，可容错。
    """
    if context is None:
        return ""

    if isinstance(context, str):
        return context.strip()

    if isinstance(context, list):
        pieces: List[str] = []
        for item in context:
            try:
                if isinstance(item, str):
                    val = item.strip()
                elif isinstance(item, dict):
                    if 'text' in item and isinstance(item['text'], str):
                        val = item['text'].strip()
                    elif 'passage_text' in item and isinstance(item['passage_text'], str):
                        val = item['passage_text'].strip()
                    else:
                        val = json.dumps(item, ensure_ascii=False)
                else:
                    val = json.dumps(item, ensure_ascii=False)
                if val:
                    pieces.append(val)
            except Exception:
                continue
        return "\n\n".join(pieces)

    if isinstance(context, dict):
        for key in ("text", "context", "passage", "content"):
            if key in context and isinstance(context[key], str):
                return context[key].strip()
        try:
            return json.dumps(context, ensure_ascii=False)
        except Exception:
            return ""

    try:
        return str(context)
    except Exception:
        return ""


def _chunk_text_to_kb_chunks(
    text: str,
    doc_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
    min_chunk_len: int = 10,
    separators: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    将一段文本切分为 KB 块，结构：{doc_name, page_number, chunk_id, text}
    """
    if not text or not text.strip():
        return []

    if separators is None:
        separators = DEFAULT_SEPARATORS

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=separators,
    )

    chunks: List[Dict[str, Any]] = []
    parts = splitter.split_text(text)
    for i, part in enumerate(parts):
        part = (part or "").strip()
        if len(part) < min_chunk_len:
            continue
        chunks.append({
            "doc_name": doc_name,
            "page_number": 1,
            "chunk_id": f"{doc_name}_c{i+1}",
            "text": part,
        })
    return chunks


def _iter_samples_from_file(input_file: Path) -> Iterable[Dict[str, Any]]:
    """
    统一迭代器：支持两种输入格式
    - NDJSON：每行一个 JSON 对象
    - JSON 列表：整个文件是一个数组
    """
    # 先读取前若干字节判断是 NDJSON 还是 JSON 数组
    with input_file.open('r', encoding='utf-8') as f:
        # 读取少量字符以判断结构
        head = f.read(2048)
        # 回到文件起始
        f.seek(0)
        stripped = head.lstrip()
        if stripped.startswith('['):
            # JSON 数组
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            yield item
                return
            except Exception:
                # 如果整体解析失败，回退到逐行
                f.seek(0)
        # NDJSON 按行解析
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def process_ndjson_or_list_json(
    input_file: Path,
    dataset_name: str,
    split_name: str,
    output_file: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
    min_chunk_len: int = 10,
    separators: List[str] | None = None,
) -> int:
    """
    将 input_file 中的每条样本(context)切分为 KB 块，保存到 output_file (JSON 列表)。
    返回生成的块数量。
    """
    if not input_file.is_file():
        print(f"[跳过] 输入文件不存在: {input_file}")
        return 0

    print(f"读取: {input_file}")
    total_chunks = 0
    all_chunks: List[Dict[str, Any]] = []

    for idx, sample in enumerate(_iter_samples_from_file(input_file)):
        sample_id = str(sample.get("id", idx))
        context_text = normalize_context_to_text(sample.get("context", ""))
        if not context_text:
            continue
        doc_name = f"{dataset_name}_{split_name}_{sample_id}"
        chunks = _chunk_text_to_kb_chunks(
            context_text, doc_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_len=min_chunk_len,
            separators=separators,
        )
        if chunks:
            all_chunks.extend(chunks)
            total_chunks += len(chunks)

        if (idx + 1) % 1000 == 0:
            print(f"  进度: {idx+1} 条样本，累计 {total_chunks} 个块")

    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"完成: {dataset_name}/{split_name} -> {output_file}，共 {total_chunks} 个块")
    return total_chunks