#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
执行脚本：在 Evaluate_Linguistic 目录下按指南实现分块与元数据增强的流水线。

主要步骤：
1) 目录准备：chunks/enhanced/indexes/evaluation
2) 测试模式：仅对 Evaluate_Linguistic/datasets/*.txt 进行分块，避免处理整个 knowledge_base；
3) 全量模式：处理整个 knowledge_base 目录；
4) 增强（pipeline）：调用 preprocess/llm_chunk_processor.py 提供的函数（测试限制 test_limit=100， 全量不限制）；
5) 归档：将优化与增强 JSON 复制到 Evaluate_Linguistic/enhanced/

注意：
- 不修改已有脚本，仅复用函数。
- 遵循用户约束，不使用 try/except；必要的逻辑判断用于步骤控制而非规避错误。
"""

import os
import sys
import shutil
import asyncio
import argparse

# 将项目根目录加入路径，便于导入现有模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from script import config_rag as config
from preprocess.preprocess_documents import process_knowledge_base, generate_document_chunks_langchain
from preprocess.llm_chunk_processor import refine_all_chunks_with_llm, enhance_chunks_with_llm_metadata


# 固定参数（依据指南）
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
MIN_CHUNK_LENGTH = 10
TEST_LIMIT = 100


def _eval_paths():
    """返回 Evaluate_Linguistic 相关目录路径字典。"""
    eval_root = os.path.join(PROJECT_ROOT, "Evaluate_Linguistic")
    return {
        "root": eval_root,
        "datasets": os.path.join(eval_root, "datasets"),
        "chunks": os.path.join(eval_root, "chunks"),
        "enhanced": os.path.join(eval_root, "enhanced"),
        "indexes": os.path.join(eval_root, "indexes"),
        "evaluation": os.path.join(eval_root, "evaluation"),
    }


def ensure_directories():
    """创建 Evaluate_Linguistic 下的管理目录，以及 config 定义的处理目录与知识库目录。"""
    paths = _eval_paths()
    os.makedirs(paths["chunks"], exist_ok=True)
    os.makedirs(paths["enhanced"], exist_ok=True)
    os.makedirs(paths["indexes"], exist_ok=True)
    os.makedirs(paths["evaluation"], exist_ok=True)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(config.KNOWLEDGE_BASE_DIR, exist_ok=True)


def copy_txts_to_knowledge_base():
    """将 Evaluate_Linguistic/datasets 下的 .txt 文件复制到项目知识库目录（config.KNOWLEDGE_BASE_DIR）。"""
    paths = _eval_paths()
    src_dir = paths["datasets"]
    filenames = os.listdir(src_dir)
    targets = [f for f in filenames if f.endswith(".txt")]
    for name in targets:
        src = os.path.join(src_dir, name)
        dst = os.path.join(config.KNOWLEDGE_BASE_DIR, name)
        shutil.copy2(src, dst)


def run_chunking_for_dataset_files():
    """仅对 Evaluate_Linguistic/datasets 下的 .txt 文件进行分块，写入 PROCESSED_DATA_DIR。"""
    paths = _eval_paths()
    src_dir = paths["datasets"]
    filenames = os.listdir(src_dir)
    targets = [f for f in filenames if f.endswith(".txt")]

    all_chunks = []
    separators = ["\n\n", "。", "！", "？", "，", "、", ". ", "! ", "? ", ", ", " ", ""]

    for name in targets:
        file_path = os.path.join(src_dir, name)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = generate_document_chunks_langchain(
            content,
            name,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
            MIN_CHUNK_LENGTH,
            separators=separators,
        )
        all_chunks.extend(chunks)

    output_json_path = os.path.join(config.PROCESSED_DATA_DIR, "processed_knowledge_base_chunks.json")
    with open(output_json_path, "w", encoding="utf-8") as outfile:
        import json
        json.dump(all_chunks, outfile, ensure_ascii=False, indent=2)


def run_chunking_to_processed_dir():
    """调用分块函数，输出到 PROCESSED_DATA_DIR（兼容 pipeline 默认输入）。"""
    output_json_path = os.path.join(config.PROCESSED_DATA_DIR, "processed_knowledge_base_chunks.json")
    separators = ["\n\n", "。", "！", "？", "，", "、", ". ", "! ", "? ", ", ", " ", ""]
    process_knowledge_base(
        config.KNOWLEDGE_BASE_DIR,
        output_json_path,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        MIN_CHUNK_LENGTH,
        langchain_separators=separators,
    )


def run_llm_pipeline(test_limit: int | None):
    """调用优化与元数据增强的两步函数；测试模式限制 test_limit=100，全量模式不限制。"""
    input_chunks_path = os.path.join(config.PROCESSED_DATA_DIR, "processed_knowledge_base_chunks.json")
    optimized_chunks_path = os.path.join(config.PROCESSED_DATA_DIR, "optimized_knowledge_base_chunks.json")
    enhanced_chunks_path = os.path.join(config.PROCESSED_DATA_DIR, "enhanced_knowledge_base_chunks_llm.json")

    asyncio.run(refine_all_chunks_with_llm(
        input_chunks_json_path=input_chunks_path,
        output_refined_chunks_json_path=optimized_chunks_path,
        limit=test_limit,
    ))

    asyncio.run(enhance_chunks_with_llm_metadata(
        input_chunks_json_path=optimized_chunks_path,
        output_chunks_json_path=enhanced_chunks_path,
        test_limit=test_limit,
        use_dynamic_batching=False,
    ))


def archive_outputs_to_eval_enhanced():
    """将 PROCESSED_DATA_DIR 下的优化与增强产物复制到 Evaluate_Linguistic/enhanced。"""
    paths = _eval_paths()
    optimized_src = os.path.join(config.PROCESSED_DATA_DIR, "optimized_knowledge_base_chunks.json")
    enhanced_src = os.path.join(config.PROCESSED_DATA_DIR, "enhanced_knowledge_base_chunks_llm.json")
    optimized_dst = os.path.join(paths["enhanced"], os.path.basename(optimized_src))
    enhanced_dst = os.path.join(paths["enhanced"], os.path.basename(enhanced_src))
    shutil.copy2(optimized_src, optimized_dst)
    shutil.copy2(enhanced_src, enhanced_dst)


def main():
    parser = argparse.ArgumentParser(description="Evaluate_Linguistic 流水线执行：测试或全量")
    parser.add_argument("--mode", choices=["test", "full"], default="test", help="运行模式：test=仅处理datasets下的TXT并限制100；full=处理 Evaluate_Linguistic/datasets 下的TXT（不限制）")
    args = parser.parse_args()

    print("=== 流水线开始 ===")
    ensure_directories()
    print("目录准备完成。")

    if args.mode == "test":
        print("测试模式：仅分块 Evaluate_Linguistic/datasets 下的TXT")
        run_chunking_for_dataset_files()
        print("分块完成，已写入 PROCESSED_DATA_DIR。")

        print("执行优化与元数据增强 (test_limit=100)...")
        run_llm_pipeline(test_limit=TEST_LIMIT)
    else:
        print("全量模式：直接处理 Evaluate_Linguistic/datasets 下的TXT（不限制）")
        run_chunking_for_dataset_files()
        print("分块完成，已写入 PROCESSED_DATA_DIR。")

        print("执行优化与元数据增强（全量，不限制）...")
        run_llm_pipeline(test_limit=None)
    print("优化与增强完成，产物已生成到 PROCESSED_DATA_DIR。")

    print("归档产物到 Evaluate_Linguistic/enhanced ...")
    archive_outputs_to_eval_enhanced()
    print("归档完成。")

    print("=== 流水线结束 ===")


if __name__ == "__main__":
    main()