#!/usr/bin/env python3
"""
Evaluate_Linguistic 索引构建包装脚本

目的：
- 复用 preprocess/build_core_indexes.py，在 Evaluate_Linguistic 目录下完成索引构建；
- 输入固定为 Evaluate_Linguistic/enhanced/enhanced_knowledge_base_chunks_llm.json；
- 输出统一归档到 Evaluate_Linguistic/indexes/；
- 支持两种模式：test（小样本）/ full（全量）。

注意：
- 不修改底层构建逻辑与日志输出策略，仅做路径与文件名统一；
- 测试模式使用正式产物文件名，便于下游脚本验证加载；
"""

from pathlib import Path
import sys
import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate_Linguistic 索引构建包装器")
    parser.add_argument("--mode", choices=["test", "full"], default="test", help="运行模式：test 或 full")
    parser.add_argument("--test-limit", type=int, default=100, help="测试模式下限制处理的块数量（默认 100）")
    args = parser.parse_args()

    # 项目根目录：.../RAG
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))

    # 导入底层构建函数与配置
    from preprocess.build_core_indexes import build_all_search_indexes
    from script.config_rag import EMBEDDING_MODEL_PATH

    # 路径设定（固定，不随模式变化）
    eval_dir = project_root / "Evaluate_Linguistic"
    input_file = eval_dir / "enhanced" / "enhanced_knowledge_base_chunks_llm.json"
    output_dir = eval_dir / "indexes"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 模式分支
    if args.mode == "test":
        # 小样本：截取前 N 个块，但使用正式文件名进行输出
        with open(input_file, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        limited_chunks = all_chunks[: args.test_limit]

        temp_input = output_dir / "_temp_test_chunks.json"
        with open(temp_input, "w", encoding="utf-8") as f:
            json.dump(limited_chunks, f, ensure_ascii=False, indent=2)

        print(f"[build_indexes] 测试模式：输入 {temp_input}，输出目录 {output_dir}，块数 {len(limited_chunks)}")
        build_all_search_indexes(
            enhanced_chunks_path=str(temp_input),
            embedding_model_name_or_path=EMBEDDING_MODEL_PATH,
            output_dir=str(output_dir),
            # 正式文件名（与指南一致）
            chunk_dense_emb_filename="dense_embeddings_chunks.npy",
            chunk_faiss_idx_filename="faiss_index_chunks_ip.idx",
            indexed_chunks_meta_filename="indexed_chunks_metadata.json",
            chunk_bm25_index_filename="chunk_bm25_index.pkl",
            phrase_dense_map_filename="phrase_dense_embeddings_map.pkl",
            phrase_bm25_index_filename="phrase_bm25_index.pkl",
            question_dense_emb_filename="dense_embeddings_questions.npy",
            question_faiss_idx_filename="faiss_index_questions_ip.idx",
            question_to_chunk_id_map_filename="question_index_to_chunk_id_map.json",
            question_texts_list_filename="all_question_texts.json",
        )
    else:
        print(f"[build_indexes] 全量模式：输入 {input_file}，输出目录 {output_dir}")
        build_all_search_indexes(
            enhanced_chunks_path=str(input_file),
            embedding_model_name_or_path=EMBEDDING_MODEL_PATH,
            output_dir=str(output_dir),
            chunk_dense_emb_filename="dense_embeddings_chunks.npy",
            chunk_faiss_idx_filename="faiss_index_chunks_ip.idx",
            indexed_chunks_meta_filename="indexed_chunks_metadata.json",
            chunk_bm25_index_filename="chunk_bm25_index.pkl",
            phrase_dense_map_filename="phrase_dense_embeddings_map.pkl",
            phrase_bm25_index_filename="phrase_bm25_index.pkl",
            question_dense_emb_filename="dense_embeddings_questions.npy",
            question_faiss_idx_filename="faiss_index_questions_ip.idx",
            question_to_chunk_id_map_filename="question_index_to_chunk_id_map.json",
            question_texts_list_filename="all_question_texts.json",
        )

    print(f"[build_indexes] 索引构建完成。产物位于 {output_dir}")


if __name__ == "__main__":
    main()