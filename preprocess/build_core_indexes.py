# preprocess/build_core_indexes.py

import faiss
import numpy as np
import json
import pickle
from pathlib import Path
import os
import time
import logging
from typing import List, Dict, Optional, Set, Any
from script.config import *


from FlagEmbedding import BGEM3FlagModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 归一化向量"""
    if embeddings.ndim == 1: # 单个向量
        embeddings = embeddings.reshape(1, -1)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9  # 防止除以零
    return embeddings / norms

def build_all_search_indexes(
    enhanced_chunks_path: str,
    embedding_model_name_or_path: str,
    output_dir: str,
    # 块文本相关文件名
    chunk_dense_emb_filename: str,
    chunk_faiss_idx_filename: str,
    indexed_chunks_meta_filename: str,
    # 关键词短语相关文件名
    phrase_sparse_map_filename: str,
    # 预生成问题相关文件名
    question_dense_emb_filename: str,
    question_faiss_idx_filename: str,
    question_to_chunk_id_map_filename: str,
    question_texts_list_filename: str,
    batch_size_embed: int = 32,
    batch_size_sparse_phrases: int = 256,
    batch_size_questions: int = 128 # 为问题编码新增批大小
):
    """
    构建核心的搜索索引：
    1. 稠密向量索引 (Faiss) 基于块的 'text'。
    2. 唯一关键词短语的稀疏权重映射 (BGE-M3 lexical_weights)。
    并保存这些索引及相关的块元数据。
    """
    logger.info(f"开始从 '{enhanced_chunks_path}' 构建核心搜索索引...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 加载嵌入模型
    try:
        logger.info(f"正在加载嵌入模型: {embedding_model_name_or_path}")
        model = BGEM3FlagModel(embedding_model_name_or_path, use_fp16=True)
        logger.info("嵌入模型加载成功。")
    except Exception as e:
        logger.error(f"加载嵌入模型失败: {e}", exc_info=True)
        return

    # 2. 加载LLM增强后的块数据
    try:
        with open(enhanced_chunks_path, 'r', encoding='utf-8') as f:
            all_enhanced_chunks_input = json.load(f)
        logger.info(f"成功加载 {len(all_enhanced_chunks_input)} 个增强后的块数据。")
    except Exception as e:
        logger.error(f"加载增强块数据 '{enhanced_chunks_path}' 失败: {e}", exc_info=True)
        return

    if not all_enhanced_chunks_input:
        logger.warning("加载的增强块数据为空，无法构建索引。")
        return

    # --- 1. 准备数据 ---
    chunks_to_index_metadata_list: List[Dict[str, Any]] = []
    texts_for_chunk_dense_embedding: List[str] = []
    all_unique_keyword_phrases: Set[str] = set()

    all_question_texts_flat: List[str] = []
    question_to_chunk_id_map: List[str] = [] # 记录每个问题向量对应的原始chunk_id

    logger.info("筛选有效块并收集文本、唯一关键词短语和所有预生成问题...")
    for chunk_data in all_enhanced_chunks_input:
        if chunk_data.get('is_meaningful', True):
            text_content = chunk_data.get('text', '').strip()
            keyword_summaries = chunk_data.get('keyword_summaries', [])
            generated_questions = chunk_data.get('generated_questions', [])
            chunk_id = chunk_data.get("chunk_id", f"auto_id_{len(texts_for_chunk_dense_embedding)}") # 确保有ID

            if text_content:
                chunk_meta_for_index = {
                    "chunk_id": chunk_id,
                    "doc_name": chunk_data.get("doc_name", "unknown_doc"),
                    "page_number": chunk_data.get("page_number", -1),
                    "keyword_summaries": keyword_summaries,
                    "generated_questions": generated_questions,
                    "text": text_content
                }
                chunks_to_index_metadata_list.append(chunk_meta_for_index)
                texts_for_chunk_dense_embedding.append(text_content)

                if isinstance(keyword_summaries, list):
                    for phrase in keyword_summaries:
                        if isinstance(phrase, str) and phrase.strip():
                            all_unique_keyword_phrases.add(phrase.strip())

                if isinstance(generated_questions, list):
                    for question_text in generated_questions:
                        if isinstance(question_text, str) and question_text.strip():
                            all_question_texts_flat.append(question_text.strip())
                            question_to_chunk_id_map.append(chunk_id) # 每个问题都关联到它的源chunk_id
                else:
                    logger.debug(f"跳过块 '{chunk_data.get('chunk_id')}' 因为文本内容为空。")
            else:
                logger.info(f"跳过块 '{chunk_data.get('chunk_id')}' 因为被LLM标记为 is_meaningful:false。")

    if not texts_for_chunk_dense_embedding:
        logger.error("没有有效的文本块用于构建稠密索引。")
        return

    logger.info(f"准备了 {len(texts_for_chunk_dense_embedding)} 个块用于稠密编码 (对应 {len(chunks_to_index_metadata_list)} 条元数据)。")
    logger.info(f"收集到 {len(all_unique_keyword_phrases)} 个唯一关键词短语。")
    logger.info(f"收集到 {len(all_question_texts_flat)} 个预生成问题 (来自所有块)。")

    # --- 2. 为块文本编码稠密向量并构建Faiss索引 (使用IndexFlatIP) ---
    if texts_for_chunk_dense_embedding:
        logger.info(f"开始为 {len(texts_for_chunk_dense_embedding)} 个文本块生成稠密向量...")
        all_dense_vecs_list = []
        try:
            for i in range(0, len(texts_for_chunk_dense_embedding), batch_size_embed):
                batch_texts = texts_for_chunk_dense_embedding[i:i+batch_size_embed]
                output = model.encode(batch_texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
                all_dense_vecs_list.append(output['dense_vecs'])

            chunk_dense_embeddings_np = np.vstack(all_dense_vecs_list).astype(np.float32)
            logger.info("对块文本稠密向量进行L2归一化...")
            chunk_dense_embeddings_np_normalized = normalize_embeddings(chunk_dense_embeddings_np)

            logger.info(f"块文本稠密向量编码完成。Shape: {chunk_dense_embeddings_np_normalized.shape}")
            np.save(output_path / chunk_dense_emb_filename, chunk_dense_embeddings_np_normalized) # 保存归一化后的
            logger.info(f"块文本稠密向量已保存到: {output_path / chunk_dense_emb_filename}")

            dimension = chunk_dense_embeddings_np_normalized.shape[1]
            logger.info(f"正在创建块文本的 Faiss 索引 (IndexFlatIP, dim={dimension})...")
            chunk_faiss_index = faiss.IndexFlatIP(dimension) # 使用内积
            chunk_faiss_index.add(chunk_dense_embeddings_np_normalized)
            logger.info(f"块文本 Faiss 索引创建完成 ({chunk_faiss_index.ntotal} 个向量).")
            faiss.write_index(chunk_faiss_index, str(output_path / chunk_faiss_idx_filename))
            logger.info(f"块文本 Faiss 索引已保存到: {output_path / chunk_faiss_idx_filename}")

        except Exception as e:
            logger.error(f"块文本稠密向量处理或Faiss索引构建过程中出错: {e}", exc_info=True)
            # 如果这里失败，后续可能无法进行，或者至少稠密检索会失败
    else:
        logger.warning("没有文本块可用于稠密索引，跳过此步骤。")


    # --- 3. 为唯一关键词短语编码稀疏权重并保存映射 ---
    phrase_to_sparse_weights_map: Dict[str, Dict[int, float]] = {}
    if all_unique_keyword_phrases:
        logger.info(f"开始为 {len(all_unique_keyword_phrases)} 个唯一关键词短语生成稀疏权重...")
        unique_phrases_list_for_encoding = list(all_unique_keyword_phrases)
        all_phrase_sparse_weights_list = []
        try:
            for i in range(0, len(unique_phrases_list_for_encoding), batch_size_sparse_phrases):
                batch_phrases = unique_phrases_list_for_encoding[i:i+batch_size_sparse_phrases]
                outputs = model.encode(batch_phrases, return_dense=False, return_sparse=True, return_colbert_vecs=False)
                phrase_sparse_weights_batch = outputs.get("lexical_weights")
                if phrase_sparse_weights_batch and len(phrase_sparse_weights_batch) == len(batch_phrases):
                    all_phrase_sparse_weights_list.extend(phrase_sparse_weights_batch)
                else:
                    logger.error(f"关键词短语稀疏编码批次 {i//batch_size_sparse_phrases + 1} 输出不匹配或为空。")

            if len(all_phrase_sparse_weights_list) == len(unique_phrases_list_for_encoding):
                for i, phrase_str in enumerate(unique_phrases_list_for_encoding):
                    phrase_to_sparse_weights_map[phrase_str] = all_phrase_sparse_weights_list[i]
                logger.info("唯一关键词短语的稀疏权重映射生成完成。")
            else:
                logger.error("最终编码的短语稀疏权重数量与唯一短语数量不匹配。映射可能不完整。")

            with open(output_path / phrase_sparse_map_filename, "wb") as f:
                pickle.dump(phrase_to_sparse_weights_map, f)
            logger.info(f"关键词短语稀疏权重映射已保存到: {output_path / phrase_sparse_map_filename}")
        except Exception as e:
            logger.error(f"为关键词短语编码稀疏权重或保存映射时出错: {e}", exc_info=True)
    else:
        logger.warning("没有找到唯一关键词短语，未生成稀疏权重映射。")

    # --- 4. 为所有预生成问题编码稠密向量并构建Faiss索引 (使用IndexFlatIP) ---
    if all_question_texts_flat:
        logger.info(f"开始为 {len(all_question_texts_flat)} 个预生成问题生成稠密向量...")
        all_question_dense_vecs_list = []
        try:
            for i in range(0, len(all_question_texts_flat), batch_size_questions):
                batch_q_texts = all_question_texts_flat[i:i+batch_size_questions]
                logger.debug(f"  问题稠密编码批次 {i//batch_size_questions + 1} / { (len(all_question_texts_flat) + batch_size_questions - 1)//batch_size_questions }...")
                output = model.encode(batch_q_texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
                all_question_dense_vecs_list.append(output['dense_vecs'])

            question_dense_embeddings_np = np.vstack(all_question_dense_vecs_list).astype(np.float32)
            logger.info("对预生成问题稠密向量进行L2归一化...")
            question_dense_embeddings_np_normalized = normalize_embeddings(question_dense_embeddings_np)

            logger.info(f"预生成问题稠密向量编码完成。Shape: {question_dense_embeddings_np_normalized.shape}")
            np.save(output_path / question_dense_emb_filename, question_dense_embeddings_np_normalized) # 保存归一化后的
            logger.info(f"预生成问题稠密向量已保存到: {output_path / question_dense_emb_filename}")

            dimension_q = question_dense_embeddings_np_normalized.shape[1]
            logger.info(f"正在创建预生成问题的 Faiss 索引 (IndexFlatIP, dim={dimension_q})...")
            question_faiss_index = faiss.IndexFlatIP(dimension_q) # 使用内积
            question_faiss_index.add(question_dense_embeddings_np_normalized)
            logger.info(f"预生成问题 Faiss 索引创建完成 ({question_faiss_index.ntotal} 个向量).")
            faiss.write_index(question_faiss_index, str(output_path / question_faiss_idx_filename))
            logger.info(f"预生成问题 Faiss 索引已保存到: {output_path / question_faiss_idx_filename}")

            # 保存问题索引到 chunk_id 的映射
            with open(output_path / question_to_chunk_id_map_filename, "w", encoding="utf-8") as f_q_map:
                json.dump(question_to_chunk_id_map, f_q_map, ensure_ascii=False, indent=2)
            logger.info(f"预生成问题索引到 chunk_id 的映射已保存到: {output_path / question_to_chunk_id_map_filename}")

            # --- 新增：保存扁平化的问题文本列表 ---
            if all_question_texts_flat:  # 确保有数据可保存
                all_q_texts_save_path = output_path / question_texts_list_filename  # 从函数参数获取文件名
                try:
                    with open(all_q_texts_save_path, "w", encoding="utf-8") as f_all_q:
                        json.dump(all_question_texts_flat, f_all_q, ensure_ascii=False, indent=2)
                    logger.info(f"所有预生成问题文本列表已保存到: {all_q_texts_save_path}")
                except Exception as e:
                    logger.error(f"保存所有预生成问题文本列表时出错: {e}", exc_info=True)
            else:
                logger.warning("没有预生成问题文本可保存。")

        except Exception as e:
            logger.error(f"预生成问题稠密向量处理或Faiss索引构建过程中出错: {e}", exc_info=True)
    else:
        logger.warning("没有预生成问题可用于构建稠密索引，跳过此步骤。")

    # --- 5. 保存被索引的块的元数据列表 ---
    # (这个列表的顺序与主Faiss索引和主dense_embeddings一致)
    try:
        with open(output_path / indexed_chunks_meta_filename, "w", encoding="utf-8") as f:
            json.dump(chunks_to_index_metadata_list, f, ensure_ascii=False, indent=2)
        logger.info(f"被索引块的元数据列表已保存到: {output_path / indexed_chunks_meta_filename}")
    except Exception as e:
        logger.error(f"保存被索引块的元数据列表时出错: {e}", exc_info=True)

    logger.info("所有核心搜索索引构建流程结束。")

if __name__ == '__main__':
    logger.info("=========== 开始构建所有核心搜索索引 ===========")

    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)

    if not Path(ENHANCED_CHUNKS_JSON_PATH).is_file():
        logger.error(f"错误：输入文件 '{ENHANCED_CHUNKS_JSON_PATH}' 未找到。")
    else:
        build_all_search_indexes(
            enhanced_chunks_path=ENHANCED_CHUNKS_JSON_PATH,
            embedding_model_name_or_path=EMBEDDING_MODEL_PATH,
            output_dir=PROCESSED_DATA_DIR,
            chunk_dense_emb_filename="dense_embeddings_chunks.npy",
            chunk_faiss_idx_filename="faiss_index_chunks_ip.idx",
            indexed_chunks_meta_filename="indexed_chunks_metadata.json",
            phrase_sparse_map_filename="phrase_sparse_weights_map.pkl",
            question_dense_emb_filename="dense_embeddings_questions.npy",
            question_faiss_idx_filename="faiss_index_questions_ip.idx",
            question_to_chunk_id_map_filename="question_index_to_chunk_id_map.json",
            question_texts_list_filename=ALL_QUESTION_TEXTS_SAVE_PATH,
        )
    logger.info("=========== 所有核心搜索索引构建完成 ===========")