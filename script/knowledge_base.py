# script/knowledge_base.py (更新后的加载部分)

import faiss
import numpy as np
import json
import pickle
from pathlib import Path
import os
import time  # 仍然可以用于记录加载时间
import logging
from typing import List, Dict, Optional, Any

from .vllm_clients import EmbeddingAPIClient
from .config_rag import (
    FAISS_INDEX_CHUNKS_SAVE_PATH,
    INDEXED_CHUNKS_METADATA_SAVE_PATH,
    PHRASE_SPARSE_WEIGHTS_MAP_SAVE_PATH,
    FAISS_INDEX_QUESTIONS_SAVE_PATH,
    QUESTION_INDEX_TO_CHUNK_ID_MAP_SAVE_PATH,
    ALL_QUESTION_TEXTS_SAVE_PATH,  # 用于加载问题文本的路径

    # 检索参数，虽然主要在检索方法中使用，但放在config里是好的
    DENSE_CHUNK_RETRIEVAL_TOP_K,
    DENSE_QUESTION_RETRIEVAL_TOP_K,
    SPARSE_KEYWORD_RETRIEVAL_TOP_K,
    DENSE_CHUNK_THRESHOLD,
    DENSE_QUESTION_THRESHOLD
)

logger = logging.getLogger(__name__)


# _normalize_embeddings 辅助函数可以保留在模块级别或类内部作为静态/辅助方法
def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 归一化向量 (辅助函数)"""
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    normalized_embeddings = np.zeros_like(embeddings, dtype=np.float32)
    for i in range(embeddings.shape[0]):
        norm = np.linalg.norm(embeddings[i])
        if norm > 1e-9:
            normalized_embeddings[i] = embeddings[i] / norm
    return normalized_embeddings


class KnowledgeBase:
    def __init__(self):
        logger.info("Initializing KnowledgeBase to load pre-built indexes...")

        # --- 模型加载 ---
        logger.info(f"Initializing embedding API client")
        self.embedding_model = EmbeddingAPIClient()
        logger.info("Embedding API client initialized successfully.")

        # --- 数据容器 ---
        self.faiss_chunks_index: Optional[faiss.Index] = None
        self.indexed_chunks_metadata: List[Dict[str, Any]] = []
        self.chunk_id_to_metadata_map: Dict[str, Dict[str, Any]] = {}

        self.phrase_to_sparse_weights_map: Dict[str, Dict[int, float]] = {}

        self.faiss_questions_index: Optional[faiss.Index] = None
        self.question_idx_to_chunk_id_map: List[str] = []
        self.question_idx_to_text_map: List[str] = []  # 存储问题文本，顺序与问题Faiss索引一致

        self._load_all_search_indexes()

        logger.info(f"KnowledgeBase initialized successfully with {len(self.indexed_chunks_metadata)} indexed chunks "
                    f"and {self.faiss_questions_index.ntotal if self.faiss_questions_index else 0} indexed questions.")

    def _load_all_search_indexes(self):
        logger.info(f"Loading all search indexes using paths from config...")

        # 使用从 config.py 导入的路径常量
        required_files_paths = [
            Path(FAISS_INDEX_CHUNKS_SAVE_PATH),
            Path(INDEXED_CHUNKS_METADATA_SAVE_PATH),
            Path(PHRASE_SPARSE_WEIGHTS_MAP_SAVE_PATH),
            Path(FAISS_INDEX_QUESTIONS_SAVE_PATH),
            Path(QUESTION_INDEX_TO_CHUNK_ID_MAP_SAVE_PATH),
            Path(ALL_QUESTION_TEXTS_SAVE_PATH) # 这个是可选的
        ]

        missing = [p.name for p in required_files_paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Required index files not found: {missing}")

        # 直接加载，不使用try-except
        # 1. 加载块文本稠密检索资源
        logger.info(f"Loading Faiss index for chunk texts from '{FAISS_INDEX_CHUNKS_SAVE_PATH}'...")
        self.faiss_chunks_index = faiss.read_index(FAISS_INDEX_CHUNKS_SAVE_PATH)  # faiss需要str路径

        logger.info(f"Loading indexed chunks metadata from '{INDEXED_CHUNKS_METADATA_SAVE_PATH}'...")
        with open(INDEXED_CHUNKS_METADATA_SAVE_PATH, 'r', encoding='utf-8') as f:
            self.indexed_chunks_metadata = json.load(f)

        if self.faiss_chunks_index.ntotal != len(self.indexed_chunks_metadata):
            raise ValueError(f"Mismatch: Faiss chunk index ntotal ({self.faiss_chunks_index.ntotal}) "
                         f"vs. indexed_chunks_metadata length ({len(self.indexed_chunks_metadata)}).")
        logger.info(f"Loaded {self.faiss_chunks_index.ntotal} chunk text vectors and corresponding metadata.")

        # 构建 chunk_id -> metadata 映射
        self.chunk_id_to_metadata_map = {
            meta.get("chunk_id"): meta
            for meta in self.indexed_chunks_metadata
            if meta.get("chunk_id")  # 确保chunk_id存在
        }
        logger.info(f"Built chunk_id_to_metadata_map with {len(self.chunk_id_to_metadata_map)} entries.")

        # 2. 加载关键词短语稀疏权重映射
        logger.info(f"Loading phrase sparse weights map from '{PHRASE_SPARSE_WEIGHTS_MAP_SAVE_PATH}'...")
        with open(PHRASE_SPARSE_WEIGHTS_MAP_SAVE_PATH, "rb") as f:
            self.phrase_to_sparse_weights_map = pickle.load(f)
        logger.info(f"Loaded sparse weights for {len(self.phrase_to_sparse_weights_map)} unique phrases.")

        # 3. 加载预生成问题稠密检索资源
        logger.info(f"Loading Faiss index for questions from '{FAISS_INDEX_QUESTIONS_SAVE_PATH}'...")
        self.faiss_questions_index = faiss.read_index(FAISS_INDEX_QUESTIONS_SAVE_PATH)

        logger.info(f"Loading question index to chunk_id map from '{QUESTION_INDEX_TO_CHUNK_ID_MAP_SAVE_PATH}'...")
        with open(QUESTION_INDEX_TO_CHUNK_ID_MAP_SAVE_PATH, 'r', encoding='utf-8') as f:
            self.question_idx_to_chunk_id_map = json.load(f)

        if self.faiss_questions_index.ntotal != len(self.question_idx_to_chunk_id_map):
            raise ValueError(f"Mismatch: Faiss question index ntotal ({self.faiss_questions_index.ntotal}) "
                         f"vs. question_idx_to_chunk_id_map length ({len(self.question_idx_to_chunk_id_map)}).")
        logger.info(f"Loaded {self.faiss_questions_index.ntotal} question vectors and their chunk_id map.")

        # 4. 加载问题文本列表 (如果存在)
        all_q_texts_file_path = Path(ALL_QUESTION_TEXTS_SAVE_PATH)  # 使用Path对象进行检查
        if all_q_texts_file_path.is_file():
            logger.info(f"Loading all question texts from '{all_q_texts_file_path}'...")
            with open(all_q_texts_file_path, 'r', encoding='utf-8') as f:
                self.question_idx_to_text_map = json.load(f)  # 假设它是一个列表
            if len(self.question_idx_to_text_map) != self.faiss_questions_index.ntotal:
                logger.warning(f"Mismatch: Loaded question texts count ({len(self.question_idx_to_text_map)}) "
                               f"vs. question Faiss index ntotal ({self.faiss_questions_index.ntotal}). Map may not align.")
                # 你可以选择清空它或接受不匹配
                # self.question_idx_to_text_map = []
        else:
            logger.info(f"Optional file '{ALL_QUESTION_TEXTS_SAVE_PATH}' not found, "
                        "retrieved question texts will not be directly available for display from this map.")

        logger.info("All search indexes and necessary data loaded successfully.")

    # --- 检索方法 ---
    def search_dense_chunks(self, query_text: str, top_k: int = DENSE_CHUNK_RETRIEVAL_TOP_K,
                            threshold: float = DENSE_CHUNK_THRESHOLD) -> List[Dict[str, Any]]:
        """
        使用Faiss基于块文本的稠密向量进行检索。
        返回结果包含 chunk_id, retrieval_score (余弦相似度), 以及块的完整元数据。
        """
        if not self.embedding_model or not self.faiss_chunks_index or not self.indexed_chunks_metadata:
            logger.warning("稠密块检索：必要资源未加载。")
            return []

        start_time = time.time()
        
        query_output = self.embedding_model.encode(instruct="Given a question, retrieve relevant text passages that contain information to answer the question.", texts=query_text)
        query_vector = query_output.get("dense_vecs")
        if query_vector is None or query_vector.size == 0:
            raise RuntimeError("未能为查询生成稠密向量。")

        query_vector_normalized = _normalize_embeddings(query_vector.astype(np.float32))

        # Faiss IndexFlatIP 返回的是内积 (如果向量归一化了，就是余弦相似度)
        # 我们取比 top_k 稍多一些的候选（例如 top_k * 3 或固定数量如50），然后用阈值过滤并精确排序
        # 这是因为Faiss的 search 可能不会完美按内积排序（取决于索引类型和参数），或者我们需要严格的阈值过滤
        num_candidates_to_fetch = min(top_k * 5, self.faiss_chunks_index.ntotal)  # 可调整倍数
        if num_candidates_to_fetch == 0: return []

        scores, faiss_indices = self.faiss_chunks_index.search(query_vector_normalized, num_candidates_to_fetch)

        results = []
        for i in range(len(faiss_indices[0])):
            idx = faiss_indices[0][i]
            score = float(scores[0][i])  # score 越大越好 (余弦相似度)

            if idx != -1 and score >= threshold and 0 <= idx < len(self.indexed_chunks_metadata):
                # 获取原始块的元数据副本
                chunk_meta_copy = dict(self.indexed_chunks_metadata[idx])
                chunk_meta_copy['retrieval_score'] = score
                chunk_meta_copy['retrieval_type'] = 'dense_chunk_text'
                results.append(chunk_meta_copy)

        # 按实际的 retrieval_score 再次排序 (因为 Faiss 返回的顺序和阈值过滤可能影响)
        # 并且只取最终的 top_k
        results.sort(key=lambda x: x['retrieval_score'], reverse=True)
        final_results = results[:top_k]

        logger.info(
            f"稠密块检索 for '{query_text[:30]}...' (耗时: {time.time() - start_time:.3f}s) - 返回 {len(final_results)}/{top_k} 个结果 (阈值 {threshold})")
        return final_results

    def search_dense_questions(self, query_text: str, top_k: int = DENSE_QUESTION_RETRIEVAL_TOP_K,
                               threshold: float = DENSE_QUESTION_THRESHOLD) -> List[Dict[str, Any]]:
        """
        使用Faiss基于预生成问题的稠密向量进行检索。
        返回结果包含触发召回的 chunk_id, retrieval_score (与问题的余弦相似度), 匹配上的问题文本, 以及块的完整元数据。
        """
        if not self.embedding_model or not self.faiss_questions_index or \
                not self.question_idx_to_chunk_id_map or not self.chunk_id_to_metadata_map:  # 需要chunk_id_to_metadata_map
            logger.warning("预生成问题稠密检索：必要资源未加载。")
            return []

        start_time = time.time()
        
        query_output = self.embedding_model.encode(instruct="Given a question, retrieve similar or related questions that address the same topic or domain.", texts=query_text)
        query_vector = query_output.get("dense_vecs")
        if query_vector is None or query_vector.size == 0:
            raise RuntimeError("未能为查询生成稠密向量。")

        query_vector_normalized = _normalize_embeddings(query_vector.astype(np.float32))

        num_candidates_to_fetch = min(top_k * 5, self.faiss_questions_index.ntotal)
        if num_candidates_to_fetch == 0: return []

        scores, q_indices = self.faiss_questions_index.search(query_vector_normalized, num_candidates_to_fetch)

        # 使用字典来确保每个 chunk_id 只被添加一次（基于最高分的问题匹配）
        chunk_id_to_best_q_match_info: Dict[str, Dict[str, Any]] = {}

        for i in range(len(q_indices[0])):
            q_idx = q_indices[0][i]  # 这是匹配上的问题在扁平化列表中的索引
            score = float(scores[0][i])

            if q_idx != -1 and score >= threshold and 0 <= q_idx < len(self.question_idx_to_chunk_id_map):
                original_chunk_id = self.question_idx_to_chunk_id_map[q_idx]
                matched_q_text = self.question_idx_to_text_map[q_idx] if 0 <= q_idx < len(
                    self.question_idx_to_text_map) else "未知问题文本"

                # 如果这个chunk_id已经因为另一个问题被添加了，只在分数更高时更新
                if original_chunk_id not in chunk_id_to_best_q_match_info or \
                        score > chunk_id_to_best_q_match_info[original_chunk_id]['retrieval_score']:

                    chunk_meta_original = self.chunk_id_to_metadata_map.get(original_chunk_id)
                    if chunk_meta_original:
                        chunk_meta_copy = dict(chunk_meta_original)  # 获取原始块的完整元数据
                        chunk_meta_copy['retrieval_score'] = score
                        chunk_meta_copy['retrieval_type'] = 'dense_generated_question'
                        chunk_meta_copy['matched_question'] = matched_q_text  # 记录匹配上的问题
                        chunk_id_to_best_q_match_info[original_chunk_id] = chunk_meta_copy

        results = list(chunk_id_to_best_q_match_info.values())
        results.sort(key=lambda x: x['retrieval_score'], reverse=True)
        final_results = results[:top_k]

        logger.info(
            f"预生成问题稠密检索 for '{query_text[:30]}...' (耗时: {time.time() - start_time:.3f}s) - 返回 {len(final_results)}/{top_k} 个结果 (阈值 {threshold})")
        return final_results

    def search_dense_keywords(self, query_text: str, top_k: int = SPARSE_KEYWORD_RETRIEVAL_TOP_K,
                     threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        使用稠密向量进行检索。
        现在使用稠密向量替代原来的稀疏关键词匹配。
        
        Args:
            query_text: 查询文本
            top_k: 返回的最大结果数量
            threshold: 相似度阈值，如果为None则使用默认的块检索阈值
            
        Returns:
            检索结果列表
        """

        start_time = time.time()
        
        query_output = self.embedding_model.encode(instruct="Given a question, retrieve relevant keywords and key concepts that are related to the question topic.", texts=query_text)
        query_vector = query_output.get("dense_vecs")
        if query_vector is None or query_vector.size == 0:
            raise RuntimeError("未能为查询生成稠密向量。")

        query_vector_normalized = _normalize_embeddings(query_vector.astype(np.float32))

        num_candidates_to_fetch = min(top_k * 5, self.faiss_questions_index.ntotal)
        if num_candidates_to_fetch == 0: return []

        scores, q_indices = self.faiss_questions_index.search(query_vector_normalized, num_candidates_to_fetch)

        # 使用字典来确保每个 chunk_id 只被添加一次（基于最高分的问题匹配）
        chunk_id_to_best_q_match_info: Dict[str, Dict[str, Any]] = {}

        for i in range(len(q_indices[0])):
            q_idx = q_indices[0][i]  # 这是匹配上的问题在扁平化列表中的索引
            score = float(scores[0][i])

            if q_idx != -1 and score >= threshold and 0 <= q_idx < len(self.question_idx_to_chunk_id_map):
                original_chunk_id = self.question_idx_to_chunk_id_map[q_idx]
                matched_q_text = self.question_idx_to_text_map[q_idx] if 0 <= q_idx < len(
                    self.question_idx_to_text_map) else "未知问题文本"

                # 如果这个chunk_id已经因为另一个问题被添加了，只在分数更高时更新
                if original_chunk_id not in chunk_id_to_best_q_match_info or \
                        score > chunk_id_to_best_q_match_info[original_chunk_id]['retrieval_score']:

                    chunk_meta_original = self.chunk_id_to_metadata_map.get(original_chunk_id)
                    if chunk_meta_original:
                        chunk_meta_copy = dict(chunk_meta_original)  # 获取原始块的完整元数据
                        chunk_meta_copy['retrieval_score'] = score
                        chunk_meta_copy['retrieval_type'] = 'dense_generated_question'
                        chunk_meta_copy['matched_question'] = matched_q_text  # 记录匹配上的问题
                        chunk_id_to_best_q_match_info[original_chunk_id] = chunk_meta_copy

        results = list(chunk_id_to_best_q_match_info.values())
        results.sort(key=lambda x: x['retrieval_score'], reverse=True)
        final_results = results[:top_k]

        logger.info(
            f"关键词稠密检索 for '{query_text[:30]}...' (耗时: {time.time() - start_time:.3f}s) - 返回 {len(final_results)}/{top_k} 个结果 (阈值 {threshold})")
        return final_results
