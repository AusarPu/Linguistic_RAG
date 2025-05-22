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

from FlagEmbedding import BGEM3FlagModel
from config import (  # 从你的 config.py 导入所有需要的常量
    EMBEDDING_MODEL_PATH,
    PROCESSED_DATA_DIR,  # 这个可能在内部路径拼接时用到，或者直接用下面的完整路径

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
        try:
            logger.info(f"Loading embedding model from: {EMBEDDING_MODEL_PATH}")
            self.embedding_model = BGEM3FlagModel(EMBEDDING_MODEL_PATH, use_fp16=False)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load embedding model: {e}") from e

        # --- 数据容器 ---
        self.faiss_chunks_index: Optional[faiss.Index] = None
        self.indexed_chunks_metadata: List[Dict[str, Any]] = []
        self.chunk_id_to_metadata_map: Dict[str, Dict[str, Any]] = {}

        self.phrase_to_sparse_weights_map: Dict[str, Dict[int, float]] = {}

        self.faiss_questions_index: Optional[faiss.Index] = None
        self.question_idx_to_chunk_id_map: List[str] = []
        self.question_idx_to_text_map: List[str] = []  # 存储问题文本，顺序与问题Faiss索引一致

        if not self._load_all_search_indexes():
            logger.critical("KnowledgeBase initialization failed: Could not load all necessary search indexes.")
            logger.critical(f"Please ensure all pre-processing scripts "
                            f"(e.g., 'preprocess/build_all_search_indexes.py') "
                            f"have run successfully and all index files exist as per config.py.")
            raise RuntimeError("KnowledgeBase failed to load essential search indexes.")

        logger.info(f"KnowledgeBase initialized successfully with {len(self.indexed_chunks_metadata)} indexed chunks "
                    f"and {self.faiss_questions_index.ntotal if self.faiss_questions_index else 0} indexed questions.")

    def _load_all_search_indexes(self) -> bool:
        logger.info(f"Attempting to load all search indexes using paths from config...")

        # 使用从 config.py 导入的路径常量
        required_files_paths = [
            Path(FAISS_INDEX_CHUNKS_SAVE_PATH),
            Path(INDEXED_CHUNKS_METADATA_SAVE_PATH),
            Path(PHRASE_SPARSE_WEIGHTS_MAP_SAVE_PATH),
            Path(FAISS_INDEX_QUESTIONS_SAVE_PATH),
            Path(QUESTION_INDEX_TO_CHUNK_ID_MAP_SAVE_PATH),
            Path(ALL_QUESTION_TEXTS_SAVE_PATH) # 这个是可选的
        ]

        if not all(p.is_file() for p in required_files_paths):
            missing = [p.name for p in required_files_paths if not p.is_file()]
            logger.error(f"One or more required index files not found. Missing: {missing}.")
            return False

        try:
            # 1. 加载块文本稠密检索资源
            logger.info(f"Loading Faiss index for chunk texts from '{FAISS_INDEX_CHUNKS_SAVE_PATH}'...")
            self.faiss_chunks_index = faiss.read_index(FAISS_INDEX_CHUNKS_SAVE_PATH)  # faiss需要str路径

            logger.info(f"Loading indexed chunks metadata from '{INDEXED_CHUNKS_METADATA_SAVE_PATH}'...")
            with open(INDEXED_CHUNKS_METADATA_SAVE_PATH, 'r', encoding='utf-8') as f:
                self.indexed_chunks_metadata = json.load(f)

            if self.faiss_chunks_index.ntotal != len(self.indexed_chunks_metadata):
                logger.error(f"Mismatch: Faiss chunk index ntotal ({self.faiss_chunks_index.ntotal}) "
                             f"vs. indexed_chunks_metadata length ({len(self.indexed_chunks_metadata)}).")
                return False
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
                logger.error(f"Mismatch: Faiss question index ntotal ({self.faiss_questions_index.ntotal}) "
                             f"vs. question_idx_to_chunk_id_map length ({len(self.question_idx_to_chunk_id_map)}).")
                return False
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
            return True

        except Exception as e:
            logger.error(f"Error during loading of search indexes: {e}", exc_info=True)
            self._clear_all_loaded_data()
            return False

    def _clear_all_loaded_data(self):
        """辅助函数，用于在加载失败时清理所有数据成员。"""
        self.faiss_chunks_index = None
        self.indexed_chunks_metadata = []
        self.chunk_id_to_metadata_map = {}
        self.phrase_to_sparse_weights_map = {}
        self.faiss_questions_index = None
        self.question_idx_to_chunk_id_map = []
        self.question_idx_to_text_map = []  # 确保这里也清空
        logger.info("Cleared all loaded KnowledgeBase data members due to an error or incomplete load.")


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
        try:
            query_output = self.embedding_model.encode(query_text, return_dense=True, return_sparse=False,
                                                       return_colbert_vecs=False)
            query_vector = query_output.get("dense_vecs")
            if query_vector is None or query_vector.size == 0:
                logger.error("未能为查询生成稠密向量。")
                return []

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
        except Exception as e:
            logger.error(f"稠密块检索时出错: {e}", exc_info=True)
            return []

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
        try:
            query_output = self.embedding_model.encode(query_text, return_dense=True, return_sparse=False,
                                                       return_colbert_vecs=False)
            query_vector = query_output.get("dense_vecs")
            if query_vector is None or query_vector.size == 0:
                logger.error("未能为查询生成稠密向量。")
                return []

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
        except Exception as e:
            logger.error(f"预生成问题稠密检索时出错: {e}", exc_info=True)
            return []

    def search_sparse_keywords(self, query_text: str, top_k: int = SPARSE_KEYWORD_RETRIEVAL_TOP_K,
                               threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        使用 BGE-M3 的稀疏表示，基于查询与块的预存 keyword_summaries 进行匹配。
        采用“分别编码唯一短语，在线查找并聚合最大分”的策略。
        返回结果包含 chunk_id, retrieval_score (关键词稀疏得分), 以及块的完整元数据。
        """
        if not self.embedding_model or not self.phrase_to_sparse_weights_map or not self.indexed_chunks_metadata:
            logger.warning("关键词稀疏检索：必要资源未加载。")
            return []

        start_time = time.time()
        try:
            query_output = self.embedding_model.encode(query_text, return_dense=False, return_sparse=True,
                                                       return_colbert_vecs=False)
            query_sparse_weights = query_output.get("lexical_weights")
            if not query_sparse_weights:  # BGE-M3 encode 单个句子时，lexical_weights 不是列表，直接是字典
                logger.error("未能为查询生成稀疏权重用于关键词搜索。")
                return []
        except Exception as e:
            logger.error(f"为查询 '{query_text[:30]}...' 编码稀疏权重时出错: {e}", exc_info=True)
            return []

        chunk_scores_with_meta = []
        # 遍历 self.indexed_chunks_metadata 来获取每个块的 keyword_summaries
        for chunk_meta_original in self.indexed_chunks_metadata:
            chunk_keyword_summaries = chunk_meta_original.get('keyword_summaries', [])

            # 如果一个块没有关键词摘要，它的稀疏得分自然是0，可以跳过或赋予0分
            if not chunk_keyword_summaries:
                if threshold is None:  # 如果不过滤，也给它一个0分的机会，但通常不会被选中
                    # result_item = dict(chunk_meta_original)
                    # result_item['retrieval_score'] = 0.0
                    # result_item['retrieval_type'] = 'sparse_keywords'
                    # chunk_scores_with_meta.append(result_item)
                    pass  # 或者直接跳过，不参与排序
                continue

            max_score_for_chunk = 0.0
            for phrase in chunk_keyword_summaries:
                phrase_sparse_weights = self.phrase_to_sparse_weights_map.get(phrase)
                if phrase_sparse_weights:
                    try:
                        score = self.embedding_model.compute_lexical_matching_score(
                            query_sparse_weights,
                            phrase_sparse_weights
                        )
                        # BGE-M3的compute_lexical_matching_score返回的已经是某种形式的相似度，
                        # 在你的旧代码中你将其clip到[0,1]，这里也这样做以保持一致性
                        score = float(max(0.0, min(1.0, score)))
                        if score > max_score_for_chunk:
                            max_score_for_chunk = score
                    except Exception:
                        pass  # 单个短语计算出错，忽略它

            if (threshold is None and max_score_for_chunk > 0) or \
                    (threshold is not None and max_score_for_chunk >= threshold):
                result_item = dict(chunk_meta_original)
                result_item['retrieval_score'] = max_score_for_chunk
                result_item['retrieval_type'] = 'sparse_keywords'
                chunk_scores_with_meta.append(result_item)

        # 按关键词稀疏得分排序
        chunk_scores_with_meta.sort(key=lambda x: x.get('retrieval_score', 0.0), reverse=True)
        final_results = chunk_scores_with_meta[:top_k]

        logger.info(
            f"关键词稀疏检索 for '{query_text[:30]}...' (耗时: {time.time() - start_time:.3f}s) - 返回 {len(final_results)}/{top_k} 个结果 (阈值 {threshold if threshold is not None else 'N/A'})")
        return final_results
