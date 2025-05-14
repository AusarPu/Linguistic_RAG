# knowledge_base.py (修改提案)

import faiss
import numpy as np
import json
import pickle
from pathlib import Path
import os
import time
import logging
from typing import List, Dict, Optional, Any, Set  # 增加了 Set

from FlagEmbedding import BGEM3FlagModel
# 假设你的配置文件路径能被正确解析
from .config import (
    EMBEDDING_MODEL_PATH,
    PROCESSED_DATA_DIR,
    # 从 build_all_search_indexes.py 输出的文件名常量 (你需要确保这些在config.py中定义)
    FAISS_INDEX_CHUNKS_SAVE_PATH,  # 例如 "faiss_index_chunks_ip.idx"
    INDEXED_CHUNKS_METADATA_SAVE_PATH,  # 例如 "indexed_chunks_metadata.json"
    PHRASE_SPARSE_WEIGHTS_MAP_SAVE_PATH,  # 例如 "phrase_sparse_weights_map.pkl"
    FAISS_INDEX_QUESTIONS_SAVE_PATH,  # 例如 "faiss_index_questions_ip.idx"
    QUESTION_INDEX_TO_CHUNK_ID_MAP_SAVE_PATH,  # 例如 "question_index_to_chunk_id_map.json"
    # (可选) 如果要方便地获取问题文本，可能还需要原始的增强JSON路径
    ENHANCED_CHUNKS_JSON_PATH,  # 例如 "enhanced_chunks_with_keywords.json"
    # 检索参数
    DENSE_CHUNK_RETRIEVAL_TOP_K,
    DENSE_QUESTION_RETRIEVAL_TOP_K,
    SPARSE_KEYWORD_RETRIEVAL_TOP_K,
    DENSE_CHUNK_THRESHOLD,  # 你设定的 0.6
    DENSE_QUESTION_THRESHOLD  # 你设定的 0.8
)

logger = logging.getLogger(__name__)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 归一化向量 (辅助函数)"""
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return embeddings / norms


class KnowledgeBase:
    def __init__(self,
                 knowledge_source_dir_for_staleness: Optional[str] = None,
                 knowledge_file_pattern_for_staleness: Optional[str] = None):
        logger.info("Initializing KnowledgeBase with multi-faceted retrieval capabilities...")

        self.processed_data_path = Path(PROCESSED_DATA_DIR)

        # 用于数据新鲜度检查的原始知识库目录和模式
        self.knowledge_source_dir_staleness = Path(
            knowledge_source_dir_for_staleness) if knowledge_source_dir_for_staleness else None
        self.knowledge_file_pattern_staleness = knowledge_file_pattern_for_staleness

        # --- 模型加载 ---
        try:
            logger.info(f"Loading embedding model from: {EMBEDDING_MODEL_PATH}")
            self.embedding_model = BGEM3FlagModel(EMBEDDING_MODEL_PATH, use_fp16=True)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load embedding model: {e}") from e

        # --- 数据容器 ---
        # 1. 块文本稠密检索相关
        self.faiss_chunks_index: Optional[faiss.Index] = None
        self.indexed_chunks_metadata: List[Dict[str, Any]] = []  # 存储被索引块的元数据

        # 2. 关键词摘要稀疏检索相关
        self.phrase_to_sparse_weights_map: Dict[str, Dict[int, float]] = {}

        # 3. 预生成问题稠密检索相关
        self.faiss_questions_index: Optional[faiss.Index] = None
        self.question_idx_to_chunk_id_map: List[str] = []
        self.question_idx_to_text_map: Dict[int, str] = {}  # (可选) 存储问题文本

        if not self._load_all_search_indexes():
            logger.critical("Failed to load one or more search indexes or essential data.")
            logger.critical(
                f"Please ensure all pre-processing scripts (e.g., 'preprocess/build_all_search_indexes.py') have been run successfully and "
                f"all necessary files exist in '{self.processed_data_path}'.")
            # 在实际应用中，这里可能应该抛出更严重的错误或阻止服务启动
            raise RuntimeError("KnowledgeBase failed to load essential search indexes.")

        logger.info(f"KnowledgeBase initialized successfully with {len(self.indexed_chunks_metadata)} indexed chunks.")

    def _get_current_source_metadata_for_staleness(self) -> Dict[str, float]:
        # (这个函数与你之前的版本类似，用于获取原始 TXT 文件的修改时间)
        # ... (你的实现或我之前提供的简化版本) ...
        source_files_metadata = {}
        if self.knowledge_source_dir_staleness and self.knowledge_source_dir_staleness.is_dir() and self.knowledge_file_pattern_staleness:
            try:
                for file_path in sorted(
                        self.knowledge_source_dir_staleness.glob(self.knowledge_file_pattern_staleness)):
                    if file_path.is_file():
                        source_files_metadata[str(file_path.resolve())] = os.path.getmtime(file_path)
            except Exception as e:
                logger.error(f"Error scanning source directory for staleness check: {e}", exc_info=True)
        return source_files_metadata

    def _load_all_search_indexes(self) -> bool:
        logger.info(f"Attempting to load all search indexes from '{self.processed_data_path}'...")

        # 定义需要加载的文件路径
        # 注意：文件名应与 build_all_search_indexes.py 中保存的文件名一致
        # 你应该从 config.py 中导入这些文件名常量
        faiss_chunks_file = self.processed_data_path / FAISS_INDEX_CHUNKS_SAVE_PATH
        indexed_chunks_meta_file = self.processed_data_path / INDEXED_CHUNKS_METADATA_SAVE_PATH
        phrase_sparse_map_file = self.processed_data_path / PHRASE_SPARSE_WEIGHTS_MAP_SAVE_PATH
        faiss_questions_file = self.processed_data_path / FAISS_INDEX_QUESTIONS_SAVE_PATH
        q_idx_to_chunk_id_map_file = self.processed_data_path / QUESTION_INDEX_TO_CHUNK_ID_MAP_SAVE_PATH

        # (可选，用于获取问题文本)
        # enhanced_chunks_full_data_file = Path(ENHANCED_CHUNKS_WITH_KEYWORDS_PATH_CONFIG)

        required_files = [faiss_chunks_file, indexed_chunks_meta_file, phrase_sparse_map_file,
                          faiss_questions_file, q_idx_to_chunk_id_map_file]

        if not all(p.is_file() for p in required_files):
            missing = [p.name for p in required_files if not p.is_file()]
            logger.error(f"One or more required index files not found. Missing: {missing}. Cannot load KnowledgeBase.")
            return False

        try:
            # 0. (可选) 数据新鲜度检查，可以放在这里或 __init__ 更早的位置
            #    这里假设如果索引文件存在，我们就尝试加载，新鲜度检查可以提示用户重新构建，但不一定阻止加载。
            #    或者，如果检查到过时，这里直接 return False。
            #    为了简化，暂时省略这里的staleness check，假设外部流程已保证索引是用户想要的。

            # 1. 加载块文本稠密检索资源
            logger.info(f"Loading Faiss index for chunk texts from '{faiss_chunks_file}'...")
            self.faiss_chunks_index = faiss.read_index(str(faiss_chunks_file))
            logger.info(f"Loading indexed chunks metadata from '{indexed_chunks_meta_file}'...")
            with open(indexed_chunks_meta_file, 'r', encoding='utf-8') as f:
                self.indexed_chunks_metadata = json.load(f)
            if self.faiss_chunks_index.ntotal != len(self.indexed_chunks_metadata):
                logger.error("Mismatch: Faiss chunk index ntotal vs. indexed_chunks_metadata length.")
                return False
            logger.info(f"Loaded {self.faiss_chunks_index.ntotal} chunk text vectors and metadata.")

            # 2. 加载关键词短语稀疏权重映射
            logger.info(f"Loading phrase sparse weights map from '{phrase_sparse_map_file}'...")
            with open(phrase_sparse_map_file, "rb") as f:
                self.phrase_to_sparse_weights_map = pickle.load(f)
            logger.info(f"Loaded sparse weights for {len(self.phrase_to_sparse_weights_map)} unique phrases.")

            # 3. 加载预生成问题稠密检索资源
            logger.info(f"Loading Faiss index for questions from '{faiss_questions_file}'...")
            self.faiss_questions_index = faiss.read_index(str(faiss_questions_file))
            logger.info(f"Loading question index to chunk_id map from '{q_idx_to_chunk_id_map_file}'...")
            with open(q_idx_to_chunk_id_map_file, 'r', encoding='utf-8') as f:
                self.question_idx_to_chunk_id_map = json.load(f)
            if self.faiss_questions_index.ntotal != len(self.question_idx_to_chunk_id_map):
                logger.error("Mismatch: Faiss question index ntotal vs. question_idx_to_chunk_id_map length.")
                return False
            logger.info(f"Loaded {self.faiss_questions_index.ntotal} question vectors and their chunk_id map.")

            # 4. (可选但推荐) 构建 question_idx_to_text_map 以便调试和未来使用
            #    这需要读取 ENHANCED_CHUNKS_WITH_KEYWORDS_PATH_CONFIG 文件
            #    这里我们假设它在 build_all_search_indexes.py 中也保存了一个纯问题列表
            #    或者我们在这里重新构建它：
            logger.info(
                f"Building question_idx_to_text_map from '{ENHANCED_CHUNKS_JSON_PATH}' (if available)...")
            if Path(ENHANCED_CHUNKS_JSON_PATH).is_file():
                q_idx_counter = 0
                with open(ENHANCED_CHUNKS_JSON_PATH, 'r', encoding='utf-8') as f_enh:
                    all_enhanced_chunks = json.load(f_enh)
                for chunk_meta in all_enhanced_chunks:  # 遍历的是包含所有元数据的块
                    if chunk_meta.get('is_meaningful', True):  # 只考虑用于索引的块的问题
                        generated_questions = chunk_meta.get('generated_questions', [])
                        if isinstance(generated_questions, list):
                            for q_text in generated_questions:
                                if isinstance(q_text, str) and q_text.strip():
                                    self.question_idx_to_text_map[q_idx_counter] = q_text.strip()
                                    q_idx_counter += 1
                logger.info(f"Built map for {len(self.question_idx_to_text_map)} question texts.")
                if q_idx_counter != self.faiss_questions_index.ntotal:
                    logger.warning(f"Warning: Number of questions in map ({q_idx_counter}) "
                                   f"differs from question Faiss index ({self.faiss_questions_index.ntotal}). Map may be incomplete or mismatched if filtering changed.")

            logger.info("All search indexes and necessary data loaded successfully.")
            return True

        except Exception as e:
            logger.error(f"Error during loading of search indexes: {e}", exc_info=True)
            # 清理可能部分加载的数据
            self.faiss_chunks_index = None;
            self.indexed_chunks_metadata = []
            self.phrase_to_sparse_weights_map = {}
            self.faiss_questions_index = None;
            self.question_idx_to_chunk_id_map = []
            self.question_idx_to_text_map = {}
            return False

    # --- 检索方法 ---
    def search_dense_chunks(self, query_text: str, top_k: int = DENSE_CHUNK_RETRIEVAL_TOP_K,
                            threshold: float = DENSE_CHUNK_THRESHOLD) -> List[Dict[str, Any]]:
        if not self.embedding_model or not self.faiss_chunks_index or not self.indexed_chunks_metadata:
            logger.warning("稠密块检索：必要资源未加载。")
            return []

        start_time = time.time()
        try:
            query_output = self.embedding_model.encode(query_text, return_dense=True, return_sparse=False,
                                                       return_colbert_vecs=False)
            query_vector = query_output.get("dense_vecs")
            if query_vector is None: return []

            query_vector_normalized = _normalize_embeddings(query_vector.astype(np.float32))

            # Faiss IndexFlatIP 返回的是内积 (如果向量归一化了，就是余弦相似度)
            scores, indices = self.faiss_chunks_index.search(query_vector_normalized, top_k * 3)  # 取稍多一些，然后用阈值过滤

            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                score = float(scores[0][i])  # score 越大越好
                if idx != -1 and score >= threshold and 0 <= idx < len(self.indexed_chunks_metadata):
                    # 返回完整的块元数据，因为后续reranker和LLM生成都需要
                    chunk_meta_copy = dict(self.indexed_chunks_metadata[idx])
                    chunk_meta_copy['retrieval_score'] = score
                    chunk_meta_copy['retrieval_type'] = 'dense_chunk_text'
                    results.append(chunk_meta_copy)

            # 按分数再次排序并取 top_k (因为上面的 top_k*3 和 threshold 可能导致顺序变化或数量不足/过多)
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
        if not self.embedding_model or not self.faiss_questions_index or \
                not self.question_idx_to_chunk_id_map or not self.indexed_chunks_metadata:
            logger.warning("预生成问题稠密检索：必要资源未加载。")
            return []

        start_time = time.time()
        try:
            query_output = self.embedding_model.encode(query_text, return_dense=True, return_sparse=False,
                                                       return_colbert_vecs=False)
            query_vector = query_output.get("dense_vecs")
            if query_vector is None: return []

            query_vector_normalized = _normalize_embeddings(query_vector.astype(np.float32))

            scores, q_indices = self.faiss_questions_index.search(query_vector_normalized, top_k * 3)  # 取稍多一些

            results = []
            # 使用字典来确保每个 chunk_id 只被添加一次（如果多个问题指向同一个 chunk）
            # 并保留该 chunk_id 下匹配上的最高分的问题
            chunk_id_to_best_q_match: Dict[str, Dict[str, Any]] = {}

            for i in range(len(q_indices[0])):
                q_idx = q_indices[0][i]
                score = float(scores[0][i])
                if q_idx != -1 and score >= threshold and 0 <= q_idx < len(self.question_idx_to_chunk_id_map):
                    original_chunk_id = self.question_idx_to_chunk_id_map[q_idx]
                    matched_q_text = self.question_idx_to_text_map.get(q_idx, "未知问题")

                    # 如果这个chunk_id已经因为另一个问题被添加了，只在分数更高时更新
                    if original_chunk_id not in chunk_id_to_best_q_match or \
                            score > chunk_id_to_best_q_match[original_chunk_id]['retrieval_score']:

                        # 找到这个chunk_id对应的完整元数据
                        # 为了高效，应该预先构建一个 chunk_id -> chunk_meta 的映射
                        # 暂时先遍历查找（效率较低，如果indexed_chunks_metadata很大）
                        chunk_meta_original = next(
                            (meta for meta in self.indexed_chunks_metadata if meta["chunk_id"] == original_chunk_id),
                            None)
                        if chunk_meta_original:
                            chunk_meta_copy = dict(chunk_meta_original)
                            chunk_meta_copy['retrieval_score'] = score
                            chunk_meta_copy['retrieval_type'] = 'dense_generated_question'
                            chunk_meta_copy['matched_question'] = matched_q_text  # 记录匹配上的问题
                            chunk_id_to_best_q_match[original_chunk_id] = chunk_meta_copy

            # 将字典的值转为列表，并按分数排序
            results = list(chunk_id_to_best_q_match.values())
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
        if not self.embedding_model or not self.phrase_to_sparse_weights_map or not self.indexed_chunks_metadata:
            logger.warning("关键词稀疏检索：必要资源未加载。")
            return []

        start_time = time.time()
        try:
            query_output = self.embedding_model.encode(query_text, return_dense=False, return_sparse=True,
                                                       return_colbert_vecs=False)
            query_sparse_weights = query_output.get("lexical_weights")
            if not query_sparse_weights:
                logger.error("未能为查询生成稀疏权重用于关键词搜索。")
                return []
        except Exception as e:
            logger.error(f"为查询 '{query_text[:30]}...' 编码稀疏权重时出错: {e}", exc_info=True)
            return []

        chunk_scores = []
        for chunk_meta_original in self.indexed_chunks_metadata:  # 使用已加载的、包含所有元数据的列表
            chunk_keyword_summaries = chunk_meta_original.get('keyword_summaries', [])
            if not chunk_keyword_summaries:
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
                        score = float(max(0.0, min(1.0, score)))
                        if score > max_score_for_chunk:
                            max_score_for_chunk = score
                    except Exception:
                        pass

            if (threshold is None and max_score_for_chunk > 0) or \
                    (threshold is not None and max_score_for_chunk >= threshold):
                result_item = dict(chunk_meta_original)
                result_item['retrieval_score'] = max_score_for_chunk
                result_item['retrieval_type'] = 'sparse_keywords'
                chunk_scores.append(result_item)

        chunk_scores.sort(key=lambda x: x.get('retrieval_score', 0.0), reverse=True)
        final_results = chunk_scores[:top_k]

        logger.info(
            f"关键词稀疏检索 for '{query_text[:30]}...' (耗时: {time.time() - start_time:.3f}s) - 返回 {len(final_results)}/{top_k} 个结果 (阈值 {threshold if threshold is not None else 'N/A'})")
        return final_results

    # 你原有的 _get_current_source_metadata 和 _create_index_from_sources, _save_processed_data 方法
    # 将不再由 KnowledgeBase 直接调用来创建索引，因为索引创建现在是外部预处理步骤。
    # _get_current_source_metadata_for_staleness 可以保留用于在 __init__ 或 _load_all_search_indexes 中进行数据新鲜度检查。