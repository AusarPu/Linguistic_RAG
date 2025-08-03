# script/knowledge_base.py (更新后的加载部分)

import faiss
import numpy as np
import json
import pickle
from pathlib import Path
import jieba
import time
import logging
from typing import List, Dict, Optional,Union, Any
from rank_bm25 import BM25Okapi

from .vllm_clients import EmbeddingAPIClient
from .config_rag import (
    FAISS_INDEX_CHUNKS_SAVE_PATH,
    INDEXED_CHUNKS_METADATA_SAVE_PATH,
    PHRASE_SPARSE_WEIGHTS_MAP_SAVE_PATH,
    PHRASE_DENSE_EMBEDDINGS_MAP_SAVE_PATH,
    BM25_INDEX_SAVE_PATH,
    FAISS_INDEX_QUESTIONS_SAVE_PATH,
    QUESTION_INDEX_TO_CHUNK_ID_MAP_SAVE_PATH,
    ALL_QUESTION_TEXTS_SAVE_PATH,  # 用于加载问题文本的路径

    DENSE_CHUNK_RETRIEVAL_TOP_K,
    DENSE_QUESTION_RETRIEVAL_TOP_K,
    SPARSE_KEYWORD_RETRIEVAL_TOP_K,
    DENSE_CHUNK_THRESHOLD,
    DENSE_QUESTION_THRESHOLD,
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
        self.phrase_to_dense_embeddings_map: Dict[str, np.ndarray] = {}
        
        # BM25索引
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunk_bm25_index: Optional[BM25Okapi] = None  # 文本块的BM25索引

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
            Path(PHRASE_DENSE_EMBEDDINGS_MAP_SAVE_PATH),
            Path(BM25_INDEX_SAVE_PATH),
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

        # 2.5. 加载关键词短语稠密向量映射
        logger.info(f"Loading phrase dense embeddings map from '{PHRASE_DENSE_EMBEDDINGS_MAP_SAVE_PATH}'...")
        with open(PHRASE_DENSE_EMBEDDINGS_MAP_SAVE_PATH, "rb") as f:
            self.phrase_to_dense_embeddings_map = pickle.load(f)
        logger.info(f"Loaded dense embeddings for {len(self.phrase_to_dense_embeddings_map)} unique phrases.")

        # 2. 加载BM25索引
        logger.info(f"Loading BM25 index from '{BM25_INDEX_SAVE_PATH}'...")
        with open(BM25_INDEX_SAVE_PATH, "rb") as f:
            bm25_data = pickle.load(f)
            self.bm25_index = bm25_data["bm25_model"]
            self.bm25_phrase_list = bm25_data["phrase_list"]
        logger.info(f"Loaded BM25 index with {len(self.bm25_phrase_list)} phrases.")
        
        # 2.1. 加载文本块BM25索引
        chunk_bm25_path = "/home/pushihao/RAG/processed_knowledge_base/chunk_bm25_index.pkl"
        logger.info(f"Loading chunk BM25 index from '{chunk_bm25_path}'...")
        with open(chunk_bm25_path, "rb") as f:
            chunk_bm25_data = pickle.load(f)
            self.chunk_bm25_index = chunk_bm25_data["bm25_model"]
            self.chunk_bm25_texts = chunk_bm25_data.get("chunk_texts", [])
        logger.info(f"Loaded chunk BM25 index with {len(self.chunk_bm25_texts)} chunks.")

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
    def search_dense_chunks(self, query_text: Union[str, List[str]], top_k: int = DENSE_CHUNK_RETRIEVAL_TOP_K,
                            threshold: float = DENSE_CHUNK_THRESHOLD) -> List[Dict[str, Any]]:
        """
        使用Faiss基于块文本的稠密向量进行检索。
        支持单个查询字符串或多个查询字符串列表。
        返回结果包含 chunk_id, retrieval_score (余弦相似度), 以及块的完整元数据。
        """
        if not self.embedding_model or not self.faiss_chunks_index or not self.indexed_chunks_metadata:
            logger.warning("稠密块检索：必要资源未加载。")
            return []

        start_time = time.time()
        
        # 将单个字符串转换为列表以统一处理
        queries = [query_text] if isinstance(query_text, str) else query_text
        
        # 用于存储所有查询的结果
        all_results = {}  # chunk_id -> chunk_meta 的映射，用于去重
        
        for query in queries:
            query_output = self.embedding_model.encode(
                instruct="Given a question, retrieve relevant text passages that contain information to answer the question.",
                texts=query
            )
            query_vector = query_output.get("dense_vecs")
            if query_vector is None or query_vector.size == 0:
                logger.warning(f"未能为查询 '{query[:30]}...' 生成稠密向量，跳过此查询。")
                continue

            # 确保query_vector是2D数组，FAISS需要2D输入
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

            num_candidates_to_fetch = min(top_k * 5, self.faiss_chunks_index.ntotal)
            if num_candidates_to_fetch == 0: 
                continue

            scores, faiss_indices = self.faiss_chunks_index.search(query_vector, num_candidates_to_fetch)

            # 处理当前查询的结果
            for i in range(len(faiss_indices[0])):
                idx = faiss_indices[0][i]
                score = float(scores[0][i])

                if idx != -1 and score >= threshold and 0 <= idx < len(self.indexed_chunks_metadata):
                    chunk_meta = self.indexed_chunks_metadata[idx]
                    chunk_id = chunk_meta.get("chunk_id")
                    
                    # 如果这个chunk已经被其他查询检索到，只在得分更高时更新
                    if chunk_id not in all_results or score > all_results[chunk_id]['retrieval_score']:
                        chunk_meta_copy = dict(chunk_meta)
                        chunk_meta_copy['retrieval_score'] = score
                        chunk_meta_copy['retrieval_type'] = 'dense_chunk_text'
                        all_results[chunk_id] = chunk_meta_copy

        # 将所有结果转换为列表并排序
        final_results = list(all_results.values())
        final_results.sort(key=lambda x: x['retrieval_score'], reverse=True)
        final_results = final_results[:top_k]

        query_summary = f"{len(queries)} queries" if isinstance(query_text, list) else f"'{query_text[:30]}...'"
        logger.info(
            f"稠密块检索 for {query_summary} (耗时: {time.time() - start_time:.3f}s) - "
            f"返回 {len(final_results)}/{top_k} 个结果 (阈值 {threshold})")
        
        return final_results

    def search_dense_questions(self, query_text: Union[str, List[str]], top_k: int = DENSE_QUESTION_RETRIEVAL_TOP_K,
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
        
        # 将单个字符串转换为列表以统一处理
        queries = [query_text] if isinstance(query_text, str) else query_text
        
        # 用于存储所有查询的结果
        chunk_id_to_best_q_match_info: Dict[str, Dict[str, Any]] = {}
        
        for query in queries:
            query_output = self.embedding_model.encode(
                instruct="Given a question, retrieve similar or related questions that address the same topic or domain.",
                texts=query
            )
            query_vector = query_output.get("dense_vecs")
            if query_vector is None or query_vector.size == 0:
                logger.warning(f"未能为查询 '{query[:30]}...' 生成稠密向量，跳过此查询。")
                continue

            # 确保query_vector是2D数组，FAISS需要2D输入
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

            query_vector_normalized = _normalize_embeddings(query_vector.astype(np.float32))

            num_candidates_to_fetch = min(top_k * 5, self.faiss_questions_index.ntotal)
            if num_candidates_to_fetch == 0: continue

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

        query_summary = f"{len(queries)} queries" if isinstance(query_text, list) else f"'{query_text[:30]}...'"
        logger.info(
            f"预生成问题稠密检索 for {query_summary} (耗时: {time.time() - start_time:.3f}s) - "
            f"返回 {len(final_results)}/{top_k} 个结果 (阈值 {threshold})")
        return final_results

    def search_dense_keywords(self, query_text: Union[str, List[str]], top_k: int = SPARSE_KEYWORD_RETRIEVAL_TOP_K,
                     threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        使用RRF（Reciprocal Rank Fusion）融合BM25和语义搜索的方式进行关键词检索。
        结合文本块BM25检索和稠密向量的语义匹配。
        
        Args:
            query_text: 查询文本，可以是单个字符串或字符串列表
            top_k: 返回的最大结果数量
            threshold: 相似度阈值，如果为None则使用默认的块检索阈值
            
        Returns:
            检索结果列表
        """
        start_time = time.time()
        
        if threshold is None:
            threshold = DENSE_CHUNK_THRESHOLD
        
        if not self.chunk_bm25_index or not self.embedding_model:
            logger.warning("文本块BM25索引或嵌入模型未加载，无法进行关键词检索。")
            return []
        
        # 将单个字符串转换为列表以统一处理
        queries = [query_text] if isinstance(query_text, str) else query_text
        
        # 用于存储所有查询的RRF融合结果
        chunk_id_to_rrf_info: Dict[str, Dict[str, Any]] = {}
        
        for query in queries:        
            # 1. BM25检索 - 直接对文本块进行检索
            bm25_scores = self.chunk_bm25_index.get_scores(query)
            
            # 获取BM25排序结果（按分数降序）
            bm25_ranked_indices = np.argsort(bm25_scores)[::-1]
            bm25_chunk_rankings = {}  # chunk_idx -> rank (从1开始)
            for rank, chunk_idx in enumerate(bm25_ranked_indices[:top_k * 2]):
                if bm25_scores[chunk_idx] > 0:  # 只考虑有正分数的结果
                    bm25_chunk_rankings[chunk_idx] = rank + 1
            
            # 2. 语义检索 - 使用稠密向量检索
            embedding_chunk_rankings = {}
            query_output = self.embedding_model.encode(
                instruct="Given a question, retrieve relevant text passages that contain information to answer the question.",
                texts=query
            )
            query_vector = query_output.get("dense_vecs")
            if query_vector is not None and query_vector.size > 0:
                # 确保query_vector是2D数组
                if query_vector.ndim == 1:
                    query_vector = query_vector.reshape(1, -1)
                
                # 使用Faiss进行稠密检索
                num_candidates = min(top_k * 2, self.faiss_chunks_index.ntotal)
                scores, faiss_indices = self.faiss_chunks_index.search(query_vector, num_candidates)
                
                # 获取embedding排序结果
                for rank, (idx, score) in enumerate(zip(faiss_indices[0], scores[0])):
                    if idx != -1 and score >= threshold:
                        embedding_chunk_rankings[idx] = rank + 1
            else:
                logger.warning(f"未能为查询 '{query[:30]}...' 生成稠密向量，仅使用BM25结果。")
            
            # 3. RRF融合
            k = 60  # RRF平滑因子
            chunk_rrf_scores = {}
            
            # 收集所有出现在任一排序中的chunk
            all_chunk_indices = set(bm25_chunk_rankings.keys()) | set(embedding_chunk_rankings.keys())
            
            for chunk_idx in all_chunk_indices:
                rrf_score = 0.0
                
                # BM25贡献
                if chunk_idx in bm25_chunk_rankings:
                    rrf_score += 1.0 / (k + bm25_chunk_rankings[chunk_idx])
                
                # Embedding贡献
                if chunk_idx in embedding_chunk_rankings:
                    rrf_score += 1.0 / (k + embedding_chunk_rankings[chunk_idx])
                
                chunk_rrf_scores[chunk_idx] = rrf_score
            
            # 4. 将结果映射到chunk metadata
            for chunk_idx, rrf_score in chunk_rrf_scores.items():
                if 0 <= chunk_idx < len(self.indexed_chunks_metadata):
                    chunk_meta = self.indexed_chunks_metadata[chunk_idx]
                    chunk_id = chunk_meta.get('chunk_id')
                    
                    if chunk_id:
                        # 如果这个chunk已经被其他查询检索到，只在得分更高时更新
                        if chunk_id not in chunk_id_to_rrf_info or rrf_score > chunk_id_to_rrf_info[chunk_id]['retrieval_score']:
                            chunk_meta_copy = dict(chunk_meta)
                            chunk_meta_copy['retrieval_score'] = rrf_score
                            chunk_meta_copy['retrieval_type'] = 'rrf_bm25_embedding_fusion'
                            chunk_meta_copy['bm25_rank'] = bm25_chunk_rankings.get(chunk_idx, None)
                            chunk_meta_copy['embedding_rank'] = embedding_chunk_rankings.get(chunk_idx, None)
                            chunk_meta_copy['rrf_k'] = k
                            chunk_id_to_rrf_info[chunk_id] = chunk_meta_copy
        
        # 5. 对RRF分数进行最小-最大归一化
        results = list(chunk_id_to_rrf_info.values())
        if results:
            # 获取所有RRF分数
            rrf_scores = [result['retrieval_score'] for result in results]
            min_score = min(rrf_scores)
            max_score = max(rrf_scores)
            
            # 进行最小-最大归一化
            if max_score > min_score:  # 避免除零错误
                for result in results:
                    original_score = result['retrieval_score']
                    normalized_score = (original_score - min_score) / (max_score - min_score)
                    result['retrieval_score'] = normalized_score
                    result['original_rrf_score'] = original_score  # 保留原始分数用于调试
            else:
                # 如果所有分数相同，设置为1.0
                for result in results:
                    result['original_rrf_score'] = result['retrieval_score']
                    result['retrieval_score'] = 1.0
        
        # 6. 排序并返回结果
        results.sort(key=lambda x: x['retrieval_score'], reverse=True)
        final_results = results[:top_k]

        query_summary = f"{len(queries)} queries" if isinstance(query_text, list) else f"'{query_text[:30]}...'"
        logger.info(
            f"RRF融合检索 for {query_summary} (耗时: {time.time() - start_time:.3f}s) - "
            f"返回 {len(final_results)}/{top_k} 个结果 (阈值 {threshold}, k={60})")
        return final_results
