import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel  # 导入 BGE-M3 模型
from text_processing import split_text
from config import *

class KnowledgeBase:
    def __init__(self, similarity_mode="dense"):
        """
        初始化 KnowledgeBase 类。

        Args:
            similarity_mode: 相似度计算模式，可选 "dense", "lexical", "hybrid"。
        """
        self.embedding_model = BGEM3FlagModel(
            EMBEDDING_MODEL_PATH,  # 使用 config.py 中的 EMBEDDING_MODEL_PATH
            use_fp16=True  # 启用 FP16 加速（根据需要）
        )
        self.index = None  # Faiss 索引（用于稠密向量检索）
        self.dense_embeddings = None  # 用于存储稠密向量
        self.chunks = []  # 存储文本块
        self.sparse_weights = [] #存储稀疏向量
        self.similarity_mode = similarity_mode  # 相似度计算模式
        if self.similarity_mode not in ["dense", "lexical", "hybrid"]:
            raise ValueError("Invalid similarity_mode. Choose 'dense', 'lexical', or 'hybrid'.")


    def create_index(self, knowledge_file):
        """创建知识库索引（包括稠密索引和稀疏权重）"""
        try:
            with open(knowledge_file, "r", encoding="utf-8") as f:
                full_text = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Knowledge base file '{knowledge_file}' not found")

        self.chunks = split_text(
            full_text,
            CHUNK_SIZE,
            OVERLAP,
            MIN_CHUNK_LENGTH
        )
        self.chunks = [c for c in self.chunks if len(c.strip()) >= MIN_CHUNK_LENGTH]

        # 使用 BGE-M3 编码，同时获取稠密向量和稀疏权重
        bge_output = self.embedding_model.encode(
            self.chunks,
            return_dense=True,       # 需要稠密向量
            return_sparse=True,      # 需要稀疏向量
            return_colbert_vecs=False  # 不需要 ColBERT 向量
        )
        embeddings = bge_output["dense_vecs"]      # 获取稠密向量
        self.sparse_weights = bge_output["lexical_weights"]    #获取稀疏向量
        self.dense_embeddings = embeddings.astype(np.float32)  # 存储稠密向量

        if self.similarity_mode in ["dense", "hybrid"]:  # 如果需要稠密向量检索
            dimension = self.dense_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.dense_embeddings)  # 使用存储的向量添加索引

    def _compute_hybrid_score(self, query_dense, query_sparse, chunk_dense, chunk_sparse, dense_weight=0.6):
        """
        计算混合相似度得分。
        默认为dense部分占0.6, lexical部分占0.4
        """
        if chunk_dense is None or len(chunk_dense) == 0:  # 如果没有稠密向量
            dense_score = 0.0
        else:
            dense_score = np.dot(query_dense, chunk_dense) / (np.linalg.norm(query_dense) * np.linalg.norm(chunk_dense))  # 余弦相似度

        lexical_score = self.embedding_model.compute_lexical_matching_score(query_sparse, chunk_sparse)

        return dense_weight * dense_score + (1 - dense_weight) * lexical_score


    def retrieve_chunks(self, query, top_k=TOP_K):
        """检索相关知识块"""
        if self.similarity_mode in ["dense", "hybrid"] and not self.index:
            raise ValueError("Knowledge base index (dense) not initialized")
        if not self.chunks:
            raise ValueError("Knowledge base chunks are empty")


        query_output = self.embedding_model.encode(
            query,
            return_dense=self.similarity_mode in ["dense", "hybrid"],
            return_sparse=self.similarity_mode in ["lexical", "hybrid"],
            return_colbert_vecs=False
        )
        query_dense = query_output.get("dense_vecs", None)  # 获取查询的稠密向量（如果有）
        query_sparse = query_output.get("lexical_weights", None)  # 获取查询的稀疏权重（如果有）
        if query_dense is not None: #如果query_dense不是None，则进行reshape操作。
          query_dense = query_dense.reshape(1, -1).astype(np.float32)

        if self.similarity_mode == "dense":
            _, indices = self.index.search(query_dense, top_k)
            return [self.chunks[i] for i in indices[0]]

        elif self.similarity_mode == "lexical":
            scores = []
            for chunk_sparse in self.sparse_weights:
                score = self.embedding_model.compute_lexical_matching_score(query_sparse, chunk_sparse)
                scores.append(score)
            # 获取得分最高的 top_k 个索引
            top_k_indices = np.argsort(scores)[-top_k:][::-1]
            return [self.chunks[i] for i in top_k_indices]



        elif self.similarity_mode == "hybrid":
            if not self.index:  # 添加检查，确保混合模式下索引存在
                raise ValueError("Dense index required for hybrid mode is not initialized.")
            if self.dense_embeddings is None:  # 添加检查，确保稠密向量已存储
                raise ValueError("Dense embeddings required for hybrid mode are not stored.")
            _, indices = self.index.search(query_dense, top_k * 10)  # 检索更多候选，以便混合计算
            candidate_indices = indices[0]
            hybrid_scores = []
            for i in candidate_indices:
                # 直接从存储的列表中获取文本块的稠密向量
                chunk_dense = self.dense_embeddings[i]
                chunk_sparse = self.sparse_weights[i]  # 获取文本块的稀疏向量
                # 确保 query_dense 是单个向量 (1D array) 传递给 _compute_hybrid_score
                hybrid_score = self._compute_hybrid_score(query_dense[0], query_sparse, chunk_dense, chunk_sparse)
                hybrid_scores.append((i, hybrid_score))
            # 根据混合得分重新排序，选择 top_k
            hybrid_scores.sort(key=lambda x: x[1], reverse=True)
            top_k_indices = [idx for idx, _ in hybrid_scores[:top_k]]
            return [self.chunks[idx] for idx in top_k_indices]
        return None