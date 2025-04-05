# test/knowledge_base.py

import faiss
import numpy as np
import json
import pickle # 用于保存/加载 sparse_weights
from pathlib import Path # 用于处理路径

from FlagEmbedding import BGEM3FlagModel
from script.text_processing import split_text
# 从 config 导入所需配置
from script.config import (
    EMBEDDING_MODEL_PATH,
    CHUNK_SIZE,
    OVERLAP,
    MIN_CHUNK_LENGTH,
    PROCESSED_DATA_DIR, # 新增导入
    # 确保导入 SEARCH_MODE 如果在类内部需要用
)

class KnowledgeBase:
    def __init__(self, similarity_mode="dense", knowledge_file=None):
        """
        初始化 KnowledgeBase。
        会尝试从 PROCESSED_DATA_DIR 加载预处理数据，
        如果失败或数据不存在，并且提供了 knowledge_file，则会创建索引并保存。
        """
        print("Initializing KnowledgeBase...")
        self.embedding_model = BGEM3FlagModel(EMBEDDING_MODEL_PATH, use_fp16=True) # Embedding 模型总是需要加载的，用于查询编码
        self.index = None
        self.dense_embeddings = None
        self.chunks = []
        self.sparse_weights = []
        self.similarity_mode = similarity_mode
        if self.similarity_mode not in ["dense", "lexical", "hybrid"]:
            raise ValueError("Invalid similarity_mode.")

        self.processed_data_dir = Path(PROCESSED_DATA_DIR)

        # 尝试加载数据
        if self._load_processed_data():
            print(f"Successfully loaded pre-processed data from '{self.processed_data_dir}'")
        else:
            # 加载失败，尝试创建
            print(f"Pre-processed data not found or failed to load from '{self.processed_data_dir}'.")
            if knowledge_file and Path(knowledge_file).exists():
                print(f"Attempting to create index from '{knowledge_file}'...")
                self._create_index_from_file(knowledge_file)
                print("Index creation finished. Saving processed data...")
                self._save_processed_data()
            else:
                # 如果加载失败且没有有效的源文件，则 KB 无法工作
                raise RuntimeError(f"Failed to load processed data and no valid knowledge_file ('{knowledge_file}') provided for creation.")
        print("KnowledgeBase initialized successfully.")


    def _save_processed_data(self):
        """将 chunks, embeddings, sparse weights, 和 Faiss index 保存到磁盘。"""
        if not self.chunks:
            print("Warning: No data available to save.")
            return False
        try:
            self.processed_data_dir.mkdir(parents=True, exist_ok=True) # 创建目录（如果不存在）
            print(f"Saving processed data to '{self.processed_data_dir}'...")

            # 1. 保存 chunks (使用 JSON)
            chunks_path = self.processed_data_dir / "chunks.json"
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            print(f"Saved chunks to {chunks_path}")

            # 2. 保存 dense_embeddings (使用 NumPy)
            if self.dense_embeddings is not None:
                dense_emb_path = self.processed_data_dir / "dense_embeddings.npy"
                np.save(dense_emb_path, self.dense_embeddings)
                print(f"Saved dense embeddings to {dense_emb_path}")

            # 3. 保存 sparse_weights (使用 pickle)
            # 注意：pickle 有安全风险，确保只加载自己信任的文件
            if self.sparse_weights:
                 sparse_weights_path = self.processed_data_dir / "sparse_weights.pkl"
                 with open(sparse_weights_path, "wb") as f:
                     pickle.dump(self.sparse_weights, f)
                 print(f"Saved sparse weights to {sparse_weights_path}")

            # 4. 保存 Faiss index
            if self.index is not None:
                index_path = self.processed_data_dir / "faiss_index.idx"
                faiss.write_index(self.index, str(index_path)) # faiss 需要字符串路径
                print(f"Saved Faiss index to {index_path}")

            print("Processed data saved successfully.")
            return True
        except Exception as e:
            print(f"Error saving processed data: {e}")
            return False

    def _load_processed_data(self):
        """从磁盘加载预处理的数据。"""
        if not self.processed_data_dir.is_dir():
            # print(f"Processed data directory '{self.processed_data_dir}' not found.")
            return False # 目录不存在，无法加载

        print(f"Attempting to load processed data from '{self.processed_data_dir}'...")
        try:
            all_loaded = True # 标记是否所有需要的部分都加载成功

            # 1. 加载 chunks
            chunks_path = self.processed_data_dir / "chunks.json"
            if chunks_path.exists():
                with open(chunks_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                print(f"Loaded chunks from {chunks_path}")
                if not self.chunks:
                    print("Warning: Loaded chunks list is empty.")
                    all_loaded = False
            else:
                print(f"Chunks file not found: {chunks_path}")
                all_loaded = False

            # 2. 加载 dense_embeddings (如果模式需要)
            dense_emb_path = self.processed_data_dir / "dense_embeddings.npy"
            if self.similarity_mode in ["dense", "hybrid"]:
                if dense_emb_path.exists():
                    self.dense_embeddings = np.load(dense_emb_path)
                    print(f"Loaded dense embeddings from {dense_emb_path}")
                    # 校验维度是否匹配 (可选，但推荐)
                    # if self.chunks and len(self.chunks) != self.dense_embeddings.shape[0]:
                    #    print("Warning: Mismatch between number of chunks and loaded dense embeddings.")
                    #    all_loaded = False
                else:
                    print(f"Dense embeddings file not found (required for mode '{self.similarity_mode}'): {dense_emb_path}")
                    all_loaded = False

            # 3. 加载 sparse_weights (如果模式需要)
            sparse_weights_path = self.processed_data_dir / "sparse_weights.pkl"
            if self.similarity_mode in ["lexical", "hybrid"]:
                if sparse_weights_path.exists():
                     with open(sparse_weights_path, "rb") as f:
                         self.sparse_weights = pickle.load(f)
                     print(f"Loaded sparse weights from {sparse_weights_path}")
                     if not self.sparse_weights:
                          print("Warning: Loaded sparse weights list is empty.")
                          all_loaded = False
                     # 可选校验
                     # elif self.chunks and len(self.chunks) != len(self.sparse_weights):
                     #     print("Warning: Mismatch between number of chunks and loaded sparse weights.")
                     #     all_loaded = False
                else:
                    print(f"Sparse weights file not found (required for mode '{self.similarity_mode}'): {sparse_weights_path}")
                    all_loaded = False

            # 4. 加载 Faiss index (如果模式需要)
            index_path = self.processed_data_dir / "faiss_index.idx"
            if self.similarity_mode in ["dense", "hybrid"]:
                if index_path.exists():
                    self.index = faiss.read_index(str(index_path))
                    print(f"Loaded Faiss index from {index_path}")
                    # 校验向量数量是否匹配 (重要)
                    if self.dense_embeddings is not None and self.index.ntotal != self.dense_embeddings.shape[0]:
                         print(f"Error: Faiss index size ({self.index.ntotal}) does not match loaded dense embeddings size ({self.dense_embeddings.shape[0]}).")
                         all_loaded = False
                else:
                    print(f"Faiss index file not found (required for mode '{self.similarity_mode}'): {index_path}")
                    all_loaded = False

            if not all_loaded:
                print("Loading processed data failed due to missing files or errors.")
                # 清理可能部分加载的数据，确保状态一致
                self.index = None
                self.dense_embeddings = None
                self.chunks = []
                self.sparse_weights = []
                return False

            print("Pre-processed data loaded successfully.")
            return True

        except Exception as e:
            print(f"Error loading processed data: {e}")
            # 出错时也清理状态
            self.index = None
            self.dense_embeddings = None
            self.chunks = []
            self.sparse_weights = []
            return False


    # 将原来的 create_index 方法重命名为内部方法
    def _create_index_from_file(self, knowledge_file):
        """从原始知识库文件创建索引。"""
        try:
            with open(knowledge_file, "r", encoding="utf-8") as f:
                full_text = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Knowledge base file '{knowledge_file}' not found")

        self.chunks = split_text(
            full_text, CHUNK_SIZE, OVERLAP, MIN_CHUNK_LENGTH
        )
        self.chunks = [c for c in self.chunks if len(c.strip()) >= MIN_CHUNK_LENGTH]
        if not self.chunks:
             raise ValueError("No valid chunks generated from the knowledge file.")
        print(f"Split into {len(self.chunks)} chunks.")

        print(f"Encoding {len(self.chunks)} chunks using {EMBEDDING_MODEL_PATH}...")
        # 注意：如果 chunks 数量非常大，可能需要分批编码以避免 OOM
        batch_size = 32 # 可根据显存调整
        all_dense_embeddings = []
        all_sparse_weights = []
        for i in range(0, len(self.chunks), batch_size):
             batch_chunks = self.chunks[i:i+batch_size]
             print(f"Encoding batch {i//batch_size + 1}...")
             bge_output = self.embedding_model.encode(
                 batch_chunks,
                 return_dense=True,
                 return_sparse=True,
                 return_colbert_vecs=False
             )
             all_dense_embeddings.append(bge_output["dense_vecs"])
             all_sparse_weights.extend(bge_output["lexical_weights"]) # extend list
             print(f"Batch {i//batch_size + 1} encoded.")

        # 合并结果
        self.dense_embeddings = np.vstack(all_dense_embeddings).astype(np.float32)
        self.sparse_weights = all_sparse_weights # 已经是合并后的 list
        print("Encoding finished.")

        # 如果需要稠密索引
        if self.similarity_mode in ["dense", "hybrid"]:
            if self.dense_embeddings is None or self.dense_embeddings.shape[0] == 0:
                 raise ValueError("Cannot create Faiss index with no dense embeddings.")
            dimension = self.dense_embeddings.shape[1]
            print(f"Creating Faiss index (IndexFlatL2) with dimension {dimension}...")
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.dense_embeddings)
            print(f"Faiss index created with {self.index.ntotal} vectors.")

    # _compute_hybrid_score 和 retrieve_chunks 方法基本保持不变，
    # 因为它们依赖于已经加载或创建好的 self.index, self.chunks, self.dense_embeddings, self.sparse_weights
    # 但需要确保 retrieve_chunks 在访问这些属性前检查它们是否为 None (已在之前版本添加)
    def _compute_hybrid_score(self, query_dense, query_sparse, chunk_dense, chunk_sparse, dense_weight=0.6):
        # (代码与上一版本相同，注意向量维度处理和归一化)
         if chunk_dense is None or len(chunk_dense) == 0:
             dense_score = 0.0
         else:
             if query_dense.ndim > 1: query_dense = query_dense.squeeze()
             if chunk_dense.ndim > 1: chunk_dense = chunk_dense.squeeze()
             norm_query = np.linalg.norm(query_dense)
             norm_chunk = np.linalg.norm(chunk_dense)
             if norm_query == 0 or norm_chunk == 0: dense_score = 0.0
             else: dense_score = np.dot(query_dense, chunk_dense) / (norm_query * norm_chunk)

         lexical_score = self.embedding_model.compute_lexical_matching_score(query_sparse, chunk_sparse)
         return dense_weight * dense_score + (1 - dense_weight) * lexical_score

    def retrieve_chunks(self, query, top_k=None):
        from script.config import TOP_K as config_top_k
        top_k = top_k if top_k is not None else config_top_k

        # (代码与上一版本相同，包含各种检查和不同模式的处理逻辑)
        if not self.chunks: raise ValueError("Knowledge base chunks are not loaded or created.")
        # ... (其他检查和检索逻辑) ...
        # 确保所有模式的检索代码都使用了类成员变量 (self.index, self.dense_embeddings, etc.)

        # --- 粘贴上一版本中完整的 retrieve_chunks 实现 ---
        # --- 注意确保所有路径都正确处理 self.similarity_mode ---
        # --- 并正确使用了 self.index, self.chunks, self.dense_embeddings, self.sparse_weights ---
        # --- 以及 query_dense, query_sparse ---
        query_output = self.embedding_model.encode(query, return_dense=self.similarity_mode in ["dense", "hybrid"], return_sparse=self.similarity_mode in ["lexical", "hybrid"], return_colbert_vecs=False)
        query_dense = query_output.get("dense_vecs", None); query_sparse = query_output.get("lexical_weights", None)
        if query_dense is not None: query_dense = query_dense.astype(np.float32).reshape(1, -1)

        if self.similarity_mode == "dense":
            if self.index is None or query_dense is None: raise ValueError("Index or query dense vector missing for dense search.")
            _, indices = self.index.search(query_dense, top_k)
            valid_indices = [i for i in indices[0] if 0 <= i < len(self.chunks)]
            return [self.chunks[i] for i in valid_indices]
        elif self.similarity_mode == "lexical":
            if not self.sparse_weights or query_sparse is None: raise ValueError("Sparse weights missing for lexical search.")
            scores = [(i, self.embedding_model.compute_lexical_matching_score(query_sparse, chunk_sparse)) for i, chunk_sparse in enumerate(self.sparse_weights)]
            scores.sort(key=lambda x: x[1], reverse=True)
            top_k_indices = [idx for idx, _ in scores[:top_k]]
            valid_indices = [i for i in top_k_indices if 0 <= i < len(self.chunks)]
            return [self.chunks[i] for i in valid_indices]
        elif self.similarity_mode == "hybrid":
            if self.index is None or self.dense_embeddings is None or not self.sparse_weights or query_dense is None or query_sparse is None: raise ValueError("Required components missing for hybrid search.")
            candidate_multiplier = 10; num_candidates = min(top_k * candidate_multiplier, self.index.ntotal)
            _, dense_indices = self.index.search(query_dense, num_candidates)
            candidate_indices = [i for i in dense_indices[0] if 0 <= i < len(self.dense_embeddings) and 0 <= i < len(self.sparse_weights)] # Filter valid indices early
            hybrid_scores = []
            query_dense_1d = query_dense.squeeze()
            for i in candidate_indices:
                chunk_dense = self.dense_embeddings[i]; chunk_sparse = self.sparse_weights[i]
                hybrid_score = self._compute_hybrid_score(query_dense_1d, query_sparse, chunk_dense, chunk_sparse)
                hybrid_scores.append((i, hybrid_score))
            hybrid_scores.sort(key=lambda x: x[1], reverse=True)
            top_k_indices = [idx for idx, _ in hybrid_scores[:top_k]]
            valid_indices = [i for i in top_k_indices if 0 <= i < len(self.chunks)]
            return [self.chunks[idx] for idx in valid_indices]
        return []