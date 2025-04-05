# test/knowledge_base.py

import faiss
import numpy as np
import json
import pickle
from pathlib import Path
import os # 需要 os 来获取文件修改时间
import time

from FlagEmbedding import BGEM3FlagModel
from text_processing import split_text
from config import (
    EMBEDDING_MODEL_PATH, CHUNK_SIZE, OVERLAP, MIN_CHUNK_LENGTH,
    PROCESSED_DATA_DIR, KNOWLEDGE_BASE_DIR, KNOWLEDGE_FILE_PATTERN, # 导入新配置
    SEARCH_MODE
)

class KnowledgeBase:
    # 修改 __init__ 参数
    def __init__(self, similarity_mode="dense", knowledge_dir=None, knowledge_file_pattern=None):
        print("Initializing KnowledgeBase...")
        self.embedding_model = BGEM3FlagModel(EMBEDDING_MODEL_PATH, use_fp16=True)
        self.index = None
        self.dense_embeddings = None
        self.chunks = []
        self.sparse_weights = []
        self.source_file_metadata = {} # 用于存储源文件信息以检测变化
        self.similarity_mode = similarity_mode
        if self.similarity_mode not in ["dense", "lexical", "hybrid"]:
            raise ValueError("Invalid similarity_mode.")

        self.processed_data_dir = Path(PROCESSED_DATA_DIR)
        # 保存传入的源目录和模式配置
        self.knowledge_dir = Path(knowledge_dir) if knowledge_dir else None
        self.knowledge_file_pattern = knowledge_file_pattern

        if not self.knowledge_dir or not self.knowledge_file_pattern:
             print("Warning: Knowledge source directory or pattern not configured.")
             # 尝试加载，但如果加载失败将无法重新构建

        # 尝试加载数据 (加载函数内部会进行新鲜度检查)
        if not self._load_processed_data():
            print(f"Pre-processed data not found, invalid, or stale in '{self.processed_data_dir}'.")
            # 检查是否可以从源文件创建
            if self.knowledge_dir and self.knowledge_dir.is_dir() and self.knowledge_file_pattern:
                print(f"Attempting to create index from sources in '{self.knowledge_dir}' matching '{self.knowledge_file_pattern}'...")
                self._create_index_from_sources(self.knowledge_dir, self.knowledge_file_pattern) # 调用新的创建函数
                print("Index creation finished. Saving processed data...")
                self._save_processed_data() # 保存新创建的数据
            else:
                raise RuntimeError(f"Failed to load processed data and cannot create index: Knowledge source directory '{self.knowledge_dir}' or pattern '{self.knowledge_file_pattern}' is invalid or not provided.")
        else:
             print(f"Successfully loaded pre-processed data from '{self.processed_data_dir}'")
        print("KnowledgeBase initialized successfully.")


    def _get_current_source_metadata(self, knowledge_dir, pattern):
        """获取当前源目录下匹配文件的路径和最后修改时间"""
        source_files_metadata = {}
        if not knowledge_dir or not knowledge_dir.is_dir():
            print(f"Warning: Knowledge source directory '{knowledge_dir}' is invalid or does not exist.")
            return source_files_metadata
        try:
            # 使用 sorted 确保文件顺序一致性
            for file_path in sorted(knowledge_dir.glob(pattern)):
                 if file_path.is_file():
                      mtime = os.path.getmtime(file_path)
                      # 使用绝对路径作为 key，确保唯一性
                      source_files_metadata[str(file_path.resolve())] = mtime
        except Exception as e:
             print(f"Error scanning source directory {knowledge_dir}: {e}")
        return source_files_metadata

    def _save_processed_data(self):
        """保存处理后的数据，包括源文件元数据。"""
        if not self.chunks:
            print("Warning: No data available to save.")
            return False
        try:
            self.processed_data_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving processed data to '{self.processed_data_dir}'...")

            # 保存 chunks, dense_embeddings, sparse_weights, index (与之前类似)
            # ... (省略了这部分代码，假设与上一版本相同) ...
            # 1. 保存 chunks (JSON)
            chunks_path = self.processed_data_dir / "chunks.json"; json.dump(self.chunks, open(chunks_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2); print(f"Saved chunks to {chunks_path}")
            # 2. 保存 dense embeddings (NumPy)
            if self.dense_embeddings is not None: dense_emb_path = self.processed_data_dir / "dense_embeddings.npy"; np.save(dense_emb_path, self.dense_embeddings); print(f"Saved dense embeddings to {dense_emb_path}")
            # 3. 保存 sparse weights (Pickle)
            if self.sparse_weights: sparse_weights_path = self.processed_data_dir / "sparse_weights.pkl"; pickle.dump(self.sparse_weights, open(sparse_weights_path, "wb")); print(f"Saved sparse weights to {sparse_weights_path}")
            # 4. 保存 Faiss index
            if self.index is not None: index_path = self.processed_data_dir / "faiss_index.idx"; faiss.write_index(self.index, str(index_path)); print(f"Saved Faiss index to {index_path}")


            # === 新增：保存元数据 ===
            metadata = {
                # 使用创建时记录的源文件元数据
                "source_files": self.source_file_metadata,
                "similarity_mode": self.similarity_mode,
                "creation_time": time.time() # 记录创建时间
            }
            metadata_path = self.processed_data_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata to {metadata_path}")
            # ======================

            print("Processed data saved successfully.")
            return True
        except Exception as e:
            print(f"Error saving processed data: {e}")
            return False

    def _load_processed_data(self):
        """加载预处理数据，并检查源文件是否已更改。"""
        if not self.processed_data_dir.is_dir(): return False # 目录不存在

        metadata_path = self.processed_data_dir / "metadata.json"
        if not metadata_path.exists():
            print(f"Metadata file '{metadata_path}' not found. Cannot load processed data.")
            return False

        try:
            print(f"Loading metadata from '{metadata_path}'...")
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # === 新增：检查数据新鲜度 ===
            print("Checking data staleness against source files...")
            saved_source_metadata = metadata.get("source_files", {})
            # 获取当前的源文件状态
            current_source_metadata = self._get_current_source_metadata(self.knowledge_dir, self.knowledge_file_pattern)

            # 检查文件数量是否一致
            if len(saved_source_metadata) != len(current_source_metadata):
                print("Data is stale: Number of source files has changed.")
                print(f"  Saved: {len(saved_source_metadata)} files, Current: {len(current_source_metadata)} files.")
                return False # 强制重新构建

            # 逐一检查文件修改时间
            is_stale = False
            for file_path_str, current_mtime in current_source_metadata.items():
                saved_mtime = saved_source_metadata.get(file_path_str)
                if saved_mtime is None:
                    print(f"Data is stale: Source file '{file_path_str}' is new or was not indexed previously.")
                    is_stale = True; break
                # 比较浮点数时间戳，允许微小误差
                if abs(saved_mtime - current_mtime) > 1e-6:
                     print(f"Data is stale: Source file '{file_path_str}' has been modified.")
                     print(f"  Saved mtime: {saved_mtime}, Current mtime: {current_mtime}")
                     is_stale = True; break
            if is_stale:
                 return False # 强制重新构建

            # 检查是否有文件被删除 (存在于 saved 但不存在于 current)
            for saved_path_str in saved_source_metadata:
                 if saved_path_str not in current_source_metadata:
                      print(f"Data is stale: Source file '{saved_path_str}' seems to have been removed.")
                      return False # 强制重新构建

            print("Processed data appears up-to-date.")
            # ==========================

            # (可选) 检查 similarity_mode 是否匹配
            saved_mode = metadata.get("similarity_mode")
            if saved_mode and saved_mode != self.similarity_mode:
                print(f"Warning: Saved data mode '{saved_mode}' differs from current mode '{self.similarity_mode}'.")
                # return False # 如果要求模式必须匹配，则取消注释

            # 如果数据是新鲜的，继续加载 chunks, embeddings, index 等
            # ... (粘贴上一版本中加载 chunks, dense_embeddings, sparse_weights, index 的代码) ...
            # ... (确保对文件存在性做了检查) ...
            # 1. 加载 chunks
            chunks_path = self.processed_data_dir / "chunks.json"; self.chunks = json.load(open(chunks_path, "r", encoding="utf-8")); print(f"Loaded chunks from {chunks_path}")
            # 2. 加载 dense embeddings (如果需要)
            if self.similarity_mode in ["dense", "hybrid"]: dense_emb_path = self.processed_data_dir / "dense_embeddings.npy"; self.dense_embeddings = np.load(dense_emb_path); print(f"Loaded dense embeddings from {dense_emb_path}")
            # 3. 加载 sparse weights (如果需要)
            if self.similarity_mode in ["lexical", "hybrid"]: sparse_weights_path = self.processed_data_dir / "sparse_weights.pkl"; self.sparse_weights = pickle.load(open(sparse_weights_path, "rb")); print(f"Loaded sparse weights from {sparse_weights_path}")
            # 4. 加载 Faiss index (如果需要)
            if self.similarity_mode in ["dense", "hybrid"]: index_path = self.processed_data_dir / "faiss_index.idx"; self.index = faiss.read_index(str(index_path)); print(f"Loaded Faiss index from {index_path}")

            # 基本校验
            if not self.chunks: raise ValueError("Loaded chunks are empty.")
            if self.similarity_mode in ["dense", "hybrid"] and (self.index is None or self.dense_embeddings is None): raise ValueError("Dense/Hybrid mode requires index and dense embeddings.")
            if self.similarity_mode in ["lexical", "hybrid"] and not self.sparse_weights: raise ValueError("Lexical/Hybrid mode requires sparse weights.")
            if self.similarity_mode in ["dense", "hybrid"] and self.index.ntotal != len(self.chunks): raise ValueError("Index size does not match number of chunks.")


            print("Pre-processed data loaded successfully.")
            return True # 加载成功

        except Exception as e:
            print(f"Error loading processed data: {e}")
            # 清理可能部分加载的数据
            self.index = None; self.dense_embeddings = None; self.chunks = []; self.sparse_weights = []
            return False


    # 重命名并修改创建函数以处理多个源文件
    def _create_index_from_sources(self, knowledge_dir, pattern):
        """从指定目录中匹配模式的所有文件创建索引。"""
        print(f"Scanning '{knowledge_dir}' for files matching '{pattern}'...")
        # 获取当前源文件列表及其元数据，用于后续保存
        current_metadata = self._get_current_source_metadata(knowledge_dir, pattern)
        if not current_metadata:
            raise FileNotFoundError(f"No files found in '{knowledge_dir}' matching pattern '{pattern}'.")

        self.source_file_metadata = current_metadata # 保存元数据，以便 _save_processed_data 使用
        source_file_paths = list(current_metadata.keys()) # 获取文件路径列表

        all_chunks = []
        print(f"Processing {len(source_file_paths)} source files...")
        for file_path_str in source_file_paths:
             file_path = Path(file_path_str) # 转换为 Path 对象
             print(f"  Reading and splitting: {file_path.name}")
             try:
                  with open(file_path, "r", encoding="utf-8") as f:
                       full_text = f.read()
                  # 使用 text_processing.py 中的函数进行分块
                  file_chunks = split_text(full_text, CHUNK_SIZE, OVERLAP, MIN_CHUNK_LENGTH)
                  # 清理空块
                  file_chunks = [c for c in file_chunks if len(c.strip()) >= MIN_CHUNK_LENGTH]
                  all_chunks.extend(file_chunks) # 合并所有文件的块
             except Exception as e:
                  print(f"  Error processing file {file_path}: {e}. Skipping this file.")

        self.chunks = all_chunks
        if not self.chunks:
             raise ValueError("No valid chunks generated from any source files.")
        print(f"Total chunks generated from all files: {len(self.chunks)}")

        # 对所有合并后的 chunks 进行编码和索引构建 (与之前类似，注意使用 self.chunks)
        print(f"Encoding {len(self.chunks)} chunks...")
        # ... (粘贴上一版本中的批量编码逻辑) ...
        batch_size = 32; all_dense = []; all_sparse = []
        for i in range(0, len(self.chunks), batch_size):
             batch_chunks = self.chunks[i:i+batch_size]; print(f"Encoding batch {i//batch_size+1}...")
             bge_output = self.embedding_model.encode(batch_chunks, return_dense=True, return_sparse=True, return_colbert_vecs=False)
             all_dense.append(bge_output["dense_vecs"]); all_sparse.extend(bge_output["lexical_weights"])
        self.dense_embeddings = np.vstack(all_dense).astype(np.float32); self.sparse_weights = all_sparse
        print("Encoding finished.")

        # ... (粘贴上一版本中构建 Faiss index 的逻辑) ...
        if self.similarity_mode in ["dense", "hybrid"]:
            dimension = self.dense_embeddings.shape[1]; print(f"Creating Faiss index (dim={dimension})...");
            self.index = faiss.IndexFlatL2(dimension); self.index.add(self.dense_embeddings)
            print(f"Faiss index created ({self.index.ntotal} vectors).")

    # _compute_hybrid_score 和 retrieve_chunks 方法保持不变
    def _compute_hybrid_score(self, query_dense, query_sparse, chunk_dense, chunk_sparse, dense_weight=0.6):
        # (与上版本相同)
         if chunk_dense is None or len(chunk_dense) == 0: dense_score = 0.0
         else:
             if query_dense.ndim > 1: query_dense = query_dense.squeeze()
             if chunk_dense.ndim > 1: chunk_dense = chunk_dense.squeeze()
             norm_query = np.linalg.norm(query_dense); norm_chunk = np.linalg.norm(chunk_dense)
             if norm_query == 0 or norm_chunk == 0: dense_score = 0.0
             else: dense_score = np.dot(query_dense, chunk_dense) / (norm_query * norm_chunk)
         lexical_score = self.embedding_model.compute_lexical_matching_score(query_sparse, chunk_sparse)
         return dense_weight * dense_score + (1 - dense_weight) * lexical_score

    def retrieve_chunks(self, query, top_k=None):
        # (与上版本相同)
        from config import TOP_K as config_top_k
        top_k = top_k if top_k is not None else config_top_k
        if not self.chunks: raise ValueError("Knowledge base chunks are not loaded or created.")
        # ... (粘贴上版本完整的检查和检索逻辑) ...
        query_output = self.embedding_model.encode(query, return_dense=self.similarity_mode in ["dense", "hybrid"], return_sparse=self.similarity_mode in ["lexical", "hybrid"], return_colbert_vecs=False)
        query_dense = query_output.get("dense_vecs", None); query_sparse = query_output.get("lexical_weights", None)
        if query_dense is not None: query_dense = query_dense.astype(np.float32).reshape(1, -1)
        if self.similarity_mode == "dense": #... (dense logic)
             if self.index is None or query_dense is None: raise ValueError("Index/query dense missing for dense search.")
             _, indices = self.index.search(query_dense, top_k); valid_indices = [i for i in indices[0] if 0 <= i < len(self.chunks)]; return [self.chunks[i] for i in valid_indices]
        elif self.similarity_mode == "lexical": # ... (lexical logic)
             if not self.sparse_weights or query_sparse is None: raise ValueError("Sparse weights missing for lexical search.")
             scores = [(i, self.embedding_model.compute_lexical_matching_score(query_sparse, cs)) for i, cs in enumerate(self.sparse_weights)]; scores.sort(key=lambda x: x[1], reverse=True)
             top_k_indices = [idx for idx, _ in scores[:top_k]]; valid_indices = [i for i in top_k_indices if 0 <= i < len(self.chunks)]; return [self.chunks[i] for i in valid_indices]
        elif self.similarity_mode == "hybrid": # ... (hybrid logic)
             if self.index is None or self.dense_embeddings is None or not self.sparse_weights or query_dense is None or query_sparse is None: raise ValueError("Components missing for hybrid search.")
             candidate_multiplier = 10; num_candidates = min(top_k * candidate_multiplier, self.index.ntotal)
             _, dense_indices = self.index.search(query_dense, num_candidates)
             candidate_indices = [i for i in dense_indices[0] if 0 <= i < len(self.dense_embeddings) and 0 <= i < len(self.sparse_weights)]
             hybrid_scores = []; query_dense_1d = query_dense.squeeze()
             for i in candidate_indices:
                 chunk_dense = self.dense_embeddings[i]; chunk_sparse = self.sparse_weights[i]
                 hybrid_score = self._compute_hybrid_score(query_dense_1d, query_sparse, chunk_dense, chunk_sparse); hybrid_scores.append((i, hybrid_score))
             hybrid_scores.sort(key=lambda x: x[1], reverse=True); top_k_indices = [idx for idx, _ in hybrid_scores[:top_k]]
             valid_indices = [i for i in top_k_indices if 0 <= i < len(self.chunks)]; return [self.chunks[idx] for idx in valid_indices]
        return []