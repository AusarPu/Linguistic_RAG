# test/knowledge_base.py

import faiss
import numpy as np
import json
import pickle
from pathlib import Path
import os
import time
import logging # 导入 logging
from typing import List, Dict, Optional, Tuple, Any # 增加类型注解

from FlagEmbedding import BGEM3FlagModel
from text_processing import split_text # 使用相对导入
from config import ( # 使用相对导入
    EMBEDDING_MODEL_PATH,
    PROCESSED_DATA_DIR,
    CHUNK_SIZE,
    OVERLAP,
    MIN_CHUNK_LENGTH,
    TOP_K_INITIAL_RETRIEVAL, # 使用新的配置名
    RETRIEVAL_STRATEGY,
    HYBRID_SIMILARITY_THRESHOLD,
    MAX_THRESHOLD_RESULTS,
    # KNOWLEDGE_BASE_DIR and KNOWLEDGE_FILE_PATTERN will be passed during init
)

# 获取 logger 实例
logger = logging.getLogger(__name__)

class KnowledgeBase:
    """
    管理知识库的加载、创建、索引和检索。
    固定使用 Hybrid (稠密+稀疏) 模式进行检索。
    """
    def __init__(self, knowledge_dir: str, knowledge_file_pattern: str):
        """
        初始化知识库。

        Args:
            knowledge_dir (str): 包含知识库源文件的目录路径。
            knowledge_file_pattern (str): 在 knowledge_dir 中匹配知识库文件的模式 (例如 "*.txt")。
        """
        logger.info("Initializing KnowledgeBase (mode: hybrid)...")
        if not knowledge_dir or not knowledge_file_pattern:
            raise ValueError("Knowledge source directory and pattern must be provided.")

        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_file_pattern = knowledge_file_pattern
        self.processed_data_dir = Path(PROCESSED_DATA_DIR)

        logger.debug(f"Knowledge source directory: {self.knowledge_dir}")
        logger.debug(f"Knowledge file pattern: {self.knowledge_file_pattern}")
        logger.debug(f"Processed data directory: {self.processed_data_dir}")

        try:
            logger.info(f"Loading embedding model from: {EMBEDDING_MODEL_PATH}")
            # 注意：模型加载可能耗时且消耗资源，放在 __init__ 中需谨慎
            # 考虑是否可以惰性加载或在外部加载后传入 model 实例
            self.embedding_model = BGEM3FlagModel(EMBEDDING_MODEL_PATH, use_fp16=True)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model from {EMBEDDING_MODEL_PATH}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load embedding model: {e}") from e

        self.index: Optional[faiss.Index] = None
        self.dense_embeddings: Optional[np.ndarray] = None
        self.chunks_data: List[Dict[str, str]] = [] # 存储 {"text": ..., "source": ...}
        self.sparse_weights: List[Dict[int, float]] = []
        self.source_file_metadata: Dict[str, float] = {} # 用于存储源文件信息以检测变化

        # 尝试加载或创建索引
        if not self._load_processed_data():
            logger.warning(f"Pre-processed data not found, invalid, or stale in '{self.processed_data_dir}'.")
            if self.knowledge_dir.is_dir():
                logger.info(f"Attempting to create index from sources in '{self.knowledge_dir}' matching '{self.knowledge_file_pattern}'...")
                try:
                    self._create_index_from_sources(self.knowledge_dir, self.knowledge_file_pattern)
                    logger.info("Index creation finished. Saving processed data...")
                    if not self._save_processed_data():
                         logger.warning("Failed to save newly created processed data.")
                except Exception as e:
                    logger.error(f"Failed to create index from sources: {e}", exc_info=True)
                    raise RuntimeError(f"Failed to create index from sources: {e}") from e
            else:
                logger.error(f"Knowledge source directory '{self.knowledge_dir}' is invalid or not provided.")
                raise RuntimeError(f"Failed to load processed data and cannot create index: Knowledge source directory '{self.knowledge_dir}' is invalid.")
        else:
             logger.info(f"Successfully loaded pre-processed data from '{self.processed_data_dir}'")

        # 最终校验 (因为固定 hybrid)
        if not self.chunks_data or self.index is None or self.dense_embeddings is None or not self.sparse_weights:
             logger.error("Knowledge base initialization failed: Missing required components (chunks, index, embeddings, or weights).")
             raise RuntimeError("Knowledge base initialization failed: Missing required components after load/create.")

        logger.info("KnowledgeBase initialized successfully.")

    def _get_current_source_metadata(self, knowledge_dir: Path, pattern: str) -> Dict[str, float]:
        """获取当前源目录下匹配文件的路径和最后修改时间"""
        source_files_metadata = {}
        if not knowledge_dir or not knowledge_dir.is_dir():
            logger.warning(f"Knowledge source directory '{knowledge_dir}' is invalid or does not exist.")
            return source_files_metadata
        try:
            logger.debug(f"Scanning source directory: {knowledge_dir} with pattern: {pattern}")
            found_files = sorted(knowledge_dir.glob(pattern))
            logger.debug(f"Found {len(found_files)} potential files matching pattern.")
            for file_path in found_files:
                 if file_path.is_file():
                      try:
                           mtime = os.path.getmtime(file_path)
                           source_files_metadata[str(file_path.resolve())] = mtime
                           logger.debug(f"  - Found source file: {file_path.resolve()} (mtime: {mtime})")
                      except OSError as e:
                           logger.warning(f"Could not get modification time for file {file_path}: {e}")
                 else:
                      logger.debug(f"  - Skipping non-file entry: {file_path}")
        except Exception as e:
             logger.error(f"Error scanning source directory {knowledge_dir}: {e}", exc_info=True)
        if not source_files_metadata:
             logger.warning(f"No source files found matching pattern '{pattern}' in directory '{knowledge_dir}'.")
        return source_files_metadata

    def _save_processed_data(self) -> bool:
        """保存处理后的数据（固定为 Hybrid 模式所需数据）。"""
        if not self.chunks_data or self.dense_embeddings is None or not self.sparse_weights or self.index is None:
            logger.warning("No complete data available to save (missing chunks, embeddings, weights, or index).")
            return False
        try:
            self.processed_data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving processed data to '{self.processed_data_dir}'...")

            # 1. 保存 chunks (JSON)
            chunks_path = self.processed_data_dir / "chunks.json"
            with open(chunks_path, "w", encoding="utf-8") as f:
                 json.dump(self.chunks_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved chunks to {chunks_path}")

            # 2. 保存 dense embeddings (NumPy)
            dense_emb_path = self.processed_data_dir / "dense_embeddings.npy"
            np.save(dense_emb_path, self.dense_embeddings)
            logger.info(f"Saved dense embeddings to {dense_emb_path}")

            # 3. 保存 sparse weights (Pickle)
            sparse_weights_path = self.processed_data_dir / "sparse_weights.pkl"
            with open(sparse_weights_path, "wb") as f:
                 pickle.dump(self.sparse_weights, f)
            logger.info(f"Saved sparse weights to {sparse_weights_path}")

            # 4. 保存 Faiss index
            index_path = self.processed_data_dir / "faiss_index.idx"
            faiss.write_index(self.index, str(index_path))
            logger.info(f"Saved Faiss index to {index_path}")

            # 5. 保存元数据
            metadata = {
                "source_files": self.source_file_metadata,
                "embedding_model": EMBEDDING_MODEL_PATH, # 记录使用的模型
                "creation_time": time.time()
            }
            metadata_path = self.processed_data_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")

            logger.info("Processed data saved successfully.")
            return True
        except IOError as e:
             logger.error(f"I/O error saving processed data to {self.processed_data_dir}: {e}", exc_info=True)
        except pickle.PicklingError as e:
             logger.error(f"Error pickling sparse weights: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error saving processed data: {e}", exc_info=True)
            # 尝试清理可能部分写入的文件（可选）
            return False
        return False


    def _load_processed_data(self) -> bool:
        """加载预处理数据（固定为 Hybrid 模式），并检查源文件是否已更改。"""
        if not self.processed_data_dir.is_dir():
            logger.debug(f"Processed data directory '{self.processed_data_dir}' not found.")
            return False

        metadata_path = self.processed_data_dir / "metadata.json"
        chunks_path = self.processed_data_dir / "chunks.json"
        dense_emb_path = self.processed_data_dir / "dense_embeddings.npy"
        sparse_weights_path = self.processed_data_dir / "sparse_weights.pkl"
        index_path = self.processed_data_dir / "faiss_index.idx"

        required_files = [metadata_path, chunks_path, dense_emb_path, sparse_weights_path, index_path]
        if not all(p.exists() for p in required_files):
            logger.warning(f"One or more required processed files not found in '{self.processed_data_dir}'. Cannot load.")
            missing = [p.name for p in required_files if not p.exists()]
            logger.debug(f"Missing files: {missing}")
            return False

        try:
            logger.info(f"Loading metadata from '{metadata_path}'...")
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # === 数据新鲜度检查 ===
            logger.info("Checking data staleness against source files...")
            saved_source_metadata = metadata.get("source_files", {})
            current_source_metadata = self._get_current_source_metadata(self.knowledge_dir, self.knowledge_file_pattern)

            if len(saved_source_metadata) != len(current_source_metadata):
                logger.warning("Data is stale: Number of source files has changed.")
                logger.info(f"  Saved: {len(saved_source_metadata)} files, Current: {len(current_source_metadata)} files.")
                return False

            is_stale = False
            for file_path_str, current_mtime in current_source_metadata.items():
                saved_mtime = saved_source_metadata.get(file_path_str)
                if saved_mtime is None:
                    logger.warning(f"Data is stale: Source file '{file_path_str}' is new or was not indexed previously.")
                    is_stale = True; break
                if abs(saved_mtime - current_mtime) > 1e-6: # 允许微小误差
                     logger.warning(f"Data is stale: Source file '{file_path_str}' has been modified.")
                     logger.debug(f"  Saved mtime: {saved_mtime}, Current mtime: {current_mtime}")
                     is_stale = True; break
            if is_stale: return False

            for saved_path_str in saved_source_metadata:
                 if saved_path_str not in current_source_metadata:
                      logger.warning(f"Data is stale: Source file '{saved_path_str}' seems to have been removed.")
                      return False

            logger.info("Processed data appears up-to-date.")
            # ==========================

            # === 加载数据 ===
            logger.info("Loading processed data components...")
            # 1. Chunks
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks_data = json.load(f)
            logger.info(f"Loaded {len(self.chunks_data)} structured chunks from {chunks_path}")
            if not self.chunks_data:
                logger.error("Loaded chunks data is empty.")
                return False # 加载失败

            # 2. Dense Embeddings
            self.dense_embeddings = np.load(dense_emb_path)
            logger.info(f"Loaded dense embeddings (shape: {self.dense_embeddings.shape}) from {dense_emb_path}")

            # 3. Sparse Weights
            with open(sparse_weights_path, "rb") as f:
                self.sparse_weights = pickle.load(f)
            logger.info(f"Loaded {len(self.sparse_weights)} sparse weights from {sparse_weights_path}")

            # 4. Faiss Index
            self.index = faiss.read_index(str(index_path))
            logger.info(f"Loaded Faiss index ({self.index.ntotal} vectors) from {index_path}")

            # === 基本校验 ===
            num_chunks = len(self.chunks_data)
            if not (num_chunks > 0 and
                    self.dense_embeddings is not None and len(self.dense_embeddings) == num_chunks and
                    self.sparse_weights is not None and len(self.sparse_weights) == num_chunks and
                    self.index is not None and self.index.ntotal == num_chunks):
                logger.error("Inconsistency detected in loaded data components (counts mismatch).")
                logger.debug(f"Num chunks: {num_chunks}, Dense shape: {self.dense_embeddings.shape if self.dense_embeddings is not None else 'None'}, "
                             f"Sparse count: {len(self.sparse_weights) if self.sparse_weights is not None else 'None'}, Index ntotal: {self.index.ntotal if self.index is not None else 'None'}")
                # 清理已加载的数据，防止状态不一致
                self.index = None; self.dense_embeddings = None; self.chunks_data = []; self.sparse_weights = []
                return False
            # =================

            # 记录当前的源文件元数据，以便后续保存时使用最新的
            self.source_file_metadata = current_source_metadata
            logger.info("Pre-processed data loaded and verified successfully.")
            return True

        except FileNotFoundError as e:
            logger.error(f"Required file not found during loading: {e}", exc_info=True)
        except json.JSONDecodeError as e:
             logger.error(f"Error decoding JSON file (likely chunks or metadata): {e}", exc_info=True)
        except pickle.UnpicklingError as e:
             logger.error(f"Error unpickling sparse weights file: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error loading processed data: {e}", exc_info=True)

        # 如果加载过程中出错，清理可能部分加载的数据
        self.index = None; self.dense_embeddings = None; self.chunks_data = []; self.sparse_weights = []
        return False


    def _create_index_from_sources(self, knowledge_dir: Path, pattern: str):
        """从指定目录中匹配模式的所有文件创建 Hybrid 索引。"""
        logger.info(f"Scanning '{knowledge_dir}' for files matching '{pattern}'...")
        current_metadata = self._get_current_source_metadata(knowledge_dir, pattern)
        if not current_metadata:
            logger.error(f"No source files found in '{knowledge_dir}' matching pattern '{pattern}'. Cannot create index.")
            raise FileNotFoundError(f"No source files found for index creation.")

        self.source_file_metadata = current_metadata # 保存元数据
        source_file_paths = list(current_metadata.keys())

        all_chunks_data: List[Dict[str, str]] = []
        logger.info(f"Processing {len(source_file_paths)} source files...")
        for file_path_str in source_file_paths:
             file_path = Path(file_path_str)
             logger.info(f"  Reading and splitting: {file_path.name}")
             try:
                  with open(file_path, "r", encoding="utf-8") as f:
                       full_text = f.read()
                  # 使用 text_processing.py 中的函数进行分块
                  file_chunks_text = split_text(full_text, CHUNK_SIZE, OVERLAP, MIN_CHUNK_LENGTH)
                  source_filename = file_path.name
                  added_count = 0
                  for chunk_text in file_chunks_text:
                      # 再次检查最小长度，确保 split_text 结果符合预期
                      if len(chunk_text.strip()) >= MIN_CHUNK_LENGTH:
                          all_chunks_data.append({
                              "text": chunk_text.strip(), # 去除首尾空白
                              "source": source_filename
                          })
                          added_count += 1
                  logger.info(f"    -> Generated {added_count} valid chunks from {file_path.name}.")
             except FileNotFoundError:
                  logger.error(f"  Error processing file {file_path}: File not found (unexpected). Skipping.")
             except IOError as e:
                  logger.error(f"  I/O error processing file {file_path}: {e}. Skipping.", exc_info=True)
             except Exception as e:
                  logger.error(f"  Unexpected error processing file {file_path}: {e}. Skipping.", exc_info=True)

        if not all_chunks_data:
             logger.error("No valid chunks generated from any source files. Index creation aborted.")
             raise ValueError("No valid chunks generated from source files.")

        self.chunks_data = all_chunks_data
        logger.info(f"Total valid chunks generated from all files: {len(self.chunks_data)}")

        # 提取纯文本列表用于编码
        chunks_text_list = [item["text"] for item in self.chunks_data]

        # --- 编码 (需要稠密和稀疏) ---
        logger.info(f"Encoding {len(chunks_text_list)} chunks (for dense and sparse)...")
        batch_size = 32 # 可配置
        all_dense = []
        all_sparse = []
        try:
            for i in range(0, len(chunks_text_list), batch_size):
                 batch_chunks = chunks_text_list[i:i+batch_size]
                 logger.debug(f"Encoding batch {i//batch_size+1}/{ (len(chunks_text_list) + batch_size - 1)//batch_size }...")
                 # 同时获取 dense 和 sparse 输出
                 bge_output = self.embedding_model.encode(
                     batch_chunks,
                     return_dense=True,
                     return_sparse=True,
                     return_colbert_vecs=False # 假设不需要 colbert
                 )
                 # 检查输出是否符合预期
                 if "dense_vecs" not in bge_output or "lexical_weights" not in bge_output:
                      raise ValueError(f"Embedding model did not return expected 'dense_vecs' or 'lexical_weights' in batch {i//batch_size+1}")
                 if len(bge_output["dense_vecs"]) != len(batch_chunks) or len(bge_output["lexical_weights"]) != len(batch_chunks):
                      raise ValueError(f"Embedding output length mismatch in batch {i//batch_size+1}")

                 all_dense.append(bge_output["dense_vecs"])
                 all_sparse.extend(bge_output["lexical_weights"])

            self.dense_embeddings = np.vstack(all_dense).astype(np.float32)
            self.sparse_weights = all_sparse
            logger.info("Encoding finished.")
            logger.debug(f"Dense embeddings shape: {self.dense_embeddings.shape}")
            logger.debug(f"Sparse weights count: {len(self.sparse_weights)}")

        except Exception as e:
            logger.error(f"Error during chunk encoding: {e}", exc_info=True)
            raise RuntimeError(f"Failed during chunk encoding: {e}") from e

        # --- 构建 Faiss 索引 (仅需稠密向量) ---
        try:
            dimension = self.dense_embeddings.shape[1]
            logger.info(f"Creating Faiss index (IndexFlatL2, dim={dimension})...")
            # 使用 IndexFlatL2，适合精确搜索，如果数据量巨大可考虑 IVF 等近似索引
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.dense_embeddings)
            logger.info(f"Faiss index created ({self.index.ntotal} vectors).")
        except Exception as e:
            logger.error(f"Error creating Faiss index: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create Faiss index: {e}") from e


    def _compute_hybrid_score(self,
                              query_dense: np.ndarray,
                              query_sparse: Dict[int, float],
                              chunk_dense: np.ndarray,
                              chunk_sparse: Dict[int, float],
                              dense_weight: float = 0.9 # 可配置
                              ) -> float:
         """计算单个查询与单个块之间的混合相似度得分。"""
         # 稠密得分 (余弦相似度)
         dense_score = 0.0
         try:
             # 确保是一维向量
             if query_dense.ndim > 1: query_dense = query_dense.squeeze()
             if chunk_dense.ndim > 1: chunk_dense = chunk_dense.squeeze()
             # L2 归一化，然后点积即为余弦相似度 (更稳定)
             norm_query = np.linalg.norm(query_dense)
             norm_chunk = np.linalg.norm(chunk_dense)
             if norm_query > 1e-9 and norm_chunk > 1e-9: # 避免除以零
                 normalized_query = query_dense / norm_query
                 normalized_chunk = chunk_dense / norm_chunk
                 dense_score = np.dot(normalized_query, normalized_chunk)
                 dense_score = max(0.0, min(1.0, dense_score)) # 限制在 [0, 1]
             else:
                  logger.debug("Zero vector encountered in dense score calculation.")
         except Exception as e:
              logger.warning(f"Error calculating dense score: {e}", exc_info=True)


         # 稀疏得分 (词汇匹配)
         lexical_score = 0.0
         try:
              # 注意：确保 BGE 模型实例可用
              if self.embedding_model:
                  lexical_score = self.embedding_model.compute_lexical_matching_score(query_sparse, chunk_sparse)
                  lexical_score = max(0.0, min(1.0, lexical_score)) # 限制在 [0, 1]
              else:
                   logger.warning("Embedding model not available for lexical score calculation.")
         except Exception as e:
              logger.warning(f"Error calculating lexical score: {e}", exc_info=True)

         # 混合得分
         hybrid_score = dense_weight * dense_score + (1 - dense_weight) * lexical_score
         # logger.debug(f"Scores - Dense: {dense_score:.4f}, Lexical: {lexical_score:.4f}, Hybrid: {hybrid_score:.4f}")
         return hybrid_score

    def retrieve_chunks(self, query: str) -> List[Dict[str, str]]:
        """
        使用 Hybrid 模式检索与查询相关的知识块。

        Args:
            query (str):用户的查询语句。

        Returns:
            List[Dict[str, str]]: 检索到的知识块列表，每个块是一个字典 {"text": ..., "source": ...}。
                                   列表根据混合得分排序（如果使用 top_k 策略）或过滤（如果使用 threshold 策略）。
        """
        if not self.chunks_data:
            logger.error("Cannot retrieve chunks: Knowledge base chunks are not loaded or created.")
            raise ValueError("Knowledge base chunks are not available.")
        if self.index is None or self.dense_embeddings is None or not self.sparse_weights:
            logger.error("Cannot retrieve chunks: Required components (index, embeddings, weights) are missing.")
            raise ValueError("Knowledge base components are missing for retrieval.")

        logger.info(f"Retrieving chunks for query: '{query[:50]}...'") # 日志中截断长查询

        # 1. 对查询进行编码 (获取稠密和稀疏表示)
        try:
            logger.debug("Encoding query...")
            query_output = self.embedding_model.encode(
                query,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False
            )
            query_dense = query_output.get("dense_vecs")
            query_sparse = query_output.get("lexical_weights")

            if query_dense is None or query_sparse is None:
                logger.error("Failed to encode query properly (missing dense or sparse).")
                raise ValueError("Query encoding failed.")

            query_dense = query_dense.astype(np.float32).reshape(1, -1) # Faiss 需要 2D 输入
            logger.debug("Query encoded successfully.")

        except Exception as e:
            logger.error(f"Error encoding query: {e}", exc_info=True)
            raise RuntimeError(f"Failed to encode query: {e}") from e

        # 2. 初始候选集检索 (Faiss 基于稠密向量)
        # 使用配置的 TOP_K_INITIAL_RETRIEVAL 获取足够多的候选
        num_candidates_to_fetch = min(TOP_K_INITIAL_RETRIEVAL, self.index.ntotal)
        logger.info(f"Performing initial dense retrieval (Faiss) for top {num_candidates_to_fetch} candidates...")
        try:
            # distances, dense_indices = self.index.search(query_dense, num_candidates_to_fetch)
            _, dense_indices = self.index.search(query_dense, num_candidates_to_fetch) # 通常只需要索引
            # Faiss 返回的是 [[idx1, idx2, ...]]
            candidate_indices = dense_indices[0]
            # 过滤掉可能的无效索引 (例如 -1)
            candidate_indices = [i for i in candidate_indices if 0 <= i < len(self.chunks_data)]
            logger.info(f"Found {len(candidate_indices)} initial valid candidates from dense search.")
            if not candidate_indices:
                 logger.warning("No candidates found after initial dense retrieval.")
                 return [] # 没有候选，直接返回空
        except Exception as e:
             logger.error(f"Error during Faiss search: {e}", exc_info=True)
             raise RuntimeError(f"Faiss search failed: {e}") from e

        # 3. 计算候选集的混合分数
        logger.info(f"Calculating hybrid scores for {len(candidate_indices)} candidates...")
        hybrid_scores: List[Tuple[int, float]] = []
        query_dense_1d = query_dense.squeeze() # 用于计算得分
        calculation_errors = 0
        for i in candidate_indices:
            try:
                 # 确保索引有效 (双重检查)
                 if not (0 <= i < len(self.dense_embeddings) and 0 <= i < len(self.sparse_weights)):
                      logger.warning(f"Skipping invalid index {i} during score calculation.")
                      continue

                 chunk_dense = self.dense_embeddings[i]
                 chunk_sparse = self.sparse_weights[i]
                 hybrid_score = self._compute_hybrid_score(query_dense_1d, query_sparse, chunk_dense, chunk_sparse)
                 hybrid_scores.append((i, hybrid_score)) # 保存 (索引, 分数)
            except Exception as e:
                 calculation_errors += 1
                 logger.warning(f"Error calculating hybrid score for chunk index {i}: {e}", exc_info=False) # 避免过多日志

        if calculation_errors > 0:
             logger.warning(f"Encountered {calculation_errors} errors during hybrid score calculation.")
        logger.info(f"Calculated scores for {len(hybrid_scores)} candidates.")

        # 4. 根据策略进行最终筛选
        final_indices: List[int] = []
        if RETRIEVAL_STRATEGY == "threshold":
            logger.info(f"Applying threshold filtering (threshold: {HYBRID_SIMILARITY_THRESHOLD})")
            threshold_results = [(i, score) for i, score in hybrid_scores if score >= HYBRID_SIMILARITY_THRESHOLD]
            logger.info(f"Found {len(threshold_results)} results above threshold.")
            # 按分数排序 (降序)
            threshold_results.sort(key=lambda x: x[1], reverse=True)
            # 应用最大结果数量限制
            if len(threshold_results) > MAX_THRESHOLD_RESULTS:
                logger.info(f"Truncating threshold results from {len(threshold_results)} to {MAX_THRESHOLD_RESULTS}")
                final_results = threshold_results[:MAX_THRESHOLD_RESULTS]
            else:
                final_results = threshold_results
            final_indices = [idx for idx, _ in final_results]

        elif RETRIEVAL_STRATEGY == "top_k":
            # 按混合分数排序，取有效 top K (这里 K 由 MAX_THRESHOLD_RESULTS 控制，名字可能需要调整)
            logger.info(f"Applying top_k filtering (k: {MAX_THRESHOLD_RESULTS}) based on hybrid score")
            hybrid_scores.sort(key=lambda x: x[1], reverse=True)
            top_k_results = hybrid_scores[:MAX_THRESHOLD_RESULTS] # 使用 MAX_THRESHOLD_RESULTS 作为 top_k 的 K 值
            final_indices = [idx for idx, _ in top_k_results]
        else:
            logger.error(f"Unknown RETRIEVAL_STRATEGY: {RETRIEVAL_STRATEGY}. Defaulting to empty result.")
            # 或者可以抛出异常 raise ValueError(...)

        logger.info(f"Selected {len(final_indices)} final chunks based on strategy '{RETRIEVAL_STRATEGY}'.")

        # 5. 根据最终索引列表，从 self.chunks_data 返回结果
        # 再次校验索引有效性 (防御性编程)
        valid_final_indices = [i for i in final_indices if 0 <= i < len(self.chunks_data)]
        if len(valid_final_indices) != len(final_indices):
             logger.warning("Some final indices were invalid after filtering/sorting.")

        final_retrieved_chunks = [self.chunks_data[i] for i in valid_final_indices]
        logger.info(f"Returning {len(final_retrieved_chunks)} chunks.")
        # 可以在这里打印或记录返回的具体块信息（使用 DEBUG 级别）
        # for i, chunk_data in enumerate(final_retrieved_chunks):
        #     logger.debug(f"  Result {i+1}: Source='{chunk_data.get('source', 'N/A')}', Text='{chunk_data.get('text', '')[:100]}...'")

        return final_retrieved_chunks