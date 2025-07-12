# preprocess/build_core_indexes.py

import faiss
import numpy as np
import json
import pickle
from pathlib import Path
import os
import sys
import time
import logging
from typing import List, Dict, Optional, Set, Any
from tqdm import tqdm
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from script.config import *


import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry



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

def call_embedding_api(texts: List[str], api_url: str, model_name: str, max_retries: int = 3) -> np.ndarray:
    """
    调用嵌入API获取向量
    
    Args:
        texts: 待编码的文本列表
        api_url: API地址
        model_name: 模型名称
        max_retries: 最大重试次数
    
    Returns:
        np.ndarray: 嵌入向量数组
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    payload = {
        "model": model_name,
        "input": texts
    }
    
    try:
        response = session.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        if "data" not in result:
            raise ValueError(f"API响应格式错误: {result}")
        
        # 提取嵌入向量
        embeddings = []
        for item in result["data"]:
            if "embedding" in item:
                embeddings.append(item["embedding"])
            else:
                raise ValueError(f"API响应中缺少embedding字段: {item}")
        
        return np.array(embeddings, dtype=np.float32)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API调用失败: {e}")
        raise
    except Exception as e:
        logger.error(f"处理API响应时出错: {e}")
        raise

def build_all_search_indexes(
    enhanced_chunks_path: str,
    embedding_model_name_or_path: str,
    output_dir: str,
    # 块文本相关文件名
    chunk_dense_emb_filename: str,
    chunk_faiss_idx_filename: str,
    indexed_chunks_meta_filename: str,
    # 关键词短语相关文件名
    phrase_dense_map_filename: str,
    # 预生成问题相关文件名
    question_dense_emb_filename: str,
    question_faiss_idx_filename: str,
    question_to_chunk_id_map_filename: str,
    question_texts_list_filename: str,
    batch_size_embed: int = 128,
    batch_size_phrases: int = 2048,
    batch_size_questions: int = 1024 # 为问题编码新增批大小
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

    # 1. 准备API调用参数
    try:
        logger.info(f"准备使用嵌入API: {EMBEDDING_API_URL}")
        logger.info(f"使用模型: {EMBEDDING_MODEL_NAME_FOR_API}")
        # 测试API连接
        test_response = requests.get(EMBEDDING_API_URL.replace('/v1/embeddings', '/health'), timeout=10)
        logger.info("嵌入API连接测试成功。")
    except Exception as e:
        logger.warning(f"嵌入API连接测试失败，但将继续尝试: {e}")

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
            # 计算总批次数
            total_batches = (len(texts_for_chunk_dense_embedding) + batch_size_embed - 1) // batch_size_embed
            
            # 使用tqdm显示进度条
            with tqdm(total=total_batches, desc="块文本编码", unit="batch") as pbar:
                for i in range(0, len(texts_for_chunk_dense_embedding), batch_size_embed):
                    batch_texts = texts_for_chunk_dense_embedding[i:i+batch_size_embed]
                    batch_num = i // batch_size_embed + 1
                    pbar.set_postfix({"批次": f"{batch_num}/{total_batches}", "文本数": len(batch_texts)})
                    
                    batch_embeddings = call_embedding_api(batch_texts, EMBEDDING_API_URL, EMBEDDING_MODEL_NAME_FOR_API)
                    all_dense_vecs_list.append(batch_embeddings)
                    pbar.update(1)

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


    # --- 3. 为唯一关键词短语编码稠密向量 ---
    phrase_to_dense_embeddings_map: Dict[str, np.ndarray] = {}
    if all_unique_keyword_phrases:
        logger.info(f"开始为 {len(all_unique_keyword_phrases)} 个唯一关键词短语生成稠密向量...")
        unique_phrases_list_for_encoding = list(all_unique_keyword_phrases)
        try:
            all_phrase_dense_embeddings = []
            # 计算总批次数
            total_phrase_batches = (len(unique_phrases_list_for_encoding) + batch_size_phrases - 1) // batch_size_phrases
            
            # 使用tqdm显示进度条
            with tqdm(total=total_phrase_batches, desc="关键词编码", unit="batch") as pbar:
                for i in range(0, len(unique_phrases_list_for_encoding), batch_size_phrases):
                    batch_phrases = unique_phrases_list_for_encoding[i:i+batch_size_phrases]
                    batch_num = i // batch_size_phrases + 1
                    pbar.set_postfix({"批次": f"{batch_num}/{total_phrase_batches}", "短语数": len(batch_phrases)})
                    
                    batch_embeddings = call_embedding_api(batch_phrases, EMBEDDING_API_URL, EMBEDDING_MODEL_NAME_FOR_API)
                    all_phrase_dense_embeddings.append(batch_embeddings)
                    pbar.update(1)
            
            # 保存稠密向量映射
            phrase_dense_embeddings_np = np.vstack(all_phrase_dense_embeddings).astype(np.float32)
            
            logger.info("创建关键词短语到稠密向量的映射...")
            with tqdm(total=len(unique_phrases_list_for_encoding), desc="向量映射", unit="短语") as pbar:
                for i, phrase_str in enumerate(unique_phrases_list_for_encoding):
                    dense_vec = phrase_dense_embeddings_np[i]
                    phrase_to_dense_embeddings_map[phrase_str] = dense_vec
                    pbar.update(1)
            
            logger.info("唯一关键词短语的稠密向量映射生成完成。")

            with open(output_path / phrase_dense_map_filename, "wb") as f:
                pickle.dump(phrase_to_dense_embeddings_map, f)
            logger.info(f"关键词短语稠密向量映射已保存到: {output_path / phrase_dense_map_filename}")
        except Exception as e:
            logger.error(f"为关键词短语编码向量或保存映射时出错: {e}", exc_info=True)
    else:
        logger.warning("没有找到唯一关键词短语，未生成向量映射。")

    # --- 4. 为所有预生成问题编码稠密向量并构建Faiss索引 (使用IndexFlatIP) ---
    if all_question_texts_flat:
        logger.info(f"开始为 {len(all_question_texts_flat)} 个预生成问题生成稠密向量...")
        all_question_dense_vecs_list = []
        try:
            # 计算总批次数
            total_question_batches = (len(all_question_texts_flat) + batch_size_questions - 1) // batch_size_questions
            
            # 使用tqdm显示进度条
            with tqdm(total=total_question_batches, desc="问题编码", unit="batch") as pbar:
                for i in range(0, len(all_question_texts_flat), batch_size_questions):
                    batch_q_texts = all_question_texts_flat[i:i+batch_size_questions]
                    batch_num = i // batch_size_questions + 1
                    pbar.set_postfix({"批次": f"{batch_num}/{total_question_batches}", "问题数": len(batch_q_texts)})
                    
                    batch_embeddings = call_embedding_api(batch_q_texts, EMBEDDING_API_URL, EMBEDDING_MODEL_NAME_FOR_API)
                    all_question_dense_vecs_list.append(batch_embeddings)
                    pbar.update(1)

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

def build_test_indexes_with_limit(enhanced_chunks_path, test_limit=None):
    """
    测试模式：构建小批量索引用于测试API连接和功能验证
    
    Args:
        enhanced_chunks_path (str): 输入的增强文本块JSON文件路径
        test_limit (int, optional): 测试模式下限制处理的块数量
    """
    import json
    import tempfile
    import os
    
    logger.info(f"=========== 开始测试模式索引构建 (限制: {test_limit or '无限制'}) ===========")
    
    # 创建临时输出目录
    temp_output_dir = tempfile.mkdtemp(prefix="test_indexes_")
    logger.info(f"测试输出目录: {temp_output_dir}")
    
    try:
        # 如果指定了测试限制，创建临时的小批量数据文件
        if test_limit:
            logger.info(f"加载原始数据并限制为前 {test_limit} 个块...")
            with open(enhanced_chunks_path, 'r', encoding='utf-8') as f:
                all_chunks = json.load(f)
            
            # 限制数据量
            limited_chunks = all_chunks[:test_limit]
            logger.info(f"原始块数: {len(all_chunks)}, 测试块数: {len(limited_chunks)}")
            
            # 创建临时输入文件
            temp_input_file = os.path.join(temp_output_dir, "test_chunks.json")
            with open(temp_input_file, 'w', encoding='utf-8') as f:
                json.dump(limited_chunks, f, ensure_ascii=False, indent=2)
            
            input_file_path = temp_input_file
        else:
            input_file_path = enhanced_chunks_path
        
        # 构建索引
        build_all_search_indexes(
            enhanced_chunks_path=input_file_path,
            embedding_model_name_or_path=EMBEDDING_MODEL_PATH,
            output_dir=temp_output_dir,
            chunk_dense_emb_filename="test_dense_embeddings_chunks.npy",
            chunk_faiss_idx_filename="test_faiss_index_chunks_ip.idx",
            indexed_chunks_meta_filename="test_indexed_chunks_metadata.json",
            phrase_dense_map_filename="test_phrase_dense_embeddings_map.pkl",
            question_dense_emb_filename="test_dense_embeddings_questions.npy",
            question_faiss_idx_filename="test_faiss_index_questions_ip.idx",
            question_to_chunk_id_map_filename="test_question_index_to_chunk_id_map.json",
            question_texts_list_filename="test_all_question_texts.json",
        )
        
        # 验证生成的文件
        logger.info("\n=== 测试结果验证 ===")
        output_files = [
            "test_dense_embeddings_chunks.npy",
            "test_faiss_index_chunks_ip.idx", 
            "test_indexed_chunks_metadata.json",
            "test_phrase_dense_embeddings_map.pkl",
            "test_dense_embeddings_questions.npy",
            "test_faiss_index_questions_ip.idx",
            "test_question_index_to_chunk_id_map.json",
            "test_all_question_texts.json"
        ]
        
        for filename in output_files:
            filepath = os.path.join(temp_output_dir, filename)
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                logger.info(f"✓ {filename}: {file_size} bytes")
            else:
                logger.warning(f"✗ {filename}: 文件未生成")
        
        logger.info(f"\n测试完成！临时文件保存在: {temp_output_dir}")
        logger.info("如需清理测试文件，请手动删除上述目录")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="构建核心搜索索引")
    parser.add_argument("--test", action="store_true", help="启用测试模式")
    parser.add_argument("--test-limit", type=int, default=10, help="测试模式下限制处理的块数量（默认：10）")
    parser.add_argument("--input-file", type=str, help="自定义输入文件路径（可选）")
    
    args = parser.parse_args()
    
    # 设置日志级别
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 确定输入文件路径
    input_file = args.input_file or ENHANCED_CHUNKS_JSON_PATH
    
    # 检查输入文件是否存在
    if not Path(input_file).is_file():
        logger.error(f"错误：输入文件 '{input_file}' 未找到。")
        exit(1)
    
    if args.test:
        # 测试模式
        logger.info(f"启动测试模式，限制处理 {args.test_limit} 个块")
        build_test_indexes_with_limit(input_file, test_limit=args.test_limit)
    else:
        # 正常模式
        logger.info("=========== 开始构建所有核心搜索索引 ===========")
        Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
        
        build_all_search_indexes(
            enhanced_chunks_path=input_file,
            embedding_model_name_or_path=EMBEDDING_MODEL_PATH,
            output_dir=PROCESSED_DATA_DIR,
            chunk_dense_emb_filename="dense_embeddings_chunks.npy",
            chunk_faiss_idx_filename="faiss_index_chunks_ip.idx",
            indexed_chunks_meta_filename="indexed_chunks_metadata.json",
            phrase_dense_map_filename="phrase_dense_embeddings_map.pkl",
            question_dense_emb_filename="dense_embeddings_questions.npy",
            question_faiss_idx_filename="faiss_index_questions_ip.idx",
            question_to_chunk_id_map_filename="question_index_to_chunk_id_map.json",
            question_texts_list_filename=ALL_QUESTION_TEXTS_SAVE_PATH,
        )
        logger.info("=========== 所有核心搜索索引构建完成 ===========")