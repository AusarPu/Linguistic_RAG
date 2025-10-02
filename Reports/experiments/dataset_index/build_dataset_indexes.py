#!/usr/bin/env python3
"""
构建数据集索引脚本
从enhanced数据集构建独立的知识库索引，每个数据集对应一个知识库文件夹
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from preprocess.build_core_indexes import build_all_search_indexes
from script.config_rag import *

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据集配置
DATASETS = [
    "hotpotqa",
    "ms_marco", 
    "natural_questions",
    "triviaqa"
]

# 路径配置
ENHANCED_DATA_DIR = "/home/pushihao/RAG/Reports/experiments/datasets/enhanced"
OUTPUT_BASE_DIR = "/home/pushihao/RAG/Reports/experiments/datasets/knowledge_bases"

def get_enhanced_file_path(dataset_name: str) -> str:
    """获取增强数据集文件路径"""
    return os.path.join(ENHANCED_DATA_DIR, f"{dataset_name}_enhanced.json")

def get_output_dir(dataset_name: str) -> str:
    """获取输出目录路径"""
    return os.path.join(OUTPUT_BASE_DIR, dataset_name)

def validate_enhanced_data(file_path: str) -> bool:
    """验证增强数据文件是否存在且格式正确"""
    if not os.path.exists(file_path):
        logger.error(f"增强数据文件不存在: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) == 0:
            logger.error(f"增强数据文件格式错误或为空: {file_path}")
            return False
        
        # 检查第一个条目的关键字段
        sample = data[0]
        required_keys = ['text', 'chunk_id', 'keyword_summaries', 'generated_questions', 'is_meaningful']
        missing_keys = [key for key in required_keys if key not in sample]
        
        if missing_keys:
            logger.warning(f"数据中缺少关键字段 {missing_keys}，但将继续处理: {file_path}")
        
        logger.info(f"验证通过: {file_path} (包含 {len(data)} 个条目)")
        return True
        
    except Exception as e:
        logger.error(f"验证增强数据文件时出错 {file_path}: {e}")
        return False

def build_single_dataset_index(dataset_name: str, test_mode: bool = False, test_limit: int = 10) -> bool:
    """为单个数据集构建索引"""
    logger.info(f"========== 开始构建 {dataset_name} 数据集索引 ==========")
    
    # 获取文件路径
    enhanced_file = get_enhanced_file_path(dataset_name)
    output_dir = get_output_dir(dataset_name)
    
    # 验证输入文件
    if not validate_enhanced_data(enhanced_file):
        return False
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    try:
        # 如果是测试模式，先创建测试数据
        if test_mode:
            logger.info(f"测试模式: 限制处理 {test_limit} 个条目")
            with open(enhanced_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            
            test_data = all_data[:test_limit]
            test_file = os.path.join(output_dir, f"{dataset_name}_test_enhanced.json")
            
            with open(test_file, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            
            enhanced_file = test_file
            logger.info(f"创建测试数据文件: {test_file}")
        
        # 构建索引
        build_all_search_indexes(
            enhanced_chunks_path=enhanced_file,
            embedding_model_name_or_path=EMBEDDING_MODEL_PATH,
            output_dir=output_dir,
            # 块文本相关文件名
            chunk_dense_emb_filename="dense_embeddings_chunks.npy",
            chunk_faiss_idx_filename="faiss_index_chunks_ip.idx", 
            indexed_chunks_meta_filename="indexed_chunks_metadata.json",
            chunk_bm25_index_filename="chunk_bm25_index.pkl",
            # 关键词短语相关文件名
            phrase_dense_map_filename="phrase_dense_embeddings_map.pkl",
            phrase_bm25_index_filename="phrase_bm25_index.pkl",
            # 预生成问题相关文件名
            question_dense_emb_filename="dense_embeddings_questions.npy",
            question_faiss_idx_filename="faiss_index_questions_ip.idx",
            question_to_chunk_id_map_filename="question_index_to_chunk_id_map.json",
            question_texts_list_filename="all_question_texts.json",
            # 批处理大小
            batch_size_embed=128,
            batch_size_phrases=2048,
            batch_size_questions=1024
        )
        
        logger.info(f"========== {dataset_name} 数据集索引构建完成 ==========")
        return True
        
    except Exception as e:
        logger.error(f"构建 {dataset_name} 索引时出错: {e}", exc_info=True)
        return False

def build_all_dataset_indexes(test_mode: bool = False, test_limit: int = 10, datasets: List[str] = None) -> Dict[str, bool]:
    """构建所有数据集的索引"""
    if datasets is None:
        datasets = DATASETS
    
    logger.info(f"开始构建 {len(datasets)} 个数据集的索引...")
    logger.info(f"数据集列表: {datasets}")
    
    results = {}
    
    for dataset_name in datasets:
        try:
            success = build_single_dataset_index(dataset_name, test_mode, test_limit)
            results[dataset_name] = success
            
            if success:
                logger.info(f"✓ {dataset_name} 索引构建成功")
            else:
                logger.error(f"✗ {dataset_name} 索引构建失败")
                
        except Exception as e:
            logger.error(f"处理数据集 {dataset_name} 时出现异常: {e}", exc_info=True)
            results[dataset_name] = False
    
    # 输出总结
    logger.info("========== 索引构建总结 ==========")
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]
    
    logger.info(f"成功构建索引的数据集 ({len(successful)}): {successful}")
    if failed:
        logger.error(f"构建索引失败的数据集 ({len(failed)}): {failed}")
    
    return results

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="构建数据集索引")
    parser.add_argument("--test", action="store_true", help="启用测试模式")
    parser.add_argument("--test-limit", type=int, default=10, help="测试模式下限制处理的条目数量（默认：10）")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, help="指定要处理的数据集")
    parser.add_argument("--single", type=str, choices=DATASETS, help="只处理单个数据集")
    
    args = parser.parse_args()
    
    # 确定要处理的数据集
    if args.single:
        datasets_to_process = [args.single]
    elif args.datasets:
        datasets_to_process = args.datasets
    else:
        datasets_to_process = DATASETS
    
    logger.info("========== 数据集索引构建工具 ==========")
    logger.info(f"测试模式: {'是' if args.test else '否'}")
    if args.test:
        logger.info(f"测试限制: {args.test_limit} 个条目")
    logger.info(f"要处理的数据集: {datasets_to_process}")
    logger.info(f"输出基础目录: {OUTPUT_BASE_DIR}")
    
    # 构建索引
    results = build_all_dataset_indexes(
        test_mode=args.test,
        test_limit=args.test_limit,
        datasets=datasets_to_process
    )
    
    # 返回状态码
    all_success = all(results.values())
    if all_success:
        logger.info("所有数据集索引构建成功！")
        sys.exit(0)
    else:
        logger.error("部分数据集索引构建失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()