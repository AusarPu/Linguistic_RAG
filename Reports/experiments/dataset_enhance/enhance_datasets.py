#!/usr/bin/env python3
"""
数据集增强脚本
使用 llm_chunk_processor.py 对分块后的数据集进行优化和元数据生成
支持三种模式：优化、元数据生成、流水线模式
"""

import os
import sys
import json
import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录到路径
sys.path.append("/home/pushihao/RAG")

# 导入配置和处理器
from script import config_rag as config
from preprocess.llm_chunk_processor import refine_all_chunks_with_llm, enhance_chunks_with_llm_metadata

# 设置日志
config.setup_logging()
logger = logging.getLogger(__name__)

# 目录配置，支持环境变量覆盖以隔离不同运行
CHUNKED_DIR = os.environ.get("ENHANCE_INPUT_DIR", "/home/pushihao/RAG/Reports/experiments/datasets/chunked")
ENHANCED_DIR = os.environ.get("ENHANCE_OUTPUT_DIR", "/home/pushihao/RAG/Reports/experiments/datasets/enhanced")

def ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(ENHANCED_DIR, exist_ok=True)
    logger.info(f"输出目录已准备: {ENHANCED_DIR}")

def get_chunked_files() -> List[str]:
    """获取所有分块文件的路径"""
    chunked_files = []
    if not os.path.exists(CHUNKED_DIR):
        logger.error(f"分块目录不存在: {CHUNKED_DIR}")
        return chunked_files
    
    for filename in os.listdir(CHUNKED_DIR):
        if filename.endswith('_chunked.json'):
            file_path = os.path.join(CHUNKED_DIR, filename)
            chunked_files.append(file_path)
            logger.info(f"找到分块文件: {filename}")
    
    return chunked_files

def get_output_filename(input_path: str, mode: str) -> str:
    """根据输入文件和模式生成输出文件名"""
    basename = os.path.basename(input_path)
    dataset_name = basename.replace('_chunked.json', '')
    
    if mode == 'optimize':
        return f"{dataset_name}_optimized.json"
    elif mode == 'metadata':
        return f"{dataset_name}_enhanced.json"
    elif mode == 'pipeline':
        return f"{dataset_name}_enhanced.json"
    else:
        return f"{dataset_name}_processed.json"

async def process_single_file(input_path: str, mode: str, test_limit: Optional[int] = None) -> str:
    """处理单个文件"""
    dataset_name = os.path.basename(input_path).replace('_chunked.json', '')
    logger.info(f"开始处理数据集: {dataset_name}")
    
    if mode == 'optimize':
        # 仅优化模式
        output_filename = get_output_filename(input_path, 'optimize')
        output_path = os.path.join(ENHANCED_DIR, output_filename)
        
        logger.info(f"执行文本块优化: {dataset_name}")
        await refine_all_chunks_with_llm(
            input_chunks_json_path=input_path,
            output_refined_chunks_json_path=output_path,
            limit=test_limit
        )
        logger.info(f"优化完成: {output_path}")
        return output_path
        
    elif mode == 'metadata':
        # 仅元数据生成模式
        output_filename = get_output_filename(input_path, 'metadata')
        output_path = os.path.join(ENHANCED_DIR, output_filename)
        
        logger.info(f"执行元数据生成: {dataset_name}")
        await enhance_chunks_with_llm_metadata(
            input_chunks_json_path=input_path,
            output_chunks_json_path=output_path,
            test_limit=test_limit,
            use_dynamic_batching=False
        )
        logger.info(f"元数据生成完成: {output_path}")
        return output_path
        
    elif mode == 'pipeline':
        # 流水线模式：先优化，再生成元数据
        # 步骤1：优化
        optimized_filename = f"{dataset_name}_optimized.json"
        optimized_path = os.path.join(ENHANCED_DIR, optimized_filename)
        
        logger.info(f"流水线步骤1 - 文本块优化: {dataset_name}")
        await refine_all_chunks_with_llm(
            input_chunks_json_path=input_path,
            output_refined_chunks_json_path=optimized_path,
            limit=test_limit
        )
        logger.info(f"优化完成: {optimized_path}")
        
        # 步骤2：元数据生成
        enhanced_filename = get_output_filename(input_path, 'pipeline')
        enhanced_path = os.path.join(ENHANCED_DIR, enhanced_filename)
        
        logger.info(f"流水线步骤2 - 元数据生成: {dataset_name}")
        await enhance_chunks_with_llm_metadata(
            input_chunks_json_path=optimized_path,
            output_chunks_json_path=enhanced_path,
            test_limit=test_limit,
            use_dynamic_batching=False
        )
        logger.info(f"元数据生成完成: {enhanced_path}")
        
        # 可选：删除中间文件（优化后的文件）
        # os.remove(optimized_path)
        # logger.info(f"已删除中间文件: {optimized_path}")
        
        return enhanced_path
    
    else:
        raise ValueError(f"不支持的模式: {mode}")

async def process_all_datasets(mode: str, test_limit: Optional[int] = None, specific_dataset: Optional[str] = None):
    """处理所有数据集或指定数据集"""
    ensure_output_dir()
    
    chunked_files = get_chunked_files()
    if not chunked_files:
        logger.error("未找到任何分块文件")
        return
    
    # 如果指定了特定数据集，过滤文件列表
    if specific_dataset:
        chunked_files = [f for f in chunked_files if specific_dataset in os.path.basename(f)]
        if not chunked_files:
            logger.error(f"未找到数据集: {specific_dataset}")
            return
    
    logger.info(f"将处理 {len(chunked_files)} 个数据集文件，模式: {mode}")
    
    processed_files = []
    for input_path in chunked_files:
        try:
            output_path = await process_single_file(input_path, mode, test_limit)
            processed_files.append(output_path)
        except Exception as e:
            logger.error(f"处理文件 {input_path} 时出错: {e}")
            continue
    
    logger.info(f"处理完成！共处理 {len(processed_files)} 个文件")
    for output_path in processed_files:
        logger.info(f"输出文件: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="数据集增强脚本 - 使用LLM优化文本块和生成元数据")
    parser.add_argument(
        "mode", 
        choices=["optimize", "metadata", "pipeline"], 
        help="处理模式：optimize=仅优化文本块，metadata=仅生成元数据，pipeline=完整流水线（先优化再生成元数据）"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        help="指定要处理的数据集名称（如：hotpotqa），不指定则处理所有数据集"
    )
    parser.add_argument(
        "--test-limit", 
        type=int, 
        default=None, 
        help="测试模式：限制处理的块数量（用于测试，不指定则处理全部）"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="启用详细日志输出"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info(f"开始数据集增强处理")
    logger.info(f"模式: {args.mode}")
    logger.info(f"数据集: {args.dataset if args.dataset else '全部'}")
    logger.info(f"测试限制: {args.test_limit if args.test_limit else '无限制'}")
    
    # 运行处理
    asyncio.run(process_all_datasets(
        mode=args.mode,
        test_limit=args.test_limit,
        specific_dataset=args.dataset
    ))
    
    logger.info("数据集增强处理完成")

if __name__ == "__main__":
    main()