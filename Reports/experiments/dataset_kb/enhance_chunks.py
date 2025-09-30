#!/usr/bin/env python3
"""
文本块增强脚本
使用LLM对知识库文本块进行优化和元数据生成
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent  # 需要回到RAG目录
sys.path.insert(0, str(project_root))

# 检查并添加必要的路径
preprocess_path = project_root / "preprocess"
if preprocess_path.exists():
    sys.path.insert(0, str(preprocess_path.parent))

try:
    from preprocess.llm_chunk_processor import enhance_chunks_with_llm_metadata, refine_all_chunks_with_llm
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"项目根目录: {project_root}")
    print(f"预处理路径: {preprocess_path}")
    print("请确保VLLM服务已启动且配置正确")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 数据集配置
DATASETS = {
    'triviaqa': {
        'input_file': 'triviaqa/validation_kb_chunks_1k.json',
        'optimized_file': 'triviaqa/validation_kb_chunks_1k_optimized.json',
        'enhanced_file': 'triviaqa/validation_kb_chunks_1k_enhanced.json'
    },
    'hotpotqa': {
        'input_file': 'hotpotqa/validation_kb_chunks_1k.json',
        'optimized_file': 'hotpotqa/validation_kb_chunks_1k_optimized.json',
        'enhanced_file': 'hotpotqa/validation_kb_chunks_1k_enhanced.json'
    },
    'ms_marco': {
        'input_file': 'ms_marco/validation_kb_chunks_1k.json',
        'optimized_file': 'ms_marco/validation_kb_chunks_1k_optimized.json',
        'enhanced_file': 'ms_marco/validation_kb_chunks_1k_enhanced.json'
    },
    'natural_questions': {
        'input_file': 'natural_questions/validation_kb_chunks_1k.json',
        'optimized_file': 'natural_questions/validation_kb_chunks_1k_optimized.json',
        'enhanced_file': 'natural_questions/validation_kb_chunks_1k_enhanced.json'
    }
}

async def enhance_dataset_chunks(dataset_name, config, test_limit=None):
    """增强单个数据集的文本块"""
    logger.info(f"开始处理数据集: {dataset_name}")
    
    input_path = config['input_file']
    optimized_path = config['optimized_file']
    enhanced_path = config['enhanced_file']
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        logger.error(f"输入文件不存在: {input_path}")
        return False
    
    try:
        # 步骤1: 优化文本块
        logger.info(f"步骤1: 优化文本块 {input_path} -> {optimized_path}")
        await refine_all_chunks_with_llm(
            input_path,
            optimized_path,
            limit=test_limit
        )
        
        # 步骤2: 生成元数据
        logger.info(f"步骤2: 生成元数据 {optimized_path} -> {enhanced_path}")
        await enhance_chunks_with_llm_metadata(
            optimized_path,
            enhanced_path,
            test_limit=test_limit,
            use_dynamic_batching=True
        )
        
        logger.info(f"数据集 {dataset_name} 处理完成")
        return True
        
    except Exception as e:
        logger.error(f"处理数据集 {dataset_name} 时出错: {e}")
        return False

async def enhance_all_datasets(test_limit=None):
    """增强所有数据集的文本块"""
    logger.info("开始增强所有数据集的文本块")
    
    success_count = 0
    total_count = len(DATASETS)
    
    for dataset_name, config in DATASETS.items():
        success = await enhance_dataset_chunks(dataset_name, config, test_limit)
        if success:
            success_count += 1
    
    logger.info(f"文本块增强完成: {success_count}/{total_count} 个数据集处理成功")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="增强知识库文本块")
    parser.add_argument('--dataset', choices=list(DATASETS.keys()) + ['all'], 
                       default='all', help='要处理的数据集')
    parser.add_argument('--test-limit', type=int, help='测试模式：限制处理的块数量')
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        asyncio.run(enhance_all_datasets(args.test_limit))
    else:
        config = DATASETS[args.dataset]
        asyncio.run(enhance_dataset_chunks(args.dataset, config, args.test_limit))

if __name__ == '__main__':
    main()
