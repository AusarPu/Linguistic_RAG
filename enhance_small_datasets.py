#!/usr/bin/env python3
"""
对小规模数据集进行文本块增强的脚本
调用llm_chunk_processor.py中的元数据生成功能，跳过chunk B优化
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preprocess'))

# 导入LLM处理模块
from preprocess.llm_chunk_processor import enhance_chunks_with_llm_metadata
from script import config_rag as config

# 设置日志
config.setup_logging()
logger = logging.getLogger(__name__)

def main():
    """主函数：处理所有小规模数据集"""
    
    # 定义路径
    input_base_dir = '/home/pushihao/RAG/Reports/experiments/dataset_kb/small_size_kb'
    output_base_dir = '/home/pushihao/RAG/Reports/experiments/dataset_preprocess'
    
    # 数据集列表
    datasets = ['hotpotqa', 'ms_marco', 'natural_questions', 'triviaqa']
    
    logger.warning(f"开始处理小规模数据集的文本块增强...")
    logger.warning(f"输入目录: {input_base_dir}")
    logger.warning(f"输出目录: {output_base_dir}")
    
    # 确保输出目录存在
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 处理每个数据集
    for dataset in datasets:
        input_file = os.path.join(input_base_dir, f'{dataset}_validation_kb_chunks.json')
        output_file = os.path.join(output_base_dir, f'{dataset}_enhanced_validation_kb_chunks.json')
        
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            logger.error(f"输入文件不存在: {input_file}")
            continue
        
        logger.warning(f"\n=== 开始处理数据集: {dataset} ===")
        logger.warning(f"输入文件: {input_file}")
        logger.warning(f"输出文件: {output_file}")
        
        try:
            # 异步调用元数据增强功能（使用动态批处理优化）
            # 注意：这里只调用元数据生成，不进行chunk B优化
            asyncio.run(enhance_chunks_with_llm_metadata(
                input_chunks_json_path=input_file,
                output_chunks_json_path=output_file,
                test_limit=None,  # 处理全部数据，不限制数量
                use_dynamic_batching=True  # 启用动态批处理优化
            ))
            
            logger.warning(f"数据集 {dataset} 处理完成！")
            
        except Exception as e:
            logger.error(f"处理数据集 {dataset} 时发生错误: {e}", exc_info=True)
            continue
    
    logger.warning(f"\n=== 所有数据集处理完成 ===")
    logger.warning(f"增强后的数据集已保存到: {output_base_dir}")

if __name__ == '__main__':
    main()