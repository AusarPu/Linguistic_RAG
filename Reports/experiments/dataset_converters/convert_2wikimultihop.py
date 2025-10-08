#!/usr/bin/env python3
"""
2WikiMultiHop数据集格式转换脚本
将2WikiMultiHop数据集转换为统一格式：{id, question, answer, context}
"""

import argparse
import json
import os
from pathlib import Path
import sys

# 导入统一配置
sys.path.append(str(Path(__file__).parent))
from convert_all import get_output_dir, get_output_filename

def convert_2wikimultihop_sample(sample):
    """
    转换单个2WikiMultiHop样本为统一格式
    
    Args:
        sample: 2WikiMultiHop原始样本
        
    Returns:
        dict: 统一格式的样本
    """
    # 提取context信息，将多个文档合并为一个字符串
    context_parts = []
    if 'context' in sample:
        for i, (title, content) in enumerate(sample['context']):
            context_parts.append(f"Document {i+1}: {title}")
            # content是一个包含段落的列表
            if isinstance(content, list):
                for j, paragraph in enumerate(content):
                    context_parts.append(f"  {j+1}. {paragraph}")
            else:
                context_parts.append(f"  1. {content}")
            context_parts.append("")  # 空行分隔文档
    
    context = "\n".join(context_parts).strip()
    
    return {
        "id": sample.get('_id', ''),
        "question": sample.get('question', ''),
        "answer": sample.get('answer', ''),
        "context": context
    }

def convert_2wikimultihop_dataset(input_file, output_file, max_samples=None, filter_no_answer=True):
    """
    转换整个2WikiMultiHop数据集文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        max_samples: 最大样本数量，None表示不限制
        filter_no_answer: 是否过滤没有答案的数据
    """
    print(f"正在转换 {input_file} -> {output_file}")
    if max_samples:
        print(f"限制样本数量: {max_samples}")
    if filter_no_answer:
        print("过滤模式: 丢弃没有答案的数据")
    
    # 使用流式处理，从头开始逐行读取
    from streaming_processor import process_data_streaming
    
    converted_samples, stats = process_data_streaming(
        input_file, 
        convert_2wikimultihop_sample, 
        max_samples, 
        filter_no_answer
    )
    
    # 输出统计信息
    print(f"处理了 {stats['total_processed']} 个样本")
    print(f"成功转换 {stats['converted_count']} 个样本")
    if filter_no_answer:
        print(f"过滤掉 {stats['filtered_count']} 个没有答案的样本")
    if stats.get('shortage'):
        print(f"注意: 数据不足，缺少 {stats['shortage']} 个样本")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_samples, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成: {len(converted_samples)} 个样本")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='转换2WikiMultiHop数据集为统一格式')
    parser.add_argument('--input-dir', type=str, 
                       default='/home/pushihao/RAG/Reports/experiments/datasets/2wikimultihop/data',
                       help='输入数据目录')
    parser.add_argument('--output-dir', type=str, 
                       default=None,
                       help='输出目录 (默认使用统一配置)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='每个数据集的最大样本数量')
    parser.add_argument('--no-filter', action='store_true',
                       help='不过滤没有答案的数据')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 使用统一配置的输出目录
    output_dir = args.output_dir or get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    
    input_dir = Path(args.input_dir)
    filter_no_answer = not args.no_filter
    
    print("=" * 50)
    print("2WikiMultiHop数据集转换")
    print("=" * 50)
    
    # 处理train数据集
    train_file = input_dir / "train.json"
    if train_file.exists():
        output_file = Path(output_dir) / "2wikimultihop_train_kb_chunks.json"
        convert_2wikimultihop_dataset(
            str(train_file), 
            str(output_file), 
            args.max_samples, 
            filter_no_answer
        )
        print()
    else:
        print(f"警告: 找不到训练数据文件 {train_file}")
    
    # 处理dev数据集，重命名为validation
    dev_file = input_dir / "dev.json"
    if dev_file.exists():
        output_file = Path(output_dir) / "2wikimultihop_validation_kb_chunks.json"
        convert_2wikimultihop_dataset(
            str(dev_file), 
            str(output_file), 
            args.max_samples, 
            filter_no_answer
        )
        print()
    else:
        print(f"警告: 找不到开发数据文件 {dev_file}")
    
    # 跳过test数据集（按要求去除）
    test_file = input_dir / "test.json"
    if test_file.exists():
        print(f"跳过测试数据文件 {test_file} (按要求去除)")
    
    print("2WikiMultiHop数据集转换完成!")

if __name__ == "__main__":
    main()