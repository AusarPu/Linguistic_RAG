#!/usr/bin/env python3
"""
从数据集验证集中随机提取1000条数据的脚本
"""

import json
import random
import os
from pathlib import Path

def extract_sample_data(input_file, output_file, sample_size=1000):
    """
    从输入文件中随机提取指定数量的数据并保存到输出文件
    
    Args:
        input_file (str): 输入JSON文件路径
        output_file (str): 输出JSON文件路径
        sample_size (int): 要提取的样本数量
    """
    print(f"正在处理: {input_file}")
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据量: {len(data)}")
    
    # 如果数据量少于要求的样本数量，则取全部数据
    if len(data) <= sample_size:
        sampled_data = data
        print(f"数据量不足{sample_size}条，取全部{len(data)}条数据")
    else:
        # 随机采样
        sampled_data = random.sample(data, sample_size)
        print(f"随机提取了{sample_size}条数据")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存采样数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    
    print(f"数据已保存到: {output_file}\n")

def main():
    # 设置随机种子以确保结果可重现
    random.seed(42)
    
    # 定义数据集名称
    datasets = ['hotpotqa', 'ms_marco', 'natural_questions', 'triviaqa']
    
    # 定义路径
    base_input_dir = '/home/pushihao/RAG/Reports/experiments/dataset_kb/kb'
    base_output_dir = '/home/pushihao/RAG/Reports/experiments/dataset_kb/small_size_kb'
    
    print("开始提取数据集样本...\n")
    
    for dataset in datasets:
        # 构建输入和输出文件路径
        input_file = os.path.join(base_input_dir, dataset, 'validation_kb_chunks.json')
        output_file = os.path.join(base_output_dir, f'{dataset}_validation_kb_chunks.json')
        
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"警告: 输入文件不存在 - {input_file}")
            continue
        
        # 提取样本数据
        try:
            extract_sample_data(input_file, output_file, sample_size=1000)
        except Exception as e:
            print(f"处理{dataset}时出错: {str(e)}\n")
    
    print("所有数据集处理完成!")

if __name__ == '__main__':
    main()