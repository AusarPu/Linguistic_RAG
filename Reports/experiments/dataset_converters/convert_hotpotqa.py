#!/usr/bin/env python3
"""
HotpotQA数据集格式转换脚本
将HotpotQA数据集转换为统一格式：{id, question, answer, context}
"""

import json
import os
from pathlib import Path

def convert_hotpotqa_sample(sample):
    """
    转换单个HotpotQA样本为统一格式
    
    Args:
        sample: HotpotQA原始样本
        
    Returns:
        dict: 统一格式的样本
    """
    # 提取context信息，将多个文档合并为一个字符串
    context_parts = []
    if 'context' in sample:
        titles = sample['context'].get('title', [])
        sentences = sample['context'].get('sentences', [])
        
        for i, (title, sents) in enumerate(zip(titles, sentences)):
            context_parts.append(f"Document {i+1}: {title}")
            for j, sent in enumerate(sents):
                context_parts.append(f"  {j+1}. {sent}")
            context_parts.append("")  # 空行分隔文档
    
    context = "\n".join(context_parts).strip()
    
    return {
        "id": sample.get('id', ''),
        "question": sample.get('question', ''),
        "answer": sample.get('answer', ''),
        "context": context
    }

def convert_hotpotqa_dataset(input_file, output_file):
    """
    转换整个HotpotQA数据集文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    print(f"正在转换 {input_file} -> {output_file}")
    
    converted_samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                converted_sample = convert_hotpotqa_sample(sample)
                converted_samples.append(converted_sample)
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析失败: {e}")
                continue
            except Exception as e:
                print(f"警告: 第{line_num}行处理失败: {e}")
                continue
    
    # 写入转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in converted_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"转换完成: {len(converted_samples)} 个样本")

def main():
    """
    主函数：转换HotpotQA数据集
    """
    # 设置路径
    base_dir = Path("/home/pushihao/RAG/Reports/experiments")
    input_dir = base_dir / "datasets" / "hotpotqa"
    output_dir = base_dir / "dataset_converters" / "converted" / "hotpotqa"
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换训练集和验证集
    datasets = {
        "train.json": "train_converted.json",
        "validation.json": "validation_converted.json"
    }
    
    for input_name, output_name in datasets.items():
        input_file = input_dir / input_name
        output_file = output_dir / output_name
        
        if input_file.exists():
            convert_hotpotqa_dataset(input_file, output_file)
        else:
            print(f"警告: 输入文件不存在: {input_file}")
    
    print("HotpotQA数据集转换完成！")

if __name__ == "__main__":
    main()