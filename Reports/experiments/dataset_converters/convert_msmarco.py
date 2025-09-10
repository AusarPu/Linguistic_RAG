#!/usr/bin/env python3
"""
MS MARCO数据集格式转换脚本
将MS MARCO数据集转换为统一格式：{id, question, answer, context}
"""

import json
import os
from pathlib import Path

def convert_msmarco_sample(sample):
    """
    转换单个MS MARCO样本为统一格式
    
    Args:
        sample: MS MARCO原始样本
        
    Returns:
        dict: 统一格式的样本
    """
    # 提取答案（取第一个答案）
    answer = ""
    if 'answers' in sample and sample['answers']:
        answer = sample['answers'][0]
    elif 'wellFormedAnswers' in sample and sample['wellFormedAnswers']:
        answer = sample['wellFormedAnswers'][0]
    
    # 提取context信息，将所有passages合并
    context_parts = []
    if 'passages' in sample:
        passages = sample['passages']
        passage_texts = passages.get('passage_text', [])
        is_selected = passages.get('is_selected', [])
        urls = passages.get('url', [])
        
        for i, (text, selected, url) in enumerate(zip(passage_texts, is_selected, urls)):
            status = "[SELECTED]" if selected else "[CANDIDATE]"
            context_parts.append(f"Passage {i+1} {status}:")
            context_parts.append(f"  {text}")
            if url:
                context_parts.append(f"  Source: {url}")
            context_parts.append("")  # 空行分隔段落
    
    context = "\n".join(context_parts).strip()
    
    # 使用query_id作为id，如果没有则使用query的hash
    sample_id = sample.get('query_id', '')
    if not sample_id:
        sample_id = str(hash(sample.get('query', '')))
    
    return {
        "id": str(sample_id),
        "question": sample.get('query', ''),
        "answer": answer,
        "context": context
    }

def convert_msmarco_dataset(input_file, output_file):
    """
    转换整个MS MARCO数据集文件
    
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
                converted_sample = convert_msmarco_sample(sample)
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
    主函数：转换MS MARCO数据集
    """
    # 设置路径
    base_dir = Path("/home/pushihao/RAG/Reports/experiments")
    input_dir = base_dir / "datasets" / "ms_marco"
    output_dir = base_dir / "dataset_converters" / "converted" / "ms_marco"
    
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
            convert_msmarco_dataset(input_file, output_file)
        else:
            print(f"警告: 输入文件不存在: {input_file}")
    
    print("MS MARCO数据集转换完成！")

if __name__ == "__main__":
    main()