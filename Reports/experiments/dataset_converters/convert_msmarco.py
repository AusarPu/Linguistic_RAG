#!/usr/bin/env python3
"""
MS MARCO数据集格式转换脚本
将MS MARCO数据集转换为统一格式：{id, question, answer, context}
"""

import argparse
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

def convert_msmarco_dataset(input_file, output_file, max_samples=None, filter_no_answer=True):
    """
    转换整个MS MARCO数据集文件
    
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
    
    # MS MARCO 使用 JSONL 格式（每行一个 JSON 对象）
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"总共读取 {len(data)} 个样本")
    
    converted_samples = []
    processed_count = 0
    filtered_count = 0
    
    for sample in data:
        try:
            converted_sample = convert_msmarco_sample(sample)
            
            # 检查是否需要过滤没有答案的数据
            if filter_no_answer and (not converted_sample["answer"] or converted_sample["answer"].strip() == ""):
                filtered_count += 1
                continue
                
            converted_samples.append(converted_sample)
            processed_count += 1
            
            # 如果达到最大样本数，停止处理
            if max_samples and processed_count >= max_samples:
                break
                
        except Exception as e:
            print(f"警告: 样本处理失败: {e}")
            continue
    
    print(f"成功转换 {len(converted_samples)} 个样本")
    if filter_no_answer:
        print(f"过滤掉 {filtered_count} 个没有答案的样本")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_samples, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成: {len(converted_samples)} 个样本")

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="转换MS MARCO数据集")
    parser.add_argument('--max-samples', type=int, default=None, help='最大样本数量')
    parser.add_argument('--filter-no-answer', action='store_true', help='过滤没有答案的数据')
    return parser.parse_args()


def main():
    """
    主函数：转换MS MARCO数据集
    """
    args = parse_args()
    
    # 设置路径
    base_dir = Path("/home/pushihao/RAG/Reports/experiments")
    input_dir = base_dir / "datasets" / "ms_marco"  # 修正目录名
    output_dir = base_dir / "dataset_converters" / "converted" / "msmarco"
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换数据集
    datasets = {
        "train": "train.json",
        "validation": "validation.json"
    }
    
    for split, filename in datasets.items():
        input_file = input_dir / filename
        
        # 使用统一的输出文件名
        output_file = output_dir / f"{split}_converted.json"
        
        if input_file.exists():
            convert_msmarco_dataset(
                input_file, 
                output_file, 
                max_samples=args.max_samples,
                filter_no_answer=args.filter_no_answer
            )
        else:
            print(f"警告: 输入文件不存在: {input_file}")
    
    print("MS MARCO数据集转换完成！")

if __name__ == "__main__":
    main()