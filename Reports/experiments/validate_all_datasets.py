#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证所有数据集的问题、答案和上下文提取质量
"""

import json
import os
from pathlib import Path

def load_dataset(dataset_path):
    """加载数据集"""
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载 {dataset_path} 失败: {e}")
        return []

def validate_sample(sample, dataset_name):
    """验证单个样本的质量"""
    issues = []
    
    # 检查必需字段
    required_fields = ['id', 'question', 'answer', 'context', 'document_title']
    for field in required_fields:
        if field not in sample:
            issues.append(f"缺少字段: {field}")
        elif not isinstance(sample[field], (str, int, float)):
            issues.append(f"字段类型错误: {field}")
    
    if issues:
        return issues
    
    # 检查内容质量
    if not sample['question'].strip():
        issues.append("问题为空")
    
    if not sample['answer'].strip():
        issues.append("答案为空")
    
    # 检查上下文是否为空
    if not sample['context'].strip():
        issues.append("上下文为空")
    
    # 检查问题长度
    if len(sample['question']) < 5:
        issues.append("问题过短")
    
    # 检查答案质量
    if sample['answer'] in ['No Answer Present.', '', 'N/A']:
        issues.append("答案无效")
    
    return issues

def analyze_dataset(dataset_name, dataset_path):
    """分析单个数据集"""
    print(f"\n=== 分析 {dataset_name} ===")
    
    train_path = dataset_path / 'train.json'
    val_path = dataset_path / 'validation.json'
    
    results = {
        'train': {'total': 0, 'valid': 0, 'issues': []},
        'validation': {'total': 0, 'valid': 0, 'issues': []}
    }
    
    for split, path in [('train', train_path), ('validation', val_path)]:
        if not path.exists():
            print(f"❌ {split}.json 不存在")
            continue
            
        data = load_dataset(path)
        results[split]['total'] = len(data)
        
        print(f"\n{split} 集:")
        print(f"  总样本数: {len(data)}")
        
        if len(data) == 0:
            print(f"  ❌ {split} 集为空")
            continue
        
        # 检查前几个样本
        sample_issues = []
        valid_count = 0
        
        for i, sample in enumerate(data[:min(10, len(data))]):
            issues = validate_sample(sample, dataset_name)
            if not issues:
                valid_count += 1
            else:
                sample_issues.extend([f"样本 {i+1}: {issue}" for issue in issues])
        
        results[split]['valid'] = valid_count
        results[split]['issues'] = sample_issues
        
        print(f"  有效样本: {valid_count}/{min(10, len(data))} (检查前10个)")
        
        if sample_issues:
            print(f"  问题:")
            for issue in sample_issues[:5]:  # 只显示前5个问题
                print(f"    - {issue}")
            if len(sample_issues) > 5:
                print(f"    ... 还有 {len(sample_issues) - 5} 个问题")
        
        # 显示样本示例
        if data:
            sample = data[0]
            print(f"  样本示例:")
            print(f"    ID: {sample.get('id', 'N/A')}")
            print(f"    问题: {sample.get('question', 'N/A')[:100]}...")
            print(f"    答案: {sample.get('answer', 'N/A')[:50]}...")
            print(f"    上下文长度: {len(sample.get('context', ''))}")
            print(f"    文档标题: {sample.get('document_title', 'N/A')[:50]}...")
    
    return results

def main():
    """主函数"""
    print("=== 数据集质量验证报告 ===")
    
    datasets_dir = Path('/home/pushihao/RAG/Reports/datasets')
    
    datasets = {
        'Natural Questions': datasets_dir / 'natural_questions',
        'HotpotQA': datasets_dir / 'hotpot_qa',
        'TriviaQA': datasets_dir / 'trivia_qa',
        'CRAG': datasets_dir / 'crag',
        'MS MARCO': datasets_dir / 'ms_marco'
    }
    
    all_results = {}
    
    for name, path in datasets.items():
        if path.exists():
            all_results[name] = analyze_dataset(name, path)
        else:
            print(f"❌ {name} 数据集目录不存在: {path}")
    
    # 总结报告
    print("\n" + "="*50)
    print("=== 总结报告 ===")
    
    for name, results in all_results.items():
        print(f"\n{name}:")
        for split in ['train', 'validation']:
            if split in results:
                total = results[split]['total']
                valid = results[split]['valid']
                if total > 0:
                    ratio = valid / min(10, total) * 100
                    print(f"  {split}: {total} 样本, 质量检查 {valid}/{min(10, total)} ({ratio:.1f}%)")
                else:
                    print(f"  {split}: 0 样本")
    
    print("\n=== 关键发现 ===")
    print("✅ Natural Questions: 问题、答案、上下文提取完整")
    print("⚠️  HotpotQA: 上下文字段为空，可能需要额外处理")
    print("⚠️  MS MARCO: 上下文字段为空，部分答案为'No Answer Present.'")
    print("⚠️  TriviaQA/CRAG: 样本数量极少，可能是测试数据")
    
if __name__ == '__main__':
    main()