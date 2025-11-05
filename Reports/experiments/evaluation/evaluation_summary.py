#!/usr/bin/env python3
"""
评估结果汇总脚本
用于汇总和分析RAG系统的评估结果
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any

def load_evaluation_results(results_dir: str) -> Dict[str, Dict]:
    """
    加载所有数据集的评估结果
    
    Args:
        results_dir: 评估结果目录
        
    Returns:
        数据集名称到评估结果的映射
    """
    datasets = ['hotpotqa', 'ms_marco', 'natural_questions', 'triviaqa']
    results = {}
    
    for dataset in datasets:
        dataset_dir = os.path.join(results_dir, dataset)
        
        # 加载高级评估结果
        advanced_file = os.path.join(dataset_dir, "evaluation_results.json")
        if os.path.exists(advanced_file):
            with open(advanced_file, 'r', encoding='utf-8') as f:
                results[dataset] = json.load(f)
        else:
            print(f"警告: 未找到 {dataset} 的高级评估结果文件")
            
    return results

def analyze_answer_correctness(results: Dict[str, List[Dict]]) -> List[Dict]:
    """
    分析答案正确性
    
    Args:
        results: 评估结果
        
    Returns:
        答案正确性统计列表
    """
    stats = []
    
    for dataset, data in results.items():
        correct_count = 0
        total_count = len(data)
        confidence_scores = []
        
        for item in data:
            answer_correctness = item.get('answer_correctness', {})
            if answer_correctness.get('is_correct', False):
                correct_count += 1
            confidence_scores.append(answer_correctness.get('confidence', 0.0))
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        stats.append({
            'Dataset': dataset,
            'Total Questions': total_count,
            'Correct Answers': correct_count,
            'Accuracy (%)': accuracy * 100,
            'Avg Confidence': avg_confidence
        })
    
    return stats

def analyze_retrieval_accuracy(results: Dict[str, List[Dict]]) -> List[Dict]:
    """
    分析检索准确性
    
    Args:
        results: 评估结果
        
    Returns:
        检索准确性统计列表
    """
    stats = []
    
    for dataset, data in results.items():
        retrieved_count = 0
        total_count = len(data)
        chunk_counts = []
        
        for item in data:
            retrieval_accuracy = item.get('retrieval_accuracy', {})
            if retrieval_accuracy.get('is_retrieved', False):
                retrieved_count += 1
            chunk_counts.append(retrieval_accuracy.get('total_retrieved', 0))
        
        retrieval_rate = retrieved_count / total_count if total_count > 0 else 0
        avg_chunks = sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0
        
        stats.append({
            'Dataset': dataset,
            'Total Questions': total_count,
            'Successfully Retrieved': retrieved_count,
            'Retrieval Rate (%)': retrieval_rate * 100,
            'Avg Chunks Retrieved': avg_chunks
        })
    
    return stats

def generate_detailed_report(results: Dict[str, List[Dict]], output_file: str):
    """
    生成详细的评估报告
    
    Args:
        results: 评估结果
        output_file: 输出文件路径
    """
    import datetime
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# RAG系统评估报告\n\n")
        f.write(f"评估时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 总体统计
        f.write("## 总体统计\n\n")
        
        # 答案正确性统计
        answer_stats = analyze_answer_correctness(results)
        f.write("### 答案正确性统计\n\n")
        f.write("| Dataset | Total Questions | Correct Answers | Accuracy (%) | Avg Confidence |\n")
        f.write("|---------|-----------------|-----------------|--------------|----------------|\n")
        for stat in answer_stats:
            f.write(f"| {stat['Dataset']} | {stat['Total Questions']} | {stat['Correct Answers']} | {stat['Accuracy (%)']:.1f} | {stat['Avg Confidence']:.3f} |\n")
        f.write("\n\n")
        
        # 检索准确性统计
        retrieval_stats = analyze_retrieval_accuracy(results)
        f.write("### 检索准确性统计\n\n")
        f.write("| Dataset | Total Questions | Successfully Retrieved | Retrieval Rate (%) | Avg Chunks Retrieved |\n")
        f.write("|---------|-----------------|------------------------|--------------------|-----------------------|\n")
        for stat in retrieval_stats:
            f.write(f"| {stat['Dataset']} | {stat['Total Questions']} | {stat['Successfully Retrieved']} | {stat['Retrieval Rate (%)']:.1f} | {stat['Avg Chunks Retrieved']:.1f} |\n")
        f.write("\n\n")
        
        # 各数据集详细分析
        f.write("## 各数据集详细分析\n\n")
        
        for dataset, data in results.items():
            f.write(f"### {dataset.upper()}\n\n")
            
            # 基本统计
            total_questions = len(data)
            correct_answers = sum(1 for item in data 
                                if item.get('answer_correctness', {}).get('is_correct', False))
            successful_retrievals = sum(1 for item in data 
                                      if item.get('retrieval_accuracy', {}).get('is_retrieved', False))
            
            f.write(f"- **总问题数**: {total_questions}\n")
            f.write(f"- **答案正确数**: {correct_answers} ({correct_answers/total_questions*100:.1f}%)\n")
            f.write(f"- **检索成功数**: {successful_retrievals} ({successful_retrievals/total_questions*100:.1f}%)\n\n")
            
            # 错误案例分析
            f.write("#### 错误案例分析\n\n")
            error_count = 0
            for i, item in enumerate(data):
                answer_correctness = item.get('answer_correctness', {})
                if not answer_correctness.get('is_correct', False):
                    error_count += 1
                    if error_count <= 3:  # 只显示前3个错误案例
                        f.write(f"**错误案例 {error_count}:**\n")
                        f.write(f"- 问题: {item.get('question', 'N/A')}\n")
                        f.write(f"- 系统回答: {item.get('system_answer', 'N/A')[:200]}...\n")
                        f.write(f"- 标准答案: {item.get('ground_truth_answer', 'N/A')}\n")
                        f.write(f"- 评估说明: {answer_correctness.get('explanation', 'N/A')}\n\n")
            
            if error_count > 3:
                f.write(f"... 还有 {error_count - 3} 个错误案例\n\n")
            
            f.write("\n")
        
        # 总结和建议
        f.write("## 总结和建议\n\n")
        
        # 计算总体指标
        total_questions = sum(len(data) for data in results.values())
        total_correct = sum(sum(1 for item in data 
                              if item.get('answer_correctness', {}).get('is_correct', False))
                          for data in results.values())
        total_retrieved = sum(sum(1 for item in data 
                                if item.get('retrieval_accuracy', {}).get('is_retrieved', False))
                            for data in results.values())
        
        overall_accuracy = total_correct / total_questions * 100 if total_questions > 0 else 0
        overall_retrieval = total_retrieved / total_questions * 100 if total_questions > 0 else 0
        
        f.write(f"### 总体表现\n\n")
        f.write(f"- **总体答案准确率**: {overall_accuracy:.1f}%\n")
        f.write(f"- **总体检索成功率**: {overall_retrieval:.1f}%\n\n")
        
        f.write("### 主要发现\n\n")
        
        # 找出表现最好和最差的数据集
        best_dataset = max(results.keys(), 
                          key=lambda d: sum(1 for item in results[d] 
                                          if item.get('answer_correctness', {}).get('is_correct', False)) / len(results[d]))
        worst_dataset = min(results.keys(), 
                           key=lambda d: sum(1 for item in results[d] 
                                           if item.get('answer_correctness', {}).get('is_correct', False)) / len(results[d]))
        
        f.write(f"1. **表现最好的数据集**: {best_dataset}\n")
        f.write(f"2. **表现最差的数据集**: {worst_dataset}\n")
        f.write(f"3. **检索系统整体表现良好**: 平均检索成功率为 {overall_retrieval:.1f}%\n")
        f.write(f"4. **答案生成需要改进**: 平均答案准确率为 {overall_accuracy:.1f}%\n\n")
        
        f.write("### 改进建议\n\n")
        f.write("1. **优化答案生成模型**: 考虑使用更大的模型或改进提示词\n")
        f.write("2. **改进检索策略**: 对于检索成功率较低的数据集，优化检索算法\n")
        f.write("3. **增强知识库**: 补充相关领域的知识内容\n")
        f.write("4. **调整评估标准**: 考虑更细粒度的评估指标\n")

def main():
    """主函数"""
    results_dir = "/home/pushihao/RAG/Reports/experiments/rag_evaluation_results"
    output_dir = "/home/pushihao/RAG/Reports/experiments/evaluation"
    
    # 加载评估结果
    print("加载评估结果...")
    results = load_evaluation_results(results_dir)
    
    if not results:
        print("错误: 未找到任何评估结果")
        return
    
    print(f"成功加载 {len(results)} 个数据集的评估结果")
    
    # 生成统计表格
    print("生成统计表格...")
    answer_stats = analyze_answer_correctness(results)
    retrieval_stats = analyze_retrieval_accuracy(results)
    
    print("\n答案正确性统计:")
    print("Dataset\t\tTotal\tCorrect\tAccuracy(%)\tConfidence")
    print("-" * 60)
    for stat in answer_stats:
        print(f"{stat['Dataset']:<15}\t{stat['Total Questions']}\t{stat['Correct Answers']}\t{stat['Accuracy (%)']:.1f}\t\t{stat['Avg Confidence']:.3f}")
    
    print("\n检索准确性统计:")
    print("Dataset\t\tTotal\tRetrieved\tRate(%)\t\tAvg Chunks")
    print("-" * 60)
    for stat in retrieval_stats:
        print(f"{stat['Dataset']:<15}\t{stat['Total Questions']}\t{stat['Successfully Retrieved']}\t\t{stat['Retrieval Rate (%)']:.1f}\t\t{stat['Avg Chunks Retrieved']:.1f}")
    
    # 生成详细报告
    report_file = os.path.join(output_dir, "evaluation_report.md")
    print(f"\n生成详细报告: {report_file}")
    generate_detailed_report(results, report_file)
    
    print("评估报告生成完成!")

if __name__ == "__main__":
    main()