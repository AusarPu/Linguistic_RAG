#!/usr/bin/env python3
"""
RAG系统高级评估结果分析脚本
分析advanced_evaluation_results目录下的评估结果，计算各种指标并生成报告
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse

class AdvancedEvaluationAnalyzer:
    def __init__(self, results_dir: str = None):
        """初始化分析器"""
        if results_dir is None:
            # 默认使用相对于当前脚本的路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.results_dir = os.path.join(current_dir, '../datasets', 'advanced_evaluation_results')
        else:
            self.results_dir = results_dir
        self.datasets = ['hotpotqa', 'ms_marco', 'natural_questions', 'triviaqa']
        
    def load_results(self, dataset: str) -> List[Dict]:
        """加载指定数据集的结果文件"""
        file_path = os.path.join(self.results_dir, dataset, 'advanced_results.json')
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def calculate_answer_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """计算答案准确性相关指标"""
        if not results:
            return {'accuracy': 0.0, 'total_questions': 0}
        
        correct_answers = 0
        total_questions = len(results)
        confidence_scores = []
        
        for result in results:
            if 'answer_correctness' in result:
                if result['answer_correctness'].get('is_correct', False):
                    correct_answers += 1
                confidence_scores.append(result['answer_correctness'].get('confidence', 0.0))
        
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return {
            'accuracy': accuracy,
            'correct_answers': correct_answers,
            'total_questions': total_questions,
            'average_confidence': avg_confidence
        }
    
    def calculate_retrieval_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """计算检索相关指标"""
        if not results:
            return {'retrieval_success_rate': 0.0, 'avg_retrieved_chunks': 0.0}
        
        successful_retrievals = 0
        total_retrieved_chunks = 0
        total_questions = len(results)
        has_retrieval_data = 0
        
        for result in results:
            if 'retrieval_accuracy' in result:
                has_retrieval_data += 1
                if result['retrieval_accuracy'].get('is_retrieved', False):
                    successful_retrievals += 1
                total_retrieved_chunks += result['retrieval_accuracy'].get('total_retrieved', 0)
        
        retrieval_success_rate = successful_retrievals / has_retrieval_data if has_retrieval_data > 0 else 0.0
        avg_retrieved_chunks = total_retrieved_chunks / total_questions if total_questions > 0 else 0.0
        
        return {
            'retrieval_success_rate': retrieval_success_rate,
            'successful_retrievals': successful_retrievals,
            'total_questions': total_questions,
            'has_retrieval_data': has_retrieval_data,
            'avg_retrieved_chunks': avg_retrieved_chunks,
            'total_retrieved_chunks': total_retrieved_chunks
        }
    
    def calculate_combined_metrics(self, answer_metrics: Dict, retrieval_metrics: Dict) -> Dict[str, float]:
        """计算综合指标"""
        # 计算F1分数 (基于准确率和检索成功率)
        precision = answer_metrics['accuracy']
        recall = retrieval_metrics['retrieval_success_rate']
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall
        }
    
    def analyze_dataset(self, dataset: str) -> Dict:
        """分析单个数据集"""
        print(f"正在分析数据集: {dataset}")
        results = self.load_results(dataset)
        
        if not results:
            return {
                'dataset': dataset,
                'error': '无法加载数据或数据为空'
            }
        
        answer_metrics = self.calculate_answer_metrics(results)
        retrieval_metrics = self.calculate_retrieval_metrics(results)
        combined_metrics = self.calculate_combined_metrics(answer_metrics, retrieval_metrics)
        
        return {
            'dataset': dataset,
            'answer_metrics': answer_metrics,
            'retrieval_metrics': retrieval_metrics,
            'combined_metrics': combined_metrics
        }
    
    def analyze_all_datasets(self) -> Dict:
        """分析所有数据集"""
        all_results = {}
        overall_stats = {
            'total_questions': 0,
            'total_correct': 0,
            'total_successful_retrievals': 0,
            'total_retrieved_chunks': 0
        }
        
        for dataset in self.datasets:
            dataset_results = self.analyze_dataset(dataset)
            all_results[dataset] = dataset_results
            
            if 'error' not in dataset_results:
                overall_stats['total_questions'] += dataset_results['answer_metrics']['total_questions']
                overall_stats['total_correct'] += dataset_results['answer_metrics']['correct_answers']
                overall_stats['total_successful_retrievals'] += dataset_results['retrieval_metrics']['successful_retrievals']
                overall_stats['total_retrieved_chunks'] += dataset_results['retrieval_metrics']['total_retrieved_chunks']
        
        # 计算总体指标
        overall_accuracy = overall_stats['total_correct'] / overall_stats['total_questions'] if overall_stats['total_questions'] > 0 else 0.0
        overall_retrieval_rate = overall_stats['total_successful_retrievals'] / overall_stats['total_questions'] if overall_stats['total_questions'] > 0 else 0.0
        overall_f1 = 2 * (overall_accuracy * overall_retrieval_rate) / (overall_accuracy + overall_retrieval_rate) if (overall_accuracy + overall_retrieval_rate) > 0 else 0.0
        avg_chunks_per_question = overall_stats['total_retrieved_chunks'] / overall_stats['total_questions'] if overall_stats['total_questions'] > 0 else 0.0
        
        all_results['overall'] = {
            'total_questions': overall_stats['total_questions'],
            'overall_accuracy': overall_accuracy,
            'overall_retrieval_success_rate': overall_retrieval_rate,
            'overall_f1_score': overall_f1,
            'avg_chunks_per_question': avg_chunks_per_question,
            'has_retrieval_data': sum(dataset_results['retrieval_metrics'].get('has_retrieval_data', 0) for dataset_results in all_results.values() if 'error' not in dataset_results)
        }
        
        return all_results

def format_results_to_markdown(results: Dict) -> str:
    """将结果格式化为Markdown"""
    md_content = []
    
    # 标题
    md_content.append("# RAG系统高级评估报告\n")
    
    # 总体性能
    overall = results['overall']
    md_content.append("## 总体性能\n")
    md_content.append(f"- **总问题数**: {overall['total_questions']}")
    md_content.append(f"- **总体准确率**: {overall['overall_accuracy']:.2%}")
    md_content.append(f"- **总体检索成功率**: {overall['overall_retrieval_success_rate']:.2%}")
    md_content.append(f"- **总体F1分数**: {overall['overall_f1_score']:.2f}")
    md_content.append(f"- **平均检索块数**: {overall['avg_chunks_per_question']:.2f}")
    md_content.append(f"- **有检索数据的问题数**: {overall.get('has_retrieval_data', 'N/A')}")
    md_content.append("")
    
    # 各数据集详细结果
    md_content.append("## 各数据集详细结果\n")
    
    for dataset in ['hotpotqa', 'ms_marco', 'natural_questions', 'triviaqa']:
        if dataset in results and 'error' not in results[dataset]:
            data = results[dataset]
            md_content.append(f"### {dataset.upper()}\n")
            
            # 回答准确性
            answer_metrics = data['answer_metrics']
            md_content.append("#### 回答准确性")
            md_content.append(f"- 准确率: {answer_metrics['accuracy']:.2%}")
            md_content.append(f"- 正确回答数: {answer_metrics['correct_answers']}")
            md_content.append(f"- 总问题数: {answer_metrics['total_questions']}")
            md_content.append(f"- 平均置信度: {answer_metrics['average_confidence']:.2f}")
            md_content.append("")
            
            # 检索性能
            retrieval_metrics = data['retrieval_metrics']
            md_content.append("#### 检索性能")
            md_content.append(f"- 检索成功率: {retrieval_metrics['retrieval_success_rate']:.2%}")
            md_content.append(f"- 成功检索数: {retrieval_metrics['successful_retrievals']}")
            md_content.append(f"- 有检索数据的问题数: {retrieval_metrics.get('has_retrieval_data', 'N/A')}")
            md_content.append(f"- 平均检索块数: {retrieval_metrics['avg_retrieved_chunks']:.2f}")
            md_content.append("")
            
            # 综合指标
            combined_metrics = data['combined_metrics']
            md_content.append("#### 综合指标")
            md_content.append(f"- F1分数: {combined_metrics['f1_score']:.2f}")
            md_content.append(f"- 精确率: {combined_metrics['precision']:.2%}")
            md_content.append(f"- 召回率: {combined_metrics['recall']:.2%}")
            md_content.append("")
        
        elif dataset in results and 'error' in results[dataset]:
            md_content.append(f"### {dataset.upper()}\n")
            md_content.append(f"**错误**: {results[dataset]['error']}\n")
    
    # 报告生成信息
    md_content.append("---")
    md_content.append(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    md_content.append(f"*使用脚本: evaluate_results.py*")
    
    return "\n".join(md_content)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分析RAG系统高级评估结果')
    parser.add_argument('--results-dir', type=str, help='评估结果目录路径')
    parser.add_argument('--output-dir', type=str, help='输出目录路径')
    
    args = parser.parse_args()
    
    print("开始分析RAG系统评估结果...")
    
    # 初始化分析器
    analyzer = AdvancedEvaluationAnalyzer(args.results_dir)
    
    # 分析所有数据集
    results = analyzer.analyze_all_datasets()
    
    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = analyzer.results_dir
    
    # 生成Markdown报告
    md_report = format_results_to_markdown(results)
    md_file_path = os.path.join(output_dir, 'evaluation_report.md')
    
    with open(md_file_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    
    print(f"评估报告已生成: {md_file_path}")
    
    # 打印摘要
    print("=" * 50)
    print("评估结果摘要:")
    print("=" * 50)
    overall = results['overall']
    print(f"总问题数: {overall['total_questions']}")
    print(f"总体准确率: {overall['overall_accuracy']:.4f} ({overall['overall_accuracy']*100:.2f}%)")
    print(f"总体检索成功率: {overall['overall_retrieval_success_rate']:.4f} ({overall['overall_retrieval_success_rate']*100:.2f}%)")
    print(f"总体F1分数: {overall['overall_f1_score']:.4f}")
    print(f"平均每问题检索块数: {overall['avg_chunks_per_question']:.2f}")
    print(f"有检索数据的问题数: {overall['has_retrieval_data']}")
    
    # 生成简要文本报告
    txt_report = f"""RAG系统高级评估结果摘要

总体性能:
- 总问题数: {overall['total_questions']}
- 总体准确率: {overall['overall_accuracy']:.2%}
- 总体检索成功率: {overall['overall_retrieval_success_rate']:.2%}
- 总体F1分数: {overall['overall_f1_score']:.2f}
- 平均检索块数: {overall['avg_chunks_per_question']:.2f}
- 有检索数据的问题数: {overall['has_retrieval_data']}

各数据集性能:
"""
    
    for dataset in ['hotpotqa', 'ms_marco', 'natural_questions', 'triviaqa']:
        if dataset in results and 'error' not in results[dataset]:
            data = results[dataset]
            txt_report += f"\n{dataset.upper()}:\n"
            txt_report += f"  - 准确率: {data['answer_metrics']['accuracy']:.2%}\n"
            txt_report += f"  - 检索成功率: {data['retrieval_metrics']['retrieval_success_rate']:.2%}\n"
            txt_report += f"  - F1分数: {data['combined_metrics']['f1_score']:.2f}\n"
    
    txt_report += f"\n报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    txt_file_path = os.path.join(output_dir, 'evaluation_summary.txt')
    with open(txt_file_path, 'w', encoding='utf-8') as f:
        f.write(txt_report)
    
    print(f"简要报告已生成: {txt_file_path}")

if __name__ == "__main__":
    main()