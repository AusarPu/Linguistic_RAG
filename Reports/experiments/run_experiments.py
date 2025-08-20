#!/usr/bin/env python3
"""
实验运行脚本
执行完整的对比实验和消融研究
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import asyncio
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.asyncio import tqdm

from dataset_preparation import DatasetPreparator
from evaluation_metrics import RAGEvaluator
from baseline_models import ModelFactory
# 从baseline_models导入RAGConfig类
from baseline_models import RAGConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """
    实验运行器
    """
    
    def __init__(self, config: RAGConfig, output_dir: str = "Reports/experiments/results"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据集管理器和评估器
        self.dataset_manager = DatasetPreparator()
        self.evaluator = RAGEvaluator()
        
        # 实验配置
        self.models_to_test = [
            'basic_rag',           # 基础RAG
            'hybrid_rag',          # 传统混合检索
            'reranker_rag',        # 重排序RAG
            'multipath_no_filter', # 多路径无过滤（消融）
            'full_rag'             # 完整系统
        ]
        
        self.datasets_to_test = ['quac', 'natural_questions']
        self.top_k_values = [5, 10, 20]
        
        logger.info(f"实验运行器初始化完成，输出目录: {self.output_dir}")
    
    def load_quac_eval(self) -> List[Dict[str, Any]]:
        """
        加载QuAC评测数据
        """
        eval_file = Path("./datasets/quac/validation.json")
        if not eval_file.exists():
            raise FileNotFoundError(f"QuAC评测文件不存在: {eval_file}")
        
        with open(eval_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 转换为统一格式
        questions = []
        for item in data:
            if item.get('questions') and item.get('answers', {}).get('texts'):
                questions.append({
                    'question': item['questions'][0],
                    'answer': item['answers']['texts'][0][0] if item['answers']['texts'][0] else '',
                    'context_passages': [item.get('context', '')]
                })
        
        return questions
    
    def load_nq_eval(self) -> List[Dict[str, Any]]:
        """
        加载Natural Questions评测数据
        """
        eval_file = Path("./datasets/natural_questions/validation.json")
        if not eval_file.exists():
            raise FileNotFoundError(f"Natural Questions评测文件不存在: {eval_file}")
        
        with open(eval_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 转换为统一格式
        questions = []
        for item in data:
            questions.append({
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'context_passages': [item.get('context', '')]
            })
        
        return questions
    
    def prepare_datasets(self):
        """
        准备实验数据集
        """
        logger.info("开始准备数据集...")
        
        # 检查数据集是否已存在
        quac_file = Path("./datasets/quac/validation.json")
        nq_file = Path("./datasets/natural_questions/validation.json")
        
        if quac_file.exists() and nq_file.exists():
            logger.info("数据集文件已存在，跳过下载")
            return
        
        try:
            # 下载和处理数据集
            self.dataset_manager.prepare_all_datasets()
            logger.info("数据集准备完成")
        except Exception as e:
            logger.error(f"数据集准备失败: {e}")
            raise
    
    async def run_single_experiment(self, model_type: str, dataset_name: str, 
                                  questions: List[Dict], top_k: int = 10) -> Dict[str, Any]:
        """
        运行单个实验
        
        Args:
            model_type: 模型类型
            dataset_name: 数据集名称
            questions: 问题列表
            top_k: 检索文档数量
        
        Returns:
            实验结果
        """
        logger.info(f"开始实验: {model_type} on {dataset_name} (top_k={top_k})")
        
        try:
            # 创建模型
            model = ModelFactory.create_model(model_type, self.config)
            
            # 存储结果
            results = []
            retrieval_results = []
            qa_results = []
            
            # 记录时间
            start_time = time.time()
            total_retrieval_time = 0
            total_generation_time = 0
            
            # 处理每个问题
            for i, item in enumerate(tqdm(questions, desc=f"{model_type}-{dataset_name}")):
                try:
                    question = item['question']
                    ground_truth = item.get('answer', '')
                    context_passages = item.get('context_passages', [])
                    
                    # 记录检索时间
                    retrieval_start = time.time()
                    
                    # 执行问答
                    result = await model.answer_question(question, top_k)
                    
                    retrieval_time = time.time() - retrieval_start
                    total_retrieval_time += retrieval_time
                    
                    # 评估检索效果（如果有ground truth passages）
                    if context_passages:
                        # 提取文档ID列表
                        retrieved_doc_ids = [doc.get('id', doc.get('chunk_id', str(i))) 
                                            for i, doc in enumerate(result['retrieved_docs'])]
                        # 构造评估数据格式
                        eval_data = [{
                            'retrieved_docs': retrieved_doc_ids,
                            'relevant_docs': context_passages
                        }]
                        retrieval_metrics = self.evaluator.evaluate_retrieval(
                            results=eval_data,
                            k_values=[5, 10, 20]
                        )
                        retrieval_results.append(retrieval_metrics)
                    
                    # 评估问答效果
                    if ground_truth:
                        qa_metrics = self.evaluator.evaluate_qa(
                            predictions=[result['answer']],
                            ground_truths=[ground_truth]
                        )
                        qa_results.append(qa_metrics)
                    
                    # 存储详细结果
                    results.append({
                        'question_id': i,
                        'question': question,
                        'predicted_answer': result['answer'],
                        'ground_truth_answer': ground_truth,
                        'num_retrieved': result['num_retrieved'],
                        'retrieval_time': retrieval_time,
                        'retrieval_metrics': retrieval_metrics if context_passages else None,
                        'qa_metrics': qa_metrics if ground_truth else None
                    })
                    
                except Exception as e:
                    logger.warning(f"处理问题 {i} 失败: {e}")
                    continue
            
            total_time = time.time() - start_time
            
            # 计算平均指标
            avg_retrieval_metrics = {}
            if retrieval_results:
                for k in [5, 10, 20]:
                    avg_retrieval_metrics[f'recall@{k}'] = np.mean([r[f'recall@{k}'] for r in retrieval_results])
                    avg_retrieval_metrics[f'mrr@{k}'] = np.mean([r[f'mrr@{k}'] for r in retrieval_results])
                    avg_retrieval_metrics[f'ndcg@{k}'] = np.mean([r[f'ndcg@{k}'] for r in retrieval_results])
            
            avg_qa_metrics = {}
            if qa_results:
                avg_qa_metrics['exact_match'] = np.mean([r['exact_match'] for r in qa_results])
                avg_qa_metrics['f1_score'] = np.mean([r['f1_score'] for r in qa_results])
                avg_qa_metrics['rouge_l'] = np.mean([r['rouge_l'] for r in qa_results])
            
            # 汇总结果
            experiment_result = {
                'model_type': model_type,
                'dataset_name': dataset_name,
                'top_k': top_k,
                'num_questions': len(questions),
                'num_processed': len(results),
                'avg_retrieval_metrics': avg_retrieval_metrics,
                'avg_qa_metrics': avg_qa_metrics,
                'timing': {
                    'total_time': total_time,
                    'avg_time_per_question': total_time / len(results) if results else 0,
                    'total_retrieval_time': total_retrieval_time,
                    'avg_retrieval_time': total_retrieval_time / len(results) if results else 0
                },
                'detailed_results': results
            }
            
            logger.info(f"实验完成: {model_type} on {dataset_name}")
            logger.info(f"处理了 {len(results)}/{len(questions)} 个问题")
            if avg_qa_metrics:
                logger.info(f"平均EM: {avg_qa_metrics.get('exact_match', 0):.3f}, 平均F1: {avg_qa_metrics.get('f1_score', 0):.3f}")
            
            return experiment_result
            
        except Exception as e:
            logger.error(f"实验失败: {model_type} on {dataset_name}: {e}")
            raise
    
    async def run_comparative_experiments(self):
        """
        运行对比实验
        """
        logger.info("开始运行对比实验...")
        
        all_results = []
        
        for dataset_name in self.datasets_to_test:
            logger.info(f"\n=== 在 {dataset_name} 数据集上进行实验 ===")
            
            # 加载数据集
            try:
                if dataset_name == 'quac':
                    questions = self.load_quac_eval()
                elif dataset_name == 'natural_questions':
                    questions = self.load_nq_eval()
                else:
                    logger.warning(f"未知数据集: {dataset_name}")
                    continue
                
                logger.info(f"加载了 {len(questions)} 个问题")
                
            except Exception as e:
                logger.error(f"加载数据集 {dataset_name} 失败: {e}")
                continue
            
            # 对每个模型进行测试
            for model_type in self.models_to_test:
                for top_k in self.top_k_values:
                    try:
                        result = await self.run_single_experiment(
                            model_type, dataset_name, questions, top_k
                        )
                        all_results.append(result)
                        
                        # 保存单个实验结果
                        result_file = self.output_dir / f"{model_type}_{dataset_name}_k{top_k}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(result_file, 'w', encoding='utf-8') as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                        
                        logger.info(f"结果已保存到: {result_file}")
                        
                    except Exception as e:
                        logger.error(f"实验失败: {model_type} on {dataset_name} (k={top_k}): {e}")
                        continue
        
        # 保存汇总结果
        summary_file = self.output_dir / f"comparative_experiments_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"对比实验完成，汇总结果保存到: {summary_file}")
        return all_results
    
    def generate_results_table(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        生成结果表格
        
        Args:
            results: 实验结果列表
        
        Returns:
            结果DataFrame
        """
        rows = []
        
        for result in results:
            row = {
                'Model': result['model_type'],
                'Dataset': result['dataset_name'],
                'Top-K': result['top_k'],
                'Questions': result['num_processed']
            }
            
            # 添加检索指标
            if result['avg_retrieval_metrics']:
                for k in [5, 10, 20]:
                    row[f'Recall@{k}'] = result['avg_retrieval_metrics'].get(f'recall@{k}', 0)
                    row[f'MRR@{k}'] = result['avg_retrieval_metrics'].get(f'mrr@{k}', 0)
                    row[f'NDCG@{k}'] = result['avg_retrieval_metrics'].get(f'ndcg@{k}', 0)
            
            # 添加问答指标
            if result['avg_qa_metrics']:
                row['EM'] = result['avg_qa_metrics'].get('exact_match', 0)
                row['F1'] = result['avg_qa_metrics'].get('f1_score', 0)
                row['ROUGE-L'] = result['avg_qa_metrics'].get('rouge_l', 0)
            
            # 添加效率指标
            row['Avg_Time'] = result['timing']['avg_time_per_question']
            row['Avg_Retrieval_Time'] = result['timing']['avg_retrieval_time']
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    async def run_ablation_study(self):
        """
        运行消融研究
        """
        logger.info("开始运行消融研究...")
        
        # 消融研究的模型组合
        ablation_models = [
            'full_rag',              # 完整系统
            'multipath_no_filter',   # 无有用性判断
            'basic_rag',             # 无查询重写 + 单路径检索
            'hybrid_rag'             # 传统混合检索
        ]
        
        ablation_results = []
        
        # 选择一个主要数据集进行消融研究
        dataset_name = 'quac'  # 使用QuAC作为主要评估数据集
        
        try:
            questions = self.load_quac_eval()
            logger.info(f"消融研究使用 {len(questions)} 个问题")
        except Exception as e:
            logger.error(f"加载消融研究数据集失败: {e}")
            return []
        
        # 对每个消融模型进行测试
        for model_type in ablation_models:
            try:
                result = await self.run_single_experiment(
                    model_type, dataset_name, questions, top_k=10
                )
                ablation_results.append(result)
                
                # 保存消融研究结果
                result_file = self.output_dir / f"ablation_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                logger.info(f"消融研究结果已保存: {result_file}")
                
            except Exception as e:
                logger.error(f"消融研究失败: {model_type}: {e}")
                continue
        
        # 保存消融研究汇总
        ablation_summary_file = self.output_dir / f"ablation_study_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(ablation_summary_file, 'w', encoding='utf-8') as f:
            json.dump(ablation_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"消融研究完成，结果保存到: {ablation_summary_file}")
        return ablation_results
    
    async def run_all_experiments(self):
        """
        运行所有实验
        """
        logger.info("开始运行完整实验套件...")
        
        try:
            # 准备数据集
            self.prepare_datasets()
            
            # 2. 运行对比实验
            comparative_results = await self.run_comparative_experiments()
            
            # 3. 运行消融研究
            ablation_results = await self.run_ablation_study()
            
            # 4. 生成汇总报告
            await self.generate_summary_report(comparative_results, ablation_results)
            
            logger.info("所有实验完成！")
            
        except Exception as e:
            logger.error(f"实验运行失败: {e}")
            raise
    
    async def generate_summary_report(self, comparative_results: List[Dict], ablation_results: List[Dict]):
        """
        生成实验汇总报告
        
        Args:
            comparative_results: 对比实验结果
            ablation_results: 消融研究结果
        """
        logger.info("生成实验汇总报告...")
        
        try:
            # 生成对比实验表格
            if comparative_results:
                comparative_df = self.generate_results_table(comparative_results)
                comparative_csv = self.output_dir / f"comparative_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                comparative_df.to_csv(comparative_csv, index=False)
                logger.info(f"对比实验表格保存到: {comparative_csv}")
            
            # 生成消融研究表格
            if ablation_results:
                ablation_df = self.generate_results_table(ablation_results)
                ablation_csv = self.output_dir / f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                ablation_df.to_csv(ablation_csv, index=False)
                logger.info(f"消融研究表格保存到: {ablation_csv}")
            
            # 生成Markdown报告
            report_content = self.generate_markdown_report(comparative_results, ablation_results)
            report_file = self.output_dir / f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"实验报告保存到: {report_file}")
            
        except Exception as e:
            logger.error(f"生成汇总报告失败: {e}")
    
    def generate_markdown_report(self, comparative_results: List[Dict], ablation_results: List[Dict]) -> str:
        """
        生成Markdown格式的实验报告
        
        Args:
            comparative_results: 对比实验结果
            ablation_results: 消融研究结果
        
        Returns:
            Markdown报告内容
        """
        report = f"""# RAG系统实验报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 实验概述

本报告包含了RAG系统的完整实验结果，包括对比实验和消融研究。

### 测试模型

1. **Basic RAG**: 基础RAG模型（单一向量检索）
2. **Hybrid RAG**: 传统混合检索RAG（BM25 + 向量检索）
3. **Reranker RAG**: 基于重排序的RAG（检索 + Cross-Encoder重排序）
4. **MultiPath No Filter**: 多路径检索但无有用性判断
5. **Full RAG**: 完整系统（多路径检索 + 有用性判断）

### 测试数据集

- **QuAC**: 对话式问答数据集
- **Natural Questions**: 自然问题数据集

## 对比实验结果

"""
        
        if comparative_results:
            # 按数据集分组显示结果
            datasets = set(r['dataset_name'] for r in comparative_results)
            
            for dataset in datasets:
                report += f"\n### {dataset.upper()} 数据集结果\n\n"
                
                dataset_results = [r for r in comparative_results if r['dataset_name'] == dataset]
                
                # 创建结果表格
                report += "| Model | Top-K | EM | F1 | ROUGE-L | Recall@10 | MRR@10 | Avg Time(s) |\n"
                report += "|-------|-------|----|----|---------|-----------|--------|-------------|\n"
                
                for result in dataset_results:
                    model = result['model_type']
                    top_k = result['top_k']
                    
                    qa_metrics = result.get('avg_qa_metrics', {})
                    retrieval_metrics = result.get('avg_retrieval_metrics', {})
                    timing = result.get('timing', {})
                    
                    em = qa_metrics.get('exact_match', 0)
                    f1 = qa_metrics.get('f1_score', 0)
                    rouge = qa_metrics.get('rouge_l', 0)
                    recall10 = retrieval_metrics.get('recall@10', 0)
                    mrr10 = retrieval_metrics.get('mrr@10', 0)
                    avg_time = timing.get('avg_time_per_question', 0)
                    
                    report += f"| {model} | {top_k} | {em:.3f} | {f1:.3f} | {rouge:.3f} | {recall10:.3f} | {mrr10:.3f} | {avg_time:.2f} |\n"
        
        report += "\n## 消融研究结果\n\n"
        
        if ablation_results:
            report += "| Component | EM | F1 | ROUGE-L | Recall@10 | MRR@10 | Improvement |\n"
            report += "|-----------|----|----|---------|-----------|--------|-------------|\n"
            
            # 找到完整系统的结果作为基准
            full_system = next((r for r in ablation_results if r['model_type'] == 'full_rag'), None)
            
            for result in ablation_results:
                model = result['model_type']
                qa_metrics = result.get('avg_qa_metrics', {})
                retrieval_metrics = result.get('avg_retrieval_metrics', {})
                
                em = qa_metrics.get('exact_match', 0)
                f1 = qa_metrics.get('f1_score', 0)
                rouge = qa_metrics.get('rouge_l', 0)
                recall10 = retrieval_metrics.get('recall@10', 0)
                mrr10 = retrieval_metrics.get('mrr@10', 0)
                
                # 计算相对于完整系统的改进
                improvement = ""
                if full_system and model != 'full_rag':
                    full_f1 = full_system.get('avg_qa_metrics', {}).get('f1_score', 0)
                    if full_f1 > 0:
                        improvement = f"{((f1 - full_f1) / full_f1 * 100):+.1f}%"
                
                report += f"| {model} | {em:.3f} | {f1:.3f} | {rouge:.3f} | {recall10:.3f} | {mrr10:.3f} | {improvement} |\n"
        
        report += "\n## 结论\n\n"
        report += "1. **有用性判断模块的效果**: 通过对比Full RAG和MultiPath No Filter的结果，可以看出有用性判断模块的贡献。\n"
        report += "2. **多路径检索的效果**: 通过对比MultiPath No Filter和传统方法，可以看出多路径检索的优势。\n"
        report += "3. **整体系统性能**: Full RAG系统在各项指标上的表现。\n"
        report += "4. **效率分析**: 各模型的响应时间对比。\n\n"
        
        report += "## 实验文件\n\n"
        report += f"- 详细结果文件保存在: `{self.output_dir}`\n"
        report += "- 包含每个问题的详细预测结果和评估指标\n"
        
        return report

async def main():
    """
    主函数
    """
    try:
        # 初始化配置
        config = RAGConfig()
        
        # 创建实验运行器
        runner = ExperimentRunner(config)
        
        # 运行所有实验
        await runner.run_all_experiments()
        
        print("\n=== 实验完成 ===")
        print(f"结果保存在: {runner.output_dir}")
        
    except Exception as e:
        logger.error(f"实验运行失败: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())