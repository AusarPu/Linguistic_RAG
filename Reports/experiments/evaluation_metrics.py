#!/usr/bin/env python3
"""
评测指标实现
实现RAG系统评测所需的各种指标：Recall@k, MRR@k, NDCG@k, EM, F1, ROUGE
"""

import re
import string
import numpy as np
from typing import List, Dict, Any, Tuple, Set
from collections import Counter
import logging
from rouge_score import rouge_scorer
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalMetrics:
    """
    检索阶段的评测指标
    """
    
    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """
        计算Recall@k
        
        Args:
            retrieved_docs: 检索到的文档ID列表（按相关性排序）
            relevant_docs: 相关文档ID列表
            k: 截断位置
        
        Returns:
            Recall@k分数
        """
        if not relevant_docs:
            return 0.0
        
        retrieved_at_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        intersection = retrieved_at_k.intersection(relevant_set)
        recall = len(intersection) / len(relevant_set)
        
        return recall
    
    @staticmethod
    def mrr_at_k(retrieved_docs_list: List[List[str]], relevant_docs_list: List[List[str]], k: int) -> float:
        """
        计算MRR@k (Mean Reciprocal Rank)
        
        Args:
            retrieved_docs_list: 多个查询的检索结果列表
            relevant_docs_list: 多个查询的相关文档列表
            k: 截断位置
        
        Returns:
            MRR@k分数
        """
        if len(retrieved_docs_list) != len(relevant_docs_list):
            raise ValueError("检索结果和相关文档列表长度不匹配")
        
        reciprocal_ranks = []
        
        for retrieved_docs, relevant_docs in zip(retrieved_docs_list, relevant_docs_list):
            if not relevant_docs:
                reciprocal_ranks.append(0.0)
                continue
            
            relevant_set = set(relevant_docs)
            rr = 0.0
            
            for i, doc_id in enumerate(retrieved_docs[:k]):
                if doc_id in relevant_set:
                    rr = 1.0 / (i + 1)
                    break
            
            reciprocal_ranks.append(rr)
        
        return np.mean(reciprocal_ranks)
    
    @staticmethod
    def ndcg_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int, 
                  relevance_scores: Dict[str, float] = None) -> float:
        """
        计算NDCG@k (Normalized Discounted Cumulative Gain)
        
        Args:
            retrieved_docs: 检索到的文档ID列表
            relevant_docs: 相关文档ID列表
            k: 截断位置
            relevance_scores: 文档相关性分数字典，如果为None则使用二元相关性
        
        Returns:
            NDCG@k分数
        """
        if not relevant_docs:
            return 0.0
        
        # 如果没有提供相关性分数，使用二元相关性（相关=1，不相关=0）
        if relevance_scores is None:
            relevance_scores = {doc_id: 1.0 for doc_id in relevant_docs}
        
        # 计算DCG@k
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs[:k]):
            if doc_id in relevance_scores:
                rel = relevance_scores[doc_id]
                dcg += (2**rel - 1) / math.log2(i + 2)
        
        # 计算IDCG@k (理想情况下的DCG)
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_scores):
            idcg += (2**rel - 1) / math.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg

class QAMetrics:
    """
    问答任务的评测指标
    """
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    @staticmethod
    def normalize_answer(s: str) -> str:
        """
        标准化答案文本
        """
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def exact_match(self, prediction: str, ground_truth: str) -> float:
        """
        计算精确匹配分数
        
        Args:
            prediction: 预测答案
            ground_truth: 标准答案
        
        Returns:
            EM分数（0或1）
        """
        return float(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))
    
    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """
        计算F1分数
        
        Args:
            prediction: 预测答案
            ground_truth: 标准答案
        
        Returns:
            F1分数
        """
        pred_tokens = self.normalize_answer(prediction).split()
        truth_tokens = self.normalize_answer(ground_truth).split()
        
        if len(pred_tokens) == 0 and len(truth_tokens) == 0:
            return 1.0
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return 0.0
        
        pred_counter = Counter(pred_tokens)
        truth_counter = Counter(truth_tokens)
        
        # 计算交集
        intersection = pred_counter & truth_counter
        num_same = sum(intersection.values())
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(pred_tokens)
        recall = num_same / len(truth_tokens)
        
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    def rouge_scores(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """
        计算ROUGE分数
        
        Args:
            prediction: 预测答案
            ground_truth: 标准答案
        
        Returns:
            包含ROUGE-1, ROUGE-2, ROUGE-L分数的字典
        """
        scores = self.rouge_scorer.score(ground_truth, prediction)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def evaluate_batch(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """
        批量评测
        
        Args:
            predictions: 预测答案列表
            ground_truths: 标准答案列表
        
        Returns:
            包含各种指标平均分数的字典
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("预测答案和标准答案数量不匹配")
        
        em_scores = []
        f1_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, truth in zip(predictions, ground_truths):
            # EM和F1
            em_scores.append(self.exact_match(pred, truth))
            f1_scores.append(self.f1_score(pred, truth))
            
            # ROUGE
            rouge_scores = self.rouge_scores(pred, truth)
            rouge1_scores.append(rouge_scores['rouge1'])
            rouge2_scores.append(rouge_scores['rouge2'])
            rougeL_scores.append(rouge_scores['rougeL'])
        
        return {
            'exact_match': np.mean(em_scores),
            'f1': np.mean(f1_scores),
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }

class RAGEvaluator:
    """
    RAG系统综合评测器
    """
    
    def __init__(self):
        self.retrieval_metrics = RetrievalMetrics()
        self.qa_metrics = QAMetrics()
    
    def evaluate_retrieval(self, results: List[Dict[str, Any]], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        评测检索性能
        
        Args:
            results: 检索结果列表，每个元素包含 'retrieved_docs' 和 'relevant_docs'
            k_values: 要计算的k值列表
        
        Returns:
            包含各种检索指标的字典
        """
        metrics = {}
        
        # 准备数据
        retrieved_docs_list = [r['retrieved_docs'] for r in results]
        relevant_docs_list = [r['relevant_docs'] for r in results]
        
        # 计算各种指标
        for k in k_values:
            # Recall@k
            recall_scores = []
            for retrieved_docs, relevant_docs in zip(retrieved_docs_list, relevant_docs_list):
                recall = self.retrieval_metrics.recall_at_k(retrieved_docs, relevant_docs, k)
                recall_scores.append(recall)
            metrics[f'recall@{k}'] = np.mean(recall_scores)
            
            # NDCG@k
            ndcg_scores = []
            for retrieved_docs, relevant_docs in zip(retrieved_docs_list, relevant_docs_list):
                ndcg = self.retrieval_metrics.ndcg_at_k(retrieved_docs, relevant_docs, k)
                ndcg_scores.append(ndcg)
            metrics[f'ndcg@{k}'] = np.mean(ndcg_scores)
        
        # MRR@k
        for k in k_values:
            mrr = self.retrieval_metrics.mrr_at_k(retrieved_docs_list, relevant_docs_list, k)
            metrics[f'mrr@{k}'] = mrr
        
        return metrics
    
    def evaluate_qa(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """
        评测问答性能
        
        Args:
            predictions: 预测答案列表
            ground_truths: 标准答案列表
        
        Returns:
            包含各种问答指标的字典
        """
        return self.qa_metrics.evaluate_batch(predictions, ground_truths)
    
    def evaluate_end_to_end(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        端到端评测
        
        Args:
            results: 结果列表，每个元素包含:
                - 'question': 问题
                - 'prediction': 预测答案
                - 'ground_truth': 标准答案
                - 'retrieved_docs': 检索到的文档
                - 'relevant_docs': 相关文档
        
        Returns:
            包含检索和问答指标的综合评测结果
        """
        # 分离检索和问答数据
        retrieval_results = []
        predictions = []
        ground_truths = []
        
        for result in results:
            retrieval_results.append({
                'retrieved_docs': result['retrieved_docs'],
                'relevant_docs': result['relevant_docs']
            })
            predictions.append(result['prediction'])
            ground_truths.append(result['ground_truth'])
        
        # 评测检索性能
        retrieval_metrics = self.evaluate_retrieval(retrieval_results)
        
        # 评测问答性能
        qa_metrics = self.evaluate_qa(predictions, ground_truths)
        
        # 合并结果
        combined_metrics = {
            'retrieval': retrieval_metrics,
            'qa': qa_metrics,
            'num_samples': len(results)
        }
        
        return combined_metrics
    
    def print_evaluation_results(self, results: Dict[str, Any]):
        """
        打印评测结果
        
        Args:
            results: evaluate_end_to_end返回的结果
        """
        print("\n=== RAG系统评测结果 ===")
        print(f"样本数量: {results['num_samples']}")
        
        print("\n--- 检索性能 ---")
        for metric, score in results['retrieval'].items():
            print(f"{metric}: {score:.4f}")
        
        print("\n--- 问答性能 ---")
        for metric, score in results['qa'].items():
            print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    # 测试评测指标
    evaluator = RAGEvaluator()
    
    # 示例数据
    test_results = [
        {
            'question': '什么是机器学习？',
            'prediction': '机器学习是人工智能的一个分支，它使计算机能够从数据中学习。',
            'ground_truth': '机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。',
            'retrieved_docs': ['doc1', 'doc2', 'doc3'],
            'relevant_docs': ['doc1', 'doc4']
        }
    ]
    
    # 运行评测
    results = evaluator.evaluate_end_to_end(test_results)
    evaluator.print_evaluation_results(results)