#!/usr/bin/env python3
"""
评估结果总结脚本
分析生成的JSON文件，提供统计信息和示例
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any

def analyze_results(file_path: str) -> Dict[str, Any]:
    """
    分析单个结果文件
    """
    if not os.path.exists(file_path):
        return {"error": "文件不存在"}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if not results:
            return {"error": "文件为空"}
        
        # 基本统计
        total_questions = len(results)
        questions_with_chunks = sum(1 for r in results if r.get('retrieved_chunk_ids'))
        questions_without_chunks = total_questions - questions_with_chunks
        
        # 检索到的chunk数量统计
        chunk_counts = [len(r.get('retrieved_chunk_ids', [])) for r in results]
        avg_chunks_per_question = sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0
        max_chunks = max(chunk_counts) if chunk_counts else 0
        
        # 回答长度统计
        answer_lengths = [len(r.get('system_answer', '')) for r in results]
        avg_answer_length = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
        
        # 示例问题
        sample_questions = []
        for i, result in enumerate(results[:3]):  # 取前3个作为示例
            sample_questions.append({
                "question": result.get('question', '')[:100] + "..." if len(result.get('question', '')) > 100 else result.get('question', ''),
                "retrieved_chunks_count": len(result.get('retrieved_chunk_ids', [])),
                "answer_length": len(result.get('system_answer', '')),
                "has_ground_truth": bool(result.get('ground_truth_answer')),
                "pipeline_end_reason": result.get('pipeline_end_reason', 'unknown'),
                "has_rewritten_query": bool(result.get('rewritten_query')),
                "has_reasoning": result.get('has_reasoning', False)
            })
        
        return {
            "total_questions": total_questions,
            "questions_with_chunks": questions_with_chunks,
            "questions_without_chunks": questions_without_chunks,
            "retrieval_success_rate": questions_with_chunks / total_questions if total_questions > 0 else 0,
            "avg_chunks_per_question": round(avg_chunks_per_question, 2),
            "max_chunks_retrieved": max_chunks,
            "avg_answer_length": round(avg_answer_length, 2),
            "sample_questions": sample_questions
        }
    
    except Exception as e:
        return {"error": f"处理文件时出错: {str(e)}"}

def main():
    """
    主函数：分析所有数据集的评估结果
    """
    base_dir = "/home/pushihao/RAG/Reports/experiments/rag_evaluation_results"
    datasets = ["hotpotqa", "ms_marco", "natural_questions", "triviaqa"]
    
    print("=" * 80)
    print("RAG系统数据集评估结果总结")
    print("=" * 80)
    
    for dataset in datasets:
        print(f"\n📊 数据集: {dataset.upper()}")
        print("-" * 50)
        
        result_file = os.path.join(base_dir, dataset, "evaluation_results.json")
        if not os.path.exists(result_file):
            print("❌ 未找到结果文件")
            continue
        
        analysis = analyze_results(result_file)
        
        if "error" in analysis:
            print(f"❌ 错误: {analysis['error']}")
            continue
        
        print(f"📁 文件类型: 完整结果")
        print(f"📝 总问题数: {analysis['total_questions']}")
        print(f"✅ 成功检索到内容的问题: {analysis['questions_with_chunks']} ({analysis['retrieval_success_rate']:.1%})")
        print(f"❌ 未检索到内容的问题: {analysis['questions_without_chunks']}")
        print(f"📊 平均每问题检索块数: {analysis['avg_chunks_per_question']}")
        print(f"🔝 最大检索块数: {analysis['max_chunks_retrieved']}")
        print(f"📏 平均回答长度: {analysis['avg_answer_length']} 字符")
        
        print("\n🔍 示例问题:")
        for i, sample in enumerate(analysis['sample_questions'], 1):
            print(f"  {i}. {sample['question']}")
            print(f"     检索块数: {sample['retrieved_chunks_count']}, 回答长度: {sample['answer_length']}")
            print(f"     流程结束原因: {sample['pipeline_end_reason']}, 有重写查询: {'是' if sample['has_rewritten_query'] else '否'}, 有推理: {'是' if sample['has_reasoning'] else '否'}")
    
    print("\n" + "=" * 80)
    print("📋 输出文件说明:")
    print("• 每个JSON文件包含以下字段:")
    print("  - question: 原始问题")
    print("  - retrieved_chunk_ids: 检索到的知识库内容ID列表")
    print("  - system_answer: 系统生成的回答（不包含思考过程）")
    print("  - original_id: 原始数据集中的问题ID")
    print("  - ground_truth_answer: 标准答案（如果有）")
    print("  - rewritten_query: 查询重写器生成的重写结果")
    print("  - pipeline_end_reason: RAG流程结束的原因")
    print("  - has_reasoning: 是否包含推理过程")
    print("\n📍 文件位置和命名:")
    for dataset in datasets:
        print(f"  - {dataset}:")
        print(f"    • 完整结果: {base_dir}/{dataset}/evaluation_results.json")
        print(f"    • 示例结果: {base_dir}/{dataset}/sample_results.json")
    print("=" * 80)

if __name__ == "__main__":
    main()