#!/usr/bin/env python3
"""
LLM基线评估脚本（并发版本）
直接向LLM提问而不使用RAG系统，用于对比RAG系统的效果
输出格式：与RAG评估保持一致的JSON文件格式
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any
import sys
import time
import threading

# 添加项目根目录到路径
project_root = "/home/pushihao/RAG"
sys.path.insert(0, project_root)

from script.vllm_clients import call_generator_vllm_stream
from script.config_rag import GENERATION_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据集配置 - 复用原有配置但修改输出目录
def get_output_filename(is_sample: bool = False) -> str:
    """根据是否为示例模式返回相应的文件名"""
    return "sample_results.json" if is_sample else "evaluation_results.json"

DATASETS = {
    "hotpotqa": {
        "questions_file": "/home/pushihao/RAG/Reports/experiments/datasets/converted/hotpotqa_validation_kb_chunks.json",
        "output_dir": "/home/pushihao/RAG/Reports/experiments/datasets/llm_baseline_results/hotpotqa"
    },
    "ms_marco": {
        "questions_file": "/home/pushihao/RAG/Reports/experiments/datasets/converted/ms_marco_validation_kb_chunks.json",
        "output_dir": "/home/pushihao/RAG/Reports/experiments/datasets/llm_baseline_results/ms_marco"
    },
    "natural_questions": {
        "questions_file": "/home/pushihao/RAG/Reports/experiments/datasets/converted/natural_questions_validation_kb_chunks.json",
        "output_dir": "/home/pushihao/RAG/Reports/experiments/datasets/llm_baseline_results/natural_questions"
    },
    "triviaqa": {
        "questions_file": "/home/pushihao/RAG/Reports/experiments/datasets/converted/triviaqa_validation_kb_chunks.json",
        "output_dir": "/home/pushihao/RAG/Reports/experiments/datasets/llm_baseline_results/triviaqa"
    }
}

# 并发配置
DEFAULT_BATCH_SIZE = 5  # 默认批处理大小
DEFAULT_MAX_QUESTIONS = 20  # 默认问题数量限制

# 线程锁用于保护共享资源
result_lock = threading.Lock()

async def process_single_question_llm(question: str, question_id: str = "") -> Dict[str, Any]:
    """
    直接向LLM提问，不使用RAG系统
    
    Args:
        question: 用户问题
        question_id: 问题ID（用于日志）
    
    Returns:
        Dict: 包含问题和LLM回答的结果字典
    """
    system_answer = ""
    reasoning_text = ""
    pipeline_end_reason = "direct_llm_response"
    
    try:
        # 直接向LLM提问，不使用系统提示词
        messages = [{"role": "user", "content": question}]
        
        # 使用与RAG系统相同的生成配置
        async for event in call_generator_vllm_stream(
            messages=messages,
            generation_config=GENERATION_CONFIG
        ):
            # 收集系统回答（不包括思考内容）
            if event.get("type") == "content_delta":
                system_answer += event.get("text", "")
            
            # 收集思考过程（但不包含在最终答案中）
            elif event.get("type") == "reasoning_delta":
                reasoning_text += event.get("text", "")
            
            # 处理错误
            elif event.get("type") == "error":
                logger.error(f"处理问题 {question_id} 时出错: {event.get('message', '')}")
                system_answer = f"处理错误: {event.get('message', '')}"
                pipeline_end_reason = "error"
                break
            
            # 流程结束
            elif event.get("type") == "stream_end":
                pipeline_end_reason = "direct_llm_response"
                break
                
    except Exception as e:
        logger.error(f"处理问题 {question_id} 时出错: {str(e)}")
        system_answer = f"处理异常: {str(e)}"
        pipeline_end_reason = "error"
    
    # 如果system_answer为空，提供默认回答
    if not system_answer.strip():
        system_answer = "LLM未能生成有效回答。"
        pipeline_end_reason = "no_response"
    
    return {
        "question": question,
        "retrieved_chunk_ids": [],  # LLM基线不涉及检索，始终为空
        "system_answer": system_answer.strip(),
        "rewritten_query": {},  # LLM基线不涉及查询重写，始终为空
        "pipeline_end_reason": pipeline_end_reason,
        "has_reasoning": bool(reasoning_text.strip())
    }

async def process_questions_batch_llm(questions_batch: List[Dict[str, Any]], batch_id: int) -> List[Dict[str, Any]]:
    """
    并发处理一批问题（LLM基线版本）
    """
    logger.info(f"开始处理批次 {batch_id}，包含 {len(questions_batch)} 个问题")
    start_time = time.time()
    
    # 创建并发任务
    tasks = []
    for i, item in enumerate(questions_batch):
        question = item.get("question", "")
        if not question:
            continue
        
        question_id = f"batch_{batch_id}_q_{i+1}"
        task = process_single_question_llm(question, question_id)
        tasks.append((task, item))
    
    # 并发执行所有任务
    results = []
    completed_tasks = await asyncio.gather(*[task for task, _ in tasks], return_exceptions=True)
    
    # 处理结果
    for i, (result, original_item) in enumerate(zip(completed_tasks, [item for _, item in tasks])):
        if isinstance(result, Exception):
            logger.error(f"批次 {batch_id} 中的问题 {i+1} 处理失败: {str(result)}")
            result = {
                "question": original_item.get("question", ""),
                "retrieved_chunk_ids": [],
                "system_answer": f"处理异常: {str(result)}",
                "rewritten_query": {},
                "pipeline_end_reason": "error",
                "has_reasoning": False
            }
        
        # 添加原始数据中的其他信息
        result.update({
            "original_id": original_item.get("id", ""),
            "ground_truth_answer": original_item.get("answer", "")
        })
        
        results.append(result)
    
    duration = time.time() - start_time
    logger.info(f"批次 {batch_id} 处理完成，耗时 {duration:.2f}s，成功处理 {len(results)} 个问题")
    
    return results

async def evaluate_dataset_llm_baseline(dataset_name: str, config: Dict[str, str], 
                                      batch_size: int = DEFAULT_BATCH_SIZE, 
                                      max_questions: int = None,
                                      is_sample: bool = False) -> None:
    """
    评估单个数据集（LLM基线版本）
    """
    mode_desc = "示例模式" if is_sample else "完整模式"
    logger.info(f"开始LLM基线评估数据集: {dataset_name} ({mode_desc}, 批大小: {batch_size})")
    
    # 创建输出目录和文件路径
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / get_output_filename(is_sample)
    
    try:
        # 加载问题数据
        logger.info(f"加载问题文件: {config['questions_file']}")
        with open(config["questions_file"], 'r', encoding='utf-8') as f:
            # 尝试加载为标准JSON数组格式
            try:
                questions_data = json.load(f)
                logger.info(f"成功加载JSON数组格式文件")
            except json.JSONDecodeError:
                # 如果失败，尝试JSONL格式（逐行JSON）
                f.seek(0)  # 重置文件指针
                questions_data = []
                for line in f:
                    line = line.strip()
                    if line:
                        questions_data.append(json.loads(line))
                logger.info(f"成功加载JSONL格式文件")
        
        # 限制问题数量
        if max_questions:
            questions_data = questions_data[:max_questions]
            logger.info(f"限制处理问题数量为: {max_questions}")
        
        logger.info(f"总共需要处理 {len(questions_data)} 个问题")
        
        # 分批处理问题
        all_results = []
        total_batches = (len(questions_data) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(questions_data), batch_size):
            batch_num = batch_idx // batch_size + 1
            batch_questions = questions_data[batch_idx:batch_idx + batch_size]
            
            logger.info(f"处理批次 {batch_num}/{total_batches}")
            
            # 并发处理当前批次
            batch_results = await process_questions_batch_llm(batch_questions, batch_num)
            all_results.extend(batch_results)
            
            # 每处理完一个批次就保存结果（防止数据丢失）
            logger.info(f"批次 {batch_num} 完成，保存中间结果...")
            with result_lock:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            # 简短休息，避免过度占用资源
            await asyncio.sleep(0.5)
        
        # 保存最终结果
        logger.info(f"保存最终结果到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据集 {dataset_name} LLM基线评估完成，共处理 {len(all_results)} 个问题")
        
    except Exception as e:
        logger.error(f"评估数据集 {dataset_name} 时出错: {str(e)}")
        raise

async def evaluate_all_datasets_llm_baseline(datasets_to_process: List[tuple], 
                                           batch_size: int = DEFAULT_BATCH_SIZE,
                                           max_questions: int = None,
                                           dataset_concurrent: bool = False) -> None:
    """
    评估所有数据集（LLM基线版本），支持数据集级别的并发
    """
    # 判断是否为示例模式
    is_sample = max_questions is not None and max_questions <= 100
    
    if dataset_concurrent:
        # 数据集级别并发处理
        logger.info(f"开始并发评估所有数据集（LLM基线），批大小: {batch_size}")
        tasks = [
            evaluate_dataset_llm_baseline(dataset_name, config, batch_size, max_questions, is_sample)
            for dataset_name, config in datasets_to_process
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    else:
        # 数据集串行处理，但问题并发处理
        logger.info(f"开始串行评估数据集（LLM基线，问题并发），批大小: {batch_size}")
        for dataset_name, config in datasets_to_process:
            try:
                await evaluate_dataset_llm_baseline(dataset_name, config, batch_size, max_questions, is_sample)
            except Exception as e:
                logger.error(f"数据集 {dataset_name} 评估失败: {str(e)}")
                continue

async def main():
    """
    主函数：LLM基线评估
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM基线评估：直接向LLM提问而不使用RAG系统")
    parser.add_argument("--dataset", type=str, choices=list(DATASETS.keys()) + ["all"], 
                       default="all", help="要评估的数据集")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, 
                       help=f"并发批处理大小（默认：{DEFAULT_BATCH_SIZE}）")
    parser.add_argument("--max-questions", type=int, default=DEFAULT_MAX_QUESTIONS, 
                       help=f"限制每个数据集处理的问题数量（默认：{DEFAULT_MAX_QUESTIONS}）")
    parser.add_argument("--dataset-concurrent", action="store_true", 
                       help="启用数据集级别的并发处理（默认：串行处理数据集）")
    
    args = parser.parse_args()
    
    if args.dataset == "all":
        datasets_to_process = list(DATASETS.items())
    else:
        datasets_to_process = [(args.dataset, DATASETS[args.dataset])]
    
    logger.info(f"开始LLM基线评估")
    logger.info(f"数据集: {[name for name, _ in datasets_to_process]}")
    logger.info(f"批处理大小: {args.batch_size}")
    logger.info(f"最大问题数: {args.max_questions}")
    logger.info(f"数据集并发: {'是' if args.dataset_concurrent else '否'}")
    
    start_time = time.time()
    
    await evaluate_all_datasets_llm_baseline(
        datasets_to_process, 
        args.batch_size, 
        args.max_questions,
        args.dataset_concurrent
    )
    
    total_time = time.time() - start_time
    logger.info(f"所有数据集LLM基线评估完成，总耗时: {total_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())