#!/usr/bin/env python3
"""
数据集评估脚本（并发版本）
对4个数据集进行独立测试，实现问题级别的并发处理
输出格式：包含知识库内容ID、系统回答和问题的JSON文件
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# 添加项目根目录到路径
project_root = "/home/pushihao/RAG"
sys.path.insert(0, project_root)

from script.rag_pipeline import execute_rag_flow
from script.knowledge_base import KnowledgeBase
from script.config_rag import PROCESSED_DATA_DIR

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据集配置
def get_output_filename(is_sample: bool = False) -> str:
    """根据是否为示例模式返回相应的文件名"""
    return "sample_results.json" if is_sample else "evaluation_results.json"

DATASETS = {
    "hotpotqa": {
        "questions_file": "/home/pushihao/RAG/Reports/experiments/datasets/converted/hotpotqa_validation_kb_chunks.json",
        "index_dir": "/home/pushihao/RAG/Reports/experiments/datasets/knowledge_bases/hotpotqa",
        "output_dir": "/home/pushihao/RAG/Reports/experiments/datasets/rag_evaluation_results/hotpotqa"
    },
    "ms_marco": {
        "questions_file": "/home/pushihao/RAG/Reports/experiments/datasets/converted/ms_marco_validation_kb_chunks.json",
        "index_dir": "/home/pushihao/RAG/Reports/experiments/datasets/knowledge_bases/ms_marco",
        "output_dir": "/home/pushihao/RAG/Reports/experiments/datasets/rag_evaluation_results/ms_marco"
    },
    "natural_questions": {
        # 使用1k子集，确保评估只针对构建索引用到的1000条问题
        "questions_file": "/home/pushihao/RAG/Reports/experiments/datasets/converted/natural_questions_validation_kb_chunks.json",
        "index_dir": "/home/pushihao/RAG/Reports/experiments/datasets/knowledge_bases/natural_questions",
        "output_dir": "/home/pushihao/RAG/Reports/experiments/datasets/rag_evaluation_results/natural_questions"
    },
    "triviaqa": {
        "questions_file": "/home/pushihao/RAG/Reports/experiments/datasets/converted/triviaqa_validation_kb_chunks.json",
        "index_dir": "/home/pushihao/RAG/Reports/experiments/datasets/knowledge_bases/triviaqa",
        "output_dir": "/home/pushihao/RAG/Reports/experiments/datasets/rag_evaluation_results/triviaqa"
    }
}

# 并发配置
DEFAULT_BATCH_SIZE = 5  # 默认批处理大小
DEFAULT_MAX_QUESTIONS = 20  # 默认不限制问题数量（按数据集配置与文件决定）

# 线程锁用于保护共享资源
result_lock = threading.Lock()

async def process_single_question(question: str, kb_instance: KnowledgeBase, question_id: str = "",
                                use_query_rewriter: bool = True,
                                use_dense_chunks: bool = True,
                                use_dense_keywords: bool = True,
                                use_dense_questions: bool = True,
                                use_usefulness_judger: bool = True) -> Dict[str, Any]:
    """
    处理单个问题，返回结果
    """
    retrieved_chunk_ids = []
    system_answer = ""
    reasoning_text = ""
    rewritten_query = {}
    pipeline_end_reason = ""
    
    try:
        # 执行RAG流程
        async for event in execute_rag_flow(
            user_query=question,
            chat_history_openai=[],  # 空的聊天历史
            kb_instance=kb_instance,
            use_query_rewriter=use_query_rewriter,
            use_dense_chunks=use_dense_chunks,
            use_dense_keywords=use_dense_keywords,
            use_dense_questions=use_dense_questions,
            use_usefulness_judger=use_usefulness_judger
        ):
            # 收集查询重写结果
            if event.get("type") == "rewritten_query_result":
                rewritten_query = event.get("rewritten_text", {})
            
            # 收集检索到的chunk IDs
            elif event.get("type") == "useful_chunks_preview":
                retrieved_chunk_ids = [chunk.get("chunk_id") for chunk in event.get("preview", [])]
            
            # 收集系统回答（不包括思考内容）
            elif event.get("type") == "content_delta":
                system_answer += event.get("text", "")
            
            # 收集思考过程（但不包含在最终答案中）
            elif event.get("type") == "reasoning_delta":
                reasoning_text += event.get("text", "")
            
            # 流程结束
            elif event.get("type") == "pipeline_end":
                pipeline_end_reason = event.get("reason", "completed")
                break
                
    except Exception as e:
        # 使用 logger.exception 记录完整的堆栈信息，便于调试空错误消息问题
        logger.exception(f"处理问题 {question_id} 时出错", exc_info=True)
        import traceback
        tb_text = traceback.format_exc()
        system_answer = f"处理错误: {str(e) or type(e).__name__}. Traceback: {tb_text}"
        pipeline_end_reason = "error"
    

    with open("/home/pushihao/RAG/001.txt", "a+") as f:
            f.write("reasoning_text:\n"+reasoning_text+"\n"+"system_answer:\n"+system_answer+"\n"+"-"*40+"\n"+"pipeline_end_reason:\n"+pipeline_end_reason+"\n")
    if not system_answer.strip():
        if pipeline_end_reason == "no_context_found_after_retrieval":
            system_answer = "抱歉，我没有找到与您问题相关的直接信息。"
        elif pipeline_end_reason == "no_context_found_after_usefulness":
            system_answer = "抱歉，我没有找到与您问题直接相关的有用信息。"
        elif pipeline_end_reason == "error":
            system_answer = "处理过程中发生错误，无法生成回答。"
        else:
            system_answer = "未能生成有效回答。"
    
    return {
        "question": question,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "system_answer": system_answer.strip(),
        "rewritten_query": rewritten_query,
        "pipeline_end_reason": pipeline_end_reason,
        "has_reasoning": bool(reasoning_text.strip())
    }

async def process_questions_batch(questions_batch: List[Dict[str, Any]], kb_instance: KnowledgeBase, batch_id: int,
                                use_query_rewriter: bool = True,
                                use_dense_chunks: bool = True,
                                use_dense_keywords: bool = True,
                                use_dense_questions: bool = True,
                                use_usefulness_judger: bool = True) -> List[Dict[str, Any]]:
    """
    并发处理一批问题
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
        task = process_single_question(question, kb_instance, question_id, 
                                     use_query_rewriter, use_dense_chunks, 
                                     use_dense_keywords, use_dense_questions, 
                                     use_usefulness_judger)
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
                "system_answer": f"处理异常: {str(result)}"
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

async def evaluate_dataset_concurrent(dataset_name: str, config: Dict[str, str], 
                                    batch_size: int = DEFAULT_BATCH_SIZE, 
                                    max_questions: int = None,
                                    is_sample: bool = False,
                                    use_query_rewriter: bool = True,
                                    use_dense_chunks: bool = True,
                                    use_dense_keywords: bool = True,
                                    use_dense_questions: bool = True,
                                    use_usefulness_judger: bool = True) -> None:
    """
    并发评估单个数据集
    """
    mode_desc = "示例模式" if is_sample else "完整模式"
    logger.info(f"开始并发评估数据集: {dataset_name} ({mode_desc}, 批大小: {batch_size})")
    
    # 创建输出目录和文件路径
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / get_output_filename(is_sample)
    
    # 临时设置PROCESSED_DATA_DIR为当前数据集的索引目录
    original_processed_dir = PROCESSED_DATA_DIR
    
    # 重新导入并设置配置
    import importlib
    import script.config_rag as config_module
    
    # 更新配置
    config_module.PROCESSED_DATA_DIR = config["index_dir"]
    config_module.FAISS_INDEX_CHUNKS_SAVE_PATH = os.path.join(config["index_dir"], "faiss_index_chunks_ip.idx")
    config_module.INDEXED_CHUNKS_METADATA_SAVE_PATH = os.path.join(config["index_dir"], "indexed_chunks_metadata.json")
    config_module.PHRASE_SPARSE_WEIGHTS_MAP_SAVE_PATH = os.path.join(config["index_dir"], "phrase_sparse_weights_map.pkl")
    config_module.PHRASE_DENSE_EMBEDDINGS_MAP_SAVE_PATH = os.path.join(config["index_dir"], "phrase_dense_embeddings_map.pkl")
    config_module.BM25_INDEX_SAVE_PATH = os.path.join(config["index_dir"], "phrase_bm25_index.pkl")
    config_module.FAISS_INDEX_QUESTIONS_SAVE_PATH = os.path.join(config["index_dir"], "faiss_index_questions_ip.idx")
    config_module.QUESTION_INDEX_TO_CHUNK_ID_MAP_SAVE_PATH = os.path.join(config["index_dir"], "question_index_to_chunk_id_map.json")
    config_module.ALL_QUESTION_TEXTS_SAVE_PATH = os.path.join(config["index_dir"], "all_question_texts.json")
    
    # 重新导入知识库模块以使用新的配置
    import script.knowledge_base
    importlib.reload(script.knowledge_base)
    from script.knowledge_base import KnowledgeBase
    
    try:
        # 初始化知识库
        logger.info(f"初始化知识库: {config['index_dir']}")
        kb_instance = KnowledgeBase()
        
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
            batch_results = await process_questions_batch(
                batch_questions, 
                kb_instance, 
                batch_num,
                use_query_rewriter,
                use_dense_chunks,
                use_dense_keywords,
                use_dense_questions,
                use_usefulness_judger
            )
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
        
        logger.info(f"数据集 {dataset_name} 并发评估完成，共处理 {len(all_results)} 个问题")
        
    except Exception as e:
        logger.error(f"评估数据集 {dataset_name} 时出错: {str(e)}")
        raise
    finally:
        # 恢复原始配置
        config_module.PROCESSED_DATA_DIR = original_processed_dir

async def evaluate_all_datasets_concurrent(datasets_to_process: List[tuple], 
                                         batch_size: int = DEFAULT_BATCH_SIZE,
                                         max_questions: int = None,
                                         dataset_concurrent: bool = False,
                                         use_query_rewriter: bool = True,
                                         use_dense_chunks: bool = True,
                                         use_dense_keywords: bool = True,
                                         use_dense_questions: bool = True,
                                         use_usefulness_judger: bool = True) -> None:
    """
    评估所有数据集，支持数据集级别的并发
    """
    # 判断是否为示例模式
    is_sample = max_questions is not None and max_questions <= 100
    
    if dataset_concurrent:
        # 数据集级别并发处理
        logger.info(f"开始并发评估所有数据集，批大小: {batch_size}")
        tasks = [
            evaluate_dataset_concurrent(dataset_name, config, batch_size, max_questions, is_sample,
                                      use_query_rewriter, use_dense_chunks, use_dense_keywords, 
                                      use_dense_questions, use_usefulness_judger)
            for dataset_name, config in datasets_to_process
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    else:
        # 数据集串行处理，但问题并发处理
        logger.info(f"开始串行评估数据集（问题并发），批大小: {batch_size}")
        for dataset_name, config in datasets_to_process:
            try:
                await evaluate_dataset_concurrent(dataset_name, config, batch_size, max_questions, is_sample,
                                                use_query_rewriter, use_dense_chunks, use_dense_keywords, 
                                                use_dense_questions, use_usefulness_judger)
            except Exception as e:
                logger.error(f"数据集 {dataset_name} 评估失败: {str(e)}")
                continue

async def main():
    """
    主函数：并发评估数据集
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="并发评估RAG系统在多个数据集上的性能")
    parser.add_argument("--dataset", type=str, choices=list(DATASETS.keys()) + ["all"], 
                       default="all", help="要评估的数据集")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, 
                       help=f"并发批处理大小（默认：{DEFAULT_BATCH_SIZE}）")
    parser.add_argument("--max-questions", type=int, default=DEFAULT_MAX_QUESTIONS, 
                       help=f"限制每个数据集处理的问题数量（默认：{DEFAULT_MAX_QUESTIONS}）")
    parser.add_argument("--dataset-concurrent", action="store_true", 
                       help="启用数据集级别的并发处理（默认：串行处理数据集）")
    
    # 消融实验参数
    parser.add_argument("--no-query-rewriter", action="store_true", 
                       help="禁用查询重写模块")
    parser.add_argument("--no-dense-chunks", action="store_true", 
                       help="禁用密集块检索路径")
    parser.add_argument("--no-dense-keywords", action="store_true", 
                       help="禁用密集关键字检索路径")
    parser.add_argument("--no-dense-questions", action="store_true", 
                       help="禁用密集问题检索路径")
    parser.add_argument("--no-usefulness-judger", action="store_true", 
                       help="禁用有用性判断模块")
    
    args = parser.parse_args()
    
    if args.dataset == "all":
        datasets_to_process = list(DATASETS.items())
    else:
        datasets_to_process = [(args.dataset, DATASETS[args.dataset])]
    
    logger.info(f"开始并发评估")
    logger.info(f"数据集: {[name for name, _ in datasets_to_process]}")
    logger.info(f"批处理大小: {args.batch_size}")
    logger.info(f"最大问题数: {args.max_questions}")
    logger.info(f"数据集并发: {'是' if args.dataset_concurrent else '否'}")
    
    # 消融实验配置
    use_query_rewriter = not args.no_query_rewriter
    use_dense_chunks = not args.no_dense_chunks
    use_dense_keywords = not args.no_dense_keywords
    use_dense_questions = not args.no_dense_questions
    use_usefulness_judger = not args.no_usefulness_judger
    
    logger.info(f"消融实验配置:")
    logger.info(f"  查询重写: {'启用' if use_query_rewriter else '禁用'}")
    logger.info(f"  密集块检索: {'启用' if use_dense_chunks else '禁用'}")
    logger.info(f"  密集关键字检索: {'启用' if use_dense_keywords else '禁用'}")
    logger.info(f"  密集问题检索: {'启用' if use_dense_questions else '禁用'}")
    logger.info(f"  有用性判断: {'启用' if use_usefulness_judger else '禁用'}")
    
    start_time = time.time()
    
    await evaluate_all_datasets_concurrent(
        datasets_to_process, 
        args.batch_size, 
        args.max_questions,
        args.dataset_concurrent,
        use_query_rewriter,
        use_dense_chunks,
        use_dense_keywords,
        use_dense_questions,
        use_usefulness_judger
    )
    
    total_time = time.time() - start_time
    logger.info(f"所有数据集并发评估完成，总耗时: {total_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())