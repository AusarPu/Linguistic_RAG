#!/usr/bin/env python3
"""
数据集评估脚本（示例版本）
对4个数据集进行独立测试，每个数据集只处理前50个问题作为示例
输出格式：包含知识库内容ID、系统回答和问题的JSON文件
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from script.rag_pipeline import execute_rag_flow
from script.knowledge_base import KnowledgeBase
from script.config_rag import PROCESSED_DATA_DIR

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据集配置
DATASETS = {
    "hotpotqa": {
        "questions_file": "/home/pushihao/RAG/Reports/experiments/dataset_converters/converted/hotpotqa/validation_converted.json",
        "index_dir": "/home/pushihao/RAG/Reports/experiments/dataset_indexs/hotpotqa",
        "output_file": "/home/pushihao/RAG/Reports/experiments/rag_evaluation_results/hotpotqa/sample_evaluation_results.json"
    },
    "ms_marco": {
        "questions_file": "/home/pushihao/RAG/Reports/experiments/dataset_converters/converted/ms_marco/validation_converted.json",
        "index_dir": "/home/pushihao/RAG/Reports/experiments/dataset_indexs/ms_marco",
        "output_file": "/home/pushihao/RAG/Reports/experiments/rag_evaluation_results/ms_marco/sample_evaluation_results.json"
    },
    "natural_questions": {
        "questions_file": "/home/pushihao/RAG/Reports/experiments/dataset_converters/converted/natural_questions/validation_converted.json",
        "index_dir": "/home/pushihao/RAG/Reports/experiments/dataset_indexs/natural_questions",
        "output_file": "/home/pushihao/RAG/Reports/experiments/rag_evaluation_results/natural_questions/sample_evaluation_results.json"
    },
    "triviaqa": {
        "questions_file": "/home/pushihao/RAG/Reports/experiments/dataset_converters/converted/triviaqa/validation_converted.json",
        "index_dir": "/home/pushihao/RAG/Reports/experiments/dataset_indexs/triviaqa",
        "output_file": "/home/pushihao/RAG/Reports/experiments/rag_evaluation_results/triviaqa/sample_evaluation_results.json"
    }
}

# 每个数据集处理的问题数量
SAMPLE_SIZE = 50

async def process_single_question(question: str, kb_instance: KnowledgeBase) -> Dict[str, Any]:
    """
    处理单个问题，返回结果
    """
    retrieved_chunk_ids = []
    system_answer = ""
    reasoning_text = ""
    
    try:
        # 执行RAG流程
        async for event in execute_rag_flow(
            user_query=question,
            chat_history_openai=[],  # 空的聊天历史
            kb_instance=kb_instance
        ):
            # 收集检索到的chunk IDs
            if event.get("type") == "useful_chunks_preview":
                retrieved_chunk_ids = [chunk.get("chunk_id") for chunk in event.get("preview", [])]
            
            # 收集系统回答（不包括思考内容）
            elif event.get("type") == "content_delta":
                system_answer += event.get("text", "")
            
            # 收集思考过程（但不包含在最终答案中）
            elif event.get("type") == "reasoning_delta":
                reasoning_text += event.get("text", "")
            
            # 流程结束
            elif event.get("type") == "pipeline_end":
                break
                
    except Exception as e:
        logger.error(f"处理问题时出错: {str(e)}")
        system_answer = f"处理错误: {str(e)}"
    
    return {
        "question": question,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "system_answer": system_answer.strip()
    }

async def evaluate_dataset(dataset_name: str, config: Dict[str, str]) -> None:
    """
    评估单个数据集（示例版本）
    """
    logger.info(f"开始评估数据集: {dataset_name} (示例模式，处理前{SAMPLE_SIZE}个问题)")
    
    # 创建输出目录
    output_file = Path(config["output_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
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
            questions_data = []
            for line in f:
                line = line.strip()
                if line:
                    questions_data.append(json.loads(line))
        
        # 限制问题数量
        questions_data = questions_data[:SAMPLE_SIZE]
        logger.info(f"处理前 {len(questions_data)} 个问题")
        
        # 处理每个问题
        results = []
        for i, item in enumerate(questions_data):
            question = item.get("question", "")
            if not question:
                continue
                
            logger.info(f"处理问题 {i+1}/{len(questions_data)}: {question[:50]}...")
            
            result = await process_single_question(question, kb_instance)
            
            # 添加原始数据中的其他信息
            result.update({
                "original_id": item.get("id", ""),
                "ground_truth_answer": item.get("answer", "")
            })
            
            results.append(result)
            
            # 每处理10个问题保存一次（防止数据丢失）
            if (i + 1) % 10 == 0:
                logger.info(f"已处理 {i+1} 个问题，中间保存结果...")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存最终结果
        logger.info(f"保存最终结果到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据集 {dataset_name} 评估完成，共处理 {len(results)} 个问题")
        
    except Exception as e:
        logger.error(f"评估数据集 {dataset_name} 时出错: {str(e)}")
        raise
    finally:
        # 恢复原始配置
        config_module.PROCESSED_DATA_DIR = original_processed_dir

async def main():
    """
    主函数：评估所有数据集（示例版本）
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="评估RAG系统在多个数据集上的性能（示例版本）")
    parser.add_argument("--dataset", type=str, choices=list(DATASETS.keys()) + ["all"], 
                       default="all", help="要评估的数据集")
    
    args = parser.parse_args()
    
    if args.dataset == "all":
        datasets_to_process = DATASETS.items()
    else:
        datasets_to_process = [(args.dataset, DATASETS[args.dataset])]
    
    logger.info(f"开始示例评估，每个数据集处理前{SAMPLE_SIZE}个问题")
    logger.info(f"将处理数据集: {[name for name, _ in datasets_to_process]}")
    
    for dataset_name, config in datasets_to_process:
        try:
            await evaluate_dataset(dataset_name, config)
        except Exception as e:
            logger.error(f"数据集 {dataset_name} 评估失败: {str(e)}")
            continue
    
    logger.info("所有数据集示例评估完成")

if __name__ == "__main__":
    asyncio.run(main())