# script/rag_pipeline.py

import asyncio
import os
import sys
import time
import logging
import json
from typing import List, Dict, Any, Tuple, AsyncGenerator, Optional

import aiohttp  # 仅当 VLLM 客户端函数也在此文件时直接需要

# --- 从项目中导入 ---
from .knowledge_base import KnowledgeBase
from .query_rewriter import generate_rewritten_query  # 假设它在 query_rewriter.py 中且是异步的
from .vllm_clients import call_reranker_vllm, call_generator_vllm_stream


from .config_rag import (
    # 检索参数
    DENSE_CHUNK_RETRIEVAL_TOP_K, DENSE_QUESTION_RETRIEVAL_TOP_K, SPARSE_KEYWORD_RETRIEVAL_TOP_K,
    DENSE_CHUNK_THRESHOLD, DENSE_QUESTION_THRESHOLD, SPARSE_KEYWORD_THRESHOLD,
    # Reranker 参数
    RERANKER_TOP_N_INPUT_MAX,
    # Generator 参数
    GENERATOR_CONTEXT_TOP_N, GENERATOR_SYSTEM_PROMPT_FILE,
    # (其他如API URL, MODEL NAME等由各客户端函数内部从config获取)
)

logger = logging.getLogger(__name__)

# 全局加载系统提示
with open(GENERATOR_SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
    GENERATOR_SYSTEM_PROMPT_CONTENT = f.read()


async def execute_rag_flow(
        user_query: str,
        chat_history_openai: List[Dict[str, str]],
        kb_instance: KnowledgeBase,
        # 你也可以将 reranker_client_fn, generator_client_fn 作为参数传入，以增加灵活性
        # 或者让它们直接从本模块或 vllm_clients.py 导入
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    执行完整的RAG流程，并异步yield事件。
    """
    start_time_total = time.time()
    flow_request_id = f"ragflow_{int(start_time_total)}_{hash(user_query + str(time.time_ns())) % 10000}"
    logger.info(f"[{flow_request_id}] RAG Flow STAGE: Pipeline Start. Query: '{user_query[:50]}...'")

    # 辅助函数用于 yield 状态事件
    def _build_status_event(stage: str, message: str, data: Optional[Any] = None) -> Dict[str, Any]:
        event = {"type": "status", "stage": stage, "message": message}
        if data: event["data"] = data
        logger.info(f"[{flow_request_id}] [STATUS] {stage}: {message}" + (
            f" Data (preview): {str(data)[:100]}..." if data else ""))
        return event

    yield _build_status_event("pipeline_start", "RAG流程启动")

    # 1. 查询重写
    yield _build_status_event("query_rewriting", "步骤1: 正在进行查询重构...")
    rewritten_query = generate_rewritten_query(messages=chat_history_openai, user_input=user_query)
    _QUESTION = rewritten_query["question"]
    _BROADENED_QUESTION = rewritten_query["broadened_question"]
    _KEYWORD = rewritten_query["keyword"]

    yield {"type": "rewritten_query_result", "original_query": user_query, "rewritten_text": rewritten_query}
    logger.info(
        f"[{flow_request_id}] RAG Flow STAGE: Query Rewriting complete. Rewritten: '{rewritten_query}'")

    # 2. 多路并行召回
    yield _build_status_event("retrieval_start", "步骤2: 开始多路并行召回...")
    retrieval_start_time = time.time()

    # 创建三个异步任务
    tasks = [
        asyncio.to_thread(kb_instance.search_dense_chunks, 
                         [_QUESTION] + _BROADENED_QUESTION,
                         DENSE_CHUNK_RETRIEVAL_TOP_K, 
                         DENSE_CHUNK_THRESHOLD),
        asyncio.to_thread(kb_instance.search_dense_keywords,
                         [_KEYWORD],
                         SPARSE_KEYWORD_RETRIEVAL_TOP_K,
                         SPARSE_KEYWORD_THRESHOLD),
        asyncio.to_thread(kb_instance.search_dense_questions,
                         [_QUESTION] + _BROADENED_QUESTION,
                         DENSE_QUESTION_RETRIEVAL_TOP_K,
                         DENSE_QUESTION_THRESHOLD)
    ]
    
    # 并行执行所有任务
    full_text_chunks, keyword_chunks, question_chunks = await asyncio.gather(*tasks)

    all_retrieved_chunks_map: Dict[str, Dict[str, Any]] = {}
    retrieval_paths_display_names = ["文本召回", "关键词召回","问题召回", ]
    retrieval_outputs = [full_text_chunks, keyword_chunks, question_chunks]

    for i, res_or_exc in enumerate(retrieval_outputs):
        path_name = retrieval_paths_display_names[i]
        if isinstance(res_or_exc, Exception):
            logger.error(f"[{flow_request_id}] 召回路径 '{path_name}' 执行时发生错误: {res_or_exc}")
            yield {"type": "error", "stage": f"retrieval_{path_name}",
                    "message": f"召回路径 '{path_name}' 失败."}
        elif isinstance(res_or_exc, list):
            logger.info(f"[{flow_request_id}] 召回路径 '{path_name}' 返回 {len(res_or_exc)} 个结果。")
            for chunk_data in res_or_exc:
                chunk_id = chunk_data.get("chunk_id")
                chunk_content = chunk_data.get("content", "")[:50]  # 获取前50个字
                retrieval_score = chunk_data.get('retrieval_score')
                
                # 打印召回信息
                print(f"块ID: {chunk_id}")
                print(f"内容预览: {chunk_content}...")
                print(f"召回方式: {path_name}")
                print(f"召回分数: {retrieval_score}\n")
                
                if chunk_id not in all_retrieved_chunks_map:
                    all_retrieved_chunks_map[chunk_id] = chunk_data
                    all_retrieved_chunks_map[chunk_id].setdefault('retrieved_from_paths', {})[
                        path_name] = retrieval_score
                else:
                    # 如果块已通过其他路径召回，添加来源并记录分数
                    all_retrieved_chunks_map[chunk_id].setdefault('retrieved_from_paths', {})[
                        path_name] = retrieval_score
   


if __name__ == "__main__":
    # 测试用例
    async def test_rag_flow():
        # 初始化知识库实例
        kb = KnowledgeBase()
        
        # 模拟用户查询和聊天历史
        test_query = "从孔子的视角怎么看待焚书坑儒?"
        test_chat_history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好!我能帮你什么?"}
        ]
        
        # 执行RAG流程并打印结果
        print("开始测试RAG流程...")
        async for event in execute_rag_flow(
            user_query=test_query,
            chat_history_openai=test_chat_history,
            kb_instance=kb
        ):
            print(f"收到事件: {json.dumps(event, ensure_ascii=False, indent=2)}")
    
    # 运行测试
    asyncio.run(test_rag_flow())
