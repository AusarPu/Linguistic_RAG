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
from .query_rewriter import generate_rewritten_query
from .useful_judger import judge_knowledge_usefulness
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
                         _KEYWORD + _BROADENED_QUESTION,
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
        if res_or_exc:
            logger.info(f"[{flow_request_id}] 召回路径 '{path_name}' 返回 {len(res_or_exc)} 个结果。")
            for chunk_data in res_or_exc:
                chunk_id = chunk_data.get("chunk_id")
                retrieval_score = chunk_data.get('retrieval_score')
                if chunk_id not in all_retrieved_chunks_map:
                    all_retrieved_chunks_map[chunk_id] = chunk_data
                    all_retrieved_chunks_map[chunk_id].setdefault('retrieved_from_paths', {})[
                        path_name] = retrieval_score
                else:
                    # 如果块已通过其他路径召回，添加来源并记录分数
                    all_retrieved_chunks_map[chunk_id].setdefault('retrieved_from_paths', {})[
                        path_name] = retrieval_score
    

    candidate_chunks_for_reranker = list(all_retrieved_chunks_map.values())
    retrieval_duration = time.time() - retrieval_start_time
    logger.info(
        f"[{flow_request_id}] RAG Flow STAGE: Retrieval complete. Found {len(candidate_chunks_for_reranker)} unique candidates. Duration: {retrieval_duration:.3f}s")

    preview_for_ui_retrieved = [{"id": c.get("chunk_id"),
                                    "text_preview": c.get("text", ""),
                                    "from_paths": list(c.get("retrieved_from_paths", {}).keys()),
                                    "scores": c.get("retrieved_from_paths", {})  # 也发送原始分数
                                    } for c in candidate_chunks_for_reranker]
    yield {"type": "retrieved_chunks_preview", "count": len(candidate_chunks_for_reranker),
            "preview": preview_for_ui_retrieved}

    if not candidate_chunks_for_reranker:
        yield _build_status_event("no_context_found_after_retrieval", "未能从知识库中找到与查询相关的上下文信息。")
        yield {"type": "content_delta", "text": "抱歉，我没有找到与您问题相关的直接信息。"}  # 给前端一个友好的提示
        yield {"type": "pipeline_end", "reason": "no_context_found_after_retrieval"}
        return

    # 3. 并发判断知识块的有用性
    yield _build_status_event("usefulness_judging", "步骤3: 正在判断知识块的相关性...")
    usefulness_start_time = time.time()

    # 创建判断任务列表
    judge_tasks = []
    for chunk in candidate_chunks_for_reranker:
        task = asyncio.create_task(
            asyncio.to_thread(
                judge_knowledge_usefulness,
                questions=[_QUESTION]+_BROADENED_QUESTION,
                knowledge_content=chunk.get("text", "")
            )
        )
        judge_tasks.append((chunk, task))

    # 等待所有判断任务完成
    useful_chunks = []
    for chunk, task in judge_tasks:
        try:
            result = await task
            if result == "useful":
                useful_chunks.append(chunk)
        except Exception as e:
            logger.error(f"[{flow_request_id}] 判断知识块有用性时发生错误: {str(e)}")

    usefulness_duration = time.time() - usefulness_start_time
    logger.info(
        f"[{flow_request_id}] RAG Flow STAGE: Usefulness judging complete. {len(useful_chunks)}/{len(candidate_chunks_for_reranker)} chunks kept. Duration: {usefulness_duration:.3f}s")

    # 更新候选chunks列表
    candidate_chunks_for_reranker = useful_chunks

    if not candidate_chunks_for_reranker:
        yield _build_status_event("no_context_found_after_usefulness", "筛选后未找到有用的上下文信息。")
        yield {"type": "content_delta", "text": "抱歉，我没有找到与您问题直接相关的有用信息。"}
        yield {"type": "pipeline_end", "reason": "no_context_found_after_usefulness"}
        return

    preview_for_ui_useful = [{"id": c.get("chunk_id"),
                             "text_preview": c.get("text", ""),
                             "from_paths": list(c.get("retrieved_from_paths", {}).keys()),
                             } for c in candidate_chunks_for_reranker]
    yield {"type": "useful_chunks_preview", "count": len(candidate_chunks_for_reranker),
           "preview": preview_for_ui_useful}

    # 4. 构建最终上下文并生成答案
    yield _build_status_event("generation_start", "步骤4: 正在构建提示并生成答案...")
    final_context_chunks_for_llm = preview_for_ui_useful
    context_segments = []
    for idx, chunk_data in enumerate(final_context_chunks_for_llm):
        # 确保 chunk_data 包含 'text'， 'doc_name', 'page_number', 'chunk_id'
        # call_reranker_vllm 返回的块应该已经包含了这些原始元数据
        segment = (f"【相关片段 {idx + 1} "
                    f"(文档: {chunk_data.get('doc_name', '未知')}, "
                    f"页码: {chunk_data.get('page_number', '未知')}, "
                    f"块ID: {chunk_data.get('chunk_id')})】\n"
                    f"{chunk_data.get('text', '')}")
        context_segments.append(segment)
    context_text_for_llm = "\n\n---\n\n".join(context_segments)

    user_content_for_generator = (
        f"上下文信息：\n\"\"\"\n{context_text_for_llm}\n\"\"\"\n\n"
        f"请仔细阅读并理解以上所有上下文信息，然后用中文回答以下用户问题。"
        f"请确保你的回答完全基于以上提供的上下文信息，不要凭空捏造或使用你自己的外部知识。"
        f"如果上下文中没有足够信息来回答，请明确指出。\n"
        f"用户问题：\n{rewritten_query}"
    )

    logger.info(
        f"[{flow_request_id}] [GENERATION_INPUT_PREVIEW] System: '{GENERATOR_SYSTEM_PROMPT_CONTENT[:100] if GENERATOR_SYSTEM_PROMPT_CONTENT else 'None'}...', User (context+query): '{user_content_for_generator[:100]}...'")
    yield {"type": "llm_input_preview",
            "system_prompt_used": bool(GENERATOR_SYSTEM_PROMPT_CONTENT),
            "user_content_length": len(user_content_for_generator)}

    full_final_answer_text = ""
    full_reasoning_text = ""  # 用于累积思考过程

    async for gen_event in call_generator_vllm_stream(
            GENERATOR_SYSTEM_PROMPT_CONTENT,  # 使用已加载的系统提示
            user_content_for_generator
            # 其他参数如API URL, model_name, generation_config, request_timeout 会用函数默认值 (来自config)
    ):
        yield gen_event  # 直接转发给调用者 (app_gradio.py 会处理 'reasoning_delta' 和 'content_delta')

        if gen_event.get("type") == "content_delta":
            full_final_answer_text += gen_event.get("text", "")
        elif gen_event.get("type") == "reasoning_delta":
            full_reasoning_text += gen_event.get("text", "")
        elif gen_event.get("type") == "error" or \
                (gen_event.get("type") == "stream_end" and gen_event.get("reason") != "stop"):
            logger.error(f"[{flow_request_id}] LLM Generation stream ended prematurely or with error: {gen_event}")
            if not full_final_answer_text and not full_reasoning_text:
                full_final_answer_text = f"(LLM生成错误或提前终止: {gen_event.get('message', gen_event.get('reason'))})"
            if gen_event.get("type") != "stream_end":
                yield {"type": "stream_end", "reason": gen_event.get('reason', "error_in_generation")}
            return

    logger.info(f"[{flow_request_id}] RAG Flow STAGE: Generation complete.")
    yield {"type": "final_answer_complete",
            "full_text": full_final_answer_text.strip(),
            "full_reasoning": full_reasoning_text.strip(),
            "final_context_chunk_ids": [c.get("chunk_id") for c in final_context_chunks_for_llm]}
    yield _build_status_event("generation_complete", "答案生成完毕。")
    total_duration = time.time() - start_time_total
    logger.info(f"[{flow_request_id}] RAG流程处理完毕 (总耗时: {total_duration:.3f}s)。")
    yield {"type": "pipeline_end", "reason": "flow_completed"}
    

   


if __name__ == "__main__":
    # 测试用例
    async def test_rag_flow():
        # 初始化知识库实例
        kb = KnowledgeBase()
        
        # 模拟用户查询和聊天历史
        test_query = "从他的视角怎么看待焚书坑儒?"
        test_chat_history = [
            {"role": "user", "content": "孔子是谁"},
            {"role": "assistant", "content": "孔子是中国的一个重要的思想家、哲学家、教育家。"}
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
