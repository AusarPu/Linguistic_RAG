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

# 全局加载系统提示 (或者在 app_gradio.py 的 load_all_resources 中加载并传入)
GENERATOR_SYSTEM_PROMPT_CONTENT: Optional[str] = None
try:
    if GENERATOR_SYSTEM_PROMPT_FILE and os.path.exists(GENERATOR_SYSTEM_PROMPT_FILE):
        with open(GENERATOR_SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            GENERATOR_SYSTEM_PROMPT_CONTENT = f.read()
        logger.info(f"成功从 {GENERATOR_SYSTEM_PROMPT_FILE} 加载生成器系统提示。")
    else:
        logger.warning(f"生成器系统提示文件路径 '{GENERATOR_SYSTEM_PROMPT_FILE}' 未配置或文件不存在。")
        GENERATOR_SYSTEM_PROMPT_CONTENT = "你是一个耐心、友好、乐于助人的AI助手。"  # 默认或备用
except Exception as e:
    logger.error(f"加载生成器系统提示时出错: {e}", exc_info=True)
    GENERATOR_SYSTEM_PROMPT_CONTENT = "你是一个耐心、友好、乐于助人的AI助手。"


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

    try:
        yield _build_status_event("pipeline_start", "RAG流程启动")

        # 1. 查询重写
        yield _build_status_event("query_rewriting", "步骤1: 正在进行查询重构...")
        rewritten_query = await generate_rewritten_query(messages=chat_history_openai, user_input=user_query)
        yield {"type": "rewritten_query_result", "original_query": user_query, "rewritten_text": rewritten_query}
        logger.info(
            f"[{flow_request_id}] RAG Flow STAGE: Query Rewriting complete. Rewritten: '{rewritten_query[:50]}...'")

        # 2. 多路并行召回
        yield _build_status_event("retrieval_start", "步骤2: 开始多路并行召回...")
        retrieval_start_time = time.time()

        # 假设 KnowledgeBase 的 search_* 方法是同步的，用 asyncio.to_thread 包装
        # 如果已经是 async def，则直接 await kb_instance.search_...(...)
        task_dense_chunks = asyncio.to_thread(kb_instance.search_dense_chunks, rewritten_query,
                                              DENSE_CHUNK_RETRIEVAL_TOP_K, DENSE_CHUNK_THRESHOLD)
        task_dense_questions = asyncio.to_thread(kb_instance.search_dense_questions, user_query,
                                                 DENSE_QUESTION_RETRIEVAL_TOP_K, DENSE_QUESTION_THRESHOLD)
        task_sparse_keywords = asyncio.to_thread(kb_instance.search_sparse_keywords, rewritten_query,
                                                 SPARSE_KEYWORD_RETRIEVAL_TOP_K, SPARSE_KEYWORD_THRESHOLD)

        retrieval_outputs = await asyncio.gather(task_dense_chunks, task_dense_questions, task_sparse_keywords,
                                                 return_exceptions=True)

        all_retrieved_chunks_map: Dict[str, Dict[str, Any]] = {}
        retrieval_paths_display_names = ["文本稠密召回", "问题稠密召回", "关键词稀疏召回"]

        for i, res_or_exc in enumerate(retrieval_outputs):
            path_name = retrieval_paths_display_names[i]
            if isinstance(res_or_exc, Exception):
                logger.error(f"[{flow_request_id}] 召回路径 '{path_name}' 执行时发生错误: {res_or_exc}")
                yield {"type": "error", "stage": f"retrieval_{path_name}",
                       "message": f"召回路径 '{path_name}' 失败."}  # 简化错误消息
            elif isinstance(res_or_exc, list):
                logger.info(f"[{flow_request_id}] 召回路径 '{path_name}' 返回 {len(res_or_exc)} 个结果。")
                for chunk_data in res_or_exc:
                    chunk_id = chunk_data.get("chunk_id")
                    if not chunk_id: continue
                    if chunk_id not in all_retrieved_chunks_map:
                        all_retrieved_chunks_map[chunk_id] = chunk_data
                        # 确保 'retrieved_from_paths' 存在并且是字典
                        all_retrieved_chunks_map[chunk_id].setdefault('retrieved_from_paths', {})[
                            path_name] = chunk_data.get('retrieval_score')
                    else:
                        # 如果块已通过其他路径召回，添加来源并记录分数
                        all_retrieved_chunks_map[chunk_id].setdefault('retrieved_from_paths', {})[
                            path_name] = chunk_data.get('retrieval_score')

        candidate_chunks_for_reranker = list(all_retrieved_chunks_map.values())
        retrieval_duration = time.time() - retrieval_start_time
        logger.info(
            f"[{flow_request_id}] RAG Flow STAGE: Retrieval complete. Found {len(candidate_chunks_for_reranker)} unique candidates. Duration: {retrieval_duration:.3f}s")

        preview_for_ui_retrieved = [{"id": c.get("chunk_id"),
                                     "text_preview": c.get("text", "")[:30] + "...",  # 缩短预览
                                     "from_paths": list(c.get("retrieved_from_paths", {}).keys()),
                                     "scores": c.get("retrieved_from_paths", {})  # 也发送原始分数
                                     } for c in candidate_chunks_for_reranker[:RERANKER_TOP_N_INPUT_MAX][:5]]
        yield {"type": "retrieved_chunks_preview", "count": len(candidate_chunks_for_reranker),
               "preview": preview_for_ui_retrieved}

        if not candidate_chunks_for_reranker:
            yield _build_status_event("no_context_found_after_retrieval", "未能从知识库中找到与查询相关的上下文信息。")
            yield {"type": "content_delta", "text": "抱歉，我没有找到与您问题相关的直接信息。"}  # 给前端一个友好的提示
            yield {"type": "pipeline_end", "reason": "no_context_found_after_retrieval"}
            return

        # 3. 重排 (Reranking)
        reranker_input_candidates = candidate_chunks_for_reranker[:RERANKER_TOP_N_INPUT_MAX]
        yield _build_status_event("reranking_start",
                                  f"步骤3: 开始对 {len(reranker_input_candidates)} 个候选块进行重排...")
        rerank_start_time = time.time()

        reranked_chunks_list = await call_reranker_vllm(
            query=rewritten_query,
            chunks_to_rerank=reranker_input_candidates
        )  # call_reranker_vllm 内部会处理配置
        rerank_duration = time.time() - rerank_start_time
        logger.info(f"[{flow_request_id}] RAG Flow STAGE: Reranking complete. Duration: {rerank_duration:.3f}s")

        final_context_chunks_for_llm = reranked_chunks_list[:GENERATOR_CONTEXT_TOP_N]

        context_for_display_list_reranked = []
        for chunk in final_context_chunks_for_llm:
            context_for_display_list_reranked.append({
                "chunk_id": chunk.get("chunk_id"),
                "doc_name": chunk.get("doc_name", "N/A"),
                "page_number": chunk.get("page_number"),
                "text_preview": chunk.get("text", "")[:100] + "...",
                "rerank_score": chunk.get("rerank_score"),
                "retrieved_from_paths": chunk.get("retrieved_from_paths", {})
            })
        yield {"type": "reranked_context_for_display", "chunks": context_for_display_list_reranked}

        if not final_context_chunks_for_llm:
            yield _build_status_event("no_context_found_after_reranking", "重排后未能筛选出合适的上下文信息。")
            yield {"type": "content_delta", "text": "抱歉，对信息进行排序后未能找到足够相关的内容来回答您的问题。"}
            yield {"type": "pipeline_end", "reason": "no_context_found_after_reranking"}
            return

        # 4. 构建最终上下文并生成答案
        yield _build_status_event("generation_start", "步骤4: 正在构建提示并生成答案...")

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

    except Exception as e:
        logger.error(f"[{flow_request_id}] RAG流程中发生顶层未知错误: {e}", exc_info=True)
        yield {"type": "error", "stage": "unknown_pipeline_error", "message": f"RAG流程发生严重内部错误: {str(e)}"}
    finally:
        total_duration = time.time() - start_time_total
        logger.info(f"[{flow_request_id}] RAG流程处理完毕 (总耗时: {total_duration:.3f}s)。")
        yield {"type": "pipeline_end", "reason": "flow_completed"}

