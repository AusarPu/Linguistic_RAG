# script/rag_pipeline.py

import asyncio
import time
import logging
import json
from typing import List, Dict, Any,AsyncGenerator, Optional
# --- 从项目中导入 ---
from .knowledge_base import KnowledgeBase
from .query_rewriter import  generate_rewritten_query_async
from .useful_judger import judge_knowledge_usefulness
from .vllm_clients import call_generator_vllm_stream, async_rank_with_reranker


from .config_rag import (
    # 检索参数
    DENSE_CHUNK_RETRIEVAL_TOP_K, DENSE_QUESTION_RETRIEVAL_TOP_K, SPARSE_KEYWORD_RETRIEVAL_TOP_K,
    DENSE_CHUNK_THRESHOLD, DENSE_QUESTION_THRESHOLD, SPARSE_KEYWORD_THRESHOLD, GENERATOR_SYSTEM_PROMPT_FILE,
    # 软保留策略参数
    SOFT_KEEP_MIN_CHUNKS, SOFT_KEEP_RATIO,
    # 有用性判断并发上限
    USEFULNESS_MAX_CONCURRENT_REQUESTS,
    FINAL_CONTEXT_TOP_K,
    RERANKER_SCORE_THRESHOLD,
)

logger = logging.getLogger(__name__)

# 全局加载系统提示
with open(GENERATOR_SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
    GENERATOR_SYSTEM_PROMPT_CONTENT = f.read()


async def execute_rag_flow(
        user_query: str,
        chat_history_openai: List[Dict[str, str]],
        kb_instance: KnowledgeBase,
        use_query_rewriter: bool = True,
        use_dense_chunks: bool = True,
        use_dense_keywords: bool = True,
        use_dense_questions: bool = True,
        use_usefulness_judger: bool = True,
        use_bm25_chunks_only: bool = False,
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
    if use_query_rewriter:
        yield _build_status_event("query_rewriting", "步骤1: 正在进行查询重构...")
        rewritten_query = await generate_rewritten_query_async(messages=chat_history_openai, user_input=user_query)
        _QUESTION = rewritten_query["question"]
        _BROADENED_QUESTION = rewritten_query["broadened_question"]
        _KEYWORD = rewritten_query["keyword"]

        yield {"type": "rewritten_query_result", "original_query": user_query, "rewritten_text": rewritten_query}
        logger.info(
            f"[{flow_request_id}] RAG Flow STAGE: Query Rewriting complete. Rewritten: '{rewritten_query}'")
    else:
        yield _build_status_event("query_rewriting", "步骤1: 跳过查询重构（消融实验）")
        # 不进行查询重写，直接使用原始查询
        _QUESTION = user_query
        _BROADENED_QUESTION = []
        _KEYWORD = []
        
        yield {"type": "rewritten_query_result", "original_query": user_query, "rewritten_text": {"question": user_query, "broadened_question": [], "keyword": []}}
        logger.info(f"[{flow_request_id}] RAG Flow STAGE: Query Rewriting skipped (ablation study)")

    # 2. 多路并行召回
    yield _build_status_event("retrieval_start", "步骤2: 开始多路并行召回...")
    retrieval_start_time = time.time()

    # 创建检索任务列表，根据消融实验参数决定启用哪些检索路径
    tasks = []
    retrieval_paths_display_names = []
    
    if use_dense_chunks:
        tasks.append(kb_instance.search_dense_chunks(
            [_QUESTION],
            DENSE_CHUNK_RETRIEVAL_TOP_K,
            DENSE_CHUNK_THRESHOLD
        ))
        retrieval_paths_display_names.append("文本召回")
    
    if use_dense_keywords:
        tasks.append(kb_instance.search_dense_keywords(
            _KEYWORD,
            SPARSE_KEYWORD_RETRIEVAL_TOP_K,
            SPARSE_KEYWORD_THRESHOLD
        ))
        retrieval_paths_display_names.append("关键词召回")
    
    if use_dense_questions:
        tasks.append(kb_instance.search_dense_questions(
            [_QUESTION] + _BROADENED_QUESTION,
            DENSE_QUESTION_RETRIEVAL_TOP_K,
            DENSE_QUESTION_THRESHOLD
        ))
        retrieval_paths_display_names.append("问题召回")

    if use_bm25_chunks_only:
        tasks.append(kb_instance.search_bm25_chunks_only([
            _QUESTION
        ], 10))
        retrieval_paths_display_names.append("BM25召回")
    
    # 如果所有检索路径都被禁用，返回错误
    if not tasks:
        yield _build_status_event("no_retrieval_paths", "所有检索路径都被禁用，无法进行检索。")
        yield {"type": "content_delta", "text": "错误：所有检索路径都被禁用，无法进行检索。"}
        yield {"type": "pipeline_end", "reason": "no_retrieval_paths_enabled"}
        return
    
    # 并行执行启用的检索任务
    retrieval_outputs = await asyncio.gather(*tasks)

    all_retrieved_chunks_map: Dict[str, Dict[str, Any]] = {}

    path_rankings = {}
    for i, res_or_exc in enumerate(retrieval_outputs):
        path_name = retrieval_paths_display_names[i]
        if res_or_exc:
            logger.info(f"[{flow_request_id}] 召回路径 '{path_name}' 返回 {len(res_or_exc)} 个结果。")
            ranks_for_path = {}
            for rank, chunk_data in enumerate(res_or_exc, start=1):
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
                if chunk_id and chunk_id not in ranks_for_path:
                    ranks_for_path[chunk_id] = rank
            path_rankings[path_name] = ranks_for_path
    

    candidate_chunks_for_reranker = list(all_retrieved_chunks_map.values())
    if len(candidate_chunks_for_reranker) > 0:
        rerank_start_time = time.time()
        pairs = [{"query": _QUESTION, "doc": c.get("text", "")} for c in candidate_chunks_for_reranker]
        scores = await async_rank_with_reranker(pairs, instruction=None)
        for c, s in zip(candidate_chunks_for_reranker, scores):
            c["reranker_score"] = float(s)
        candidate_chunks_for_reranker = [
            c for c in candidate_chunks_for_reranker
            if c.get("reranker_score", 0.0) >= RERANKER_SCORE_THRESHOLD
        ]
        candidate_chunks_for_reranker = sorted(
            candidate_chunks_for_reranker,
            key=lambda c: c.get("reranker_score", 0.0),
            reverse=True
        )
        if isinstance(FINAL_CONTEXT_TOP_K, int) and FINAL_CONTEXT_TOP_K > 0:
            candidate_chunks_for_reranker = candidate_chunks_for_reranker[:FINAL_CONTEXT_TOP_K]
        rerank_duration = time.time() - rerank_start_time
        logger.info(f"[{flow_request_id}] Reranker scoring complete. Duration: {rerank_duration:.3f}s. Kept {len(candidate_chunks_for_reranker)} candidates.")
        if candidate_chunks_for_reranker:
            summary = ", ".join([
                f"{c.get('chunk_id')}({c.get('doc_name', '未知')})={c.get('reranker_score', 0.0):.3f}"
                for c in candidate_chunks_for_reranker
            ])
            logger.info(f"[{flow_request_id}] Reranker kept: {summary}")
        
    retrieval_duration = time.time() - retrieval_start_time
    logger.info(
        f"[{flow_request_id}] RAG Flow STAGE: Retrieval complete. Found {len(candidate_chunks_for_reranker)} unique candidates. Duration: {retrieval_duration:.3f}s")
    yield {"type": "timing", "stage": "retrieval", "duration_ms": int(retrieval_duration * 1000)}

    preview_for_ui_retrieved = [{"id": c.get("chunk_id"),
                                    "text_preview": c.get("text", ""),
                                    "from_paths": list(c.get("retrieved_from_paths", {}).keys()),
                                    "scores": {**c.get("retrieved_from_paths", {}), "reranker_score": c.get("reranker_score", 0.0)}
                                    } for c in candidate_chunks_for_reranker]
    yield {"type": "retrieved_chunks_preview", "count": len(candidate_chunks_for_reranker),
            "preview": preview_for_ui_retrieved}

    if not candidate_chunks_for_reranker:
        yield _build_status_event("no_context_found_after_retrieval", "未能从知识库中找到与查询相关的上下文信息。")
        yield {"type": "content_delta", "text": "抱歉，我没有找到与您问题相关的直接信息。"}  # 给前端一个友好的提示
        yield {"type": "pipeline_end", "reason": "no_context_found_after_retrieval"}
        return

    # 在进行有用性判断之前，保存原始候选用于软保留回退
    original_candidates_for_soft_keep = list(candidate_chunks_for_reranker)

    # 3. 并发判断知识块的有用性
    if use_usefulness_judger:
        yield _build_status_event("usefulness_judging", "步骤3: 正在判断知识块的相关性...")
        usefulness_start_time = time.time()

        # 创建判断任务列表 - 使用受限并发的异步版本
        semaphore = asyncio.Semaphore(USEFULNESS_MAX_CONCURRENT_REQUESTS)

        async def _judge_chunk_usefulness(chunk_obj):
            """包装器以限制并发"""
            async with semaphore:
                return await judge_knowledge_usefulness(
                    questions=[_QUESTION],
                    knowledge_content=chunk_obj.get("text", "")
                )

        judge_tasks = [asyncio.create_task(_judge_chunk_usefulness(chunk)) for chunk in candidate_chunks_for_reranker]

        # 并发收集结果
        results = await asyncio.gather(*judge_tasks, return_exceptions=True)
        useful_chunks = []
        for chunk, result in zip(candidate_chunks_for_reranker, results):
            if isinstance(result, Exception):
                logger.error(f"[{flow_request_id}] 判断知识块有用性时发生错误: {str(result)}")
                continue
            if result == "useful":
                useful_chunks.append(chunk)

        usefulness_duration = time.time() - usefulness_start_time
        logger.info(
            f"[{flow_request_id}] RAG Flow STAGE: Usefulness judging complete. {len(useful_chunks)}/{len(candidate_chunks_for_reranker)} chunks kept. Duration: {usefulness_duration:.3f}s")
        yield {"type": "timing", "stage": "usefulness", "duration_ms": int(usefulness_duration * 1000)}

        # 更新候选chunks列表
        candidate_chunks_for_reranker = useful_chunks

        # --- 软保留策略：当筛后数量过少时，保留一部分高分原始候选，避免证据链断裂 ---
        try:
            soft_keep_target = max(SOFT_KEEP_MIN_CHUNKS, int(len(original_candidates_for_soft_keep) * SOFT_KEEP_RATIO))
        except Exception:
            soft_keep_target = SOFT_KEEP_MIN_CHUNKS

        if len(candidate_chunks_for_reranker) < soft_keep_target:
            # 计算综合分数：考虑各召回路径分数与顶层retrieval_score
            def _combined_score(c):
                scores = []
                rp = c.get("retrieved_from_paths", {})
                if isinstance(rp, dict):
                    for s in rp.values():
                        if isinstance(s, (int, float)):
                            scores.append(float(s))
                s_top = c.get("retrieval_score")
                if isinstance(s_top, (int, float)):
                    scores.append(float(s_top))
                return max(scores) if scores else 0.0

            kept_ids = {c.get("chunk_id") for c in candidate_chunks_for_reranker}
            fallback_pool = [c for c in original_candidates_for_soft_keep if c.get("chunk_id") not in kept_ids]
            fallback_sorted = sorted(fallback_pool, key=_combined_score, reverse=True)

            need = max(0, soft_keep_target - len(candidate_chunks_for_reranker))
            additional = []
            for c in fallback_sorted[:need]:
                cc = dict(c)
                cc["soft_kept"] = True
                additional.append(cc)

            if additional:
                candidate_chunks_for_reranker = candidate_chunks_for_reranker + additional
                yield _build_status_event(
                    "soft_keep_applied",
                    f"启用软保留策略，额外保留 {len(additional)} 个上下文。",
                    {"target": soft_keep_target}
                )
    else:
        yield _build_status_event("usefulness_judging", "步骤3: 跳过有用性判断（消融实验）")
        logger.info(f"[{flow_request_id}] RAG Flow STAGE: Usefulness judging skipped (ablation study)")

    if not candidate_chunks_for_reranker:
        if use_usefulness_judger:
            yield _build_status_event("no_context_found_after_usefulness", "筛选后未找到有用的上下文信息。")
            yield {"type": "content_delta", "text": "抱歉，我没有找到与您问题直接相关的有用信息。"}
            yield {"type": "pipeline_end", "reason": "no_context_found_after_usefulness"}
        else:
            yield _build_status_event("no_context_found_after_retrieval", "未能从知识库中找到与查询相关的上下文信息。")
            yield {"type": "content_delta", "text": "抱歉，我没有找到与您问题相关的直接信息。"}
            yield {"type": "pipeline_end", "reason": "no_context_found_after_retrieval"}
        return

    # 输出软保留的摘要信息（如果有）
    soft_kept_count = sum(1 for c in candidate_chunks_for_reranker if c.get("soft_kept"))
    if soft_kept_count:
        yield {"type": "soft_keep_summary", "added": soft_kept_count}

    preview_for_ui_useful = [{"id": c.get("chunk_id"),
                         "text_preview": c.get("text", ""),
                         "text": c.get("text", ""),  # 保留完整文本
                         "doc_name": c.get("doc_name", "未知"),  # 保留文档名
                         "page_number": c.get("page_number", "未知"),  # 保留页码
                         "author": c.get("author", "未知"),  # 保留作者
                         "chunk_id": c.get("chunk_id"),  # 保留块ID
                         "from_paths": list(c.get("retrieved_from_paths", {}).keys()),
                         "soft_kept": bool(c.get("soft_kept", False)),  # 标记是否为软保留补充
                         } for c in candidate_chunks_for_reranker]
    
    # 构建知识库内容作为tool role消息
    knowledge_content_for_tool = "\n\n---\n\n".join([
        f"【相关片段 {idx + 1} "
        f"(文档: {chunk_data.get('doc_name', '未知')}, "
        f"作者: {chunk_data.get('author', '未知')}, "
        f"页码: {chunk_data.get('page_number', '未知')}, "
        f"块ID: {chunk_data.get('chunk_id')})】\n"
        f"{chunk_data.get('text', '')}"
        for idx, chunk_data in enumerate(preview_for_ui_useful)
    ])
    
    # 构建基于broadened question的思考过程
    # thinking_process = f"<thinking>好的，我认为要回答这个问题，应该从这几个方面来回答：{', '.join(_BROADENED_QUESTION)}。" #</thinking>
    thinking_process = ""
    yield {"type": "useful_chunks_preview", "count": len(candidate_chunks_for_reranker),
           "preview": preview_for_ui_useful}

    # 4. 构建最终上下文并生成答案
    yield _build_status_event("generation_start", "步骤4: 正在构建提示并生成答案...")
    generation_start_time = time.time()
    final_context_chunks_for_llm = preview_for_ui_useful

    # 构建新的消息格式：system + chat history + tool role + user role + thinking process
    messages_for_generator = [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT_CONTENT}
    ]
    
    # 添加聊天历史
    messages_for_generator.extend(chat_history_openai)
    
    # 添加当前查询相关的消息
    messages_for_generator.extend([
        {
            "role": "user", 
            "content": user_query  # 使用原始用户输入
        },
        {
            "role": "tool",
            "content": f"知识库检索结果：\n{knowledge_content_for_tool}"
        },
        {
            "role": "assistant",
            "content": thinking_process
        }
    ])

    logger.info(
        f"[{flow_request_id}] [GENERATION_INPUT_PREVIEW] Messages count: {len(messages_for_generator)}, Knowledge content length: {len(knowledge_content_for_tool)}")
    yield {"type": "llm_input_preview",
            "system_prompt_used": bool(GENERATOR_SYSTEM_PROMPT_CONTENT),
            "message_count": len(messages_for_generator)}

    full_final_answer_text = ""
    full_reasoning_text = ""  # 用于累积思考过程

    async for gen_event in call_generator_vllm_stream(
            messages=messages_for_generator
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
    generation_duration = time.time() - generation_start_time
    yield {"type": "timing", "stage": "generation", "duration_ms": int(generation_duration * 1000)}
    yield {"type": "final_answer_complete",
            "full_text": full_final_answer_text.strip(),
            "full_reasoning": full_reasoning_text.strip(),
            "final_context_chunk_ids": [c.get("chunk_id") for c in final_context_chunks_for_llm]}
    yield _build_status_event("generation_complete", "答案生成完毕。")
    total_duration = time.time() - start_time_total
    logger.info(f"[{flow_request_id}] RAG流程处理完毕 (总耗时: {total_duration:.3f}s)。")
    yield {"type": "timing", "stage": "total", "duration_ms": int(total_duration * 1000)}
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
