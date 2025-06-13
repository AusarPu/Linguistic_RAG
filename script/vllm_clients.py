import asyncio
import time
import logging
import json
from typing import List, Dict, Any, Optional, AsyncGenerator

import aiohttp  # 用于异步HTTP请求

# --- 从项目中导入配置 ---
from .config import (
    RERANKER_API_URL,
    RERANKER_MODEL_NAME_FOR_API,
    RERANKER_BATCH_SIZE,
    VLLM_REQUEST_TIMEOUT,  # 通用请求超时
    GENERATOR_API_URL,
    GENERATOR_MODEL_NAME_FOR_API,
    GENERATOR_RAG_CONFIG,
    VLLM_REQUEST_TIMEOUT_GENERATION
)

logger = logging.getLogger(__name__)


async def call_reranker_vllm(
        query: str,
        chunks_to_rerank: List[Dict[str, Any]],  # 每个字典至少包含 'chunk_id' 和 'text'
        api_url: str = RERANKER_API_URL,
        model_name: str = RERANKER_MODEL_NAME_FOR_API,
        batch_size: int = RERANKER_BATCH_SIZE,  # 每批次发送给API的文档数
        request_timeout: float = VLLM_REQUEST_TIMEOUT
) -> List[Dict[str, Any]]:
    """
    异步调用VLLM Reranker服务 (Cohere兼容 /v2/rerank 端点)。

    参数:
        query (str): 用户的查询。
        chunks_to_rerank (List[Dict[str, Any]]): 需要重排的块列表。
            每个块字典应至少包含 'chunk_id' 和 'text'。
            函数会保留原始字典中的其他元数据，并添加 'rerank_score'。
        api_url (str): Reranker VLLM服务的完整API端点。
        model_name (str): 在VLLM服务中加载的Reranker模型名称。
        batch_size (int): 将候选块分批发送给Reranker API时每批的大小。
        request_timeout (float): 单个API请求的超时时间（秒）。

    返回:
        List[Dict[str, Any]]: 一个列表，其中每个元素是输入块数据字典增加了 'rerank_score' 键。
                              列表按 'rerank_score' 降序排列。
                              如果查询为空或块列表为空，或发生严重错误，可能返回空列表或带有低分的原始列表。
    """
    if not chunks_to_rerank:
        logger.debug("[VLLM_RERANKER_CLIENT] 输入的待重排块列表为空，返回空列表。")
        return []
    if not query or not query.strip():  # 增加对查询是否为空的检查
        logger.warning("[VLLM_RERANKER_CLIENT] 查询为空，无法进行重排。将为所有块返回最低分。")
        return [{**chunk, "rerank_score": -float('inf')} for chunk in chunks_to_rerank]

    request_main_id = f"rerank-{time.time_ns() // 1000000}"
    logger.info(
        f"[{request_main_id}] [VLLM_RERANKER_CLIENT] 开始对 {len(chunks_to_rerank)} 个块进行重排，查询: '{query[:50]}...'")

    all_results_with_scores_and_meta: List[Dict[str, Any]] = []

    # 内部辅助函数，用于处理单个批次的 rerank 请求
    async def _fetch_scores_for_single_batch(
            session: aiohttp.ClientSession,
            current_batch_of_chunks_data: List[Dict[str, Any]],  # 这是原始块数据字典的列表
            batch_idx: int  # 用于日志
    ) -> List[Dict[str, Any]]:

        # 从当前批次的块数据中提取文本列表，用于发送给API
        batch_passages_text = [chunk_data.get("text", "") for chunk_data in current_batch_of_chunks_data]

        payload = {
            "model": model_name,
            "query": query,
            "documents": batch_passages_text,
            # "top_n": len(batch_passages_text), # VLLM /v2/rerank 默认会返回所有文档的分数
            "return_documents": False  # 我们不需要VLLM返回文档文本，因为我们有原始块数据
        }

        # 创建一个临时列表来存储这个批次的处理结果，预填充原始块数据和默认低分
        batch_processed_chunks = [{**original_chunk_data, "rerank_score": -float('inf')}
                                  for original_chunk_data in current_batch_of_chunks_data]
        batch_req_id_log = f"{request_main_id}_b{batch_idx}"

        try:
            logger.debug(
                f"[{batch_req_id_log}] [VLLM_RERANKER_CLIENT] 发送 rerank 批次 {batch_idx + 1} (大小: {len(current_batch_of_chunks_data)}). URL: {api_url}")
            # logger.debug(f"[{batch_req_id_log}] Payload: {json.dumps(payload, ensure_ascii=False)}") # Payload可能很大

            async with session.post(api_url, json=payload, headers={"Content-Type": "application/json",
                                                                    "Accept": "application/json"}) as response:
                response.raise_for_status()  # 检查HTTP错误
                response_json = await response.json()
                logger.debug(
                    f"[{batch_req_id_log}] [VLLM_RERANKER_CLIENT] Reranker API 响应 (批次 {batch_idx + 1}): {json.dumps(response_json, ensure_ascii=False, indent=2)}")

                results_from_api = response_json.get("results", [])
                if isinstance(results_from_api, list):
                    # API 返回的 results 列表中的每个元素包含 "index" 和 "relevance_score"
                    # "index" 是对应于 payload 中 "documents" 列表的原始索引
                    for api_res_item in results_from_api:
                        original_index_in_batch = api_res_item.get("index")
                        score = api_res_item.get("relevance_score")

                        if original_index_in_batch is not None and \
                                0 <= original_index_in_batch < len(current_batch_of_chunks_data) and \
                                score is not None:
                            # 更新我们临时列表 batch_processed_chunks 中对应块的分数
                            batch_processed_chunks[original_index_in_batch]["rerank_score"] = float(score)
                        else:
                            logger.warning(
                                f"[{batch_req_id_log}] [VLLM_RERANKER_CLIENT] Reranker API返回结果中包含无效索引或分数: {api_res_item} "
                                f"(批内索引 {original_index_in_batch}, 批大小 {len(current_batch_of_chunks_data)})")
                else:
                    logger.error(
                        f"[{batch_req_id_log}] [VLLM_RERANKER_CLIENT] Reranker API 响应中的 'results' 字段不是列表或不存在。")

        except aiohttp.ClientResponseError as e_http:
            logger.error(
                f"[{batch_req_id_log}] [VLLM_RERANKER_CLIENT] Rerank批次 {batch_idx + 1} HTTP错误: Status {e_http.status}, Message: {e_http.message}, URL: {e_http.request_info.url if e_http.request_info else api_url}",
                exc_info=False)
        except asyncio.TimeoutError:
            logger.error(
                f"[{batch_req_id_log}] [VLLM_RERANKER_CLIENT] Rerank批次 {batch_idx + 1} 请求超时 (>{request_timeout}s)。")
        except Exception as e:
            logger.error(f"[{batch_req_id_log}] [VLLM_RERANKER_CLIENT] Rerank批次 {batch_idx + 1} 处理失败: {e}",
                         exc_info=True)

        return batch_processed_chunks  # 返回这个批次处理过的块（带有分数，或错误时的默认低分）

    timeout_config = aiohttp.ClientTimeout(total=request_timeout)
    async with aiohttp.ClientSession(timeout=timeout_config) as session:
        batch_coroutines = []  # 存储每个批次的协程任务
        for i in range(0, len(chunks_to_rerank), batch_size):
            current_batch_to_process = chunks_to_rerank[i: i + batch_size]
            batch_coroutines.append(_fetch_scores_for_single_batch(session, current_batch_to_process, i // batch_size))

        if batch_coroutines:
            logger.info(
                f"[{request_main_id}] [VLLM_RERANKER_CLIENT] 创建了 {len(batch_coroutines)} 个 reranking 批处理任务，并发执行...")
            start_gather_time = time.time()
            all_processed_batches_results_or_exc = await asyncio.gather(*batch_coroutines, return_exceptions=True)
            logger.info(
                f"[{request_main_id}] [VLLM_RERANKER_CLIENT] 所有 reranking 批处理任务完成 (耗时: {time.time() - start_gather_time:.2f}s)。")

            for batch_output_or_exc in all_processed_batches_results_or_exc:
                if isinstance(batch_output_or_exc, list):
                    all_results_with_scores_and_meta.extend(batch_output_or_exc)  # 收集每个批次的结果
                elif isinstance(batch_output_or_exc, Exception):
                    logger.error(
                        f"[{request_main_id}] [VLLM_RERANKER_CLIENT] 一个 reranking 批处理任务因异常失败: {batch_output_or_exc}")
                    # 失败的批次中的块会保留它们的默认低分 -float('inf')

    # 按 rerank_score 降序排列所有收集到的结果
    all_results_with_scores_and_meta.sort(key=lambda x: x.get("rerank_score", -float('inf')), reverse=True)

    if all_results_with_scores_and_meta:
        logger.info(
            f"[{request_main_id}] [VLLM_RERANKER_CLIENT] Reranking 完成. 共处理并返回 {len(all_results_with_scores_and_meta)} 个块。Top score: {all_results_with_scores_and_meta[0]['rerank_score']:.4f}")
    else:
        logger.warning(
            f"[{request_main_id}] [VLLM_RERANKER_CLIENT] Reranking 未产生任何结果 (所有批次可能都失败了，或者输入为空)。")

    return all_results_with_scores_and_meta




async def call_generator_vllm_stream(
        system_prompt: Optional[str],
        user_content_for_generator: str,
        api_url: str = GENERATOR_API_URL,  # 使用config中的默认值
        model_name: str = GENERATOR_MODEL_NAME_FOR_API,  # 使用config中的默认值
        generation_config: Dict[str, Any] = None,  # 允许覆盖默认的RAG配置
        request_timeout: float = VLLM_REQUEST_TIMEOUT_GENERATION  # 使用config中的默认值
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    异步调用VLLM Generator服务并流式返回响应。
    能够区分处理 delta 中的 'reasoning_content' 和 'content'。

    Yields dicts:
        {"type": "reasoning_delta", "text": "..."} (CoT内容)
        {"type": "content_delta", "text": "..."} (最终答案内容)
        {"type": "error", "message": "..."}
        {"type": "stream_end", "reason": "..."}
    """
    effective_generation_config = GENERATOR_RAG_CONFIG.copy()  # 从config获取基础RAG配置
    if generation_config:  # 如果调用时传入了特定配置，则更新/覆盖
        effective_generation_config.update(generation_config)

    logger.info(
        f"[VLLM_GENERATOR_CLIENT] 请求生成。系统提示: {'是' if system_prompt else '否'}，用户内容长度: {len(user_content_for_generator)}")
    logger.debug(f"[VLLM_GENERATOR_CLIENT] 使用生成参数: {effective_generation_config}")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content_for_generator})

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        **{k: v for k, v in effective_generation_config.items() if v is not None}
    }
    # 确保 payload 中没有 None 值，因为有些API不喜欢它
    payload = {k: v for k, v in payload.items() if v is not None}

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    request_id = f"gen-{time.time_ns() // 1000000}"  # 简单的毫秒级请求ID

    try:
        timeout_config = aiohttp.ClientTimeout(total=request_timeout, connect=10.0)  # connect timeout可以短一些
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            logger.debug(f"[{request_id}] [VLLM_GENERATOR_CLIENT] 发送生成请求到 {api_url}. "
                         f"Payload (部分): model='{payload['model']}', "
                         f"messages_user (首50字符)='{user_content_for_generator[:50].replace(chr(10), ' ')}...'")

            async with session.post(api_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_body = await response.text()
                    logger.error(
                        f"[{request_id}] [VLLM_GENERATOR_CLIENT] API 请求失败，状态码 {response.status}: {error_body}")
                    yield {"type": "error",
                           "message": f"LLM Generator API 错误 (状态码: {response.status}) - {error_body[:200]}"}  # 包含部分错误体
                    yield {"type": "stream_end", "reason": "error_api_status"}
                    return

                async for line_bytes in response.content:
                    line = line_bytes.decode('utf-8').strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        logger.debug(f"[{request_id}] [VLLM_GENERATOR_CLIENT] 收到 [DONE] 标记。")
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        delta = chunk_data.get("choices", [{}])[0].get("delta", {})

                        emitted_in_this_delta = False
                        reasoning_text_fragment = delta.get("reasoning_content")
                        if reasoning_text_fragment is not None and isinstance(reasoning_text_fragment, str):
                            if reasoning_text_fragment:  # 只发送非空reasoning
                                yield {"type": "reasoning_delta", "text": reasoning_text_fragment}
                                emitted_in_this_delta = True

                        final_answer_text_fragment = delta.get("content")
                        if final_answer_text_fragment is not None and isinstance(final_answer_text_fragment, str):
                            # content 的空字符串 "" 是有效的流信号，表示仍在进行中
                            yield {"type": "content_delta", "text": final_answer_text_fragment}
                            # 只有当它实际有内容时，我们才认为 "emitted_in_this_delta" 为 True（用于调试日志）
                            if final_answer_text_fragment:
                                emitted_in_this_delta = True

                                # 检查是否有其他意外的、或者需要处理的字段，例如 tool_calls
                        finish_reason = chunk_data.get("choices", [{}])[0].get("finish_reason")
                        if finish_reason and finish_reason != "stop":  # 例如 "length", "tool_calls"
                            logger.info(
                                f"[{request_id}] [VLLM_GENERATOR_CLIENT] 流结束原因非 'stop': {finish_reason}. Delta: {delta}")
                            # 如果是tool_calls，你可能想yield一个特定的事件
                            if finish_reason == "tool_calls" and "tool_calls" in delta:
                                yield {"type": "tool_calls_delta", "data": delta["tool_calls"]}
                            # 对于 "length"，通常意味着被max_tokens截断
                            yield {"type": "stream_end", "reason": finish_reason}
                            return  # 遇到明确的结束原因（非stop）就终止

                        # if not emitted_in_this_delta and delta and not finish_reason: # (可选) 调试未处理的delta
                        #     logger.debug(f"[{request_id}] [VLLM_GENERATOR_CLIENT] Delta received but no text yielded this event: {delta}")

                    except json.JSONDecodeError:
                        logger.warning(f"[{request_id}] [VLLM_GENERATOR_CLIENT] SSE流JSON解析错误: {data_str}")
                    except Exception as e_stream:
                        logger.error(
                            f"[{request_id}] [VLLM_GENERATOR_CLIENT] 处理SSE数据 '{data_str}' 时出错: {e_stream}",
                            exc_info=False)

    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] [VLLM_GENERATOR_CLIENT] API请求超时 (>{request_timeout}s)。")
        yield {"type": "error", "message": "LLM Generator 请求超时。"}
    except aiohttp.ClientConnectorError as e:  # 例如无法连接
        logger.error(f"[{request_id}] [VLLM_GENERATOR_CLIENT] 连接错误到 {api_url}: {e}")
        yield {"type": "error", "message": f"无法连接到 LLM Generator: {e}"}
    except Exception as e:  # 其他所有 aiohttp.ClientError 或未知错误
        logger.error(f"[{request_id}] [VLLM_GENERATOR_CLIENT] API调用/流处理时发生未知错误: {e}", exc_info=True)
        yield {"type": "error", "message": f"LLM Generator发生未知错误: {e}"}

    # 确保在所有情况下（包括正常结束或在循环中因 stop_reason 提前返回）都有一个最终的 stream_end
    # 如果上面已经因为特定finish_reason返回了stream_end，这里就不会执行
    # 但如果只是因为[DONE]跳出循环，这里会补上
    yield {"type": "stream_end", "reason": "stop"}
