import asyncio
import time
import logging
import json
import requests
import numpy as np
from typing import List, Dict, Any, Optional, AsyncGenerator, Union

import aiohttp  # 用于异步HTTP请求

# --- 从项目中导入配置 ---
from .config_rag import (
    VLLM_REQUEST_TIMEOUT,  # 通用请求超时
    GENERATOR_API_URL,
    GENERATOR_MODEL_NAME_FOR_API,
    GENERATOR_RAG_CONFIG,
    VLLM_REQUEST_TIMEOUT_GENERATION,
    EMBEDDING_API_URL,
    EMBEDDING_MODEL_NAME_FOR_API
)

logger = logging.getLogger(__name__)


# ===== EMBEDDING CLIENT =====
class EmbeddingAPIClient:
    """Embedding API 客户端类"""
    
    def __init__(self):
        self.api_url = EMBEDDING_API_URL
        self.model_name = EMBEDDING_MODEL_NAME_FOR_API
        logger.info(f"Initialized EmbeddingAPIClient with URL: {self.api_url}")

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """构造带指令的查询文本"""
        if not task_description:
            return query
        return f'Instruct: {task_description}\nQuery:{query}'
    
    def encode(self, texts: Union[str, List[str]], instruct: str = "") -> Dict[str, Any]:
        """
        编码文本为向量，兼容原BGE-M3的接口
        
        Args:
            texts: 单个文本或文本列表
            instruct: 指令文本
            
        Returns:
            包含dense_vecs的字典，格式兼容BGE-M3
        """
        # 确保输入是列表格式
        if isinstance(texts, str):
            input_texts = [texts]
            is_single = True
        else:
            input_texts = texts
            is_single = False
        
        # 构建API请求
        payload = {
            "model": self.model_name,
            "input": self.get_detailed_instruct(instruct, input_texts),
            "encoding_format": "float"
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # 发送请求
        response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
        
        if response.status_code != 200:
            raise RuntimeError(f"Embedding API request failed with status {response.status_code}: {response.text}")
        
        result = response.json()
        
        if "data" not in result:
            raise RuntimeError(f"Invalid API response format: {result}")
        
        # 提取向量
        embeddings = []
        for item in result["data"]:
            if "embedding" not in item:
                raise RuntimeError(f"Missing embedding in API response: {item}")
            embeddings.append(item["embedding"])
        
        # 转换为numpy数组
        dense_vecs = np.array(embeddings, dtype=np.float32)
        
        # 如果输入是单个文本，返回单个向量
        if is_single:
            dense_vecs = dense_vecs[0]
        
        # 返回兼容BGE-M3格式的结果
        return {
            "dense_vecs": dense_vecs
        }
    



async def call_generator_vllm_stream(
        system_prompt: Optional[str] = None,
        user_content_for_generator: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        api_url: str = GENERATOR_API_URL,  # 使用config中的默认值
        model_name: str = GENERATOR_MODEL_NAME_FOR_API,  # 使用config中的默认值
        generation_config: Dict[str, Any] = None,  # 允许覆盖默认的RAG配置
        request_timeout: float = VLLM_REQUEST_TIMEOUT_GENERATION  # 使用config中的默认值
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    异步调用VLLM Generator服务并流式返回响应。
    能够区分处理 delta 中的 'reasoning_content' 和 'content'。
    
    Args:
        system_prompt: 系统提示（兼容旧接口）
        user_content_for_generator: 用户内容（兼容旧接口）
        messages: 消息列表（新接口，优先使用）
        
    Yields dicts:
        {"type": "reasoning_delta", "text": "..."} (CoT内容)
        {"type": "content_delta", "text": "..."} (最终答案内容)
        {"type": "error", "message": "..."}
        {"type": "stream_end", "reason": "..."}
    """
    effective_generation_config = GENERATOR_RAG_CONFIG.copy()  # 从config获取基础RAG配置
    if generation_config:  # 如果调用时传入了特定配置，则更新/覆盖
        effective_generation_config.update(generation_config)

    # 处理消息格式：优先使用messages参数，否则使用旧的system_prompt + user_content格式
    if messages:
        final_messages = messages
        logger.info(f"[VLLM_GENERATOR_CLIENT] 请求生成。使用消息列表格式，消息数量: {len(messages)}")
    else:
        final_messages = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        if user_content_for_generator:
            final_messages.append({"role": "user", "content": user_content_for_generator})
        logger.info(
            f"[VLLM_GENERATOR_CLIENT] 请求生成。系统提示: {'是' if system_prompt else '否'}，用户内容长度: {len(user_content_for_generator) if user_content_for_generator else 0}")
    
    logger.debug(f"[VLLM_GENERATOR_CLIENT] 使用生成参数: {effective_generation_config}")
    logger.debug(f"[VLLM_GENERATOR_CLIENT] 最终消息数量: {len(final_messages)}")

    payload = {
        "model": model_name,
        "messages": final_messages,
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
            # 构建日志信息
            if messages:
                log_content = f"messages数量={len(messages)}"
            else:
                log_content = f"user_content长度={len(user_content_for_generator) if user_content_for_generator else 0}"
            
            logger.debug(f"[{request_id}] [VLLM_GENERATOR_CLIENT] 发送生成请求到 {api_url}. "
                         f"Payload (部分): model='{payload['model']}', {log_content}")

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
