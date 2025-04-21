import logging
import time

import aiohttp  # <--- 导入 aiohttp
import asyncio  # <--- 可能需要 asyncio (例如用于超时)
import json
from typing import List, Dict, Optional
from .config import *

logger = logging.getLogger(__name__)

# 提前加载指令模板内容，避免每次调用函数都读取文件
try:
    with open(REWRITER_INSTRUCTION_FILE, "r", encoding="utf-8") as f:
        # 将模板内容存储在一个全局（模块级）变量中
        _REWRITER_INSTRUCTION_TEMPLATE_CONTENT = f.read()
    logger.info(f"成功加载重写指令模板: {REWRITER_INSTRUCTION_FILE}")
except Exception as e:
    logger.error(f"无法加载重写指令模板 {REWRITER_INSTRUCTION_FILE}: {e}", exc_info=True)
    # 提供一个备用模板，确保 {user_input} 占位符存在
    _REWRITER_INSTRUCTION_TEMPLATE_CONTENT = "根据对话历史重写用户问题，使其更适合知识库检索。\n用户问题：{user_input}"


# --- 将函数修改为异步 ---
async def generate_rewritten_query(
    messages: List[Dict[str, str]],
    user_input: str,
    ) -> str: # 返回值类型不变
    """
    使用 vLLM API 端点异步 (aiohttp) 地根据对话历史重写用户当前问题。

    Args:
        messages: 包含对话历史的列表。
        user_input: 用户当前输入的原始问题。

    Returns:
        str: 重写后的查询字符串。如果失败则返回原始输入。
    """
    func_start_time = time.time() # 可选：函数计时
    logger.info(f"[{func_start_time:.3f}] 开始 ASYNC 查询重写 (vLLM API): '{user_input[:50]}...'")


    # 1. 准备对话历史 (逻辑保持不变)
    raw_rewrite_history = messages[-(MAX_HISTORY * 2):]
    filtered_history = [msg for msg in raw_rewrite_history if msg.get("role") in ["user", "assistant"]]
    logger.debug(f"使用最近 {len(filtered_history)} 条消息作为重写上下文。")

    # 2. 构建最终的重写指令 (使用预加载的模板内容)
    try:
        final_rewrite_instruction = _REWRITER_INSTRUCTION_TEMPLATE_CONTENT.format(user_input=user_input)
    except KeyError:
        logger.error("重写指令模板缺少 '{user_input}' 占位符!")
        logger.warning("因模板错误，回退到原始用户输入。")
        return user_input
    except Exception as e:
        logger.error(f"格式化重写指令模板时出错: {e}", exc_info=True)
        logger.warning("因模板格式化错误，回退到原始用户输入。")
        return user_input

    # 3. 构建发送给 vLLM API 的 messages 列表
    rewrite_api_messages = filtered_history + [
        {"role": "user", "content": final_rewrite_instruction},
    ]

    # 4. 准备 API 调用细节
    api_url = f"{VLLM_REWRITER_API_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}  # 非流式，接受 JSON

    # --- 构造 Payload (包含 LoRA 指定) ---
    # !!! 重要: 请根据你的 vLLM 版本确认指定 LoRA 的正确方式 !!!
    # 这里以方式 B (lora_request 字段) 作为示例，如果你的 vLLM 需要方式 A，请修改此处。
    payload = {
        "model": VLLM_REWRITER_MODEL,  # 基础模型名称
        "messages": rewrite_api_messages,
        # 使用 REWRITER_GENERATION_CONFIG
        **{k: v for k, v in REWRITER_GENERATION_CONFIG.items() if v is not None},
        "stream": False,  # 重写不需要流式
        # 指定 LoRA (方式 B 示例, 结构需核实)
        "lora_request": {
            "lora_name": REWRITER_LORA_NAME,
            "lora_int_id": 1  # 查阅 vLLM 文档确认是否需要以及具体值
        }
        # --- 或者使用 方式 A ---
        # "model": f"{VLLM_REWRITER_MODEL}@@{REWRITER_LORA_NAME}", # 示例分隔符 @@
        # ---------------------
    }
    # payload = {k: v for k, v in payload.items() if v is not None} # 清理 None 值 (如果上面 **kwargs 已处理则无需重复)

    rewritten_query_result = user_input  # 默认回退值

    # 5. 使用 aiohttp 调用 API
    try:
        # 设置超时 (例如 60 秒)
        timeout = aiohttp.ClientTimeout(total=60.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            req_start_api_time = time.time()
            logger.info(f"[{req_start_api_time:.3f}] 发送 ASYNC 重写请求到: {api_url}")

            async with session.post(api_url, headers=headers, json=payload) as response:
                # 检查 HTTP 状态码
                response.raise_for_status()  # 对 4xx/5xx 抛出 ClientResponseError

                # 6. 处理成功的响应 (异步读取 JSON)
                response_data = await response.json()
                api_end_time = time.time()
                logger.debug(
                    f"[{api_end_time:.3f}] vLLM API (重写) 响应 JSON (耗时: {api_end_time - req_start_api_time:.3f}s):\n{json.dumps(response_data, indent=2, ensure_ascii=False)}")

                # 提取内容 (逻辑同前)
                if response_data.get("choices") and len(response_data["choices"]) > 0:
                    content = response_data["choices"][0].get("message", {}).get("content")
                    if content:
                        # 处理 <think> 标签 (逻辑同前)
                        raw_content = content.strip()
                        think_end_tag = "</think>"
                        if think_end_tag in raw_content:
                            parts = raw_content.split(think_end_tag, 1)
                            if len(parts) > 1 and parts[1].strip():
                                rewritten_query_result = parts[1].strip()
                            else:
                                logger.warning(f"{think_end_tag} 后无有效内容。")
                        else:
                            rewritten_query_result = raw_content
                    else:
                        logger.warning("API 响应 content 为空。")
                else:
                    logger.warning("API 响应 choices 无效。")

    # 7. 捕获 aiohttp 和 asyncio 的异常
    except aiohttp.ClientResponseError as e:
        logger.error(f"[{time.time():.3f}] vLLM API (重写) HTTP 错误: Status {e.status}, Message: {e.message}",
                     exc_info=False)
    except asyncio.TimeoutError:
        logger.error(f"[{time.time():.3f}] 请求 vLLM API (重写) 超时。")
    except aiohttp.ClientConnectorError as e:
        logger.error(f"[{time.time():.3f}] 无法连接到 vLLM 重写服务: {e}", exc_info=False)
    except aiohttp.ClientError as e:  # 其他 aiohttp 客户端错误
        logger.error(f"[{time.time():.3f}] 调用 vLLM API (重写) 时发生 aiohttp 客户端错误: {e}", exc_info=True)
    except Exception as e:  # 未知错误
        logger.error(f"[{time.time():.3f}] 处理查询重写 API 调用时发生未知错误: {e}", exc_info=True)

    # 8. 后处理和最终检查 (逻辑同前)
    rewritten_query_result = rewritten_query_result.strip('\"\'').strip()

    logger.info(f"--- 查询重写结果 (来自 vLLM Async, 处理后) ---")
    logger.info(f"原始查询: {user_input}")
    logger.info(f"最终使用的重写查询: '{rewritten_query_result}'")
    logger.info(f"-----------------------------------------")

    if not rewritten_query_result or len(rewritten_query_result) < 5 or rewritten_query_result == user_input:
        if rewritten_query_result != user_input:
            logger.warning("最终重写查询结果无效或回退。")
        return user_input  # 回退到原始输入

    logger.info(f"[{time.time():.3f}] ASYNC 查询重写完成 (总耗时: {time.time() - func_start_time:.3f}s)。")
    return rewritten_query_result