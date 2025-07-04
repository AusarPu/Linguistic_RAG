import logging
import time

import aiohttp
import asyncio
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
    api_url = REWRITER_API_URL
    headers = {"Content-Type": "application/json", "Accept": "application/json"}  # 非流式，接受 JSON

    # --- 构造 Payload (包含 LoRA 指定) ---
    payload = {
        "model": REWRITER_MODEL_NAME_FOR_API,  # 基础模型名称
        "messages": rewrite_api_messages,
        # 使用 REWRITER_GENERATION_CONFIG
        **{k: v for k, v in REWRITER_GENERATION_CONFIG.items() if v is not None},
        "stream": False,  # 重写不需要流式
        "chat_template_kwargs": {"enable_thinking": False}
    }

    rewritten_query_result = user_input  # 默认回退值

    # 5. 使用 aiohttp 调用 API (不变)
    try:
        timeout = aiohttp.ClientTimeout(total=20.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            req_start_api_time = time.time()
            logger.info(f"[{req_start_api_time:.3f}] 发送 ASYNC 重写请求 (简化 JSON): {api_url}")
            async with session.post(api_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                response_data = await response.json()
                api_end_time = time.time()
                logger.debug(
                    f"[{api_end_time:.3f}] vLLM API (重写) 响应 JSON (耗时: {api_end_time - req_start_api_time:.3f}s):\n{json.dumps(response_data, indent=2, ensure_ascii=False)}")

                # 6. ★★★ 修改响应解析逻辑 ★★★
                if response_data.get("choices") and len(response_data["choices"]) > 0:
                    content_str = response_data["choices"][0].get("message", {}).get("reasoning_content") # 适配vllm指定分词器为r1格式又禁止思考。如果不应指定的话该选content
                    if content_str:
                        logger.debug(f"从 API 获取的原始 content (应为 JSON 字符串): '{content_str}'")
                        try:
                            # 尝试直接解析 content 为 JSON 对象
                            parsed_json = json.loads(content_str)
                            # 提取 "rewritten_queries" 列表
                            rewritten_queries_list = parsed_json.get("rewritten_queries")  # 不设默认值，后面判断

                            # 检查提取结果是否为列表且包含字符串
                            if isinstance(rewritten_queries_list, list) and all(
                                    isinstance(q, str) for q in rewritten_queries_list):
                                if rewritten_queries_list:
                                    # ★ 将列表合并为多行字符串返回 ★
                                    rewritten_query_result = "\n".join(rewritten_queries_list)
                                    logger.info(f"成功从 JSON 提取查询列表，共 {len(rewritten_queries_list)} 条。")
                                else:
                                    logger.warning("JSON 中的 'rewritten_queries' 列表为空。")
                                    # 保持默认回退
                            else:
                                logger.warning("JSON 'rewritten_queries' 格式错误 (非字符串列表)。JSON 内容: %s",
                                               parsed_json)
                                # 保持默认回退
                        except json.JSONDecodeError as e:
                            logger.error(f"解析 API 返回的 content 为 JSON 时失败: {e}. Content: '{content_str}'")
                            # 保持默认回退
                        except Exception as e:
                            logger.error(f"处理解析后的 JSON 时出错: {e}", exc_info=True)
                            # 保持默认回退
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