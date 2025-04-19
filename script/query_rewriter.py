# script/query_rewriter.py (修正并包含日志记录和解析逻辑)

import logging
import requests  # 导入 requests 库用于发送 HTTP 请求
import json      # 导入 json 用于处理 JSON 数据
from typing import List, Dict
from config import *

# 导入配置信息
try:
    from config import (
        VLLM_REWRITER_API_BASE_URL,         # vLLM API 地址
        VLLM_REWRITER_MODEL,       # vLLM 中的重写模型标识符
        MAX_HISTORY,               # 最大历史记录轮数
        REWRITER_INSTRUCTION_FILE, # 重写指令模板文件路径
        REWRITER_GENERATION_CONFIG # 重写任务的生成参数
    )
except ImportError:
    logging.critical("在 query_rewriter.py 中无法从 config.py 导入配置")
    # 提供默认值或抛出错误
    VLLM_API_BASE_URL = "http://localhost:8001/v1"
    VLLM_REWRITER_MODEL = "placeholder-rewriter-model" # 需要替换为实际模型名
    MAX_HISTORY = 5
    REWRITER_INSTRUCTION_FILE = "../prompts/rewriter_instruction.txt" # 需要确保路径正确
    REWRITER_GENERATION_CONFIG = {"max_tokens": 512, "temperature": 0.2}

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


# 更新函数签名，移除 model 和 tokenizer 参数
def generate_rewritten_query(
    messages: List[Dict[str, str]],
    user_input: str,
    # 不再需要从外部传入 template, max_history, generation_config，直接使用导入的配置
    ) -> str:
    """
    使用 vLLM API 端点根据对话历史重写用户当前问题。

    Args:
        messages: 包含对话历史的列表。
        user_input: 用户当前输入的原始问题。

    Returns:
        str: 重写后的查询字符串 (可能包含换行符)。如果重写失败或结果不佳，返回原始输入。
    """
    logger.info(f"开始通过 vLLM API 进行查询重写，原始输入: '{user_input[:50]}...'")

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

    # 4. 调用 vLLM API 端点
    api_url = f"{VLLM_REWRITER_API_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": VLLM_REWRITER_MODEL,
        "messages": rewrite_api_messages,
        "max_tokens": REWRITER_GENERATION_CONFIG.get("max_tokens", 512),
        "temperature": REWRITER_GENERATION_CONFIG.get("temperature", 0.2),
        "top_p": REWRITER_GENERATION_CONFIG.get("top_p", 0.95),
        "repetition_penalty": REWRITER_GENERATION_CONFIG.get("repetition_penalty", 1.1),
        "stop": REWRITER_GENERATION_CONFIG.get("stop"),
        "stream": False, # 查询重写通常不需要流式返回
        "lora_request": {  # <--- 添加这个字段
            "lora_name": REWRITER_LORA_NAME,
            "lora_int_id": 1  # 这个 ID 通常需要，具体值可能需要查看 vLLM LoRA 文档
        },
    }
    payload = {k: v for k, v in payload.items() if v is not None} # 清理 None 值

    rewritten_query_result = user_input # 初始化为原始输入，以便在出错时回退
    try:
        logger.info(f"向 vLLM API 发送重写请求: {api_url}")
        response = requests.post(api_url, headers=headers, json=payload, timeout=60) # 设置超时
        response.raise_for_status() # 检查 HTTP 错误

        # --- 增加: 记录完整的原始响应 ---
        try:
            response_data = response.json()
            # 使用 DEBUG 级别记录，避免日志过大，确保日志级别已设为 DEBUG
            logger.debug(f"vLLM API (重写) 完整响应 JSON:\n{json.dumps(response_data, indent=2, ensure_ascii=False)}")
        except json.JSONDecodeError as e:
             logger.error(f"无法解析 vLLM API (重写) 响应 JSON: {e}")
             logger.debug(f"vLLM API (重写) 原始响应文本: {response.text}") # 如果 JSON 解析失败，记录原始文本
             response_data = None # 标记解析失败
        # -----------------------------

        # 5. 处理 API 响应 (仅当 response_data 有效时)
        if response_data:
            if response_data.get("choices") and len(response_data["choices"]) > 0:
                choice = response_data["choices"][0]
                message = choice.get("message", {})
                content = message.get("content")
                if content:
                    # === 在这里调整解析逻辑 ===
                    raw_content_from_api = content.strip()
                    logger.debug(f"从 API 获取的原始 content 用于重写: '{raw_content_from_api}'")

                    # 尝试根据 </think> 标签分割
                    think_end_tag = "</think>"
                    if think_end_tag in raw_content_from_api:
                        parts = raw_content_from_api.split(think_end_tag, 1) # 只分割一次
                        if len(parts) > 1:
                            extracted_query = parts[1].strip()
                            if extracted_query: # 确保分割后有内容
                                rewritten_query_result = extracted_query
                                logger.info(f"成功提取到 {think_end_tag} 之后的内容作为重写查询。")
                            else:
                                logger.warning(f"在 {think_end_tag} 之后未找到有效内容。原始 content: '{raw_content_from_api}'。可能模型只返回了思考过程。")
                                # 保持 rewritten_query_result 为 user_input (默认回退)
                        else:
                             # 理论上如果标签存在，分割总会产生至少两部分（可能第二部分为空）
                             logger.warning(f"找到 {think_end_tag} 但分割异常。使用 API 返回的原始 content。")
                             rewritten_query_result = raw_content_from_api
                    else:
                        # 如果没有找到 </think> 标签，则假定整个 content 就是结果
                        logger.info(f"在 API 响应中未找到 {think_end_tag} 标签，将使用完整的 content。")
                        rewritten_query_result = raw_content_from_api
                    # ==========================
                else:
                    logger.warning("vLLM API 响应 (重写) 中 'content' 字段为空或不存在。")
            else:
                logger.warning("vLLM API 响应 (重写) 格式不符合预期 (缺少 'choices')。")
        else:
             logger.warning("因 JSON 解析失败，无法处理 vLLM API (重写) 响应。")

    except requests.exceptions.Timeout:
        logger.error("请求 vLLM API (重写) 超时。")
    except requests.exceptions.RequestException as e:
        logger.error(f"调用 vLLM API (重写) 时发生网络或HTTP错误: {e}", exc_info=False)
    except Exception as e:
        logger.error(f"处理查询重写 API 调用或响应时发生未知错误: {e}", exc_info=True)

    # 6. 后处理 (对提取出的 rewritten_query_result 进行)
    # 移除可能的引号等基础清理
    rewritten_query_result = rewritten_query_result.strip('\"\'').strip()
    # 这里可以添加更多针对性的清理，如果需要的话

    logger.info(f"--- 查询重写结果 (来自 vLLM, 处理后) ---")
    logger.info(f"原始查询: {user_input}")
    logger.info(f"最终使用的重写查询 (用于检索): '{rewritten_query_result}'") # 记录最终结果
    logger.info(f"-----------------------------------------")

    # 7. 最终有效性检查与回退 (检查处理后的结果)
    if not rewritten_query_result or len(rewritten_query_result) < 5:
        logger.warning("最终重写查询结果无效 (空或过短)，回退到原始查询。")
        return user_input
    # 如果 rewritten_query_result 仍然是 user_input (因为出错或解析后为空)，这里也会回退
    if rewritten_query_result == user_input:
         logger.info("重写结果与原始输入相同或已回退到原始输入。")


    return rewritten_query_result