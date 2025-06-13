# script/tests/test_generator_client.py

import asyncio
import logging
import json
import os
import sys
from typing import Dict, Any, Optional

# --- 调整导入路径 ---
SCRIPT_DIR_TEST = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PARENT_DIR = os.path.dirname(SCRIPT_DIR_TEST)
PROJECT_ROOT_FOR_TEST = os.path.dirname(SCRIPT_PARENT_DIR)
if PROJECT_ROOT_FOR_TEST not in sys.path:  # 避免重复添加
    sys.path.insert(0, PROJECT_ROOT_FOR_TEST)

try:
    # 假设 call_generator_vllm_stream 在 script.vllm_clients 或 script.rag_pipeline 中
    # 我们假设你把它放在了 vllm_clients.py
    from script.vllm_clients import call_generator_vllm_stream
    from script.config import (
        GENERATOR_API_URL,
        GENERATOR_MODEL_NAME_FOR_API,  # 你在config.py中是这个名字
        GENERATOR_SYSTEM_PROMPT_FILE,
        GENERATOR_RAG_CONFIG,
        VLLM_REQUEST_TIMEOUT_GENERATION
    )
except ImportError as e:
    logger = logging.getLogger(__name__)  # 在导入失败前获取logger
    logger.critical(f"测试脚本导入错误: {e}。请确保路径正确，并且 vllm_clients.py 和 config.py 可访问。")
    raise  # 抛出异常以停止执行

# --- 配置日志 ---
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s.%(msecs)03d - [%(levelname)s] - %(name)s - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# -----------------

# 全局加载系统提示
GENERATOR_SYSTEM_PROMPT_CONTENT: Optional[str] = None
if GENERATOR_SYSTEM_PROMPT_FILE and os.path.exists(GENERATOR_SYSTEM_PROMPT_FILE):
    try:
        with open(GENERATOR_SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            GENERATOR_SYSTEM_PROMPT_CONTENT = f.read()
        logger.info(f"成功加载生成器系统提示: {GENERATOR_SYSTEM_PROMPT_FILE}")
    except Exception as e:
        logger.error(f"无法加载生成器系统提示 {GENERATOR_SYSTEM_PROMPT_FILE}: {e}", exc_info=True)
        GENERATOR_SYSTEM_PROMPT_CONTENT = "你是一个乐于助人的AI助手。"
else:
    logger.warning(f"生成器系统提示文件路径 '{GENERATOR_SYSTEM_PROMPT_FILE}' 未配置或文件不存在，使用默认提示。")
    GENERATOR_SYSTEM_PROMPT_CONTENT = "你是一个乐于助人的AI助手。"


async def main_test_generator_stream():
    logger.info("--- 测试 Generator VLLM 客户端 (call_generator_vllm_stream) ---")

    sample_user_content = """上下文信息：
\"\"\"
【相关片段 1】索绪尔是瑞士语言学家，他的核心理论之一是语言的符号性，区分能指和所指，它们之间的联系是任意的。
【相关片段 2】语言符号的另一个重要特性是其线性特征，能指在时间上是展开的。
\"\"\"
请先进行一步步的思考（这部分会作为reasoning_content），然后清晰总结索绪尔语言符号的两个主要特性，并完全基于上下文回答。
用户问题：索绪尔提出的语言符号的两个主要特性是什么？
"""

    logger.info(f"发送给生成器的用户内容 (预览): '{sample_user_content[:150].replace(chr(10), ' ')}...'")

    full_reasoning_text = ""
    full_content_text = ""

    # 确保配置有效
    if not GENERATOR_API_URL or not GENERATOR_MODEL_NAME_FOR_API:
        logger.error("Generator API URL 或 Model ID 未在 config.py 中正确配置。测试中止。")
        return

    try:
        request_count = 0
        async for event in call_generator_vllm_stream(
                system_prompt=GENERATOR_SYSTEM_PROMPT_CONTENT,
                user_content_for_generator=sample_user_content
                # api_url, model_name, generation_config, request_timeout 会使用函数定义的默认值（来自config）
        ):
            request_count += 1
            event_type = event.get("type")
            text_fragment = event.get("text", "")

            if event_type == "reasoning_delta":
                print(f"[CoT]: {text_fragment}", end="", flush=True)
                full_reasoning_text += text_fragment
            elif event_type == "content_delta":
                print(f"[Ans]: {text_fragment}", end="", flush=True)
                full_content_text += text_fragment
            elif event_type == "error":
                logger.error(f"\n流式生成过程中发生错误: {event.get('message')}")
                break
            elif event_type == "stream_end":
                logger.info(f"\n流式生成结束，原因: {event.get('reason')}")
                break
            elif event_type == "tool_calls_delta":  # 如果你的模型支持工具调用并返回了这个
                logger.info(f"\n收到工具调用 Delta: {event.get('data')}")
            else:
                logger.warning(f"\n收到未知事件类型: {event_type}, 内容: {event}")

        if request_count == 0:
            logger.warning("call_generator_vllm_stream 没有 yield 任何事件。请检查 VLLM 服务和网络连接。")

        logger.info(f"\n\n--- 流式生成完毕 (测试脚本) ---")
        logger.info(f"--- 完整的思考过程 (Reasoning/CoT) ---")
        if full_reasoning_text:
            logger.info(f"\n{full_reasoning_text}")
        else:
            logger.info("(无思考过程内容)")

        logger.info(f"\n--- 完整的最终答案内容 (Content) ---")
        if full_content_text:
            logger.info(f"\n{full_content_text}")
        else:
            logger.info("(无最终答案内容)")

    except Exception as e:
        logger.error(f"测试 Generator 客户端时发生顶层错误: {e}", exc_info=True)


if __name__ == "__main__":
    # 确保 Generator VLLM 服务已启动
    # 确保所有相关的配置 (API URL, 模型名等) 在 config.py 中正确
    asyncio.run(main_test_generator_stream())