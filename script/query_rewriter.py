import logging
import time
import asyncio
from aiohttp import client
from openai import OpenAI
from pydantic import BaseModel
from .config_rag import MAX_HISTORY,REWRITER_INSTRUCTION_FILE,REWRITER_GENERATION_CONFIG
import json
import re
import os

logger = logging.getLogger(__name__)

# --- 语言检测功能 ---
def detect_text_language(text: str) -> str:
    """
    简单的语言检测功能，基于文本特征判断主要语言
    
    Args:
        text (str): 待检测的文本
    
    Returns:
        str: 检测到的语言代码 ('zh', 'en', 'mixed')
    """
    if not text or not text.strip():
        return 'en'  # 默认英文
    
    # 统计中文字符数量
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 统计英文字符数量（字母）
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    # 总字符数（排除空格和标点）
    total_chars = chinese_chars + english_chars
    
    if total_chars == 0:
        return 'en'  # 默认英文
    
    chinese_ratio = chinese_chars / total_chars
    english_ratio = english_chars / total_chars
    
    # 判断语言
    if chinese_ratio > 0.3:  # 中文字符占比超过30%
        if english_ratio > 0.2:  # 英文字符也占一定比例
            return 'mixed'
        else:
            return 'zh'
    else:
        return 'en'

# --- 动态提示词加载功能 ---
def get_rewriter_instruction(language: str) -> str:
    """
    根据检测到的语言返回相应的重写指令模板
    
    Args:
        language (str): 语言代码 ('zh', 'en', 'mixed')
    
    Returns:
        str: 对应语言的指令模板内容
    """
    # 获取prompts目录路径
    prompts_dir = os.path.dirname(REWRITER_INSTRUCTION_FILE)
    
    if language == 'en':
        # 使用英文版本的指令文件
        instruction_file = os.path.join(prompts_dir, "rewriter_instruction_en.txt")
    else:
        # 对于中文和混合语言，使用中文版本
        instruction_file = REWRITER_INSTRUCTION_FILE
    
    try:
        with open(instruction_file, "r", encoding="utf-8") as f:
            content = f.read()
            # 去掉最后两行（如果存在）
            lines = content.split("\n")
            if len(lines) >= 2:
                return "\n".join(lines[:-2])
            else:
                return content
    except FileNotFoundError:
        logger.warning(f"指令文件 {instruction_file} 未找到，使用默认中文版本")
        # 如果英文文件不存在，回退到中文版本
        with open(REWRITER_INSTRUCTION_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")
            if len(lines) >= 2:
                return "\n".join(lines[:-2])
            else:
                return content

# 移除原来的全局变量定义，改为动态加载
_USR_INPUT_FORMAT = """
[对话历史]
{context}
[当前问题]
{question}
"""


def get_client_and_model():
    client = OpenAI(base_url="http://localhost:8001/v1", api_key="-")
    model_id = client.models.list().data[0].id
    return client, model_id

class rewrite_output(BaseModel):
    question: str
    broadened_question: list[str]
    keyword: list[str]

def format_chat_history(messages: list[dict[str, str]]) -> str:
    """
    将对话历史记录转换为格式化的文本块
    
    Args:
        messages: 包含对话历史的字典列表
        
    Returns:
        str: 格式化后的对话历史文本
    """
    if not messages:
        return "[对话历史]\n"
        
    formatted_messages = ["[对话历史]"]
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "user":
            formatted_messages.append(f"user: {content}")
        elif role == "assistant":
            formatted_messages.append(f"assistant: {content}")
            
    return "\n".join(formatted_messages) + "\n"


async def generate_rewritten_query_async(
    messages: list[dict[str, str]],
    user_input: str,
    ) -> dict:
    """
    异步版本：使用 vLLM API 端点根据对话历史重写用户当前问题。

    Args:
        messages: 包含对话历史的列表。
        user_input: 用户当前输入的原始问题。

    Returns:
        dict: 包含重写后的查询信息的字典。如果失败则返回原始输入。
    """
    func_start_time = time.time() # 函数计时
    logger.info(f"[{func_start_time:.3f}] 开始查询重写: '{user_input}'")

    # 1. 检测用户输入的语言
    detected_language = detect_text_language(user_input)
    logger.debug(f"检测到的语言: {detected_language}")
    
    # 2. 根据语言获取相应的系统提示词
    # sys_prompt = get_rewriter_instruction(detected_language)
    # logger.debug(f"使用的系统提示词语言版本: {detected_language}")
    sys_prompt = get_rewriter_instruction("en")

    # 3. 准备对话历史
    raw_rewrite_history = messages[-(MAX_HISTORY * 2):]
    filtered_history = [msg for msg in raw_rewrite_history if msg.get("role") in ["user", "assistant"]]
    logger.debug(f"使用最近 {len(filtered_history)} 条消息作为重写上下文。")

    # 4. 格式化对话历史
    formatted_history = format_chat_history(filtered_history)
    logger.debug(f"格式化后的对话历史: {formatted_history}")

    # 5. 格式化用户输入并加上指示
    formatted_user_input = _USR_INPUT_FORMAT.format(
        context=formatted_history,
        question=user_input
    )

    # 6. 异步发送给vLLM格式化后的消息
    def _sync_call():
        client, model_id = get_client_and_model()
        # 将vLLM特有参数放入extra_body中
        extra_body = {k: v for k, v in REWRITER_GENERATION_CONFIG.items() 
                     if k not in ["max_tokens", "temperature", "top_p", "stop"]}
        
        completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": formatted_user_input},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "rewrite_output",
                "schema": rewrite_output.model_json_schema()
            }
        },
        # 使用OpenAI兼容的参数
        max_tokens=REWRITER_GENERATION_CONFIG.get("max_tokens", 30960),
        temperature=REWRITER_GENERATION_CONFIG.get("temperature", 0.2),
        top_p=REWRITER_GENERATION_CONFIG.get("top_p", 0.95),
        stop=REWRITER_GENERATION_CONFIG.get("stop"),
        extra_body=extra_body  # vLLM特有参数放在extra_body中
        )
        return completion

    # 使用asyncio.to_thread来异步执行同步调用
    completion = await asyncio.to_thread(_sync_call)

    logger.info(f"[{time.time():.3f}] ASYNC 查询重写完成 (总耗时: {time.time() - func_start_time:.3f}s)。")

    # 解析JSON响应为rewrite_output对象
    response_json = json.loads(completion.choices[0].message.content)
    return response_json


def generate_rewritten_query(
    messages: list[dict[str, str]],
    user_input: str,
    ) -> dict:
    """
    使用 vLLM API 端点根据对话历史重写用户当前问题。

    Args:
        messages: 包含对话历史的列表。
        user_input: 用户当前输入的原始问题。

    Returns:
        dict: 包含重写后的查询信息的字典。如果失败则返回原始输入。
    """
    func_start_time = time.time() # 函数计时
    logger.info(f"[{func_start_time:.3f}] 开始查询重写: '{user_input}'")

    # 1. 检测用户输入的语言
    detected_language = detect_text_language(user_input)
    logger.debug(f"检测到的语言: {detected_language}")
    
    # 2. 根据语言获取相应的系统提示词
    sys_prompt = get_rewriter_instruction("en")

    # 3. 准备对话历史
    raw_rewrite_history = messages[-(MAX_HISTORY * 2):]
    filtered_history = [msg for msg in raw_rewrite_history if msg.get("role") in ["user", "assistant"]]
    logger.debug(f"使用最近 {len(filtered_history)} 条消息作为重写上下文。")

    # 4. 格式化对话历史
    formatted_history = format_chat_history(filtered_history)
    logger.debug(f"格式化后的对话历史: {formatted_history}")

    # 5. 格式化用户输入并加上指示
    formatted_user_input = _USR_INPUT_FORMAT.format(
        context=formatted_history,
        question=user_input
    )

    # 6. 发送给vLLM格式化后的消息
    client, model_id = get_client_and_model()
    # 将vLLM特有参数放入extra_body中
    extra_body = {k: v for k, v in REWRITER_GENERATION_CONFIG.items() 
                 if k not in ["max_tokens", "temperature", "top_p", "stop"]}
    
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": formatted_user_input},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "rewrite_output",
                "schema": rewrite_output.model_json_schema()
            }
        },
        # 使用OpenAI兼容的参数
        max_tokens=REWRITER_GENERATION_CONFIG.get("max_tokens", 30960),
        temperature=REWRITER_GENERATION_CONFIG.get("temperature", 0.2),
        top_p=REWRITER_GENERATION_CONFIG.get("top_p", 0.95),
        stop=REWRITER_GENERATION_CONFIG.get("stop"),
        extra_body=extra_body  # vLLM特有参数放在extra_body中
    )

    logger.info(f"[{time.time():.3f}] ASYNC 查询重写完成 (总耗时: {time.time() - func_start_time:.3f}s)。")

    # 解析JSON响应为rewrite_output对象
    response_json = json.loads(completion.choices[0].message.content)
    return response_json


if __name__ == "__main__":
    def test_generate_rewritten_query():
        # 测试用例：模拟对话历史
        test_messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么我可以帮你的吗？"},
            {"role": "user", "content": "我想了解一下人工智能在医疗领域的应用"},
            {"role": "assistant", "content": "人工智能在医疗领域有很多重要应用，包括医学影像分析、疾病诊断、药物研发等。您想具体了解哪个方面呢？"},
            {"role": "user", "content": "主要想了解在医学影像方面的应用"},
            {"role": "assistant", "content": "AI在医学影像分析方面确实有很大突破。主要应用包括：1. CT和核磁共振图像的自动分析；2. X光片中病变识别；3. 病理切片的智能诊断。这些技术可以帮助医生更快更准确地发现问题。您对哪个具体应用感兴趣？"},
            {"role": "user", "content": "CT影像分析这块怎么样？"},
            {"role": "assistant", "content": "AI在CT影像分析方面非常强大。它可以快速处理大量CT图像，帮助检测肿瘤、骨折、肺部感染等问题。特别是在新冠疫情期间，AI辅助CT诊断发挥了重要作用。具体来说，AI系统可以：1. 自动标注异常区域；2. 进行3D重建；3. 量化分析病变进展。要不要我详细解释某个具体功能？"},
        ]
        
        # 测试用户输入
        test_user_input = "这个产品的性能怎么样？"
        
        try:
            # 调用重写函数
            result = generate_rewritten_query(test_messages, test_user_input)
            print("\n=== 查询重写测试结果 ===")
            print(f"原始输入: {test_user_input}")
            print(f"重写结果: {result}")
            print("=====================\n")
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
    
    # 运行测试
    test_generate_rewritten_query()