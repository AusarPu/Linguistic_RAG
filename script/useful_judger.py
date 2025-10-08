import logging
import time
import asyncio
import aiohttp
from openai import OpenAI  # 仅用于一次性获取模型 ID
from .config_rag import (
    USEFUL_JUDGER_INSTRUCTION_FILE,
    USEFULNESS_GENERATION_CONFIG,
    VLLM_REQUEST_TIMEOUT,
)
from preprocess.llm_chunk_processor import extract_content_from_vllm_response
import json
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------- 全局加载指令模板 ----------------
with open(USEFUL_JUDGER_INSTRUCTION_FILE, "r", encoding="utf-8") as f:
    _USR_INPUT_FORMAT = """
    [当前问题]
    {questions}
    [知识库内容]
    {knowledge_content}
    """
    _SYS_PROMPT = f.read()

# ---------------- vLLM 相关常量 ----------------
_VLLM_BASE_URL = "http://localhost:8001/v1"

# 一次性获取模型 ID，避免每次请求都列模型
try:
    _MODEL_ID = OpenAI(base_url=_VLLM_BASE_URL, api_key="-").models.list().data[0].id
except Exception as e:
    logger.warning(f"获取模型 ID 失败，将使用默认值 'unknown': {e}")
    _MODEL_ID = "unknown"


async def judge_knowledge_usefulness(
    knowledge_content: str,
    questions: list[str],
) -> str:
    """异步版本：判断知识库内容对于给定问题是否有用。

    Args:
        knowledge_content: 知识库内容字符串
        questions: 问题列表

    Returns:
        str: "useful" 或 "useless"
    """
    func_start_time = time.time()
    logger.info(f"[{func_start_time:.3f}] 开始判断知识库内容是否有用")

    # 1. 构造用户输入
    formatted_user_input = _USR_INPUT_FORMAT.format(
        knowledge_content=knowledge_content,
        questions=questions,
    )

    # 2. 组装请求数据（直接走 HTTP 调用）
    request_body = {
        "model": _MODEL_ID,
        "messages": [
            {"role": "system", "content": _SYS_PROMPT},
            {"role": "user", "content": formatted_user_input},
        ],
        "guided_choice": ["useful", "useless"],
        **USEFULNESS_GENERATION_CONFIG,
    }

    # 3. 发送请求
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=VLLM_REQUEST_TIMEOUT)
    ) as session:
        async with session.post(f"{_VLLM_BASE_URL}/chat/completions", json=request_body) as resp:
            if resp.status != 200:
                err_text = await resp.text()
                raise RuntimeError(f"vLLM useful_judge API HTTP {resp.status}: {err_text[:200]}")
            resp_json = await resp.json()

    # 4. 解析响应并提取有用性结果
    if "choices" not in resp_json or not resp_json["choices"]:
        raise ValueError("API响应缺少 choices 字段或为空")

    message_dict = resp_json["choices"][0]["message"]

    response_content = extract_content_from_vllm_response(
        message_dict,
        USEFULNESS_GENERATION_CONFIG,
    )

    # --- 使用 Pydantic 解析结构化输出 ---
    class _UsefulnessOutput(BaseModel):
        usefulness: str

    usefulness_val = response_content.strip()
    try:
        usefulness_val = _UsefulnessOutput.model_validate_json(usefulness_val).usefulness
    except Exception:
        # 如果解析失败，则直接使用原始字符串
        pass

    usefulness_val = usefulness_val.lower()

    logger.info(
        f"[{time.time():.3f}] 判断完成 (总耗时: {time.time() - func_start_time:.3f}s)。 输出: {usefulness_val}"
    )
    return usefulness_val



if __name__ == "__main__":
    # 测试用例1：相关知识库内容
    test_knowledge_1 = """
    Python是一种广泛使用的解释型、高级编程语言。
    Python的设计哲学强调代码的可读性和简洁的语法，尤其是使用空格缩进来划分代码块。
    Python支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。
    """
    
    test_questions_1 = [
        "Python的主要特点是什么？",
        "Python使用什么来划分代码块？",
        "Python支持哪些编程范式？"
    ]
    
    # 测试用例2：不相关知识库内容
    test_knowledge_2 = """
    Java是一种面向对象的编程语言。
    Java程序可以在不同的平台上运行，遵循"一次编写，到处运行"的理念。
    Java具有强大的类库支持，适合开发企业级应用。
    """
    
    test_questions_2 = [
        "Python的主要特点是什么？",
        "如何在Python中处理异常？",
        "Python的垃圾回收机制是怎样的？"
    ]
    
    # 测试用例3：部分相关知识库内容
    test_knowledge_3 = """
    编程语言可以分为解释型和编译型。
    Python属于解释型语言，而C++属于编译型语言。
    解释型语言的执行速度通常较慢，但开发效率高。
    """
    
    test_questions_3 = [
        "Python是什么类型的语言？",
        "Python和数据库如何交互？",
        "如何使用Python进行web开发？"
    ]
    
    # 测试所有用例
    print("测试用例1（相关内容）结果:")
    result1 = judge_knowledge_usefulness(test_knowledge_1, test_questions_1)
    print(result1)
    
    print("\n测试用例2（不相关内容）结果:")
    result2 = judge_knowledge_usefulness(test_knowledge_2, test_questions_2)
    print(result2)
    
    print("\n测试用例3（部分相关内容）结果:")
    result3 = judge_knowledge_usefulness(test_knowledge_3, test_questions_3)
    print(result3)
