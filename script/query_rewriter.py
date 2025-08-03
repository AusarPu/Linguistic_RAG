import logging
import time
from aiohttp import client
from openai import OpenAI
from pydantic import BaseModel
from .config_rag import MAX_HISTORY,REWRITER_INSTRUCTION_FILE
import json

logger = logging.getLogger(__name__)


with open(REWRITER_INSTRUCTION_FILE, "r", encoding="utf-8") as f:
    # 将模板内容存储在一个全局（模块级）变量中
    _USR_INPUT_FORMAT = """
    [对话历史]
    {context}
    [当前问题]
    {question}
    """
    _SYS_PROMPT = "\n".join(f.read().split("\n")[:-2])  # 去掉最后两行


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

    # 1. 准备对话历史
    raw_rewrite_history = messages[-(MAX_HISTORY * 2):]
    filtered_history = [msg for msg in raw_rewrite_history if msg.get("role") in ["user", "assistant"]]
    logger.debug(f"使用最近 {len(filtered_history)} 条消息作为重写上下文。")

    # 2. 格式化对话历史
    formatted_history = format_chat_history(filtered_history)
    logger.debug(f"格式化后的对话历史: {formatted_history}")

    # 3. 格式化用户输入并加上指示
    formatted_user_input = _USR_INPUT_FORMAT.format(
        context=formatted_history,
        question=user_input
    )

    # 4. 发送给vLLM格式化后的消息
    client, model_id = get_client_and_model()
    completion = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "system", "content": _SYS_PROMPT},
        {"role": "user", "content": formatted_user_input},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "rewrite_output",
            "schema": rewrite_output.model_json_schema()
        }
    },
    extra_body={"enable_thinking": True},
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