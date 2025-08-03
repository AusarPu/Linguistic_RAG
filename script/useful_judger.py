import logging
import time
from openai import OpenAI
from .config_rag import USEFUL_JUDGER_INSTRUCTION_FILE
import json

logger = logging.getLogger(__name__)


with open(USEFUL_JUDGER_INSTRUCTION_FILE, "r", encoding="utf-8") as f:
    # 将模板内容存储在一个全局（模块级）变量中
    _USR_INPUT_FORMAT = """
    [当前问题]
    {questions}
    [知识库内容]
    {knowledge_content}
    """
    _SYS_PROMPT = f.read()


def get_client_and_model():
    client = OpenAI(base_url="http://localhost:8001/v1", api_key="-")
    model_id = client.models.list().data[0].id
    return client, model_id


def judge_knowledge_usefulness(
    knowledge_content: str,
    questions: list[str]
    ) -> str:
    """
    判断知识库内容对于给定问题是否有用

    Args:
        knowledge_content: 知识库内容字符串
        questions: 问题列表

    Returns:
        str: 返回useful或useless
    """
    func_start_time = time.time() # 函数计时
    logger.info(f"[{func_start_time:.3f}] 开始判断知识库内容是否有用")

    # 1. 格式化用户输入并加上指示
    formatted_user_input = _USR_INPUT_FORMAT.format(
        knowledge_content=knowledge_content,
        questions=questions
    )

    # 2. 发送给vLLM格式化后的消息
    client, model_id = get_client_and_model()
    completion = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "system", "content": _SYS_PROMPT},
        {"role": "user", "content": formatted_user_input},
    ],
    extra_body={"guided_choice": ["useful", "useless"], "enable_thinking": True},
    )

    logger.info(f"[{time.time():.3f}] 判断完成 (总耗时: {time.time() - func_start_time:.3f}s)。")

    response_content = completion.choices[0].message.content.strip()
    return response_content


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
