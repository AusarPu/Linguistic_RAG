# test/query_rewriter.py

import torch
from typing import List, Dict
from script.config import *

# 注意：这个文件需要能访问到 model 和 tokenizer 实例，
# 因此我们将它们作为参数传递给函数。

def generate_rewritten_query(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    user_input: str,
    max_history: int = 10, # 允许指定使用的历史轮数
    ) -> str:
    """
    使用 LLM 根据对话历史重写用户当前问题，生成适合知识库检索的查询。

    Args:
        model: 加载的语言模型。
        tokenizer: 加载的分词器。
        messages: 包含对话历史的列表 (格式: [{"role": "user/assistant", "content": "..."}, ...])。
        user_input: 用户当前输入的原始问题。
        max_history: 用于重构查询时考虑的最大历史对话轮数 (每轮包含用户和助手的一条消息)。
        max_new_tokens: 生成重写查询时允许的最大 token 数。

    Returns:
        重写后的查询字符串。如果重写失败或结果不佳，可能返回原始输入。
    """
    # 1. 准备用于重写的对话历史
    # 只取最近 max_history * 2 条消息 (因为一轮是 user + assistant)
    raw_rewrite_history = messages[-(max_history * 2):]

    filtered_history = [msg for msg in raw_rewrite_history if msg.get("role") in ["user", "assistant"]]

    # 2. 构建重写 Prompt
    # 构造一个临时的消息列表用于重写任务
    # 可以尝试不同的指令模板，看哪个效果好
    rewrite_instruction = f"""
你的角色是专门为 **语言学知识库** 优化用户查询的专家助手。你将收到一段对话历史和用户的最新查询。
你的任务是将这个最新查询改写成一个独立的、清晰的、精确的搜索查询语句，这个语句要适合从一个专门的语言学数据库中检索信息。

**具体指令：**

1.  **分析上下文：** 仔细检查提供的对话历史，以理解当前的讨论主题，并解决用户最新查询中的任何不明确之处。
    * 根据历史记录，将代词（例如，“它”、“那个概念”、“她的理论”）替换为它们所指代的具体语言学术语或人名。
    * 补充那些从上下文中可以推断出的隐含信息（例如，如果当前讨论的主题是音系学，那么“那些规则”很可能指“音系规则”）。

2.  **结合语言学知识进行提炼与增强：** 利用你对语言学术语和概念的理解，适度地改进用户的查询，以便提高知识库的检索效果。这可能包括：
    * **术语标准化：** 将口语化或非正式的表述替换为更规范的语言学术语（例如，“单词是怎么构成的” -> “形态构成过程” 或 “构词法”）。
    * **增加具体性：** 如果原始查询过于宽泛，根据上下文或常见的语言学分支，使其指向更具体（例如，“给我讲讲语法” 在涉及乔姆斯基的语境下可能优化为 “生成语法的基本原则”，或者在讨论句法树时优化为 “依存语法关系”）。
    * **消除歧义：** 将可能引起误解的问法改写得更清晰明确。
    * **丰富关键词：** 在不改变用户核心意图的前提下，如果能显著地突出查询重点，可以适量添加相关的语言学关键词（例如，“语言变化” -> “历史语言学中的语音演变” 或 “语义演变”）。

3.  **保持核心意图：** 非常重要的一点是，改写后的查询**必须**忠实于用户原始问题的根本意图和含义。所做的增强应当是为了澄清和聚焦问题，而不是从根本上扭曲用户的查询目的。

4.  **输出格式要求：** **仅仅**输出最终改写完成的查询字符串本身。不要附带任何形式的解释、引言、道歉、问候语或者类似引号、标签（例如，不要输出“优化后的查询是：...”这样的文字）等额外内容。


现在，注意！---------------------------------------------------------
请基于提供的对话历史和用户最新查询：
```'{user_input}'```，
开始生成优化后的查询语句。"""

    rewrite_messages = filtered_history +[
        {"role": "user", "content": rewrite_instruction},
    ]

    with open("../test.txt", "w", encoding="utf-8") as f:
        for msg in rewrite_messages:
            f.write(msg["role"] + "\n")
            f.write(msg["content"] + "\n")
            f.write("-"*30)

    # 使用 apply_chat_template 构建输入
    # 注意：add_generation_prompt=True 会在末尾添加表示助手开始说话的标记，
    # 这对于续写风格的 Prompt 可能有用，对于直接指令可能需要调整。
    # 如果模型直接输出重写结果效果不好，可以尝试移除 add_generation_prompt=True，
    # 或者调整 rewrite_instruction 的格式。
    rewrite_prompt = tokenizer.apply_chat_template(rewrite_messages, tokenize=False, add_generation_prompt=True) # 假设需要助手提示

    # 3. 调用模型生成重写查询 (非流式)
    inputs = tokenizer(rewrite_prompt, return_tensors="pt", return_attention_mask=True).to(model.device)

    # 使用与最终回答不同的生成参数
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 1024,
            do_sample = False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 4. 解码得到重写后的查询
    input_token_len = inputs.input_ids.shape[1]
    rewritten_query_full = tokenizer.decode(outputs[0, input_token_len:], skip_special_tokens=True).strip()

    # 5. 后处理
    # 尝试获取think之后的回答
    rewritten_query = rewritten_query_full.split("</think>")[-1]

    # 去除可能的引号包裹
    rewritten_query = rewritten_query.strip('"').strip("'")

    print(f"\n--- Query Rewriter ---")
    print(f"Original Query: {user_input}")
    print(f"Rewritten Query: {rewritten_query}")
    print(f"----------------------\n")

    # 6. 回退机制
    # 如果重写结果为空、太短或与原句几乎一样（可选），则回退到原始查询
    if not rewritten_query or len(rewritten_query) < 5: # 基本检查
        print("--- Query Rewriter: Result seems invalid, falling back to original query. ---")
        return user_input


    return rewritten_query