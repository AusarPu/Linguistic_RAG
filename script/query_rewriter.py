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
    max_history: int = MAX_HISTORY, # 允许指定使用的历史轮数
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
你的角色是一个 **语言学查询规划助手 (Linguistics Query Planner)**。

**任务描述:**
接收一段对话历史和用户的最新查询。你的目标是 **理解用户的核心信息需求**，并将其**分解**为一系列 **关键的、可独立查询的子主题或方面**。
这些子主题应当是全面、深入地回答用户原始问题所必需的组成部分。
最终，你需要输出一个 **简洁的查询词/短语列表**，这个列表将用于后续在语言学知识库中进行多次、分别的检索。

**具体指令:**

1.  **理解上下文:** 结合对话历史，准确把握用户最新查询的真实意图，解决任何指代不明或省略的问题。
2.  **分解查询:** 基于对用户意图和语言学知识的理解，思考要完整回答原始问题，需要涵盖哪些关键方面？例如：
    * 对于历史性问题，可能需要分解为不同的**时间阶段**或**重要流派**。
    * 对于概念解释问题，可能需要分解为**定义、核心特征、代表人物、相关理论、应用领域**等。
    * 对于比较性问题，可能需要分别查询**对比双方的特点**以及**它们的异同点**。
    * 生成能对应知识库中具体信息块的、简洁明确的查询词或短语。
3.  **输出格式 (非常重要):**
    * **必须**以列表形式输出分解后的查询词/短语。
    * **每行一个**查询词或短语，使用**换行符 (`\n`)** 分隔。
    * 每个词/短语应**尽可能简洁**，适合直接作为搜索关键词。
    * **绝对禁止**包含任何额外的解释、说明、对话性文字、编号 (如 1., 2.)、项目符号 (如 -, *) 或任何标签。输出的必须是**纯粹的、以换行分隔的查询词/短语列表**。

**示例:**

* **示例 1:**
    * 用户查询: `语言学的发展历程`
    * **你的输出:**
        ```
        传统语文学研究
        历史比较语言学阶段
        结构主义语言学流派
        转换生成语法理论
        功能主义语言学观点
        认知语言学新进展
        当代语言学前沿
        ```
* **示例 2:**
    * 用户查询: `索绪尔对语言学有哪些主要贡献？`
    * **你的输出:**
        ```
        索绪尔 语言与言语
        索绪尔 语言符号理论
        索绪尔 任意性原则
        索绪尔 组合关系与聚合关系
        索绪尔 共时语言学与历时语言学
        索绪尔 结构主义语言学奠基人
        ```
* **示例 3:**
    * 用户查询: `对比一下乔姆斯基和韩礼德的语法理论`
    * **你的输出:**
        ```
        乔姆斯基生成语法核心思想
        韩礼德系统功能语法核心思想
        乔姆斯基语法的形式主义特点
        韩礼德语法的功能主义特点
        生成语法与功能语法的区别
        ```

**约束:**
* 生成的查询词/短语列表必须**紧密围绕**回答用户原始问题的核心需求。
* 输出必须严格遵守上述格式要求。

注意：-------------------------------------------------------------
现在，请分析提供的对话历史和用户最新查询: ```'{user_input}'```，并按要求输出分解后的查询词/短语列表。
--------------------------------------------------------------------
"""

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
            **GENERATION_CONFIG,
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