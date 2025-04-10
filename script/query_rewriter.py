# test/query_rewriter.py

import torch
import logging
from typing import List, Dict

# 假设 config 和模型/tokenizer 从调用处传入或在此模块可访问
# from .config import MAX_HISTORY, GENERATION_CONFIG # 如果需要直接访问配置
# from transformers import PreTrainedModel, PreTrainedTokenizerFast # 用于类型注解

logger = logging.getLogger(__name__)

def generate_rewritten_query(
    model, # : PreTrainedModel, # 添加类型注解（如果适用）
    tokenizer, # : PreTrainedTokenizerFast, # 添加类型注解（如果适用）
    messages: List[Dict[str, str]],
    user_input: str,
    rewriter_instruction_template: str, # 新增：重写指令模板字符串
    max_history: int, # = MAX_HISTORY # 可以从 config 传入
    generation_config: Dict # = GENERATION_CONFIG # 从 config 传入
    ) -> str:
    """
    使用 LLM 根据对话历史和指令模板重写用户当前问题，生成适合知识库检索的查询列表。

    Args:
        model: 加载的语言模型 (用于重写)。
        tokenizer: 加载的分词器 (用于重写)。
        messages: 包含对话历史的列表。
        user_input: 用户当前输入的原始问题。
        rewriter_instruction_template (str): 从文件加载的包含 '{user_input}' 占位符的指令模板。
        max_history (int): 用于重构查询时考虑的最大历史对话轮数。
        generation_config (Dict): 用于 LLM 生成的参数。

    Returns:
        str: 重写后的查询字符串 (以换行符分隔的列表)。如果重写失败或结果不佳，返回原始输入。
    """
    logger.info(f"Starting query rewriting for: '{user_input[:50]}...'")
    # 1. 准备对话历史
    # 只取最近 max_history * 2 条消息
    raw_rewrite_history = messages[-(max_history * 2):]
    # 过滤掉非 user/assistant 角色 (如果存在)
    filtered_history = [msg for msg in raw_rewrite_history if msg.get("role") in ["user", "assistant"]]
    logger.debug(f"Using last {len(filtered_history)} messages for rewrite context.")

    # 2. 构建重写 Prompt
    # 使用从文件加载的模板，并格式化插入当前用户输入
    try:
        # 使用 .format() 或 f-string 填充模板
        # 确保模板中的占位符是 '{user_input}'
        final_rewrite_instruction = rewriter_instruction_template.format(user_input=user_input)
    except KeyError:
        logger.error("Rewriter instruction template is missing the '{user_input}' placeholder!")
        # 可以选择回退或抛出错误
        logger.warning("Falling back to using raw user input due to template error.")
        return user_input
    except Exception as e:
        logger.error(f"Error formatting rewriter instruction template: {e}", exc_info=True)
        logger.warning("Falling back to using raw user input due to template error.")
        return user_input


    rewrite_messages = filtered_history + [
        # 将格式化后的完整指令作为最后的用户消息
        {"role": "user", "content": final_rewrite_instruction},
    ]


    try:
        # 使用 apply_chat_template 构建输入
        rewrite_prompt = tokenizer.apply_chat_template(
            rewrite_messages,
            tokenize=False,
            add_generation_prompt=True # DeepSeek/Qwen 通常需要这个来提示模型开始生成
        )
        logger.debug(f"Rewrite prompt (first 100 chars): {rewrite_prompt[:100]}...")
    except Exception as e:
        logger.error(f"Error applying chat template for rewriting: {e}", exc_info=True)
        logger.warning("Falling back to original query due to template error.")
        return user_input


    # 3. 调用模型生成重写查询 (非流式)
    try:
        inputs = tokenizer(rewrite_prompt, return_tensors="pt", return_attention_mask=True).to(model.device)

        logger.info("Generating rewritten query using LLM...")
        with torch.no_grad():
            # 使用传入的 generation_config
            outputs = model.generate(
                **inputs,
                **generation_config,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, # 确保 pad_token_id 设置正确
                eos_token_id=tokenizer.eos_token_id
            )
        logger.info("LLM generation for rewrite finished.")

        # 4. 解码得到重写后的查询
        # input_token_len = inputs.input_ids.shape[1]
        # rewritten_query_full = tokenizer.decode(outputs[0, input_token_len:], skip_special_tokens=True).strip()
        # 更可靠的解码方式：解码整个输出，然后移除输入部分（如果需要精确）
        # 或者直接解码整个输出，因为 apply_chat_template 通常包含了停止标记，模型应该在回答后停止
        rewritten_query_full = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        logger.debug(f"Full output from rewriter model: {rewritten_query_full}")

        # 从完整输出中提取助手的部分 (这依赖于模型的输出格式和 chat template)
        # 常见的方法是查找最后一个助手标记之后的内容
        # 注意：这可能需要根据你使用的模型和模板进行调整
        assistant_marker = tokenizer.apply_chat_template([{"role":"assistant", "content":""}], tokenize=False, add_generation_prompt=False) # 获取助手的起始标记
        parts = rewritten_query_full.split(assistant_marker)
        if len(parts) > 1:
            rewritten_query = parts[-1].strip()
            logger.debug(f"Extracted rewritten query part: {rewritten_query}")
        else:
            # 如果找不到助手标记，可能需要不同的提取逻辑或直接使用完整输出（去掉prompt部分）
            logger.warning("Could not reliably extract assistant part from rewriter output. Using full decoded output minus prompt (approx).")
            # 尝试另一种方法：只解码新生成的 token
            input_token_len = inputs.input_ids.shape[1]
            rewritten_query = tokenizer.decode(outputs[0, input_token_len:], skip_special_tokens=True).strip()


    except Exception as e:
        logger.error(f"Error during query rewriting generation or decoding: {e}", exc_info=True)
        logger.warning("Falling back to original query due to generation/decoding error.")
        return user_input

    # 5. 后处理
    # (之前的逻辑似乎是为了处理模型可能包含的思考过程 <think>...</think>，如果你的重写模型没有这个，可以简化)
    rewritten_query = rewritten_query_full.split("</think>")[-1] # 移除这行，除非你的重写模型确实输出 <think> 标签

    # 去除可能的引号包裹和多余空白
    rewritten_query = rewritten_query.strip('\"\'').strip()

    logger.info(f"--- Query Rewriter Result ---")
    logger.info(f"Original Query: {user_input}")
    logger.info(f"Rewritten Query (raw): \n{rewritten_query}") # Log the raw result before fallback check
    logger.info(f"-----------------------------")

    # 6. 回退机制
    # 如果重写结果为空、太短或与原句几乎一样（可选，需要更复杂的比较）
    # 或者如果重写结果看起来不像一个列表 (例如，包含很多普通文本) - 这是一个启发式检查
    is_list_like = '\n' in rewritten_query or len(rewritten_query.split()) < 10 # 简单检查是否像列表
    if not rewritten_query or len(rewritten_query) < 5 or not is_list_like:
        logger.warning("Rewritten query seems invalid (empty, too short, or not list-like). Falling back to original query.")
        return user_input

    # 返回处理过的、可能是多行的查询字符串
    return rewritten_query