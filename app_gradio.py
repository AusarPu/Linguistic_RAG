# app_gradio.py
import gradio as gr
import time
import logging
from pathlib import Path
from threading import Thread
from typing import List, Tuple, Dict, Optional

# 使用相对导入或确保 test 目录在 Python 路径中
# 如果 app_gradio.py 在 test 目录外，需要调整路径或 Python 环境设置
# 假设 test 在 PYTHONPATH 或使用包结构
try:
    from script.model_loader import load_models
    from script.knowledge_base import KnowledgeBase
    from script.query_rewriter import generate_rewritten_query
    from script import config  # 导入配置
    from script.config import (
        MAX_HISTORY, GENERATION_CONFIG, KNOWLEDGE_BASE_DIR,
        KNOWLEDGE_FILE_PATTERN, GENERATOR_SYSTEM_PROMPT_FILE,
        REWRITER_INSTRUCTION_FILE, MAX_AGGREGATED_RESULTS
    )
except ImportError as e:
    exit(999)
    # import sys
    # # 如果不在 PYTHONPATH，尝试添加 test 的父目录
    # parent_dir = Path(__file__).resolve().parent
    # test_dir_parent = parent_dir
    # if str(test_dir_parent) not in sys.path:
    #     sys.path.insert(0, str(test_dir_parent))
    # try:
    #     from test.model_loader import load_models
    #     from test.knowledge_base import KnowledgeBase
    #     from test.query_rewriter import generate_rewritten_query
    #     from test import config
    #     from test.config import (
    #         MAX_HISTORY, GENERATION_CONFIG, KNOWLEDGE_BASE_DIR,
    #         KNOWLEDGE_FILE_PATTERN, GENERATOR_SYSTEM_PROMPT_FILE,
    #         REWRITER_INSTRUCTION_FILE, MAX_AGGREGATED_RESULTS
    #     )
    # except ImportError:
    #      print(f"Error importing modules from 'test' directory: {e}")
    #      print("Please ensure 'app_gradio.py' is placed correctly relative to the 'test' directory,")
    #      print("or that the 'test' directory's parent is in your PYTHONPATH.")
    #      exit(1)


# --- 配置 logging ---
log_file = Path("rag_chat_gradio.log")
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
    filename=log_file,
    filemode='w'
)
# 添加控制台输出 handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
logging.getLogger().addHandler(console_handler)
logging.getLogger().setLevel(config.LOG_LEVEL) #确保根 logger 也设置了级别

logger = logging.getLogger(__name__)
# --------------------

# --- 全局变量/状态管理 ---
# 最好在应用启动时加载一次模型和知识库
generator_model = None
generator_tokenizer = None
rewriter_model = None
rewriter_tokenizer = None
kb: Optional[KnowledgeBase] = None
generator_system_prompt: Optional[str] = None
rewriter_instruction_template: Optional[str] = None
models_loaded = False
kb_initialized = False
prompts_loaded = False

# --- 路径处理 ---
# 使用 pathlib 处理相对路径，相对于 config.py 所在的目录
config_dir = Path(__file__).resolve().parent / "script"
prompt_dir = config_dir.parent / "prompts" # 假设 prompts 在 script 的父目录

# --- 加载资源 ---
def load_all_resources():
    global generator_model, generator_tokenizer, rewriter_model, rewriter_tokenizer
    global kb, generator_system_prompt, rewriter_instruction_template
    global models_loaded, kb_initialized, prompts_loaded

    if models_loaded and kb_initialized and prompts_loaded:
        logger.info("Resources already loaded.")
        return True

    logger.info("Loading resources...")
    # 1. 加载 Prompts
    try:
        gen_prompt_path = prompt_dir / Path(GENERATOR_SYSTEM_PROMPT_FILE).name # 使用文件名
        rew_prompt_path = prompt_dir / Path(REWRITER_INSTRUCTION_FILE).name
        logger.info(f"Attempting to load generator prompt from: {gen_prompt_path}")
        with open(gen_prompt_path, "r", encoding="utf-8") as f:
            generator_system_prompt = f.read()
        logger.info(f"Attempting to load rewriter template from: {rew_prompt_path}")
        with open(rew_prompt_path, "r", encoding="utf-8") as f:
            rewriter_instruction_template = f.read()
        prompts_loaded = True
        logger.info("Prompts loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load essential prompt files: {e}", exc_info=True)
        prompts_loaded = False

    # 2. 加载模型
    try:
        if not models_loaded:
            logger.info("Loading models...")
            generator_model, generator_tokenizer, rewriter_model, rewriter_tokenizer = load_models()
            if generator_model is None or generator_tokenizer is None:
                raise RuntimeError("Core Generator model/tokenizer failed to load.")
            models_loaded = True
            logger.info("Models loaded successfully.")
        else:
            logger.info("Models already loaded.")
    except Exception as e:
        logger.critical(f"Failed to load models: {e}", exc_info=True)
        models_loaded = False # 标记失败

    # 3. 初始化知识库
    try:
        if not kb_initialized and models_loaded: # 知识库可能依赖 embedding 模型
            logger.info("Initializing Knowledge Base...")
            # 确保知识库目录存在
            if not Path(KNOWLEDGE_BASE_DIR).is_dir():
                 raise FileNotFoundError(f"Knowledge base directory not found: {KNOWLEDGE_BASE_DIR}")
            kb = KnowledgeBase(
                knowledge_dir=KNOWLEDGE_BASE_DIR,
                knowledge_file_pattern=KNOWLEDGE_FILE_PATTERN
            )
            kb_initialized = True
            logger.info("Knowledge Base initialized successfully.")
        elif not models_loaded:
            logger.warning("Skipping KB initialization because model loading failed.")
        else:
            logger.info("Knowledge Base already initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize Knowledge Base: {e}", exc_info=True)
        kb_initialized = False # 标记失败

    all_loaded = models_loaded and kb_initialized and prompts_loaded
    if not all_loaded:
        logger.error("One or more resources failed to load. Check logs.")
    return all_loaded

# --- 辅助函数 ---
def format_context_for_display(chunks_data: List[Dict[str, str]]) -> str:
    """将检索到的上下文片段格式化为 Markdown 字符串用于显示"""
    if not chunks_data:
        return "（本次回答未直接引用知识库中的具体文本片段）"

    md_str = "### 参考知识库片段：\n\n"
    for i, chunk_data in enumerate(chunks_data):
        source = chunk_data.get("source", "未知来源")
        text = chunk_data.get("text", "内容缺失")
        # 限制显示长度避免界面过长
        display_text = text[:300] + "..." if len(text) > 300 else text
        md_str += f"**片段 {i + 1} (来源: {source})**\n"
        md_str += f"> {display_text}\n\n" # 使用 Markdown 引用块
    return md_str


def convert_gradio_to_openai(chat_history: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """将 Gradio 的历史格式转换为 OpenAI/HuggingFace messages 格式，
    并根据新的逻辑清理助手的思考过程 (提取第一个 </think> 之后的内容)"""
    messages = []
    think_end_tag = "</think>" # 定义结束标记

    for user_msg, assistant_msg in chat_history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            # 默认清理后的消息就是原始消息
            cleaned_assistant_msg = assistant_msg
            # 查找第一个 </think> 标记
            first_think_end_index = assistant_msg.find(think_end_tag)

            if first_think_end_index != -1:
                 # 如果找到了标记，则真实的回答是标记之后的部分
                 # 注意：这里直接从原始 assistant_msg 中提取，因为它包含了我们需要的信息
                 cleaned_assistant_msg = assistant_msg[first_think_end_index + len(think_end_tag):].strip()

                 # 移除我们在 final_formatted_display 中添加的格式
                 # 特别是如果思考部分和回答部分之间加了 \n\n
                 # 如果 cleaned_assistant_msg 是从包含 \n\n 的字符串中提取的，它应该已经是干净的
                 # 但如果原始 assistant_msg 本身以斜体开头，我们需要处理
                 # if assistant_msg.startswith("_") and assistant_msg.find("_\n\n") > -1:
                 #     actual_answer_start_index = assistant_msg.find("_\n\n") + len("_\n\n")
                 #     cleaned_assistant_msg = assistant_msg[actual_answer_start_index:].strip()
                 # 上面的逻辑基于特定的格式，直接提取 </think> 之后的内容更通用

            # else: 如果没找到 </think>，则假定整个消息都是回答，无需清理

            messages.append({"role": "assistant", "content": cleaned_assistant_msg})

    return messages
# --- Gradio 核心回调函数 ---
# 使用 TextIteratorStreamer 处理流式输出
# 需要安装: pip install transformers torch accelerate bitsandbytes sentencepiece # 根据你的模型需要
try:
    from transformers import TextIteratorStreamer
except ImportError:
    logger.error("transformers.TextIteratorStreamer not found. Please install necessary libraries.")
    # 可能需要定义一个虚拟的或者报错退出
    TextIteratorStreamer = None # type: ignore


def respond(
    message: str,
    chat_history_tuples: List[Tuple[str, str]],
    ):
    # ... (初始检查和资源加载检查) ...
    if not message or not message.strip():
         # 在 yield 中为新组件提供默认值
         yield {chatbot: chat_history_tuples, context_display: "", rewritten_query_display: ""}
         return

    logger.info(f"Received message: '{message[:100]}...'")

    # 1. 转换历史记录格式 (保持不变)
    messages_history = convert_gradio_to_openai(chat_history_tuples)

    # 2. 查询重写 (捕获结果用于显示)
    rewritten_query_str = message # 用于后续检索的查询串，默认为原始输入
    actual_rewritten_query_str_for_display = "(未进行重写)" # 用于显示，给个默认值
    if rewriter_model and rewriter_tokenizer and rewriter_instruction_template:
        logger.info("Performing query rewriting...")
        try:
            # 调用重写函数并保存结果
            rewritten_query_result = generate_rewritten_query(
                model=rewriter_model,
                tokenizer=rewriter_tokenizer,
                messages=messages_history,
                user_input=message,
                rewriter_instruction_template=rewriter_instruction_template,
                max_history=MAX_HISTORY,
                generation_config=GENERATION_CONFIG
            )
            # 更新用于检索的查询串
            rewritten_query_str = rewritten_query_result
            # 更新用于显示的值
            actual_rewritten_query_str_for_display = rewritten_query_result
            logger.info("Query rewriting finished.")
        except Exception as rewrite_err:
            logger.error(f"Error during query rewriting: {rewrite_err}", exc_info=True)
            logger.warning("Falling back to original query due to rewrite error.")
            rewritten_query_str = message # 出错时，检索用原始查询
            actual_rewritten_query_str_for_display = f"(重写出错: {rewrite_err})" # 显示错误信息
    else:
        logger.info("Skipping query rewriting (Rewriter model/template not available).")
        actual_rewritten_query_str_for_display = "(重写功能未启用)" # 显示未启用信息

    # 3. 解析重写结果为查询词列表 (使用 rewritten_query_str)
    search_terms = [term.strip() for term in rewritten_query_str.strip().split('\n') if term.strip()]
    if not search_terms:
        search_terms = [message]
    logger.info(f"Planned search terms ({len(search_terms)}): {search_terms}")

    # 4. 循环检索、合并与去重 (与 main.py 逻辑类似)
    all_retrieved_data: List[Dict[str, str]] = []
    processed_chunk_texts = set()
    for i, term in enumerate(search_terms):
        try:
            term_chunks_data = kb.retrieve_chunks(term) # kb 是全局初始化的
            for chunk_data in term_chunks_data:
                chunk_text = chunk_data.get("text")
                if chunk_text and chunk_text not in processed_chunk_texts:
                    all_retrieved_data.append(chunk_data)
                    processed_chunk_texts.add(chunk_text)
        except Exception as retrieval_err:
            logger.error(f"Error retrieving chunks for term '{term}': {retrieval_err}", exc_info=True)
    logger.info(f"Total unique chunks aggregated: {len(all_retrieved_data)}")

    # (可选) 上下文重排序 (这里省略)

    # 5. 控制最终上下文长度
    final_chunks_data = all_retrieved_data[:MAX_AGGREGATED_RESULTS]
    logger.info(f"Final number of chunks selected: {len(final_chunks_data)}")

    # 6. 格式化上下文用于显示
    formatted_context_md = format_context_for_display(final_chunks_data)
    # 先更新一次界面，显示上下文，并将用户消息加入 history
    # chat_history_tuples.append((message, None)) # Gradio 推荐在 yield 中更新 history
    yield {chatbot: chat_history_tuples + [(message, "...")], context_display: formatted_context_md}

    # 7. 第一次 yield: 更新所有需要立即更新的组件
    #    包括：聊天机器人占位符、上下文内容、重构查询内容
    yield {
        chatbot: chat_history_tuples + [(message, "...")], # 添加带占位符的新聊天记录
        context_display: formatted_context_md,             # 更新上下文显示
        rewritten_query_display: actual_rewritten_query_str_for_display # 更新重构查询显示
    }


    # 7. 构建最终生成 Prompt
    # 注意：这里需要维护用于 LLM 的 message 列表，与 Gradio 的 tuple list 分开
    generation_messages = [{"role": "system", "content": generator_system_prompt}]
    generation_messages.extend(messages_history) # 添加历史
    # 构建上下文文本
    context_text = "\n\n".join([f"【片段 {i + 1} | 来源文件: {d.get('source','N/A')}】| 内容: {d.get('text','')}" for i, d in enumerate(final_chunks_data)])
    if not context_text:
        context_text = "根据规划的子主题，知识库中未找到相关内容。"
    # 添加当前用户输入和上下文
    generation_messages.append({"role": "user", "content": f"User: {message}\n\n知识库：\n{context_text}"})

    # 8. 准备生成模型的输入
    try:
        prompt = generator_tokenizer.apply_chat_template(
            generation_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = generator_tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(generator_model.device)
    except Exception as e:
        logger.error(f"Error creating generation prompt or tokenizing: {e}", exc_info=True)
        yield {chatbot: chat_history_tuples + [(message, "错误：准备生成请求时出错。")], context_display: formatted_context_md}
        return

    # 9. 使用流式接口生成回复 (TextIteratorStreamer)
    if TextIteratorStreamer is None:
         yield {chatbot: chat_history_tuples + [(message, "错误：缺少 TextIteratorStreamer")], context_display: formatted_context_md}
         return

    streamer = TextIteratorStreamer(
        generator_tokenizer,
        skip_prompt=True, # 跳过输入的 prompt 部分
        skip_special_tokens=True
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": GENERATION_CONFIG.get("max_new_tokens", 4096),
        "do_sample": GENERATION_CONFIG.get("do_sample", True),
        "temperature": GENERATION_CONFIG.get("temperature", 0.6),
        "top_p": GENERATION_CONFIG.get("top_p", 0.95),
        "repetition_penalty": GENERATION_CONFIG.get("repetition_penalty", 1.1),
        "eos_token_id": generator_tokenizer.eos_token_id,
        "pad_token_id": generator_tokenizer.pad_token_id if generator_tokenizer.pad_token_id is not None else generator_tokenizer.eos_token_id
    }

    # 在单独线程中运行生成
    thread = Thread(target=generator_model.generate, kwargs=generation_kwargs)
    thread.start()

    # 10. 处理流式输出并更新 Gradio 界面 (先显示原始流)
    raw_generated_text = ""
    # 先在聊天记录中添加一个占位符或用户消息本身
    yield {chatbot: chat_history_tuples + [(message, "...")], context_display: formatted_context_md}
    try:
        for new_text in streamer:
            if new_text:  # 可能有空 token
                raw_generated_text += new_text
                # 实时更新聊天机器人，显示原始累积文本
                current_chatbot_state = chat_history_tuples + [(message, raw_generated_text)]
                yield {chatbot: current_chatbot_state, context_display: formatted_context_md}
        logger.info("Streaming finished.")
        logger.debug(f"Full raw generated text: {raw_generated_text}")

    except Exception as e:
        logger.error(f"Error during generation streaming: {e}", exc_info=True)
        # 显示错误和已生成的部分
        error_msg = raw_generated_text + "\n[生成中断]"
        current_chatbot_state = chat_history_tuples + [(message, error_msg)]
        yield {chatbot: current_chatbot_state, context_display: formatted_context_md}
        # 出错时直接返回，不再进行后续格式化
        return  # 退出函数

        # app_gradio.py (修改 respond 函数内流式结束后的处理部分)

    # --- 流式结束后 ---
    # 11. 解析最终文本，区分思考和回答 (根据新逻辑)
    final_formatted_display = raw_generated_text  # 默认显示原始内容
    final_answer_for_history = raw_generated_text  # 默认回答是原始内容

    # 查找第一个 </think> 标签的位置
    think_end_tag = "</think>"
    first_think_end_index = raw_generated_text.find(think_end_tag)

    if first_think_end_index != -1:
        # 思考部分 = 从开头到 </think> 标签结束的所有内容
        thinking_part = raw_generated_text[:first_think_end_index + len(think_end_tag)].strip()
        # 回答部分 = </think> 标签之后的所有内容
        final_answer_for_history = raw_generated_text[first_think_end_index + len(think_end_tag):].strip()

        # 格式化最终显示效果
        # 将思考部分用 Markdown 斜体表示 (更简单通用)
        formatted_thinking = f" ```{thinking_part}```"

        # 检查回答部分是否为空
        if final_answer_for_history.strip():
            # 如果有回答，组合思考(斜体) + 换行 + 回答(正常)
            final_formatted_display = f"{formatted_thinking}\n\n{final_answer_for_history}"
        else:
            # 如果 </think> 后没有内容，只显示斜体的思考部分
            final_formatted_display = formatted_thinking
            # 同时，用于历史记录的回答也应为空
            final_answer_for_history = ""

    # else: 如果没有找到 </think> 标签，则认为整个输出都是最终回答
    # final_formatted_display 和 final_answer_for_history 保持为 raw_generated_text

    # 12. 最后一次更新 Chatbot，使用格式化后的最终文本
    final_chatbot_state = chat_history_tuples + [(message, final_formatted_display)]
    yield {
        chatbot: final_chatbot_state,
        context_display: formatted_context_md,
        rewritten_query_display: actual_rewritten_query_str_for_display
    }

        # (内部 history 清理逻辑依赖于 convert_gradio_to_openai 函数)
    # Gradio 的 history 状态会通过 final_chatbot_state 自动更新，
    # 下次调用 process_and_respond 时会传入包含 final_formatted_display 的 history。
    # 我们需要在下次调用的开始处清理 history 中的思考部分。

# --- Gradio UI 定义 ---
with gr.Blocks(theme=gr.themes.Soft(), title="RAG 对话系统") as demo:
    gr.Markdown("# RAG 对话系统 (Gradio)")

    with gr.Row():
        with gr.Column(scale=3): # 聊天区域加大
            chatbot = gr.Chatbot(
                label="聊天窗口",
                bubble_full_width=False,
                height=850, # <--- 增加聊天框高度
                render_markdown=True # 保持 Markdown 渲染开启
                )
            with gr.Row():
                 msg_input = gr.Textbox(
                     label="输入消息",
                     placeholder="请输入你的问题...",
                     scale=7,
                     autofocus=True,
                     lines=1
                 )
                 send_button = gr.Button("发送", variant="primary", scale=1, min_width=0)
        with gr.Column(scale=2): # 右侧信息区域调整
             # 新增：用于显示重构查询的可折叠块
             with gr.Accordion("查看重构后的查询", open=False): # 默认折叠
                 rewritten_query_display = gr.Textbox(
                     label="重构查询列表", # Accordion自带label,这里的label可选
                     lines=5,          # 显示几行作为预览
                     interactive=False,  # 只读
                     value="重构后的查询将显示在这里..." # 初始提示
                 )
             # 修改：将知识库上下文放入可折叠块
             with gr.Accordion("查看参考知识库片段", open=True): # 默认展开
                 context_display = gr.Markdown(
                     # label="参考内容", # Accordion已有标题，内部label可省略
                     value="上下文将显示在这里..." # 初始提示
                     )
             clear_button = gr.Button("清除对话历史")

    # --- 事件绑定 ---
    # 定义一个包装函数来处理 history 截断
    def process_and_respond(message, history):
         logger.debug(f"History length before processing: {len(history)}")
         # 截断 Gradio history
         if len(history) > MAX_HISTORY:
              logger.warning(f"Truncating Gradio history from {len(history)} to {MAX_HISTORY}")
              history = history[-MAX_HISTORY:]

         # 调用核心响应函数，并传递截断后的 history
         # 使用 yield from 将生成器的结果传递出去
         yield from respond(message, history)

    # 绑定 Enter 键和发送按钮
    submit_args = {
        "fn": process_and_respond,
        "inputs": [msg_input, chatbot],
        # 在 outputs 列表中添加新的组件 rewritten_query_display
        "outputs": [chatbot, context_display, rewritten_query_display],
    }
    msg_input.submit(**submit_args)
    send_button.click(**submit_args)
    # 清除按钮功能
    # 清除按钮功能 - 需要更新返回值以清除新组件
    def clear_history():
        logger.info("Clearing chat history.")
        # 返回对应 outputs 列表的清空值
        return [], "", "上下文将显示在这里...", "重构后的查询将显示在这里..."
    # 在 outputs 列表中添加新组件 rewritten_query_display
    clear_button.click(
        clear_history,
        [],
        [chatbot, msg_input, context_display, rewritten_query_display], # 确保与 outputs 匹配
        queue=False
        )

    # --- 应用加载时执行 ---
    demo.load(load_all_resources, outputs=None) # 在界面加载时尝试加载资源

# --- 启动应用 ---
if __name__ == "__main__":
    # 尝试加载资源，如果失败则 Gradio 可能无法正常工作，但至少会启动界面
    load_successful = load_all_resources()
    if not load_successful:
        print("\n!!!警告：未能成功加载所有模型或知识库资源，系统功能可能受限!!!\n")
        # 这里可以选择是否还启动 Gradio，或者直接退出
        # 启动界面，让用户看到错误信息可能更好

    # share=True 会生成一个公开链接，如果你需要公网访问的话
    # server_name="0.0.0.0" 允许局域网访问
    demo.queue().launch(server_name="0.0.0.0", server_port=8848, share=False)
    # demo.queue() 使得 Gradio 可以处理并发请求，对于 LLM 应用很重要