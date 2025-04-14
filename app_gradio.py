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
    """将 Gradio 的历史格式转换为 OpenAI/HuggingFace messages 格式"""
    messages = []
    for user_msg, assistant_msg in chat_history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
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
    # 可以添加其他 Gradio 输入组件的状态作为参数
    ): # 返回类型注解帮助理解

    if not models_loaded or not kb_initialized or not prompts_loaded:
         # 如果资源加载失败，显示错误信息
         err_msg = "错误：系统资源未能完全加载，无法处理请求。请检查日志。"
         chat_history_tuples.append((message, err_msg))
         yield {chatbot: chat_history_tuples, context_display: "# 系统错误\n资源加载失败"}
         return

    if not message or not message.strip():
         yield {chatbot: chat_history_tuples, context_display: ""} # 空输入，不处理
         return

    logger.info(f"Received message: '{message[:100]}...'")

    # 1. 转换历史记录格式
    # 注意：我们需要维护一个内部的 messages 列表，用于 RAG 逻辑，和 Gradio 的 tuple 列表分开
    messages_history = convert_gradio_to_openai(chat_history_tuples)

    # 2. 查询重写 (如果 Rewriter 模型可用)
    rewritten_query_str = message # 默认使用原始查询
    if rewriter_model and rewriter_tokenizer and rewriter_instruction_template:
        logger.info("Performing query rewriting...")
        try:
            rewritten_query_str = generate_rewritten_query(
                model=rewriter_model,
                tokenizer=rewriter_tokenizer,
                messages=messages_history, # 使用转换后的 history
                user_input=message,
                rewriter_instruction_template=rewriter_instruction_template,
                max_history=MAX_HISTORY,
                generation_config=GENERATION_CONFIG # 可能需要为重写调整参数
            )
            logger.info("Query rewriting finished.")
        except Exception as rewrite_err:
            logger.error(f"Error during query rewriting: {rewrite_err}", exc_info=True)
            logger.warning("Falling back to original query due to rewrite error.")
            rewritten_query_str = message # 出错时回退
    else:
        logger.info("Skipping query rewriting (Rewriter model/template not available).")

    # 3. 解析重写结果为查询词列表
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

    # 10. 处理流式输出并更新 Gradio 界面
    generated_text = ""
    try:
        for new_text in streamer:
            if new_text: # 可能有空 token
                generated_text += new_text
                # 更新 Gradio Chatbot 的最新一条消息
                current_chatbot_state = chat_history_tuples + [(message, generated_text)]
                yield {chatbot: current_chatbot_state, context_display: formatted_context_md}
        logger.info("Streaming finished.")
        logger.debug(f"Full generated text: {generated_text}")

    except Exception as e:
        logger.error(f"Error during generation streaming: {e}", exc_info=True)
        # 即使流式出错，也显示已生成的部分
        current_chatbot_state = chat_history_tuples + [(message, generated_text + "\n[生成中断]"),]
        yield {chatbot: current_chatbot_state, context_display: formatted_context_md}


    # 11. 更新内部 history (为 RAG 的下一次调用准备) - 这部分需要在 Gradio 组件外部管理，或者通过 Gradio State
    # Gradio 的 chat_history_tuples 本身就是状态，下次调用会传入更新后的版本。
    # 但我们需要确保它不超过 MAX_HISTORY

    # 这里我们已经通过 yield 更新了 chatbot 的显示状态，它会自动成为下一次调用的输入 history
    # 我们只需要确保 chat_history_tuples 在传入时不超长即可，在函数开头处理。
    # (可以在函数开始时检查 len(chat_history_tuples) 并截断)

# --- Gradio UI 定义 ---
with gr.Blocks(theme=gr.themes.Soft(), title="RAG 对话系统") as demo:
    gr.Markdown("# RAG 对话系统 (Gradio)")

    with gr.Row():
        with gr.Column(scale=3): # 聊天区域占主要部分
            chatbot = gr.Chatbot(
                label="聊天窗口",
                bubble_full_width=False,
                height=600,
                # render_markdown=True # 开启以支持 Markdown
                )
            with gr.Row():
                 msg_input = gr.Textbox(
                     label="输入消息",
                     placeholder="请输入你的问题...",
                     scale=7,
                     autofocus=True,
                     lines=1 # 设置为单行输入框
                 )
                 send_button = gr.Button("发送", variant="primary", scale=1, min_width=0) # 显式添加发送按钮
        with gr.Column(scale=2): # 上下文区域占侧边
             context_display = gr.Markdown(label="参考知识库片段", value="上下文将显示在这里...")
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
        "outputs": [chatbot, context_display],
    }
    msg_input.submit(**submit_args)
    send_button.click(**submit_args)

    # 清除按钮功能
    def clear_history():
        logger.info("Clearing chat history.")
        return [], "", "上下文将显示在这里..." # 清空 chatbot, input, context display
    clear_button.click(clear_history, [], [chatbot, msg_input, context_display], queue=False)

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