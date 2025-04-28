# app_gradio.py (修正版)
import gradio as gr
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, AsyncGenerator
from web_ui.chat_interface import create_chat_interface
from web_ui.sidebar import create_sidebar

# -- New Imports --
import json
import asyncio
try:
    import aiohttp # For asynchronous HTTP requests
except ImportError:
    print("Please install aiohttp: pip install aiohttp")
    exit(1)
# -----------------

# Use relative imports or ensure correct PYTHONPATH
try:
    from script.knowledge_base import KnowledgeBase
    from script.query_rewriter import generate_rewritten_query
    from script import config # Import config
    from script.config import *
except ImportError as e:
    print(f"Error importing modules or config: {e}")
    exit(999)

# --- Logging Setup ---
log_file = Path("rag_chat_gradio.log")
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
    filename=log_file,
    filemode='w'
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
logging.getLogger().addHandler(console_handler)
logging.getLogger().setLevel(config.LOG_LEVEL)
logger = logging.getLogger(__name__)
# --------------------

# --- Global Variables (KB and Prompts only) ---
kb: Optional[KnowledgeBase] = None
kb_initialized = False
generator_system_prompt: Optional[str] = None
prompts_loaded = False
# ---------------------------------------------

# --- Path handling ---
config_dir = Path(__file__).resolve().parent / "script"
prompt_dir = config_dir.parent / "prompts"

# --- Modify Resource Loading ---
# Make load_all_resources return nothing explicitly for demo.load(..., outputs=None)
def load_all_resources():
    global kb, generator_system_prompt
    global kb_initialized, prompts_loaded

    if kb_initialized and prompts_loaded:
        # logger.info("Resources (Prompts, KB) already loaded.") # Reduce noise maybe
        return # Return None implicitly

    logger.info("Loading resources (Prompts, KB)...")

    # 1. Load Prompts
    if not prompts_loaded:
        try:
            gen_prompt_path = prompt_dir / Path(GENERATOR_SYSTEM_PROMPT_FILE).name
            logger.info(f"Attempting to load generator prompt from: {gen_prompt_path}")
            with open(gen_prompt_path, "r", encoding="utf-8") as f:
                generator_system_prompt = f.read()
            prompts_loaded = True
            logger.info("Prompts loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load essential prompt files: {e}", exc_info=True)
            prompts_loaded = False

    # 2. Initialize Knowledge Base
    if not kb_initialized:
        try:
            logger.info("Initializing Knowledge Base...")
            if not Path(KNOWLEDGE_BASE_DIR).is_dir():
                 raise FileNotFoundError(f"Knowledge base directory not found: {KNOWLEDGE_BASE_DIR}")
            kb = KnowledgeBase(
                knowledge_dir=KNOWLEDGE_BASE_DIR,
                knowledge_file_pattern=KNOWLEDGE_FILE_PATTERN
            )
            kb_initialized = True
            logger.info("Knowledge Base initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize Knowledge Base: {e}", exc_info=True)
            kb_initialized = False
    # else: # Reduce noise
        # logger.info("Knowledge Base already initialized.")

    all_loaded = kb_initialized and prompts_loaded
    if not all_loaded:
        logger.error("One or more essential resources (Prompts/KB) failed to load. Check logs.")
    else:
        logger.info("Essential resources (Prompts, KB) loaded.")
    # No return value needed for demo.load with outputs=None
    return

# --- Helper functions (keep as is) ---
def format_context_for_display(chunks_data: List[Dict[str, str]]) -> str:
    """将检索到的上下文片段格式化为 Markdown 字符串用于显示"""
    if not chunks_data:
        return "（未引用知识库片段）" # 稍微缩短
    md_str = "### 参考知识库片段：\n\n"
    for i, chunk_data in enumerate(chunks_data):
        source = chunk_data.get("source", "未知来源")
        text = chunk_data.get("text", "内容缺失")
        display_text = text[:300] + "..." if len(text) > 300 else text
        md_str += f"**片段 {i + 1} (来源: {source})**\n> {display_text}\n\n"
    return md_str

def convert_gradio_to_openai(chat_history: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """将 Gradio 历史格式转换为 OpenAI messages 格式，并清理思考过程"""
    messages = []
    think_end_tag = "</think>"
    for user_msg, assistant_msg in chat_history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            cleaned_assistant_msg = assistant_msg
            first_think_end_index = assistant_msg.find(think_end_tag)
            if first_think_end_index != -1:
                 cleaned_assistant_msg = assistant_msg[first_think_end_index + len(think_end_tag):].strip()
            # 确保清理后的消息不为空再添加 (如果只有思考过程，不添加到历史)
            if cleaned_assistant_msg:
                messages.append({"role": "assistant", "content": cleaned_assistant_msg})
    return messages


# --- Modify the Core Respond Function ---
# Added history truncation at the beginning
async def respond(
    message: str,
    chat_history_tuples: List[Tuple[str, str]],
    ) -> AsyncGenerator[Dict, None]:
    """
    Handles user message, RAG, vLLM API call, and streaming response.
    """
    # --- History Truncation ---
    logger.debug(f"History length received: {len(chat_history_tuples)}")
    original_len = len(chat_history_tuples)
    if len(chat_history_tuples) > MAX_HISTORY:
        chat_history_tuples = chat_history_tuples[-MAX_HISTORY:]
        logger.warning(f"Truncated Gradio history from {original_len} to {len(chat_history_tuples)}")
    # -------------------------

    # --- Basic Input Check ---
    if not message or not message.strip():
         # Yield a dictionary containing all output components with current/default values
         yield {
             chatbot: chat_history_tuples,
             context_display: format_context_for_display([]), # Default empty context display
             rewritten_query_display: "(无输入)"
         }
         return

    logger.info(f"Received message: '{message[:100]}...'")

    # 1. Convert History
    messages_history = convert_gradio_to_openai(chat_history_tuples)

    # 2. Query Rewriting (remains same)
    rewritten_query_str = message
    actual_rewritten_query_str_for_display = "(重写未启用或失败)"
    if VLLM_REWRITER_MODEL_ID_FOR_API: # Check if rewriter is configured
        logger.info("Performing query rewriting via API...")
        try:
            # Consider running sync requests in executor for fully async app
            # await asyncio.to_thread(generate_rewritten_query, messages_history, message)
            rewritten_query_result = await generate_rewritten_query(messages_history, message)
            if rewritten_query_result != message: # Check if rewrite actually happened
                rewritten_query_str = rewritten_query_result
                actual_rewritten_query_str_for_display = rewritten_query_result
                logger.info("Query rewriting via API successful.")
            else:
                actual_rewritten_query_str_for_display = "(重写结果与原始输入相同)"
                logger.info("Query rewriting result same as original or fallback.")
        except Exception as rewrite_err:
            logger.error(f"Error during query rewriting API call: {rewrite_err}", exc_info=True)
            rewritten_query_str = message # Fallback
            actual_rewritten_query_str_for_display = f"(重写出错: {rewrite_err})"
    else:
        logger.info("Skipping query rewriting (VLLM_REWRITER_MODEL not configured).")
        actual_rewritten_query_str_for_display = "(重写功能未配置)"


    # 3. Parse Search Terms (remains same)
    search_terms = [term.strip() for term in rewritten_query_str.strip().split('\n') if term.strip()]
    if not search_terms:
        search_terms = [message]
    logger.info(f"Planned search terms ({len(search_terms)}): {search_terms}")

    # 4. Retrieval & Aggregation (remains same)
    final_chunks_data = [] # Initialize
    formatted_context_md = format_context_for_display([]) # Default
    if kb:
        all_retrieved_data: List[Dict[str, str]] = []
        processed_chunk_texts = set()
        for i, term in enumerate(search_terms):
            try:
                term_chunks_data = kb.retrieve_chunks(term)
                for chunk_data in term_chunks_data:
                    chunk_text = chunk_data.get("text")
                    if chunk_text and chunk_text not in processed_chunk_texts:
                        all_retrieved_data.append(chunk_data)
                        processed_chunk_texts.add(chunk_text)
            except Exception as retrieval_err:
                logger.error(f"Error retrieving chunks for term '{term}': {retrieval_err}", exc_info=True)
        logger.info(f"Total unique chunks aggregated: {len(all_retrieved_data)}")
        # Context Filtering/Ranking (Truncate)
        final_chunks_data = all_retrieved_data[:MAX_AGGREGATED_RESULTS]
        logger.info(f"Final number of chunks selected: {len(final_chunks_data)}")
        # Format Context for display
        formatted_context_md = format_context_for_display(final_chunks_data)
    else:
        logger.error("KnowledgeBase not initialized, skipping retrieval.")
        formatted_context_md = "错误：知识库未初始化。"


    # 5. Yield Initial Update (Ensure all 3 outputs are included)
    yield {
        chatbot: chat_history_tuples + [(message, "...")], # Add placeholder
        context_display: formatted_context_md,
        rewritten_query_display: actual_rewritten_query_str_for_display
    }

    # 6. Prepare Generation Payload for vLLM API (remains same)
    generation_messages = [{"role": "system", "content": generator_system_prompt}]
    generation_messages.extend(messages_history)
    context_text = "\n\n".join([f"【片段 {i + 1} | 来源: {d.get('source','N/A')}】 {d.get('text','')}" for i, d in enumerate(final_chunks_data)])
    if not context_text:
        context_text = "知识库中未找到相关内容。"
    generation_messages.append({"role": "user", "content": f"User: {message}\n\n知识库：\n{context_text}"})

    api_url = f"{VLLM_GENERATOR_API_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    payload = {
        "model": VLLM_GENERATOR_MODEL_ID_FOR_API,
        "messages": generation_messages,
        "max_tokens": GENERATION_CONFIG.get("max_tokens", 4096),
        "temperature": GENERATION_CONFIG.get("temperature", 0.6),
        "top_p": GENERATION_CONFIG.get("top_p", 0.95),
        "repetition_penalty": GENERATION_CONFIG.get("repetition_penalty", 1.1),
        "stop": GENERATION_CONFIG.get("stop"),
        "stream": True,
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    # 7. Call vLLM API and Handle Streaming Response (remains same logic)
    raw_generated_text = ""
    error_occurred = False
    error_message = "抱歉，处理您的请求时发生错误。"
    final_chatbot_state = chat_history_tuples + [(message, "")] # Prepare final state structure

    try:
        timeout = aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(api_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_body = await response.text()
                    logger.error(f"vLLM API request failed status {response.status}: {error_body}")
                    error_message = f"API 请求失败 (状态码: {response.status})。"
                    error_occurred = True
                else:
                    # Process SSE stream
                    async for line_bytes in response.content:
                        line = line_bytes.decode('utf-8').strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]": break
                        try:
                            chunk_data = json.loads(data_str)
                            delta_content = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content")
                            if delta_content:
                                raw_generated_text += delta_content
                                final_chatbot_state = chat_history_tuples + [(message, raw_generated_text)]
                                yield {chatbot: final_chatbot_state} # Only update chatbot during stream
                        except json.JSONDecodeError: logger.warning(f"JSON decode error: {data_str}")
                        except Exception as e: logger.error(f"Error processing chunk: {data_str}, {e}")

    except asyncio.TimeoutError:
        logger.error(f"vLLM API request timed out.")
        error_message = "请求超时。"
        error_occurred = True
    except aiohttp.ClientError as e: # More general network error
         logger.error(f"vLLM connection error {api_url}: {e}")
         error_message = "无法连接模型服务器。"
         error_occurred = True
    except Exception as e:
        logger.error(f"Unexpected error during API call/streaming: {e}", exc_info=True)
        error_message = "内部错误。"
        error_occurred = True

    # 8. Final Processing & Yield
    if error_occurred:
        final_display_text = raw_generated_text + f"\n\n❌ **错误:** {error_message}"
    else:
        logger.info("Streaming finished successfully.")
        # Post-process Final Text (handle <think> tags)
        final_formatted_display = raw_generated_text
        final_answer_for_history = raw_generated_text # Store potentially unclean version first
        think_end_tag = "</think>"
        first_think_end_index = raw_generated_text.find(think_end_tag)
        if first_think_end_index != -1:
            thinking_part = raw_generated_text[:first_think_end_index + len(think_end_tag)].strip()
            final_answer_for_history = raw_generated_text[first_think_end_index + len(think_end_tag):].strip()
            formatted_thinking = f" ```\n{thinking_part}\n```"
            if final_answer_for_history.strip():
                final_formatted_display = f"{formatted_thinking}\n\n{final_answer_for_history}"
            else:
                final_formatted_display = formatted_thinking
                final_answer_for_history = "" # Don't add empty response to history later

        final_display_text = final_formatted_display
        # Update history with potentially cleaned message if needed by convert_gradio_to_openai
        # Currently convert_gradio_to_openai handles the cleaning based on the display text

    # Ensure the final yield updates all output components
    final_chatbot_state = chat_history_tuples + [(message, final_display_text)]
    yield {
        chatbot: final_chatbot_state,
        context_display: formatted_context_md,
        rewritten_query_display: actual_rewritten_query_str_for_display
    }


# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(), title="RAG 对话系统 (vLLM)") as demo:
    gr.Markdown("# RAG 对话系统 (vLLM 后端)")

    with gr.Row():
        # --- 调用函数创建聊天界面 ---
        chatbot, msg_input, send_button = create_chat_interface()
        # --- 修改开始: 侧边栏部分 ---
        # 调用函数创建侧边栏组件，并接收返回的实例
        rewritten_query_display, context_display, clear_button = create_sidebar()
        # --- 修改结束 ---
        # --------------------------

    # --- 事件绑定部分 ---
    submit_args = {
        "fn": respond,
        "inputs": [msg_input, chatbot],
        "outputs": [chatbot, context_display, rewritten_query_display],
    }
    msg_input.submit(**submit_args)
    send_button.click(**submit_args)


    # Clear button
    def clear_history():
        logger.info("Clearing chat history.")
        # Return default/empty values for all outputs
        return [], "", "上下文将显示在这里...", "重构后的查询将显示在这里..."
    clear_button.click(
        clear_history,
        [],
        [chatbot, msg_input, context_display, rewritten_query_display],
        queue=False
    )

# --- App Launch ---
if __name__ == "__main__":
    try:
        import aiohttp
    except ImportError:
        print("错误: 未安装 aiohttp。请运行: pip install aiohttp")
        exit(1)

    load_all_resources() # Load resources on startup
    # Optional: Add check here if load failed and maybe don't launch

    demo.queue(default_concurrency_limit=8).launch(server_name="0.0.0.0", server_port=8848, share=False)
