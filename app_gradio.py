# app_gradio.py (Complete Code - Using Tuple Chatbot Format & Yielding Lists)

import gradio as gr
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, AsyncGenerator, Union # Added Union for type hint
import json
import asyncio
try:
    import aiohttp
except ImportError:
    print("Please install aiohttp: pip install aiohttp")
    exit(1)

# --- Imports from project structure ---
try:
    # Backend logic imports
    from script.knowledge_base import KnowledgeBase
    from script.query_rewriter import generate_rewritten_query
    from script import config # Import config module itself
    from script.config import * # Import specific config variables

    # UI layout import (which handles event binding internally)
    from web_ui.layout import create_layout
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all necessary files (config.py, knowledge_base.py, etc.) exist and PYTHONPATH is correct.")
    exit(999)

# --- Logging Setup ---
log_file = Path("rag_chat_gradio.log")
# Basic config (consider moving detailed config to a setup function if needed)
logging.basicConfig(
    level=config.LOG_LEVEL, # Use log level from config
    format=config.LOG_FORMAT, # Use format from config
    datefmt=config.LOG_DATE_FORMAT, # Use date format from config
    filename=log_file,
    filemode='w'
)
# Console handler for simultaneous terminal output
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
# Add handler to the root logger
logging.getLogger().addHandler(console_handler)
# Set root logger level (optional, basicConfig might do this)
logging.getLogger().setLevel(config.LOG_LEVEL)
# Get logger for this specific file
logger = logging.getLogger(__name__)
# --------------------

# --- Global Variables ---
kb: Optional[KnowledgeBase] = None
kb_initialized = False
generator_system_prompt: Optional[str] = None
prompts_loaded = False
# -----------------------

# --- Path handling ---
# Resolve paths relative to this file's location might be more robust
_current_dir = Path(__file__).resolve().parent
config_dir = _current_dir / "script"
prompt_dir = _current_dir / "prompts"
# Use knowledge base path directly from config
_knowledge_base_dir_path = Path(KNOWLEDGE_BASE_DIR)
_processed_data_dir_path = Path(PROCESSED_DATA_DIR)
_gen_prompt_path = Path(GENERATOR_SYSTEM_PROMPT_FILE) # Use path from config
# --------------------

# --- Resource Loading Function ---
# This function is passed to layout.py to be called via demo.load()
def load_all_resources():
    """Loads prompts and initializes the Knowledge Base."""
    global kb, generator_system_prompt
    global kb_initialized, prompts_loaded

    if kb_initialized and prompts_loaded:
        logger.debug("Resources already loaded.")
        return

    logger.info("Loading resources (Prompts, KB)...")
    start_time = time.time()

    # 1. Load Prompts
    if not prompts_loaded:
        try:
            logger.info(f"Attempting to load generator prompt from: {_gen_prompt_path}")
            if not _gen_prompt_path.is_file():
                 raise FileNotFoundError(f"Generator prompt file not found: {_gen_prompt_path}")
            with open(_gen_prompt_path, "r", encoding="utf-8") as f:
                generator_system_prompt = f.read()
            if not generator_system_prompt:
                 raise ValueError("Generator prompt file is empty.")
            prompts_loaded = True
            logger.info("Prompts loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load essential prompt files: {e}", exc_info=True)
            prompts_loaded = False # Ensure flag is reset on failure

    # 2. Initialize Knowledge Base
    if not kb_initialized:
        try:
            logger.info("Initializing Knowledge Base...")
            if not _knowledge_base_dir_path.is_dir():
                 raise FileNotFoundError(f"Knowledge base directory not found: {_knowledge_base_dir_path}")
            # Ensure embedding model path is valid (optional check here)
            if not Path(EMBEDDING_MODEL_PATH).exists():
                 raise FileNotFoundError(f"Embedding model path not found: {EMBEDDING_MODEL_PATH}")

            kb = KnowledgeBase(
                knowledge_dir=KNOWLEDGE_BASE_DIR, # Use path from config
                knowledge_file_pattern=KNOWLEDGE_FILE_PATTERN # Use pattern from config
            )
            kb_initialized = True
            logger.info("Knowledge Base initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize Knowledge Base: {e}", exc_info=True)
            kb_initialized = False # Ensure flag is reset on failure

    load_duration = time.time() - start_time
    if kb_initialized and prompts_loaded:
        logger.info(f"Essential resources loaded in {load_duration:.2f} seconds.")
    else:
        logger.error(f"One or more essential resources failed to load after {load_duration:.2f} seconds. Check logs.")
        # Optionally raise an error or prevent app launch here if critical resources failed
        # raise RuntimeError("Failed to load critical resources.")

# --- Helper Functions ---

def format_context_for_display(chunks_data: List[Dict[str, str]]) -> str:
    """Formats retrieved context chunks into a Markdown string for display."""
    if not chunks_data:
        return "（未引用知识库片段）"
    md_str = "### 参考知识库片段：\n\n"
    for i, chunk_data in enumerate(chunks_data):
        source = chunk_data.get("source", "未知来源")
        text = chunk_data.get("text", "内容缺失")
        # Limit displayed text length for brevity in UI
        display_text = text[:300] + "..." if len(text) > 300 else text
        # Use blockquote for better visual separation
        md_str += f"**片段 {i + 1} (来源: {source})**\n> {display_text}\n\n"
    return md_str.strip() # Remove trailing newline

def convert_gradio_to_openai(chat_history: Optional[List[Tuple[Optional[str], Optional[str]]]]) -> List[Dict[str, str]]:
    """
    Converts Gradio's default tuple history format to OpenAI's message format (list of dicts).
    Also cleans potential <think> tags from assistant messages.
    """
    messages = []
    if not chat_history:
        return messages
    think_end_tag = "</think>"
    for user_msg, assistant_msg in chat_history:
        if user_msg is not None and user_msg.strip():
            messages.append({"role": "user", "content": user_msg.strip()})
        if assistant_msg is not None and assistant_msg.strip():
            # Clean assistant message (remove thinking part for history)
            cleaned_assistant_msg = assistant_msg
            first_think_end_index = assistant_msg.find(think_end_tag)
            if first_think_end_index != -1:
                 # Get content after the tag
                 cleaned_assistant_msg = assistant_msg[first_think_end_index + len(think_end_tag):].strip()

            # Only add if there's actual content after cleaning
            if cleaned_assistant_msg:
                messages.append({"role": "assistant", "content": cleaned_assistant_msg})
    return messages

def convert_openai_to_gradio_tuples(messages_history: List[Dict[str, str]]) -> List[Tuple[Optional[str], Optional[str]]]:
    """Converts OpenAI message format back to Gradio's default tuple format."""
    gradio_history = []
    user_msg_buffer = None
    for msg in messages_history:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            # If there was a previous user message waiting for an assistant, yield it first
            if user_msg_buffer is not None:
                 gradio_history.append((user_msg_buffer, None))
            user_msg_buffer = content # Store current user message
        elif role == "assistant":
            # Pair with the buffered user message, or yield as assistant-only
            gradio_history.append((user_msg_buffer, content))
            user_msg_buffer = None # Clear buffer after pairing
    # If the last message was from the user, yield it
    if user_msg_buffer is not None:
        gradio_history.append((user_msg_buffer, None))
    return gradio_history

# --- Core Respond Function ---
async def respond(
    message: str,
    chat_history: Optional[List[Tuple[Optional[str], Optional[str]]]],
    # Type hint for yield: List containing chatbot value (List[Tuple]), context (str), rewritten (str)
    ) -> AsyncGenerator[List[Union[List[Tuple[Optional[str], Optional[str]]], str, gr.Markdown, gr.Textbox, gr.update]], None]:
    # !!! 关键：确保每次调用 respond 时，raw_generated_text 都是全新的空字符串 !!!
    raw_generated_text = ""
    logger.debug(
        f"respond 函数开始，raw_generated_text 初始化为空: '{raw_generated_text}' (ID: {id(raw_generated_text)})")

    # 1. Input Handling and Conversion
    chat_history_tuples = chat_history or []
    messages_history = convert_gradio_to_openai(chat_history_tuples) # Internal format

    # 2. History Truncation (on internal format)
    logger.debug(f"History length (tuples): {len(chat_history_tuples)}, Internal: {len(messages_history)}")
    if len(messages_history) > MAX_HISTORY:
        # Decide which messages to keep (e.g.,system prompt + last MAX_HISTORY exchanges)
        # Simple truncation for now:
        messages_history = messages_history[-(MAX_HISTORY * 2):] # Keep last N pairs approx
        logger.warning(f"Truncated internal history to {len(messages_history)} messages.")

    # 3. Basic Input Check
    if not message or not message.strip():
         # Yield list in correct order for outputs
         yield [
             convert_openai_to_gradio_tuples(messages_history), # Chatbot (tuple format)
             format_context_for_display([]),                    # Context Display
             "(无输入)"                                          # Rewritten Query Display
         ]
         return

    logger.info(f"Received message: '{message[:100]}...'")
    user_message_internal = {"role": "user", "content": message.strip()}

    # 4. Query Rewriting
    rewritten_query_str = message.strip() # Default
    actual_rewritten_query_str_for_display = "(重写未启用或失败)"
    if VLLM_REWRITER_API_BASE_URL and generator_system_prompt: # Ensure needed components are loaded
        logger.info("Performing query rewriting via API...")
        rewrite_start_time = time.time()
        try:
            # Pass internal message history
            rewritten_query_result = await generate_rewritten_query(messages_history, message.strip())
            if rewritten_query_result != message.strip():
                rewritten_query_str = rewritten_query_result # Use rewritten query for search
                actual_rewritten_query_str_for_display = rewritten_query_result # Display the rewritten query
                logger.info(f"Query rewriting successful ({(time.time() - rewrite_start_time):.2f}s).")
            else:
                actual_rewritten_query_str_for_display = "(重写结果与原始输入相同)"
                logger.info(f"Query rewriting result same as original or fallback ({(time.time() - rewrite_start_time):.2f}s).")
        except Exception as rewrite_err:
            logger.error(f"Error during query rewriting API call: {rewrite_err}", exc_info=True)
            rewritten_query_str = message.strip() # Fallback to original
            actual_rewritten_query_str_for_display = f"(重写出错: {rewrite_err})"
            # Continue with original query on error
    else:
        logger.info("Skipping query rewriting (API URL not configured or system prompt missing).")
        actual_rewritten_query_str_for_display = "(重写功能未配置或依赖缺失)"

    # 5. Parse Search Terms
    search_terms = [term.strip() for term in rewritten_query_str.strip().split('\n') if term.strip()]
    if not search_terms:
        search_terms = [message.strip()] # Use original if rewrite is empty
    logger.info(f"Planned search terms ({len(search_terms)}): {search_terms}")

    # 6. Retrieval & Aggregation
    final_chunks_data = []
    formatted_context_md = format_context_for_display([]) # Default
    if kb and kb_initialized: # Check if KB is ready
        retrieval_start_time = time.time()
        all_retrieved_data: List[Dict[str, str]] = []
        processed_chunk_texts = set()
        logger.info("--- Starting Multi-Query Retrieval & Aggregation ---")
        for i, term in enumerate(search_terms):
            logger.info(f"Retrieving for sub-query {i+1}/{len(search_terms)}: '{term[:50]}...'")
            try:
                # retrieve_chunks should return List[Dict[str, str]]
                term_chunks_data = kb.retrieve_chunks(term)
                logger.info(f"  Retrieved {len(term_chunks_data)} chunks for this term.")
                added_count = 0
                for chunk_data in term_chunks_data:
                    chunk_text = chunk_data.get("text")
                    if chunk_text and chunk_text not in processed_chunk_texts:
                        all_retrieved_data.append(chunk_data)
                        processed_chunk_texts.add(chunk_text)
                        added_count += 1
                if added_count > 0:
                    logger.debug(f"  Added {added_count} unique chunks to aggregation pool.")
            except Exception as retrieval_err:
                logger.error(f"  Error retrieving chunks for term '{term}': {retrieval_err}", exc_info=True)
        logger.info("--- Multi-Query Retrieval Finished ---")
        logger.info(f"Total unique chunks aggregated before final filtering: {len(all_retrieved_data)}")

        # Context Filtering/Ranking (Simple truncation for now)
        # Consider adding relevance ranking here if needed
        if len(all_retrieved_data) > MAX_AGGREGATED_RESULTS:
             logger.warning(
                 f"Aggregated results ({len(all_retrieved_data)}) exceed limit ({MAX_AGGREGATED_RESULTS}). Truncating..."
             )
             final_chunks_data = all_retrieved_data[:MAX_AGGREGATED_RESULTS]
        else:
             final_chunks_data = all_retrieved_data
        logger.info(f"Final number of chunks selected for LLM context: {len(final_chunks_data)}")

        # Format Context for display
        formatted_context_md = format_context_for_display(final_chunks_data)
        logger.info(f"Retrieval and aggregation completed in {(time.time() - retrieval_start_time):.2f}s.")
    else:
        logger.error("KnowledgeBase not initialized, skipping retrieval.")
        formatted_context_md = "错误：知识库未初始化或加载失败。"

    # 7. Yield Initial Update with Context and Rewritten Query
    # Prepare placeholder for chatbot output (internal format first)
    placeholder_assistant_message = {"role": "assistant", "content": "..."}
    initial_chatbot_messages = messages_history + [user_message_internal, placeholder_assistant_message]
    yield [
        convert_openai_to_gradio_tuples(initial_chatbot_messages), # Chatbot (tuple format)
        formatted_context_md,                                      # Context Display
        actual_rewritten_query_str_for_display                     # Rewritten Query Display
    ]

    logger.debug(f"准备调用 vLLM API，此时 raw_generated_text: '{raw_generated_text}' (ID: {id(raw_generated_text)})")

    # 8. Prepare Generation Payload for vLLM API
    generation_api_messages = []
    if generator_system_prompt:
         generation_api_messages.append({"role": "system", "content": generator_system_prompt})
    # Add historical messages (internal format)
    generation_api_messages.extend(messages_history)
    # Prepare context text
    context_text = "\n\n".join(
        [f"【片段 {i + 1} | 来源: {d.get('source','N/A')}】 {d.get('text','')}" for i, d in enumerate(final_chunks_data)]
    ) if final_chunks_data else "知识库中未找到相关内容。"
    # Add current user message with context
    generation_api_messages.append({"role": "user", "content": f"User: {message.strip()}\n\n知识库：\n{context_text}"})

    api_url = f"{VLLM_GENERATOR_API_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    payload = {
        "model": VLLM_GENERATOR_MODEL_ID_FOR_API, # From config
        "messages": generation_api_messages,
        "max_tokens": GENERATION_CONFIG.get("max_tokens", 4096),
        "temperature": GENERATION_CONFIG.get("temperature", 0.6),
        "top_p": GENERATION_CONFIG.get("top_p", 0.95),
        "repetition_penalty": GENERATION_CONFIG.get("repetition_penalty", 1.1),
        "stop": GENERATION_CONFIG.get("stop"), # Use stop sequences from config if any
        "stream": True,
    }
    # Remove keys with None values as vLLM might not like them
    payload = {k: v for k, v in payload.items() if v is not None}
    logger.debug(f"Sending payload to generator API: {json.dumps(payload, ensure_ascii=False, indent=2)}")

    # 9. Call vLLM API and Handle Streaming Response
    raw_generated_text = ""
    error_occurred = False
    error_message = "抱歉，处理您的请求时发生错误。"
    # History containing only messages up to the current user query (internal format)
    current_full_history_dicts = messages_history + [user_message_internal]
    generation_start_time = time.time()

    last_yield_time = time.time()
    yield_interval = 0.5  # seconds

    try:
        # Increased timeout for potentially long generation
        timeout = aiohttp.ClientTimeout(total=300.0, connect=10.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(api_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_body = await response.text()
                    logger.error(f"vLLM Generator API request failed status {response.status}: {error_body}")
                    error_message = f"模型生成API请求失败 (状态码: {response.status})。"
                    error_occurred = True
                else:
                    # Process Server-Sent Events (SSE) stream
                    async for line_bytes in response.content:
                        line = line_bytes.decode('utf-8').strip()
                        # Skip empty lines or comments
                        if not line or not line.startswith("data:"):
                            continue

                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]":
                            logger.info("Received [DONE] marker from stream.")
                            break # End of stream

                        try:
                            chunk_data = json.loads(data_str)
                            logger.debug(f"原始SSE数据块: {data_str}")  # 记录原始数据
                            logger.debug(f"解析后 chunk_data: {chunk_data}")

                            delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                            logger.debug(f"提取的 delta 对象: {delta}")

                            delta_content = delta.get("content")  # 保持原样获取

                            # !!! 关键修改：更准确地判断 delta_content !!!
                            # 即使 delta_content 是空字符串 ""，也应该视为有效内容并尝试追加和 yield
                            # 只有当 delta_content 是 None (JSON null) 时，才可能需要跳过（取决于模型语义）
                            if delta_content is not None:  # <--- 修改这里的判断条件
                                raw_generated_text += delta_content
                                logger.debug(f"追加后 raw_generated_text (前100字符): '{raw_generated_text[:100]}...'")

                                current_assistant_message = {"role": "assistant", "content": raw_generated_text}
                                streaming_messages_value = current_full_history_dicts + [current_assistant_message]
                                converted_tuples_for_yield = convert_openai_to_gradio_tuples(streaming_messages_value)

                                logger.debug(
                                    f"准备 Yield 的数据 (部分): {str(converted_tuples_for_yield)[:200]}")  # 记录准备 yield 的数据

                                yield [
                                    converted_tuples_for_yield,
                                    gr.update(),
                                    gr.update()
                                ]
                            else:
                                logger.debug(f"收到的 delta_content 为 None，本次不追加也不 Yield。原始 delta: {delta}")

                        except json.JSONDecodeError:
                            logger.warning(f"SSE 流 JSON 解析错误: {data_str}")
                        except Exception as e:
                            # 明确记录是在处理哪个数据块时出错
                            logger.error(f"处理 SSE 数据块 '{data_str}' 时发生错误: {e}",
                                         exc_info=True)  # 改为 exc_info=True 获取完整堆栈

    except asyncio.TimeoutError:
        error_occurred = True
        error_message = "模型生成请求超时。"
        logger.error(f"vLLM Generator API request timed out after {timeout.total} seconds.")
    except aiohttp.ClientConnectorError as e:
        error_occurred = True
        error_message = f"无法连接模型生成服务器: {e}"
        logger.error(f"vLLM Generator connection error {api_url}: {e}", exc_info=False)
    except aiohttp.ClientError as e: # Catch other aiohttp client errors
         error_occurred = True
         error_message = f"模型生成API通信错误: {e}"
         logger.error(f"vLLM Generator communication error: {e}", exc_info=True)
    except Exception as e:
        error_occurred = True
        error_message = f"处理模型响应时发生意外错误: {e}"
        logger.error(f"Unexpected error during API call/streaming: {e}", exc_info=True)

    generation_duration = time.time() - generation_start_time
    logger.info(f"Streaming finished. Duration: {generation_duration:.2f}s. Error occurred: {error_occurred}")

    # 10. Final Processing & Yield
    final_display_text_for_assistant = raw_generated_text # Default to raw text
    # Apply error message if needed
    if error_occurred:
        # Append error to the generated text, or replace it
        if raw_generated_text:
             final_display_text_for_assistant = raw_generated_text.strip() + f"\n\n---\n❌ **错误:** {error_message}"
        else:
             final_display_text_for_assistant = f"❌ **错误:** {error_message}"
    else:
        # Post-process successful response (handle <think> tags for display)
        think_end_tag = "</think>"
        first_think_end_index = raw_generated_text.find(think_end_tag)
        if first_think_end_index != -1:
            thinking_part = raw_generated_text[:first_think_end_index + len(think_end_tag)].strip()
            final_answer_part = raw_generated_text[first_think_end_index + len(think_end_tag):].strip()
            # Format thinking part for display (e.g., using details/summary or code block)
            # formatted_thinking = f"<details><summary>思考过程</summary>\n\n```text\n{thinking_part}\n```\n\n</details>" # Might not render well in all MD viewers
            formatted_thinking = f"```text\n{thinking_part}\n```" # Simpler code block
            if final_answer_part:
                final_display_text_for_assistant = f"{formatted_thinking}\n\n{final_answer_part}"
            else:
                # If only thinking part is returned
                final_display_text_for_assistant = formatted_thinking
        # else: final_display_text_for_assistant remains raw_generated_text

    # Prepare final chatbot state (internal format)
    final_assistant_message = {"role": "assistant", "content": final_display_text_for_assistant}
    final_messages_value = current_full_history_dicts + [final_assistant_message]

    # Final yield updating all components
    yield [
        convert_openai_to_gradio_tuples(final_messages_value), # Chatbot (tuple format)
        formatted_context_md,                                  # Context Display
        actual_rewritten_query_str_for_display                 # Rewritten Query Display
    ]


# --- Clear History Action Function ---
# This function is passed to layout.py and called by the clear button click event
def clear_history_action():
    """Action to clear chat history and related displays."""
    logger.info("Clearing chat history action triggered.")
    # Returns default values for the components listed in clear_button.click outputs
    # chatbot, msg_input, context_display, rewritten_query_display
    return (
        [],                               # chatbot (empty list for tuple format)
        "",                               # msg_input (clear input box)
        "上下文将显示在这里...",            # context_display
        "重构后的查询将显示在这里..."      # rewritten_query_display
    )


# --- Main Application Entry Point ---
if __name__ == "__main__":
    # Ensure critical libraries are available
    try:
        import aiohttp
    except ImportError:
        print("错误: 核心依赖 aiohttp 未安装。请运行: pip install aiohttp")
        exit(1)
    try:
        import gradio
    except ImportError:
         print("错误: 核心依赖 gradio 未安装。请运行: pip install gradio")
         exit(1)

    logger.info("Starting RAG Gradio Application...")

    # Perform initial resource loading *before* creating the UI
    # This prevents UI load delays and ensures KB/prompts are ready
    try:
        load_all_resources()
        # Check if critical resources failed to load
        if not kb_initialized or not prompts_loaded:
             logger.critical("Application cannot start due to resource loading failures.")
             # Exit or display error message in UI (latter requires UI to be partially built first)
             print("\n" + "="*30)
             print("错误：应用启动失败！必要的资源（知识库/提示词）加载失败。请检查日志和配置。")
             print("="*30 + "\n")
             exit(1) # Exit if critical resources are missing
    except Exception as initial_load_err:
         logger.critical(f"Unexpected error during initial resource loading: {initial_load_err}", exc_info=True)
         print(f"\n错误：应用启动时发生意外错误: {initial_load_err}\n")
         exit(1)


    # Create the Gradio layout and bind event handlers
    # Pass the actual function objects (references) to the layout creator
    try:
        demo_instance = create_layout(
            respond_fn=respond,               # Pass the async respond function
            clear_fn=clear_history_action,    # Pass the clear action function
            load_fn=load_all_resources,       # Pass the resource load function
            theme="soft"                      # Specify desired theme (or load from config)
        )
    except Exception as layout_err:
         logger.critical(f"Failed to create Gradio layout: {layout_err}", exc_info=True)
         print(f"\n错误：创建界面布局时失败: {layout_err}\n")
         exit(1)

    # Launch the Gradio app
    logger.info("Launching Gradio interface...")
    print("Gradio 应用正在启动，请在浏览器中访问提供的 URL...")
    # Use concurrency limit from config if available, otherwise default
    concurrency = getattr(config, "GRADIO_CONCURRENCY_LIMIT", 8)
    try:
        demo_instance.queue(default_concurrency_limit=concurrency).launch(
            server_name="0.0.0.0",          # Listen on all interfaces
            server_port=8848, # Use port from config
            share=False,        # Use share setting from config
            # Add other launch options as needed from config (e.g., auth)
            # auth = (config.GRADIO_USER, config.GRADIO_PASSWORD) if config.GRADIO_USER else None
        )
        logger.info(f"Gradio app launched on http://0.0.0.0:8848")
    except Exception as launch_err:
         logger.critical(f"Failed to launch Gradio app: {launch_err}", exc_info=True)
         print(f"\n错误：启动 Gradio 服务失败: {launch_err}\n")
         exit(1)