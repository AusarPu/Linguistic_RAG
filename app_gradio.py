# app_gradio.py (第一部分：导入、全局变量、辅助函数)

import gradio as gr
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, AsyncGenerator, Union, Any
import json
import asyncio
import os  # 用于路径处理，确保导入
import sys  # 用于路径处理，确保导入

try:
    import aiohttp
except ImportError:
    print("Please install aiohttp: pip install aiohttp")
    exit(1)

# 获取当前文件的目录，然后获取项目根目录
_CURRENT_FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR_FOR_APP = _CURRENT_FILE_DIR  # 假设 app_gradio.py 在项目根目录

_SCRIPT_DIR_PATH = PROJECT_ROOT_DIR_FOR_APP / "script"
_WEB_UI_DIR_PATH = PROJECT_ROOT_DIR_FOR_APP / "web_ui"
if str(_SCRIPT_DIR_PATH) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR_PATH))
if str(PROJECT_ROOT_DIR_FOR_APP) not in sys.path:  # 确保根目录也在
    sys.path.insert(0, str(PROJECT_ROOT_DIR_FOR_APP))

try:
    # 后端逻辑导入
    from script.knowledge_base import KnowledgeBase
    from script import config  # 导入 config 模块本身
    # from script.config import * # 导入所有配置变量 (按需选择)
    # 我们会在需要的地方用 config.VARIABLE_NAME

    # 核心RAG流程
    from script.rag_pipeline import execute_rag_flow  # <--- 新增导入

    # UI 布局导入
    from web_ui.layout import create_layout  # 假设它在 web_ui/layout.py

except ImportError as e:
    # 尝试另一种导入方式，如果 app_gradio.py 就在 script 目录平级
    # 或者你直接从 script/ 运行此文件
    try:
        from knowledge_base import KnowledgeBase
        import config
        from rag_pipeline import execute_rag_flow
        from web_ui.layout import create_layout  # web_ui 目录需要能被找到
    except ImportError as e_inner:
        print(f"Error importing modules: {e} / {e_inner}")
        print("Please ensure all necessary files (config.py, knowledge_base.py, rag_pipeline.py etc.) "
              "exist and PYTHONPATH is correct, or adjust import paths.")
        print(f"Current sys.path: {sys.path}")
        print(f"Attempted SCRIPT_DIR: {_SCRIPT_DIR_PATH}, WEB_UI_DIR: {_WEB_UI_DIR_PATH}")
        exit(999)

# --- Logging Setup (与你现有逻辑保持一致) ---
log_file = PROJECT_ROOT_DIR_FOR_APP / "rag_chat_gradio.log"  # 日志文件放在项目根目录
logging.basicConfig(
    level=getattr(config, 'LOG_LEVEL', logging.INFO),  # 使用 config 中的 LOG_LEVEL
    format=getattr(config, 'LOG_FORMAT', '%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s'),
    datefmt=getattr(config, 'LOG_DATE_FORMAT', '%Y-%m-%d %H:%M:%S'),
    filename=log_file,
    filemode='a'  # 通常用 'a' (append) 而不是 'w' (overwrite)
)
console_handler = logging.StreamHandler(sys.stdout)  # 输出到控制台
console_handler.setFormatter(logging.Formatter(
    getattr(config, 'LOG_FORMAT', '%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s'),
    datefmt=getattr(config, 'LOG_DATE_FORMAT', '%Y-%m-%d %H:%M:%S')
))
root_logger = logging.getLogger()
if not root_logger.hasHandlers() or all(
        isinstance(h, logging.FileHandler) for h in root_logger.handlers):  # 避免重复添加控制台handler
    root_logger.addHandler(console_handler)
root_logger.setLevel(getattr(config, 'LOG_LEVEL', logging.INFO))
logger = logging.getLogger(__name__)  # 获取当前文件的 logger
# --------------------

# --- Global Variables (与你现有逻辑保持一致) ---
kb: Optional[KnowledgeBase] = None
kb_initialized = False
generator_system_prompt_content: Optional[str] = None  # 修改变量名以清晰表示是内容
prompts_loaded = False
# -----------------------

# --- Path handling (与你现有逻辑保持一致，确保路径正确) ---
_prompt_dir = PROJECT_ROOT_DIR_FOR_APP / "prompts"
_gen_prompt_path = _prompt_dir / \
                   getattr(config, 'GENERATOR_SYSTEM_PROMPT_FILE', "generator_system_prompt.txt").split('/')[-1]
# (上面假设GENERATOR_SYSTEM_PROMPT_FILE可能包含相对路径，我们只取文件名部分与_prompt_dir拼接)
# 更稳健的方式是，如果config.GENERATOR_SYSTEM_PROMPT_FILE是绝对路径就直接用，否则相对PROJECT_ROOT_DIR
if not os.path.isabs(config.GENERATOR_SYSTEM_PROMPT_FILE):
    _gen_prompt_path = PROJECT_ROOT_DIR_FOR_APP / config.GENERATOR_SYSTEM_PROMPT_FILE
else:
    _gen_prompt_path = Path(config.GENERATOR_SYSTEM_PROMPT_FILE)


# --- Resource Loading Function (load_all_resources) ---
# 这个函数与你之前的版本基本一致，只是KnowledgeBase的初始化参数变了
def load_all_resources():
    global kb, generator_system_prompt_content, kb_initialized, prompts_loaded

    if kb_initialized and prompts_loaded:
        logger.debug("Resources already loaded.")
        return

    logger.info("Loading resources (Prompts, KB)...")
    start_time = time.time()

    # 1. Load Generator System Prompt
    if not prompts_loaded:
        try:
            logger.info(f"Attempting to load generator prompt from: {_gen_prompt_path}")
            if not _gen_prompt_path.is_file():
                raise FileNotFoundError(f"Generator prompt file not found: {_gen_prompt_path}")
            with open(_gen_prompt_path, "r", encoding="utf-8") as f:
                generator_system_prompt_content = f.read()  # 赋值给修改后的变量名
            if not generator_system_prompt_content:  # 检查是否为空
                logger.warning("Generator prompt file is empty. Using a default.")
                # generator_system_prompt_content = "你是一个乐于助人的AI助手。" # 可选的默认值
            prompts_loaded = True
            logger.info("Generator system prompt loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load generator system prompt: {e}", exc_info=True)
            prompts_loaded = False

            # 2. Initialize Knowledge Base
    if not kb_initialized:
        try:
            logger.info("Initializing Knowledge Base...")
            # KnowledgeBase 的 __init__ 现在不接收 KNOWLEDGE_BASE_DIR 和 KNOWLEDGE_FILE_PATTERN
            # 它内部会从 config.py 读取 PROCESSED_DATA_DIR 等路径来加载索引
            kb = KnowledgeBase()
            kb_initialized = True
            logger.info("Knowledge Base initialized successfully.")
        except RuntimeError as e:  # KnowledgeBase 初始化失败时会抛出 RuntimeError
            logger.critical(f"Knowledge Base initialization failed: {e}", exc_info=True)
            kb_initialized = False
        except Exception as e:
            logger.critical(f"Unexpected error during Knowledge Base initialization: {e}", exc_info=True)
            kb_initialized = False

    load_duration = time.time() - start_time
    if kb_initialized and prompts_loaded:  # 确保两个都成功
        logger.info(f"Essential resources loaded in {load_duration:.2f} seconds.")
    else:
        # 明确指出哪个失败了
        failed_resources = []
        if not kb_initialized: failed_resources.append("Knowledge Base")
        if not prompts_loaded: failed_resources.append("Generator Prompt")
        logger.error(
            f"Resource loading failed for: {', '.join(failed_resources)} (Duration: {load_duration:.2f}s). Check logs.")
        # 应用是否能继续运行取决于这些资源的重要性。KB通常是核心。
        if not kb_initialized:
            raise RuntimeError(f"CRITICAL: KnowledgeBase failed to load. Application cannot continue.")


# --- Helper Functions (convert_gradio_to_openai, convert_openai_to_gradio_tuples, format_context_for_display) ---
# 这些与你之前的版本可以保持一致，或者根据需要微调 format_context_for_display
# format_context_for_display 现在应该能处理包含 rerank_score, retrieved_from_paths 等新元数据的块

def format_context_for_display(chunks_data: List[Dict[str, Any]]) -> str:
    """Formats retrieved (and reranked) context chunks into a Markdown string for display."""
    if not chunks_data:
        return "（未引用知识库片段或上下文为空）"

    md_str = "### 参考上下文片段 (已重排):\n\n"
    for i, chunk_data in enumerate(chunks_data):
        doc_name = chunk_data.get("doc_name", "未知文档")
        page_num = chunk_data.get("page_number", "N/A")
        chunk_id = chunk_data.get("chunk_id", "N/A")
        text = chunk_data.get("text", "内容缺失")
        rerank_score = chunk_data.get("rerank_score")
        retrieved_from = list(chunk_data.get("retrieved_from_paths", {}).keys())  # 获取召回路径列表

        display_text = text[:250] + "..." if len(text) > 250 else text  # 调整预览长度

        md_str += f"**片段 {i + 1}** (ID: `{chunk_id}`)\n"
        md_str += f"*来源*: `{doc_name}`, 页码: `{page_num}`\n"
        if rerank_score is not None:
            md_str += f"*Rerank得分*: `{rerank_score:.4f}`\n"
        if retrieved_from:
            md_str += f"*召回路径*: `{', '.join(retrieved_from)}`\n"
        md_str += f"> {display_text.replace(chr(10), ' ')}\n\n"  # 替换换行以便UI显示
    return md_str.strip()


# convert_gradio_to_openai 和 convert_openai_to_gradio_tuples 保持不变
# (从你提供的 app_gradio.py 复制过来即可)
def convert_gradio_to_openai(chat_history: Optional[List[Tuple[Optional[str], Optional[str]]]]) -> List[Dict[str, str]]:
    messages = []
    if not chat_history:
        return messages
    think_end_tag = "</think>"  # 你之前的代码中提到了这个，这里保留
    for user_msg, assistant_msg in chat_history:
        if user_msg is not None and user_msg.strip():
            messages.append({"role": "user", "content": user_msg.strip()})
        if assistant_msg is not None and assistant_msg.strip():
            cleaned_assistant_msg = assistant_msg
            # 你的代码中有一个移除 <think> 标签的逻辑，我们保留它，
            # 尽管新的流程中 CoT 可能通过独立的UI组件展示
            first_think_end_index = assistant_msg.find(think_end_tag)
            if first_think_end_index != -1:
                cleaned_assistant_msg = assistant_msg[first_think_end_index + len(think_end_tag):].strip()
            if cleaned_assistant_msg:  # 只添加清理后非空的内容
                messages.append({"role": "assistant", "content": cleaned_assistant_msg})
    return messages


def convert_openai_to_gradio_tuples(messages_history: List[Dict[str, str]]) -> List[
    Tuple[Optional[str], Optional[str]]]:
    gradio_history = []
    user_msg_buffer = None
    for msg in messages_history:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            if user_msg_buffer is not None:  # 前一个user消息没有配对的assistant消息
                gradio_history.append((user_msg_buffer, None))
            user_msg_buffer = content
        elif role == "assistant":
            gradio_history.append((user_msg_buffer, content))
            user_msg_buffer = None
    if user_msg_buffer is not None:  # 处理最后一个是用户消息的情况
        gradio_history.append((user_msg_buffer, None))
    return gradio_history


async def respond(
        message: str,
        chat_history: Optional[List[Tuple[Optional[str], Optional[str]]]],
        # 假设你的 Gradio layout outputs 顺序是: chatbot, context_display, rewritten_query_display
        # 如果你添加了 thinking_process_display, 那么 yield 的列表也需要对应增加
) -> AsyncGenerator[
    List[Union[List[Tuple[Optional[str], Optional[str]]], str, gr.Markdown, gr.Textbox, gr.update]], None]:
    global kb, kb_initialized, generator_system_prompt_content  # 使用全局加载的资源

    # --- 1. 初始化和输入处理 ---
    chat_history_tuples = chat_history or []
    # 将Gradio的聊天历史转换为OpenAI格式，用于传递给RAG pipeline
    messages_history_openai = convert_gradio_to_openai(chat_history_tuples)

    # 用于UI更新的变量
    # 注意：Gradio 的 chatbot 组件期望的是一个 [(user_msg, assistant_msg), ...] 格式的完整列表
    # 我们需要维护一个内部的 OpenAI 格式历史，并在每次 yield 时转换为 Gradio 格式
    current_openai_history_for_display = list(messages_history_openai)  # 复制一份用于显示
    if message and message.strip():
        current_openai_history_for_display.append({"role": "user", "content": message.strip()})

    assistant_response_accumulator = ""  # 用于累积 content_delta
    thinking_process_accumulator = ""  # 用于累积 reasoning_delta

    # 初始化UI组件的显示内容
    chatbot_display_list = convert_openai_to_gradio_tuples(current_openai_history_for_display + [{"role": "assistant", "content": "🤔 处理中..."}])
    context_md_str = "(上下文信息将显示在这里...)"
    rewritten_query_str = "(重写后的查询将显示在这里...)"
    # thinking_md_str = "(模型思考过程将显示在这里...)" # 如果有这个UI组件

    # 辅助函数，用于构造并 yield Gradio 更新
    # 假设你的Gradio输出顺序是: chatbot, context_display, rewritten_query_display, [可选: thinking_display]
    # 你需要根据你的 web_ui.layout.py 中 create_layout 函数定义的 outputs 顺序来调整 yield 的列表
    async def yield_gradio_update(
            current_assistant_reply: str = "🤔 处理中...",
            context_md: Optional[str] = None,
            rewritten_query: Optional[str] = None,
            thinking_md: Optional[str] = None  # 如果有思考过程显示组件
    ):
        # 准备 chatbot 的显示列表
        history_for_chatbot = current_openai_history_for_display + [
            {"role": "assistant", "content": current_assistant_reply}]
        chatbot_tuples = convert_openai_to_gradio_tuples(history_for_chatbot)

        outputs_to_yield = [chatbot_tuples]
        outputs_to_yield.append(gr.Markdown(value=context_md) if context_md is not None else gr.Markdown(update=True))
        outputs_to_yield.append(
            gr.Textbox(value=rewritten_query) if rewritten_query is not None else gr.Textbox(update=True))

        # 如果你的UI中有第四个输出组件用于显示思考过程 (例如一个名为 thinking_process_display 的 gr.Markdown)
        # if thinking_md is not None:
        #     outputs_to_yield.append(gr.Markdown(value=thinking_md, visible=True))
        # else:
        #     outputs_to_yield.append(gr.Markdown(update=True)) # 或者 gr.Markdown(visible=False)

        yield outputs_to_yield

    # --- 检查KB是否初始化 ---
    if not kb or not kb_initialized:
        error_msg = "错误：知识库未初始化或加载失败，无法处理您的请求。"
        logger.error(error_msg)
        yield_gradio_update(current_assistant_reply=error_msg, context_md="错误", rewritten_query="错误")
        return

    # --- 对空消息的快速处理 ---
    if not message or not message.strip():
        yield_gradio_update(current_assistant_reply="(请输入您的问题)", context_md="无上下文",
                                  rewritten_query="无查询")
        return

    logger.info(f"Respond function called with message: '{message[:100]}...'")

    # --- 初始UI更新：显示用户消息和“处理中” ---
    yield_gradio_update()  # 使用默认的 "🤔 处理中..."

    # --- 2. 调用核心 RAG 流程 ---
    final_rewritten_query = message.strip()  # 默认值
    final_context_md = "(未能检索到相关上下文)"

    try:
        async for event in execute_rag_flow(
                user_query=message.strip(),
                chat_history_openai=messages_history_openai,  # 传递转换后的历史
                kb_instance=kb
        ):
            event_type = event.get("type")

            if event_type == "status":
                stage = event.get("stage", "unknown_stage")
                status_message = event.get("message", "处理中...")
                logger.info(f"[UI_UPDATE] Status: [{stage}] {status_message}")
                yield_gradio_update(current_assistant_reply=f"⏳ {status_message}",
                                          context_md=final_context_md,  # 保持之前的上下文或初始值
                                          rewritten_query=final_rewritten_query)

            elif event_type == "rewritten_query_result":
                final_rewritten_query = event.get("rewritten_text", final_rewritten_query)
                yield_gradio_update(current_assistant_reply="⏳ 检索中...",
                                          context_md=final_context_md,
                                          rewritten_query=final_rewritten_query)

            elif event_type == "reranked_context_for_display":
                reranked_chunks_for_ui = event.get("chunks", [])
                final_context_md = format_context_for_display(reranked_chunks_for_ui)  # 使用你已有的格式化函数
                yield_gradio_update(current_assistant_reply="⏳ 生成答案中...",
                                          context_md=final_context_md,
                                          rewritten_query=final_rewritten_query)

            elif event_type == "reasoning_delta":
                thinking_token = event.get("text", "")
                thinking_process_accumulator += thinking_token
                # --- 更新思考过程UI组件 ---
                # 如果你有一个专门的 thinking_process_display: gr.Markdown 组件
                yield [
                    gr.Chatbot(update=True), # chatbot主回复区可以显示"正在思考..."
                    gr.Markdown(update=True), # context_display
                    gr.Textbox(update=True),  # rewritten_query_display
                    # gr.Markdown(value=format_thinking_for_display(thinking_process_accumulator)) # 假设你有这个格式化函数
                ]
                # 为了简单，我们暂时不在主聊天流中混合思考过程，但你可以记录它
                logger.debug(f"Reasoning delta: {thinking_token}")


            elif event_type == "content_delta":
                content_token = event.get("text", "")
                print(content_token,end="")
                assistant_response_accumulator += content_token
                yield_gradio_update(current_assistant_reply=assistant_response_accumulator + "▌",  # 打字机效果
                                          context_md=final_context_md,
                                          rewritten_query=final_rewritten_query)

            elif event_type == "final_answer_complete":
                assistant_response_accumulator = event.get("full_text", assistant_response_accumulator).strip()
                # full_reasoning = event.get("full_reasoning", thinking_process_accumulator).strip()
                # final_context_chunk_ids = event.get("final_context_chunk_ids", [])
                logger.info(f"Final answer generated (length: {len(assistant_response_accumulator)}).")
                # 最后的UI更新会在pipeline_end时进行，以确保是最终状态

            elif event_type == "error":
                error_stage = event.get("stage", "unknown")
                error_message = event.get("message", "未知错误")
                logger.error(f"Pipeline error at stage '{error_stage}': {error_message}")
                assistant_response_accumulator = f"❌ 在处理阶段 '{error_stage}' 发生错误: {error_message}"
                # yield 最终错误状态然后中断
                yield_gradio_update(current_assistant_reply=assistant_response_accumulator,
                                          context_md=final_context_md,
                                          rewritten_query=final_rewritten_query)
                return  # 提前结束 respond 函数

            elif event_type == "pipeline_end":
                logger.info(f"Pipeline ended with reason: {event.get('reason')}")
                # 如果是因为错误结束，assistant_response_accumulator可能已经是错误信息
                # 如果是正常结束，它应该是累积的答案
                if not assistant_response_accumulator and event.get('reason') == "no_context_found_after_retrieval":
                    assistant_response_accumulator = "抱歉，未能找到与您问题直接相关的知识。"
                elif not assistant_response_accumulator and event.get('reason') == "no_context_found_after_reranking":
                    assistant_response_accumulator = "抱歉，信息筛选后未能找到足够相关的内容。"

                yield_gradio_update(current_assistant_reply=assistant_response_accumulator.strip(),  # 移除可能的光标
                                          context_md=final_context_md,
                                          rewritten_query=final_rewritten_query)
                return  # 流程正常或处理完毕地结束

    except Exception as e:
        logger.error(f"Error in respond's main RAG flow execution: {e}", exc_info=True)
        error_text = f"抱歉，处理您的请求时发生系统内部错误。"
        # 确保 chat_history_tuples 和其他UI元素是最新的状态
        # current_openai_history_for_display 包含了当前的用户输入
        # 我们需要将错误信息作为助手的最后回复
        error_chatbot_history = current_openai_history_for_display + [{"role": "assistant", "content": error_text}]
        yield [
            convert_openai_to_gradio_tuples(error_chatbot_history),
            final_context_md if final_context_md else "发生错误，无上下文。",
            final_rewritten_query if final_rewritten_query else "(查询处理出错)"
        ]
    finally:
        logger.info(f"Respond function finished for query: '{message[:50]}...'")

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
    # 确保核心依赖已安装 (你之前的代码中已经有这个检查)
    try:
        import gradio  # type: ignore
        # import aiohttp # aiohttp 已经在 respond 函数中被导入和使用
    except ImportError as e:
        missing_lib = str(e).split("'")[-2]  # 尝试提取缺失的库名
        print(f"错误: 核心依赖 {missing_lib} 未安装。请运行: pip install {missing_lib.lower()}")
        exit(1)

    logger.info("Starting RAG Gradio Application...")

    # 1. 执行初始资源加载
    # 这个函数会设置全局的 kb_instance 和 generator_system_prompt_content
    try:
        load_all_resources()
        # 在 load_all_resources 内部，我们已经添加了对 kb_initialized 和 prompts_loaded 的检查
        # 如果关键资源加载失败，它会抛出 RuntimeError 或打印错误并可能导致应用无法正常工作
        if not kb_initialized:  # 双重检查
            logger.critical("KnowledgeBase 未能在 load_all_resources 中成功初始化。应用可能无法正常工作。")
            # 你可以选择在这里强制退出，或者让 Gradio UI 显示错误信息
            # exit(1)
        if not prompts_loaded:  # 双重检查
            logger.warning("Generator system prompt 未能在 load_all_resources 中成功加载。可能会使用默认提示。")

    except RuntimeError as e:  # 捕获 load_all_resources 中可能抛出的关键错误
        logger.critical(f"应用启动失败，必要的资源加载时发生错误: {e}", exc_info=True)
        print(f"\nCRITICAL ERROR: Application failed to start due to resource loading issue: {e}\n")
        exit(1)
    except Exception as initial_load_err:
        logger.critical(f"应用启动时发生未知错误: {initial_load_err}", exc_info=True)
        print(f"\nCRITICAL ERROR: Unexpected error during application startup: {initial_load_err}\n")
        exit(1)

    # 2. 创建 Gradio UI 布局
    # create_layout 函数会从 web_ui.layout 导入
    # 它接收 respond 函数、clear_history_action 函数和 load_all_resources 函数作为参数
    # 并返回一个 gr.Blocks 实例 (demo_instance)
    try:
        # 确保 config 对象和所有需要的常量都能被 create_layout 或其内部逻辑访问
        # (如果 create_layout 直接从 config 模块读取配置的话)
        demo_instance = create_layout(
            respond_fn=respond,  # 我们重构后的异步 respond 函数
            clear_fn=clear_history_action,  # 你已有的清空历史函数
            load_fn=load_all_resources,  # 资源加载函数，用于 "重新加载资源" 按钮
            # 根据你的 create_layout 函数定义，可能还需要其他参数
        )
        logger.info("Gradio UI layout created successfully.")
    except Exception as layout_err:
        logger.critical(f"创建 Gradio UI 布局时失败: {layout_err}", exc_info=True)
        print(f"\nCRITICAL ERROR: Failed to create Gradio UI layout: {layout_err}\n")
        exit(1)

    # 3. 启动 Gradio 应用
    # 从 config.py 中读取 Gradio 启动参数
    server_name = getattr(config, "GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = getattr(config, "GRADIO_SERVER_PORT", 8848)  # 你之前用的是8848
    share_gradio = getattr(config, "GRADIO_SHARE", False)
    gradio_user = getattr(config, "GRADIO_USER", None)
    gradio_password = getattr(config, "GRADIO_PASSWORD", None)
    concurrency_limit = getattr(config, "GRADIO_CONCURRENCY_LIMIT", 8)  # 与你之前代码一致

    auth_credentials = None
    if gradio_user and gradio_password:
        auth_credentials = (gradio_user, gradio_password)
        logger.info(f"Gradio认证已启用，用户: {gradio_user}")
    else:
        logger.info("Gradio认证未启用。")

    logger.info(f"准备在 {server_name}:{server_port} 启动 Gradio 应用...")
    print(f"Gradio 应用正在启动，请在浏览器中访问 http://{server_name}:{server_port}")
    if server_name == "0.0.0.0":
        print(f"如果在本机运行，也可以通过 http://127.0.0.1:{server_port} 访问")

    try:
        demo_instance.queue(default_concurrency_limit=concurrency_limit).launch(
            server_name=server_name,
            server_port=server_port,
            share=share_gradio,
            auth=auth_credentials,
            # prevent_thread_lock=True, # 对于异步函数，有时需要这个
            # allowed_paths=[str(PROJECT_ROOT_DIR_FOR_APP / "your_static_folder")] # 如果需要提供静态文件
        )
        # launch() 是一个阻塞调用，程序会在这里等待直到 Gradio 服务停止
        logger.info(f"Gradio app has been shut down.")

    except Exception as launch_err:
        logger.critical(f"启动 Gradio 应用时失败: {launch_err}", exc_info=True)
        print(f"\nCRITICAL ERROR: Failed to launch Gradio app: {launch_err}\n")
        exit(1)
