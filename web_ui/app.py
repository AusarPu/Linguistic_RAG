import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gradio as gr
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple

# 从项目中导入
from script.knowledge_base import KnowledgeBase  #
from script.rag_pipeline import execute_rag_flow  #
from script import config  #

# 从web_ui模块导入
from web_ui.ui_components import create_ui
from web_ui.event_handlers import update_ui_for_event, format_references_for_display

# --- 全局变量和初始化 ---
logger = logging.getLogger(__name__)
kb_instance: Optional[KnowledgeBase] = None
# 从config加载系统提示，如果rag_pipeline.py中已经加载，确保这里获取的是同一个
# GENERATOR_SYSTEM_PROMPT_CONTENT = config.GENERATOR_SYSTEM_PROMPT_CONTENT (假设config中直接定义或加载)
# 在rag_pipeline.py中，GENERATOR_SYSTEM_PROMPT_CONTENT 是在模块级别加载的
# 我们可以在这里直接引用它，或者确保config.py中的setup_logging()被调用
try:
    if config.GENERATOR_SYSTEM_PROMPT_FILE and \
            config.os.path.exists(config.GENERATOR_SYSTEM_PROMPT_FILE):  #
        with open(config.GENERATOR_SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:  #
            GENERATOR_SYSTEM_PROMPT_CONTENT = f.read()  #
    else:
        GENERATOR_SYSTEM_PROMPT_CONTENT = "你是一个耐心、友好、乐于助人的AI助手。"  #
except Exception as e:
    logger.error(f"加载生成器系统提示时出错: {e}")  #
    GENERATOR_SYSTEM_PROMPT_CONTENT = "你是一个耐心、友好、乐于助人的AI助手。"  #


def load_resources():
    """加载应用所需的全局资源，例如知识库。"""
    global kb_instance
    if kb_instance is None:
        logger.info("开始加载知识库实例...")
        try:
            kb_instance = KnowledgeBase()  #
            logger.info("知识库实例加载成功。")
        except Exception as e:
            logger.error(f"知识库实例化失败: {e}", exc_info=True)
            # 向上抛出异常或采取其他错误处理措施
            raise RuntimeError(f"无法加载KnowledgeBase: {e}") from e


async def main_chat_callback(
        user_input: str,
        # 将参数重命名，以区分 gr.State 对象和它当前的值
        current_openai_history_list: List[Dict[str, str]]
):
    if kb_instance is None:
        logger.error("知识库未加载，无法处理请求。")
        error_message = "错误：知识库未成功加载，请检查应用日志。"

        # 构建Gradio聊天显示格式
        gradio_chat_display_on_error = []
        for msg_dict in current_openai_history_list:  # 使用重命名的参数
            if msg_dict["role"] == "user":
                gradio_chat_display_on_error.append((msg_dict["content"], None))
            elif msg_dict["role"] == "assistant":
                gradio_chat_display_on_error.append((None, msg_dict["content"]))

        yield {
            ui_elements["chatbot_display"]: gradio_chat_display_on_error + [(None, error_message)],
            ui_elements["user_input_textbox"]: user_input,
            ui_elements["references_display"]: gr.skip(),
            ui_elements["references_column"]: gr.skip(),
            ui_elements["processing_view_column"]: gr.skip(),
            ui_elements["chatting_view_column"]: gr.update(visible=True),
            ui_elements["spinner_display"]: gr.skip(),
            ui_elements["status_text_display"]: gr.skip(),
            # 关键：这里的 openai_chat_history_state 指的是 build_gradio_app 中定义的 gr.State 对象
            # 而它的值是 current_openai_history_list (从参数传来，代表当前状态)
            openai_chat_history_state: current_openai_history_list
        }
        return

    # 直接修改传入的列表 current_openai_history_list，Gradio的gr.State会跟踪这个修改
    current_openai_history_list.append({"role": "user", "content": user_input})

    gradio_chat_display_after_user = []
    for msg_dict in current_openai_history_list:
        if msg_dict["role"] == "user":
            gradio_chat_display_after_user.append((msg_dict["content"], None))
        elif msg_dict["role"] == "assistant":
            gradio_chat_display_after_user.append((None, msg_dict["content"]))

    yield {
        ui_elements["chatbot_display"]: gradio_chat_display_after_user,
        ui_elements["user_input_textbox"]: "",
        ui_elements["references_display"]: format_references_for_display([]),
        ui_elements["references_column"]: gr.update(visible=True),
        ui_elements["processing_view_column"]: gr.update(visible=False),
        ui_elements["chatting_view_column"]: gr.update(visible=True),
        ui_elements["spinner_display"]: gr.update(visible=False),
        ui_elements["status_text_display"]: gr.update(value=""),
        openai_chat_history_state: current_openai_history_list  # 使用gr.State对象为键，更新后的列表为值
    }

    temp_ai_parts_for_this_turn = {"reasoning": "", "content": ""}

    # 注意：传递给 execute_rag_flow 的是 current_openai_history_list[:-1]
    # 这是因为 current_openai_history_list 此时已包含当前用户输入，而重写通常基于之前的历史
    async for event in execute_rag_flow(
            user_query=user_input,
            chat_history_openai=current_openai_history_list[:-1],  #
            kb_instance=kb_instance
    ):
        # 为 event_handler 构建当前Gradio聊天记录 (这里只是一个快照)
        # event_handler 将返回对 chatbot_display 的具体更新指令
        current_gradio_snapshot_for_handler = []
        for msg_dict in current_openai_history_list:
            if msg_dict["role"] == "user":
                current_gradio_snapshot_for_handler.append((msg_dict["content"], None))
            elif msg_dict["role"] == "assistant":
                # 如果是当前正在生成的AI消息，其内容由 temp_ai_parts_for_this_turn 决定
                # 简单的处理是直接用 msg_dict["content"]，event_handler会处理流式部分
                current_gradio_snapshot_for_handler.append((None, msg_dict["content"]))

        if event.get("type") == "generation_start":
            # 确保 current_openai_history_list 中为AI的回复添加了空壳
            if not (current_openai_history_list and current_openai_history_list[-1]["role"] == "assistant" and
                    current_openai_history_list[-1]["content"] == ""):
                current_openai_history_list.append({"role": "assistant", "content": ""})
            temp_ai_parts_for_this_turn["reasoning"] = "### 思考过程\n"
            temp_ai_parts_for_this_turn["content"] = ""

            # 更新 current_gradio_snapshot_for_handler 以包含这个新的空壳
            # 这会在 update_ui_for_event 中被用来更新 chatbot 的最后一条消息
            if len(current_gradio_snapshot_for_handler) < len(current_openai_history_list):
                current_gradio_snapshot_for_handler.append((None, "AI正在生成..."))

        ui_updates_from_handler = update_ui_for_event(
            event,
            ui_elements,
            current_gradio_snapshot_for_handler,  # 传递一个“当前所见”的chatbot历史
            temp_ai_parts_for_this_turn
        )

        full_yield_dict = {
            ui_elements["chatbot_display"]: gr.skip(),
            ui_elements["user_input_textbox"]: gr.skip(),
            ui_elements["references_display"]: gr.skip(),
            ui_elements["references_column"]: gr.skip(), # 新增
            ui_elements["processing_view_column"]: gr.skip(),
            ui_elements["chatting_view_column"]: gr.skip(),
            ui_elements["spinner_display"]: gr.skip(),
            ui_elements["status_text_display"]: gr.skip(),
            # 使用 build_gradio_app 中定义的 openai_chat_history_state 对象作为键
            openai_chat_history_state: current_openai_history_list  # 值为当前（可能已更新的）列表
        }
        full_yield_dict.update(ui_updates_from_handler)

        # 确保 state key 的值是最新的（如果 handler 也尝试更新它，虽然不应该）
        full_yield_dict[openai_chat_history_state] = current_openai_history_list

        if event.get("type") == "final_answer_complete" or \
                (event.get("type") == "pipeline_end" and event.get("reason") == "flow_completed"):
            final_reasoning = temp_ai_parts_for_this_turn.get("reasoning", "### 思考过程\n*无*")
            final_content = temp_ai_parts_for_this_turn.get("content", "*无*")
            if "### 回复" not in final_reasoning and final_content:
                final_reasoning = (final_reasoning or "") + "\n\n### 回复\n"
            final_ai_message_str = f"{final_reasoning}{final_content}".strip()

            if current_openai_history_list and current_openai_history_list[-1]["role"] == "assistant":
                current_openai_history_list[-1]["content"] = final_ai_message_str  # 更新内部状态列表

            # 确保 chatbot 的最终显示也使用这个最终版本
            final_gradio_chat_display_list = []
            for msg_dict in current_openai_history_list:  # 从更新后的内部状态列表构建
                if msg_dict["role"] == "user":
                    final_gradio_chat_display_list.append((msg_dict["content"], None))
                elif msg_dict["role"] == "assistant":
                    final_gradio_chat_display_list.append((None, msg_dict["content"]))

            # 如果 event_handler 已经更新了 chatbot_display, full_yield_dict 中会有它
            # 否则，我们在这里设置最终的 chatbot 显示
            # if ui_elements["chatbot_display"] not in ui_updates_from_handler or \
            #         (event.get("type") == "final_answer_complete"):  # 确保final_answer_complete会覆盖
            #     full_yield_dict[ui_elements["chatbot_display"]] = final_gradio_chat_display_list

            full_yield_dict[openai_chat_history_state] = current_openai_history_list  # 再次确保state是最新的

        yield full_yield_dict


# --- Gradio Blocks定义 ---
ui_elements: Dict[str, Any] = {}
# 在 build_gradio_app 中定义的 openai_chat_history_state 将是 gr.State 对象
# 我们需要确保在 main_chat_callback 中引用的是这个对象，而不是其参数值（已通过重命名解决）
openai_chat_history_state: Optional[gr.State] = None  # 声明以便在回调中通过闭包访问


def build_gradio_app():
    global ui_elements, openai_chat_history_state  # 允许此函数修改全局的ui_elements和openai_chat_history_state对象引用

    with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky),
                   css="#spinner-markdown { font-size: 2em; text-align: center; }") as demo:

        current_ui_elements = create_ui() # create_ui返回的是组件字典
        globals()['ui_elements'] = current_ui_elements # 更新全局的ui_elements

        # 定义Gradio状态变量
        # 将 openai_chat_history_state 赋值给模块级的变量，以便 main_chat_callback 可以通过闭包访问它
        openai_chat_history_state_obj_for_closure = gr.State([])
        # 更新模块级变量的引用
        globals()['openai_chat_history_state'] = openai_chat_history_state_obj_for_closure

        outputs_list = [
            current_ui_elements["chatbot_display"],
            current_ui_elements["user_input_textbox"],
            current_ui_elements["references_display"],    # Markdown 内容
            current_ui_elements["references_column"],     # Column 容器 (用于可见性) <-- 新增
            current_ui_elements["processing_view_column"],
            current_ui_elements["chatting_view_column"],
            current_ui_elements["spinner_display"],
            current_ui_elements["status_text_display"],
            openai_chat_history_state_obj_for_closure
        ] # 现在是9个输出

        ui_elements["send_button"].click(
            fn=main_chat_callback,
            inputs=[
                ui_elements["user_input_textbox"],
                openai_chat_history_state_obj_for_closure,  # gr.State 对象作为输入
            ],
            outputs=outputs_list
        )

        def clear_chat_history_fn():  # 重命名以避免与外部可能的clear_chat_history冲突
            # 返回一个字典，键是组件，值是它们的更新
            # openai_chat_history_state_obj_for_closure 是闭包捕获的State对象
            return {
                current_ui_elements["chatbot_display"]: None,
                current_ui_elements["user_input_textbox"]: "",
                current_ui_elements["references_display"]: format_references_for_display([]),
                current_ui_elements["references_column"]: gr.update(visible=True), # 重置为可见
                current_ui_elements["processing_view_column"]: gr.update(visible=False),
                current_ui_elements["chatting_view_column"]: gr.update(visible=True),
                current_ui_elements["spinner_display"]: gr.update(visible=False),
                current_ui_elements["status_text_display"]: gr.update(value=""),
                openai_chat_history_state_obj_for_closure: []
            }

        clear_button = gr.Button("清除对话 (Clear Chat)")
        clear_button.click(
            fn=clear_chat_history_fn,
            inputs=None,
            outputs=outputs_list  # 清除函数也需要更新所有相关输出
        )

    return demo


if __name__ == "__main__":
    # 配置日志 (从config.py)
    config.setup_logging()

    # 加载资源
    try:
        load_resources()
        logger.info("应用资源加载完成。")

        # 构建并启动Gradio应用
        app_interface = build_gradio_app()
        logger.info("Gradio 应用构建完成，准备启动...")
        app_interface.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)  # 使用queue()以更好地处理流式输出
        logger.info("Gradio 应用已启动。")

    except RuntimeError as e:
        logger.critical(f"应用启动失败: {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"发生未知错误导致应用无法启动: {e}", exc_info=True)