import gradio as gr
from typing import List, Dict, Any, Optional, Tuple

def format_references_for_display(chunks_data: List[Dict[str, Any]]) -> str:
    """
    将从RAG流程中获取的引用块数据格式化为Markdown字符串，用于显示。
    """
    if not chunks_data:
        return "### 引用文献\n\n*本次回答未直接引用知识库内容*"

    markdown_output = "### 引用文献\n"
    for idx, chunk in enumerate(chunks_data):
        doc_name = chunk.get('doc_name', '未知文档')
        page_number = chunk.get('page_number', '未知页码')
        chunk_id = chunk.get('chunk_id', '未知ID')
        score = chunk.get('rerank_score', 0.0)
        text_preview = chunk.get('text_preview', '无预览')  # 来自 rag_pipeline.py 中的 context_for_display_list_reranked
        retrieved_from_paths = chunk.get('retrieved_from_paths', {})
        paths_str = ", ".join(retrieved_from_paths.keys()) if retrieved_from_paths else "未知来源"

        markdown_output += f"**{idx + 1}. 文档:** {doc_name} (页码: {page_number}, 块ID: {chunk_id})\n"
        markdown_output += f"   - **相关性得分:** {score:.4f}\n"
        markdown_output += f"   - **召回路径:** {paths_str}\n"
        markdown_output += f"   - **内容预览:** {text_preview}\n\n"
    return markdown_output


def update_ui_for_event(
        event: Dict[str, Any],
        ui_elements: Dict[str, Any],
        current_chatbot_history: List[Tuple[Optional[str], Optional[str]]],
        current_ai_message_parts: Dict[str, str]
) -> Dict[Any, Any]:
    """
    根据RAG流程产生的事件，生成Gradio UI组件的更新字典。

    参数:
        event: 从 execute_rag_flow yield 的事件字典。
        ui_elements: 包含所有相关Gradio UI组件实例的字典。
        current_chatbot_history: 当前Gradio chatbot组件的聊天历史。
        current_ai_message_parts: 一个字典，包含当前AI回复的 "reasoning" 和 "content" 部分。

    返回:
        一个字典，键是Gradio组件实例，值是它们的新属性或内容。
        例如: {ui_elements["status_text_display"]: gr.update(value="新的状态...")}
    """
    event_type = event.get("type")
    updates = {}

    # --- 视图控制 ---
    if event_type == "pipeline_start":
        updates[ui_elements["processing_view_column"]] = gr.update(visible=True)
        updates[ui_elements["chatting_view_column"]] = gr.update(visible=False)  # 在处理时隐藏主聊天输入区和聊天记录
        updates[ui_elements["references_column"]] = gr.update(visible=False)  # 隐藏引用侧边栏
        updates[ui_elements["status_text_display"]] = gr.update(value=event.get("message", "流程启动中..."))
        updates[ui_elements["spinner_display"]] = gr.update(visible=True)  # 确保spinner可见

    elif event_type == "generation_start":  # 切换到聊天和引用视图
        updates[ui_elements["processing_view_column"]] = gr.update(visible=False)
        updates[ui_elements["chatting_view_column"]] = gr.update(visible=True)
        updates[ui_elements["references_column"]] = gr.update(visible=True)
        updates[ui_elements["spinner_display"]] = gr.update(visible=False)

        # 为AI的回复准备一个初始的空消息或提示，稍后用delta填充
        # 注意：实际的chatbot历史追加应该在app.py的主回调中处理，这里只负责生成更新内容
        # current_chatbot_history.append((None, "AI正在思考并生成回复...")) # 示例提示
        # updates[ui_elements["chatbot_display"]] = current_chatbot_history
        # 重置当前AI消息的组成部分
        current_ai_message_parts["reasoning"] = ""
        current_ai_message_parts["content"] = ""
        # 添加思考过程的固定前缀
        current_ai_message_parts["reasoning"] = "### 思考过程\n"


    elif event_type == "pipeline_end":
        updates[ui_elements["processing_view_column"]] = gr.update(visible=False)
        updates[ui_elements["chatting_view_column"]] = gr.update(visible=True)  # 确保聊天视图可见
        updates[ui_elements["references_column"]] = gr.update(visible=True)  # 确保引用文献可见
        updates[ui_elements["spinner_display"]] = gr.update(visible=False)
        # 可以在这里根据 event.get("reason") 更新最终状态或chatbot消息

    # --- 内容更新 ---
    if event_type == "status":  # 其他非切换视图的status事件
        if ui_elements["processing_view_column"].visible:  # 仅当处理视图可见时更新状态文本
            updates[ui_elements["status_text_display"]] = gr.update(value=event.get("message", "..."))

    elif event_type == "rewritten_query_result":
        # 可选：如果需要在某个地方显示重写后的查询
        # rewritten_text = event.get("rewritten_text", "")
        # updates[ui_elements["some_debug_display"]] = gr.update(value=f"重写查询: {rewritten_text}")
        pass  # 暂时不显示

    elif event_type == "retrieved_chunks_preview":
        # 可选：如果需要预览召回的块
        pass  # 暂时不显示

    elif event_type == "reranked_context_for_display":
        chunks_for_display = event.get("chunks", [])
        formatted_references = format_references_for_display(chunks_for_display)
        updates[ui_elements["references_display"]] = gr.update(value=formatted_references)

    elif event_type == "llm_input_preview":
        # 可选：如果需要预览给LLM的输入
        pass  # 暂时不显示

    elif event_type == "reasoning_delta":
        delta_text = event.get("text", "")
        current_ai_message_parts["reasoning"] += delta_text
        # 构建完整的当前AI消息展示（思考+回复）
        ai_message_display = f"{current_ai_message_parts['reasoning']}"
        if current_ai_message_parts["content"] or "### 回复" in current_ai_message_parts[
            "reasoning"]:  # 如果回复已经开始或reasoning中已包含回复标记
            ai_message_display += f"\n\n### 回复\n{current_ai_message_parts['content']}"

        # 更新聊天记录的最后一条（AI的回复）
        if current_chatbot_history and current_chatbot_history[-1][0] is None:  # 确保是AI的回复
            updated_history = current_chatbot_history[:-1] + [(None, ai_message_display)]
            updates[ui_elements["chatbot_display"]] = updated_history
        else:  # 如果没有AI的空壳消息，就追加一个新的（这通常由app.py在generation_start时处理）
            updates[ui_elements["chatbot_display"]] = current_chatbot_history + [(None, ai_message_display)]


    elif event_type == "content_delta":
        delta_text = event.get("text", "")
        # 确保 "### 回复" 标题只添加一次
        if not current_ai_message_parts["content"] and "### 回复" not in current_ai_message_parts["reasoning"]:
            # 如果思考过程部分没有 "### 回复" 标记，并且内容部分为空，说明回复刚开始
            if not current_ai_message_parts["reasoning"].endswith("\n\n### 回复\n"):  # 避免重复添加
                current_ai_message_parts["reasoning"] += "\n\n### 回复\n"  # 将回复标记添加到reasoning的末尾或一个独立的部分

        current_ai_message_parts["content"] += delta_text
        # 构建完整的当前AI消息展示（思考+回复）
        ai_message_display = f"{current_ai_message_parts['reasoning']}{current_ai_message_parts['content']}"

        # 更新聊天记录的最后一条（AI的回复）
        if current_chatbot_history and current_chatbot_history[-1][0] is None:  # 确保是AI的回复
            updated_history = current_chatbot_history[:-1] + [(None, ai_message_display)]
            updates[ui_elements["chatbot_display"]] = updated_history
        else:
            updates[ui_elements["chatbot_display"]] = current_chatbot_history + [(None, ai_message_display)]


    elif event_type == "final_answer_complete":
        # 确保最终的完整消息（包含思考和回复）被正确设置到chatbot中
        # 这通常意味着 current_ai_message_parts 已经是最终形态
        final_reasoning = current_ai_message_parts.get("reasoning", "### 思考过程\n*无*")
        final_content = current_ai_message_parts.get("content", "*无*")
        if "### 回复" not in final_reasoning and final_content:  # 确保回复部分有标题
            final_reasoning += "\n\n### 回复\n"

        final_ai_message_display = f"{final_reasoning}{final_content}"

        if current_chatbot_history and current_chatbot_history[-1][0] is None:
            updated_history = current_chatbot_history[:-1] + [(None, final_ai_message_display)]
            updates[ui_elements["chatbot_display"]] = updated_history
        else:  # 应急处理，理论上此时应有AI消息占位
            updates[ui_elements["chatbot_display"]] = current_chatbot_history + [(None, final_ai_message_display)]

    elif event_type == "error":
        error_message = event.get("message", "发生未知错误")
        # 在chatbot中显示错误，或者使用专门的错误提示区域
        # 为简化，暂且追加到chatbot
        error_display_message = f"**错误:** {error_message}"
        updates[ui_elements["chatbot_display"]] = current_chatbot_history + [(None, error_display_message)]
        # 也可以考虑重新显示处理视图并指示错误
        updates[ui_elements["processing_view_column"]] = gr.update(visible=False)  # 确保错误时隐藏处理动画
        updates[ui_elements["chatting_view_column"]] = gr.update(visible=True)
        updates[ui_elements["spinner_display"]] = gr.update(visible=False)

    return updates

