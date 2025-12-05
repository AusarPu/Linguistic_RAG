import os
import gradio as gr
import asyncio
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
import traceback
import sys

# 使用项目内的统一日志策略，仅输出重试与警告
sys.path.append('/home/pushihao/RAG')
from script import config_rag as config
config.setup_logging()

# 从项目中导入
sys.path.append('/home/pushihao/RAG')
from script.knowledge_base import KnowledgeBase
from script.rag_pipeline import execute_rag_flow
# config 已在上面导入并初始化日志

logger = logging.getLogger(__name__)
kb_instance: Optional[KnowledgeBase] = None

# 全局知识库实例

def load_resources():
    """加载应用所需的全局资源，例如知识库。"""
    global kb_instance
    logger.info("开始加载知识库实例...")
    if kb_instance is None:
        try:
            kb_instance = KnowledgeBase()
            logger.info("知识库实例加载成功。")
        except Exception as e:
            logger.error(f"知识库实例化失败: {e}", exc_info=True)
            # 不阻断 UI 启动；保持 kb_instance=None，后续交互时提示用户
            logger.warning("知识库加载失败，应用仍将启动但无法检索。")
    else:
        logger.info("知识库实例已存在，跳过加载。")

def format_markdown_collapsible(title: str, content: str, is_open: bool = False) -> str:
    """格式化为可折叠的markdown内容"""
    if not content.strip():
        return ""
    open_attr = " open" if is_open else ""
    return f"<details{open_attr}>\n<summary><strong>{title}</strong></summary>\n\n{content}\n\n</details>\n\n"

def format_stage_info(stage: str, message: str) -> str:
    """格式化阶段信息"""
    stage_map = {
        "pipeline_start": "🚀 流程启动",
        "query_rewriting": "✏️ 查询重构", 
        "retrieval_start": "🔍 知识检索",
        "usefulness_judging": "🎯 相关性判断",
        "generation_start": "💭 答案生成",
        "generation_complete": "✅ 生成完成"
    }
    icon_stage = stage_map.get(stage, f"📋 {stage}")
    return f"**{icon_stage}**: {message}"

def format_retrieved_chunks(chunks: List[Dict]) -> str:
    """格式化检索到的知识块 - 完整显示内容"""
    if not chunks:
        return "暂无检索结果"
    
    formatted_chunks = []
    for i, chunk in enumerate(chunks, 1):
        # 完整显示知识块内容，不截断
        chunk_content = chunk.get('text', '')
        soft_keep_flag = chunk.get('soft_kept', False)
        soft_keep_badge = " （软保留）" if soft_keep_flag else ""
        chunk_info = f"""**片段 {i}**
- **文档**: {chunk.get('doc_name', '未知')}
- **作者**: {chunk.get('author', '未知')}
- **页码**: {chunk.get('page_number', '未知')}
- **来源路径**: {', '.join(chunk.get('from_paths', []))}
- **块ID**: {chunk.get('chunk_id', '未知')}
 - **标记**: {'软保留' if soft_keep_flag else '判定为有用'}

**完整内容**:
```
{chunk_content}
```

---
"""
        formatted_chunks.append(chunk_info)
    
    return "\n".join(formatted_chunks)

def format_chat_message_with_thinking(thinking_content: str, response_content: str) -> str:
    """格式化聊天消息，包含思考过程和回答内容"""
    result = ""
    
    # 添加思考内容（可折叠）
    if thinking_content.strip():
        result += format_markdown_collapsible("💭 思考过程", thinking_content, is_open=False)
    
    # 添加回答内容（正常显示）
    if response_content.strip():
        result += f"**回答内容：**\n\n{response_content}"
    
    return result

async def chat_with_rag(
    user_input: str,
    chat_history: List[Dict[str, str]],
    openai_history: List[Dict[str, str]]
):
    """RAG聊天处理函数"""
    
    logger.info(f"开始处理用户输入: {user_input[:50]}...")
    
    if not user_input.strip():
        logger.warning("用户输入为空，返回当前状态")
        yield chat_history, openai_history, "", "暂无检索结果"
        return
    
    if kb_instance is None:
        logger.error("知识库未加载")
        error_msg = "错误：知识库未成功加载，请检查应用日志。"
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": error_msg})
        yield chat_history, openai_history, "❌ 错误", "暂无检索结果"
        return
    
    # 初始化状态变量
    current_stage = ""
    current_retrieved_chunks = []
    current_thinking_content = ""
    current_response_content = ""
    knowledge_ready = False  # 标记知识库内容是否准备好显示
    
    # 添加用户消息到历史
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": ""})
    openai_history.append({"role": "user", "content": user_input})
    
    # 立即返回初始状态，隐藏知识库内容
    yield chat_history, openai_history, "🔍 开始处理...", "🔍 正在检索知识库..."
    
    try:
        # 开始执行RAG流程
        event_count = 0
        async for event in execute_rag_flow(
            user_query=user_input,
            chat_history_openai=openai_history[:-1],
            kb_instance=kb_instance
        ):
            event_count += 1
            event_type = event.get("type")
            logger.debug(f"收到事件: {event_type}")
            
            # 处理状态更新
            if event_type == "status":
                stage = event.get("stage", "")
                message = event.get("message", "")
                current_stage = format_stage_info(stage, message)
                logger.info(f"状态更新: {current_stage}")
                
                # 只有知识库准备好后才显示内容，否则显示等待信息
                knowledge_display_content = format_retrieved_chunks(current_retrieved_chunks) if knowledge_ready else "🔍 正在检索知识库..."
                
                # 更新聊天历史中的助手回复
                if chat_history and chat_history[-1].get("role") == "assistant":
                    chat_history[-1]["content"] = format_chat_message_with_thinking(current_thinking_content, current_response_content)
                
                yield chat_history, openai_history, current_stage, knowledge_display_content
            
            # 处理检索结果
            elif event_type == "useful_chunks_preview":
                chunks = event.get("preview", [])
                current_retrieved_chunks = chunks
                knowledge_ready = True  # 标记知识库内容准备好
                logger.info(f"检索结果更新: 获得 {len(chunks)} 个知识块")
                # 如果后端有软保留摘要事件，追加到阶段信息中
                soft_kept_count = sum(1 for c in chunks if c.get("soft_kept"))
                if soft_kept_count:
                    current_stage = format_stage_info("usefulness_judging", f"已应用软保留，补充 {soft_kept_count} 个片段")
                
                # 更新聊天历史中的助手回复
                if chat_history and chat_history[-1].get("role") == "assistant":
                    chat_history[-1]["content"] = format_chat_message_with_thinking(current_thinking_content, current_response_content)
                
                yield chat_history, openai_history, current_stage, format_retrieved_chunks(current_retrieved_chunks)
            
            # 处理思考内容（流式）
            elif event_type == "reasoning_delta":
                reasoning_text = event.get("text", "")
                current_thinking_content += reasoning_text
                # 更新思考内容
                
                # 实时更新聊天历史中的助手回复
                if chat_history and chat_history[-1].get("role") == "assistant":
                    chat_history[-1]["content"] = format_chat_message_with_thinking(current_thinking_content, current_response_content)
                
                knowledge_display_content = format_retrieved_chunks(current_retrieved_chunks) if knowledge_ready else "🔍 正在检索知识库..."
                yield chat_history, openai_history, current_stage, knowledge_display_content
            
            # 处理回答内容（流式）
            elif event_type == "content_delta":
                content_text = event.get("text", "")
                current_response_content += content_text
                # 更新回答内容
                
                # 实时更新聊天历史中的助手回复
                if chat_history and chat_history[-1].get("role") == "assistant":
                    chat_history[-1]["content"] = format_chat_message_with_thinking(current_thinking_content, current_response_content)
                
                knowledge_display_content = format_retrieved_chunks(current_retrieved_chunks) if knowledge_ready else "🔍 正在检索知识库..."
                yield chat_history, openai_history, current_stage, knowledge_display_content

            # 处理软保留摘要事件
            elif event_type == "soft_keep_summary":
                added = event.get("added", 0)
                current_stage = format_stage_info("usefulness_judging", f"已应用软保留，补充 {added} 个片段")
                logger.info(f"软保留摘要: 补充 {added} 个片段")
                # 更新聊天历史中的助手回复
                if chat_history and chat_history[-1].get("role") == "assistant":
                    chat_history[-1]["content"] = format_chat_message_with_thinking(current_thinking_content, current_response_content)
                knowledge_display_content = format_retrieved_chunks(current_retrieved_chunks) if knowledge_ready else "🔍 正在检索知识库..."
                yield chat_history, openai_history, current_stage, knowledge_display_content
            
            # 处理流程结束
            elif event_type == "pipeline_end":
                logger.info("RAG流程结束")
                # 添加助手回复到openai历史
                if current_response_content.strip():
                    openai_history.append({"role": "assistant", "content": current_response_content.strip()})
                
                # 最终更新聊天历史
                if chat_history and chat_history[-1].get("role") == "assistant":
                    chat_history[-1]["content"] = format_chat_message_with_thinking(current_thinking_content, current_response_content)
                
                current_stage = "✅ 流程完成"
                knowledge_display_content = format_retrieved_chunks(current_retrieved_chunks) if knowledge_ready else "暂无检索结果"
                yield chat_history, openai_history, current_stage, knowledge_display_content
                break
            
            else:
                logger.warning(f"未处理的事件类型: {event_type}")
        
        logger.info("RAG流程完成")
    
    except Exception as e:
        error_msg = f"处理过程中发生错误: {str(e)}"
        logger.error(error_msg, exc_info=True)
        # 记录错误堆栈到日志文件
        
        if chat_history and chat_history[-1].get("role") == "assistant":
            chat_history[-1]["content"] = error_msg
        
        current_stage = "❌ 错误"
        knowledge_display_content = format_retrieved_chunks(current_retrieved_chunks) if knowledge_ready else "暂无检索结果"
        yield chat_history, openai_history, current_stage, knowledge_display_content

def create_ui():
    """创建Gradio界面"""
    with gr.Blocks(title="RAG智能问答系统", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🚀 RAG智能问答系统")
        gr.Markdown("基于检索增强生成的智能问答系统")
        
        # 状态变量
        chat_history_state = gr.State([])
        openai_history_state = gr.State([])
        
        with gr.Row():
            # 左栏：对话窗口 (70%)
            with gr.Column(scale=7):
                gr.Markdown("## 💬 对话窗口")
                
                # 聊天界面
                chatbot = gr.Chatbot(
                    label="对话历史",
                    height=500,
                    show_copy_button=True,
                    type="messages"
                )
                
                # 输入框
                with gr.Row():
                    user_input = gr.Textbox(
                        label="输入您的问题",
                        placeholder="请输入您想要询问的问题...",
                        scale=4
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)
            
            # 右栏：状态和知识库内容 (30%)
            with gr.Column(scale=3):
                gr.Markdown("## 📊 系统状态")
                
                # 当前阶段显示
                current_stage_display = gr.Markdown(
                    label="当前阶段",
                    value="等待用户输入..."
                )
                
                gr.Markdown("## 📚 检索到的知识")
                
                # 知识库内容展示
                knowledge_display = gr.Markdown(
                    label="知识库内容",
                    value="暂无检索结果",
                    height=400
                )
        

        
        # 控制按钮
        with gr.Row():
            clear_btn = gr.Button("清空对话", variant="secondary")
            reload_kb_btn = gr.Button("重新加载知识库", variant="secondary")
        
        # 事件处理函数
        def clear_chat():
            logger.info("用户点击清空对话")
            return [], [], "等待用户输入...", "暂无检索结果"
        
        def reload_knowledge_base():
            global kb_instance
            logger.info("用户点击重新加载知识库")
            try:
                kb_instance = None
                load_resources()
                return "知识库重新加载成功"
            except Exception as e:
                logger.error(f"重新加载知识库失败: {e}", exc_info=True)
                return f"知识库重新加载失败: {str(e)}"
        
        # 绑定事件
        submit_btn.click(
            fn=chat_with_rag,
            inputs=[user_input, chat_history_state, openai_history_state],
            outputs=[chatbot, openai_history_state, current_stage_display, knowledge_display]
        ).then(
            fn=lambda: "",
            inputs=[],
            outputs=[user_input]
        )
        
        user_input.submit(
            fn=chat_with_rag,
            inputs=[user_input, chat_history_state, openai_history_state],
            outputs=[chatbot, openai_history_state, current_stage_display, knowledge_display]
        ).then(
            fn=lambda: "",
            inputs=[],
            outputs=[user_input]
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chat_history_state, openai_history_state, current_stage_display, knowledge_display]
        ).then(
            fn=lambda: [],
            outputs=[chatbot]
        )
        
        reload_kb_btn.click(
            fn=reload_knowledge_base,
            outputs=[]
        )
    
    return app

if __name__ == "__main__":
    # 设置日志
    config.setup_logging()
    
    try:
        logger.info("开始启动RAG WebUI")
        load_resources()
        logger.info("应用资源加载完成。")
        
        app = create_ui()
        logger.info("Gradio应用构建完成，准备启动...")
        app.queue().launch(
            server_name="0.0.0.0", 
            server_port=8080, 
            share=False,
            show_error=True,
            debug=False
        )
        logger.info("Gradio应用已启动在端口 8080。")
    
    except RuntimeError as e:
        logger.critical(f"应用启动失败: {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"发生未知错误导致应用无法启动: {e}", exc_info=True)