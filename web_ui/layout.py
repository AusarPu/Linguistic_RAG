# web_ui/layout.py (修改版)
import gradio as gr
from web_ui.chat_interface import create_chat_interface
from web_ui.sidebar import create_sidebar

def create_layout(respond_fn, clear_fn, load_fn, theme="soft"):
    """
    创建完整的 Gradio UI 布局并绑定事件。

    Args:
        respond_fn: 处理聊天消息提交的函数 (即原来的 respond)。
        clear_fn: 处理清除历史按钮点击的函数 (即原来的 clear_history)。
        load_fn: 处理应用加载时执行的函数 (即原来的 load_all_resources)。
        theme (str): Gradio 主题名称。

    Returns:
        gr.Blocks: 构建好的 Gradio Blocks 实例 (demo)。
    """
    theme_map = {
        "soft": gr.themes.Soft(),
        "glass": gr.themes.Glass(),
        "monochrome": gr.themes.Monochrome(),
        "default": gr.themes.Default(),
    }
    selected_theme = theme_map.get(theme.lower(), gr.themes.Default())

    with gr.Blocks(theme=selected_theme, title="RAG 对话系统 (vLLM)") as demo:
        gr.Markdown("# RAG 对话系统 (vLLM 后端)")

        with gr.Row():
            chatbot, msg_input, send_button = create_chat_interface()
            rewritten_query_display, context_display, clear_button = create_sidebar()

        # --- 事件绑定逻辑移到这里 ---
        submit_args = {
            "fn": respond_fn, # 使用传入的函数
            "inputs": [msg_input, chatbot],
            "outputs": [chatbot, context_display, rewritten_query_display],
        }
        msg_input.submit(**submit_args)
        send_button.click(**submit_args)

        # 清除按钮事件绑定
        # 注意：clear_fn 应该返回一个包含对应 outputs 组件默认值的元组/列表
        clear_button.click(
            clear_fn, # 使用传入的函数
            [],
            [chatbot, msg_input, context_display, rewritten_query_display], # 确保这里的组件列表和 clear_fn 的返回值对应
            queue=False
        )

        # App 加载事件绑定
        demo.load(load_fn, inputs=None, outputs=None) # 使用传入的函数
        # --- 事件绑定结束 ---

    # 现在只需要返回 demo 对象，因为所有交互都在内部处理了
    return demo