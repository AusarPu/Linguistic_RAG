# web_ui/sidebar.py
import gradio as gr

def create_sidebar():
    """
    创建侧边栏组件：重写查询显示、上下文显示、清除按钮。
    返回需要被外部引用的组件实例。
    """
    with gr.Column(scale=2): # 保持原有的列缩放比例
        with gr.Accordion("查看重构后的查询", open=False):
            rewritten_query_display = gr.Textbox(
                label="重构查询列表",
                lines=5,
                interactive=False,
                value="重构后的查询将显示在这里..."
            )
        with gr.Accordion("查看参考知识库片段", open=True):
            context_display = gr.Markdown(
                value="上下文将显示在这里..."
            )
        clear_button = gr.Button("清除对话历史")

    # 返回需要被外部引用的组件
    return rewritten_query_display, context_display, clear_button