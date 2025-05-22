import gradio as gr

def _create_processing_display() -> dict:
    """
    创建“处理中”视图的组件。
    这些组件初始时是隐藏的，在RAG流程工作时显示。
    """
    with gr.Column(visible=False, elem_id="processing-view") as processing_view_components:
        # 加载指示器的占位符。可以是图片或基于文本的动画。
        # 如果使用图片指示器 (例如，在 'assets' 文件夹中的 spinner.gif):
        # spinner = gr.HTML("<div style='display: flex; justify-content: center;'><img src='file/assets/spinner.gif' width='50' height='50'></div>")
        # 如果使用简单的文本指示器:
        spinner = gr.Markdown("⏳", elem_id="spinner-markdown") # 简单的表情符号指示器
        status_text = gr.Textbox(
            label="当前阶段",
            interactive=False,
            lines=1,
            max_lines=1, # 保持单行显示
            show_label=True,
            elem_id="status-text-textbox"
        )
    return {
        "processing_view_column": processing_view_components, # 包含所有处理视图组件的列容器
        "spinner_display": spinner,
        "status_text_display": status_text
    }

def _create_chatting_display() -> dict:
    """
    创建主聊天界面的组件。
    """
    with gr.Column(visible=True, elem_id="chatting-view") as chatting_view_components: # 主聊天视图初始可见
        chatbot_display = gr.Chatbot(
            label="AI助手",
            elem_id="chatbot",
            height=600, # 可根据需要调整高度
            show_label=True,
            bubble_full_width=False, # 聊天气泡不占满全部宽度，更像聊天样式
            render_markdown=True # 非常重要，用于渲染包含Markdown的“思考过程”和“回复”
        )
        with gr.Row():
            user_input_textbox = gr.Textbox(
                placeholder="请输入您的问题...",
                scale=4, #占据更多水平空间
                show_label=False,
                container=False # 移除默认容器样式，使在行内布局更简洁
            )
            send_button = gr.Button(
                "发送",
                scale=1,
                variant="primary" # 使按钮更突出
            )
    return {
        "chatting_view_column": chatting_view_components, # 包含所有聊天视图组件的列容器
        "chatbot_display": chatbot_display,
        "user_input_textbox": user_input_textbox,
        "send_button": send_button
    }

def _create_references_sidebar() -> dict:
    """
    创建用于显示引用文献/参考资料的侧边栏。
    """
    with gr.Column(min_width=300, # 确保侧边栏有合适的最小宽度
                   visible=True,    # 初始可见
                   elem_id="references-sidebar-column"
                  ) as references_column:
        references_display = gr.Markdown(
            value="### 引用文献\n\n*暂无引用*", # 初始内容
            elem_id="references-markdown"
        )
    return {
        "references_column": references_column, # 引用文献侧边栏的列容器
        "references_display": references_display
    }

def create_ui() -> dict:
    """
    主函数，用于创建并组装所有UI组件。
    返回一个包含所有已创建Gradio组件的字典。
    """
    all_components = {}

    with gr.Row(equal_height=False): # 主布局行
        with gr.Column(scale=2): # 左侧列，用于聊天和处理状态
            # “处理中”显示区域 (初始隐藏)
            processing_components = _create_processing_display()
            all_components.update(processing_components)

            # 聊天显示区域 (初始可见)
            chat_components = _create_chatting_display()
            all_components.update(chat_components)

        with gr.Column(scale=1): # 右侧列，用于引用文献
            reference_components = _create_references_sidebar()
            all_components.update(reference_components)

    return all_components
