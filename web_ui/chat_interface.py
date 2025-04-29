# web_ui/chat_interface.py
import gradio as gr

def create_chat_interface():
    """
    创建聊天界面的核心组件：聊天窗口、输入框、发送按钮。
    返回这些组件的实例，以便在主布局中使用和绑定事件。
    """
    with gr.Column(scale=3): # 保持原有的列缩放比例
        chatbot = gr.Chatbot(
            label="聊天窗口",
            bubble_full_width=False,
            height=600,
            render_markdown=True,
        )
        with gr.Row():
             msg_input = gr.Textbox(
                 label="输入消息",
                 placeholder="请输入你的问题...",
                 scale=7,
                 autofocus=True,
                 lines=1
             )
             send_button = gr.Button(
                 "发送",
                 variant="primary",
                 scale=1,
                 min_width=0
             )
    # 返回需要被外部引用的组件
    return chatbot, msg_input, send_button