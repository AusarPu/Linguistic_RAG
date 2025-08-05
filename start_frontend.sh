#!/bin/bash

echo ">>> 启动 RAG 系统前端服务..."

# --- 配置 --- 
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# 设置 PYTHONPATH 以确保 Python 脚本能正确导入模块
if [[ -z "$PYTHONPATH" ]]; then
    export PYTHONPATH="$PROJECT_ROOT"
else
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
fi

PYTHON_CMD="python3" # 或你的 python 命令
GRADIO_SCRIPT="$PROJECT_ROOT/Gradio_UI/app.py"

# --- 检查前端应用文件是否存在 ---
if [ ! -f "$GRADIO_SCRIPT" ]; then
    echo "错误: Gradio 应用脚本未找到: $GRADIO_SCRIPT" >&2
    exit 1
fi

# --- 启动 Gradio Web UI (前台运行) ---
echo ">>> 启动 Gradio Web UI (前台运行)..."
cd "$PROJECT_ROOT" || { echo "错误: 无法切换到项目根目录 $PROJECT_ROOT"; exit 1; }

echo "    从 $(pwd) 启动 Gradio 应用: $GRADIO_SCRIPT"
$PYTHON_CMD "$GRADIO_SCRIPT"

echo ">>> Gradio Web UI 已退出。"