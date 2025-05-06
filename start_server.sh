#!/bin/bash

echo ">>> 启动 RAG 系统所有服务..."

# --- 配置 ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT="$SCRIPT_DIR" # 假设 start_all.sh 也在项目根目录
LOG_DIR="$PROJECT_ROOT/logs" # 日志文件目录
PID_DIR="/tmp"             # PID 文件存放目录 (或项目内)

GENERATOR_SCRIPT="$PROJECT_ROOT/start_generator_vllm.sh"
REWRITER_SCRIPT="$PROJECT_ROOT/start_rewriter_vllm.sh"
GRADIO_SCRIPT="$PROJECT_ROOT/app_gradio.py"
PYTHON_CMD="python3" # 确保指向 venv 内的 python (如果需要激活)
# 如果脚本需要激活 venv:
# source "$PROJECT_ROOT/.venv/bin/activate"

GENERATOR_LOG="$LOG_DIR/vllm_generator.log"
REWRITER_LOG="$LOG_DIR/vllm_rewriter.log"
GENERATOR_PID_FILE="$PID_DIR/vllm_generator.pid"
REWRITER_PID_FILE="$PID_DIR/vllm_rewriter.pid"

# --- 清理函数 ---
cleanup() {
    echo ">>> 收到退出信号，正在清理后台进程..."
    if [ -f "$GENERATOR_PID_FILE" ]; then
        GENERATOR_PID=$(cat "$GENERATOR_PID_FILE")
        echo "    停止 Generator (PID: $GENERATOR_PID)..."
        kill "$GENERATOR_PID" || echo "    Generator 进程 $GENERATOR_PID 可能已停止。"
        rm -f "$GENERATOR_PID_FILE"
    fi
    if [ -f "$REWRITER_PID_FILE" ]; then
        REWRITER_PID=$(cat "$REWRITER_PID_FILE")
        echo "    停止 Rewriter (PID: $REWRITER_PID)..."
        kill "$REWRITER_PID" || echo "    Rewriter 进程 $REWRITER_PID 可能已停止。"
        rm -f "$REWRITER_PID_FILE"
    fi
    echo ">>> 清理完成。"
    exit 0
}

# --- 注册信号处理 ---
# 捕获 SIGINT (Ctrl+C), SIGTERM (kill), EXIT (脚本正常退出)
trap cleanup INT TERM EXIT

# --- 创建日志目录 ---
mkdir -p "$LOG_DIR"

# --- 启动 Generator vLLM ---
echo ">>> 正在后台启动 Generator vLLM 服务..."
# 将脚本输出重定向到日志文件，& 表示后台运行
nohup "$GENERATOR_SCRIPT" > "$GENERATOR_LOG" 2>&1 &
# 获取后台进程的 PID 并保存
echo $! > "$GENERATOR_PID_FILE"
echo "    Generator 服务 PID: $(cat "$GENERATOR_PID_FILE")，日志: $GENERATOR_LOG"

# --- 等待 Generator 启动 (简单 sleep 或检查端口) ---
echo "    等待 Generator 服务启动..." # 根据实际启动时间调整


# --- 启动 Rewriter vLLM ---
echo ">>> 正在后台启动 Rewriter vLLM 服务..."
nohup "$REWRITER_SCRIPT" > "$REWRITER_LOG" 2>&1 &
echo $! > "$REWRITER_PID_FILE"
echo "    Rewriter 服务 PID: $(cat "$REWRITER_PID_FILE")，日志: $REWRITER_LOG"

# --- 等待 Rewriter 启动 ---
echo "    等待 Rewriter 服务启动..."

# --- 启动 Gradio 应用 (前台运行) ---
echo ">>> 启动 Gradio 应用 (前台运行)..."
echo "切换到项目根目录: $PROJECT_ROOT" # 假设 $PROJECT_ROOT 已正确设置
cd "$PROJECT_ROOT" || exit 1 # 如果切换失败则退出

echo "从 $(pwd) 启动 Gradio 应用..."
# 确保使用 venv 中的 python
"$PROJECT_ROOT/.venv/bin/python" "$GRADIO_SCRIPT" # 使用绝对路径或相对路径
$PYTHON_CMD "$GRADIO_SCRIPT"

# --- Gradio 退出后 ---
# 由于设置了 trap EXIT，Gradio 正常退出或被中断时，cleanup 函数会被调用
echo ">>> Gradio 应用已退出。"