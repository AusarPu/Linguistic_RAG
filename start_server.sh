#!/bin/bash

echo ">>> 启动 RAG 系统核心服务 (Rewriter, Embedding)..."

# --- 配置 --- 
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# 设置 PYTHONPATH 以确保 Python 脚本能正确导入模块
# 将项目根目录添加到PYTHONPATH的最前面，以优先加载项目内的模块
if [[ -z "$PYTHONPATH" ]]; then
    export PYTHONPATH="$PROJECT_ROOT"
else
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
fi

LOG_DIR="$PROJECT_ROOT/logs"
PYTHON_CMD="python3" # 或你的 python 命令

REWRITER_LOG="$LOG_DIR/vllm_rewriter.log"

EMBEDDING_LOG="$LOG_DIR/vllm_embedding.log"
PID_DIR="$PROJECT_ROOT/pids" # 定义 PID_DIR
REWRITER_PID_FILE="$PID_DIR/vllm_rewriter.pid"

EMBEDDING_PID_FILE="$PID_DIR/vllm_embedding.pid"

# --- 读取配置函数 ---
read_config() {
    local var_name=$1
    # 确保 PROJECT_ROOT 在 sys.path 中，以便导入 CONFIG_MODULE
    local value=$($PYTHON_CMD -c "import sys; sys.path.insert(0, '$PROJECT_ROOT'); from $CONFIG_MODULE import $var_name; print($var_name)" 2>/dev/null)
    if [ -z "$value" ]; then
        echo "错误: 无法从 $CONFIG_MODULE 读取 $var_name" >&2
        # 对于关键配置，可能需要退出或设置一个明确的无效标记
    fi
    echo "$value"
}

# --- 读取服务配置 ---
CONFIG_MODULE="script.config_rag" # 确保 CONFIG_MODULE 在此处定义
echo ">>> 读取服务配置从 $CONFIG_MODULE..."
# Rewriter 配置
REWRITER_BASE_MODEL_PATH=$(read_config VLLM_REWRITE_MODEL_LOCAL_PATH)
REWRITER_LORA_PATH=$(read_config VLLM_REWRITER_LORA_LOCAL_PATH)
REWRITER_LORA_NAME=$(read_config REWRITER_LORA_NAME)
REWRITER_PORT=$(read_config VLLM_REWRITER_PORT)
REWRITER_GPU_ID=$(read_config VLLM_REWRITER_GPU_ID)
REWRITER_GPU_MEM_UTILIZATION=$(read_config VLLM_REWRITER_MEM_UTILIZATION)
REWRITER_MAX_LORA_RANK=$(read_config VLLM_MAX_LORA_RANK)
REWRITER_TENSOR_PARALLEL_SIZE=$(read_config VLLM_REWRITER_TENSOR_PARALLEL_SIZE)
if [ -z "$REWRITER_TENSOR_PARALLEL_SIZE" ]; then REWRITER_TENSOR_PARALLEL_SIZE=2; fi # 默认值



# 读取 Embedding 配置
EMBEDDING_MODEL_PATH=$(read_config EMBEDDING_MODEL_PATH)
if [ -z "$EMBEDDING_MODEL_PATH" ]; then 
    echo "错误: EMBEDDING_MODEL_PATH 未在 $CONFIG_MODULE 中配置。" >&2;
fi

EMBEDDING_PORT=$(read_config VLLM_EMBEDDING_PORT)
if [ -z "$EMBEDDING_PORT" ]; then 
    echo "错误: VLLM_EMBEDDING_PORT 未在 $CONFIG_MODULE 中配置。" >&2;
fi

EMBEDDING_GPU_ID=$(read_config VLLM_EMBEDDING_GPU_ID)
if [ -z "$EMBEDDING_GPU_ID" ]; then 
    echo "警告: VLLM_EMBEDDING_GPU_ID 未在 $CONFIG_MODULE 中配置，将使用默认值: 0" >&2;
    EMBEDDING_GPU_ID="0"; 
fi

EMBEDDING_MEM_UTILIZATION=$(read_config VLLM_EMBEDDING_MEM_UTILIZATION)
if [ -z "$EMBEDDING_MEM_UTILIZATION" ]; then 
    echo "警告: VLLM_EMBEDDING_MEM_UTILIZATION 未在 $CONFIG_MODULE 中配置，将使用默认值: 0.3" >&2;
    EMBEDDING_MEM_UTILIZATION="0.3"; 
fi
echo "    配置读取完成。"

# --- 清理函数 ---
cleanup() {
    echo ">>> 收到退出信号，正在清理后台进程..."
    if [ -f "$REWRITER_PID_FILE" ]; then
        REWRITER_PID=$(cat "$REWRITER_PID_FILE")
        echo "    停止 Rewriter (PID: $REWRITER_PID)..."
        kill "$REWRITER_PID" &> /dev/null || echo "    Rewriter 进程 $REWRITER_PID 可能已停止。"
        rm -f "$REWRITER_PID_FILE"
    fi

    if [ -f "$EMBEDDING_PID_FILE" ]; then
        EMBEDDING_PID=$(cat "$EMBEDDING_PID_FILE")
        echo "    停止 Embedding (PID: $EMBEDDING_PID)..."
        kill "$EMBEDDING_PID" &> /dev/null || echo "    Embedding 进程 $EMBEDDING_PID 可能已停止。"
        rm -f "$EMBEDDING_PID_FILE"
    fi
    echo ">>> 清理完成。"
    exit 0
}

# --- 注册信号处理 ---
trap cleanup INT TERM EXIT

# --- 创建日志和PID目录 ---
mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"

# --- 启动 Rewriter vLLM 服务 ---
if [ -z "$REWRITER_BASE_MODEL_PATH" ] || [ -z "$REWRITER_PORT" ]; then
    echo "错误: Rewriter 服务配置不完整 (模型路径或端口缺失)，跳过启动。" >&2
else
    echo ">>> 正在后台启动 Rewriter vLLM 服务..."
    echo "    基础模型: $REWRITER_BASE_MODEL_PATH"
    echo "    端口: $REWRITER_PORT"
    echo "    分配 GPU: ${REWRITER_GPU_ID:-默认所有可见GPU}"
    echo "    显存限制: ${REWRITER_GPU_MEM_UTILIZATION:-默认}"
    echo "    张量并行数: ${REWRITER_TENSOR_PARALLEL_SIZE}"

    # 使用 bash 数组来安全地构建命令
    REWRITER_CMD_ARRAY=(
        vllm serve "$REWRITER_BASE_MODEL_PATH"
        --port "$REWRITER_PORT"
        --trust-remote-code
        --disable-log-requests
        --max-model-len 40960
        --tensor-parallel-size "$REWRITER_TENSOR_PARALLEL_SIZE"
        --max_num_seqs 2048
        --quantization fp8
        --enable-reasoning --reasoning-parser deepseek_r1
        # --rope-scaling '{"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 32768}' \
    )

    # 有条件地添加内存参数
    if [ ! -z "$REWRITER_GPU_MEM_UTILIZATION" ] && [ "$REWRITER_GPU_MEM_UTILIZATION" != "None" ]; then
      REWRITER_CMD_ARRAY+=(--gpu-memory-utilization "$REWRITER_GPU_MEM_UTILIZATION")
    fi
    
    # 使用 nohup 和正确的变量展开来启动服务
    (export CUDA_VISIBLE_DEVICES=${REWRITER_GPU_ID}; nohup "${REWRITER_CMD_ARRAY[@]}" > "$REWRITER_LOG" 2>&1 & echo $! > "$REWRITER_PID_FILE")
    echo "    Rewriter 服务 PID: $(cat "$REWRITER_PID_FILE")，日志: $REWRITER_LOG"
    echo "    等待 Rewriter 服务启动 (约30-60秒)..."
    sleep 5
fi



# --- 启动 Embedding vLLM 服务 ---
if [ -z "$EMBEDDING_MODEL_PATH" ] || [ -z "$EMBEDDING_PORT" ]; then
    echo "错误: Embedding 服务配置不完整 (模型路径或端口缺失)，跳过启动。" >&2
else
    echo ">>> 正在后台启动 Embedding vLLM 服务..."
    echo "    模型路径: $EMBEDDING_MODEL_PATH"
    echo "    端口: $EMBEDDING_PORT"
    echo "    分配 GPU: ${EMBEDDING_GPU_ID:-默认所有可见GPU}"
    echo "    显存限制: ${EMBEDDING_MEM_UTILIZATION:-默认}"

    # 使用 bash 数组来安全地构建命令
    EMBEDDING_CMD_ARRAY=(
        vllm serve "$EMBEDDING_MODEL_PATH"
        --port "$EMBEDDING_PORT"
        --trust-remote-code
        --disable-log-requests
        --max-model-len 8192
        --max_num_seqs 1024
    )

    # 有条件地添加内存参数
    if [ ! -z "$EMBEDDING_MEM_UTILIZATION" ] && [ "$EMBEDDING_MEM_UTILIZATION" != "None" ]; then
      EMBEDDING_CMD_ARRAY+=(--gpu-memory-utilization "$EMBEDDING_MEM_UTILIZATION")
    fi
    
    # 使用 nohup 和正确的变量展开来启动服务
    (export CUDA_VISIBLE_DEVICES=${EMBEDDING_GPU_ID}; nohup "${EMBEDDING_CMD_ARRAY[@]}" > "$EMBEDDING_LOG" 2>&1 & echo $! > "$EMBEDDING_PID_FILE")
    echo "    Embedding 服务 PID: $(cat "$EMBEDDING_PID_FILE")，日志: $EMBEDDING_LOG"
    echo "    等待 Embedding 服务启动 (约30-60秒)..."
    sleep 5
fi

echo ">>> vLLM 服务启动完成。"
echo ">>> 如需启动前端，请运行: ./start_frontend.sh"

# 保持脚本运行，等待信号
echo ">>> 按 Ctrl+C 停止所有服务..."
while true; do
    sleep 1
done