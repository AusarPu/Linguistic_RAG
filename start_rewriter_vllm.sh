#!/bin/bash

# --- 读取配置 ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( dirname "$SCRIPT_DIR" )
PYTHON_CMD="python3"
CONFIG_MODULE="script.config"

read_config() {
    local var_name=$1
    local value=$($PYTHON_CMD -c "import sys; sys.path.insert(0, '$PROJECT_ROOT'); from $CONFIG_MODULE import $var_name; print($var_name)" 2>/dev/null)
    if [ -z "$value" ]; then
        echo "Error: Could not read $var_name from $CONFIG_MODULE" >&2
        exit 1
    fi
    echo "$value"
}

echo ">>> 读取重写器配置从 $CONFIG_MODULE..."
BASE_MODEL_PATH=$(read_config VLLM_BASE_MODEL_LOCAL_PATH)
LORA_PATH=$(read_config VLLM_REWRITER_LORA_LOCAL_PATH)
LORA_NAME=$(read_config REWRITER_LORA_NAME)
PORT=$(read_config VLLM_REWRITER_PORT)
GPU_ID=$(read_config VLLM_REWRITER_GPU_ID)
GPU_MEM_UTILIZATION=$(read_config VLLM_REWRITER_MEM_UTILIZATION)
MAX_LORA_RANK=$(read_config VLLM_MAX_LORA_RANK)
echo "    配置读取完成。"

# --- 准备 vLLM 启动参数 ---
MEM_ARG=""
if [ ! -z "$GPU_MEM_UTILIZATION" ] && [ "$GPU_MEM_UTILIZATION" != "None" ]; then
  MEM_ARG="--gpu-memory-utilization $GPU_MEM_UTILIZATION"
fi

# --- 启动 vLLM ---
echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 启动 vLLM 服务器 (重写器)..."
echo "    基础模型: $BASE_MODEL_PATH"
echo "    LoRA 路径: $LORA_PATH"
echo "    LoRA 名称: $LORA_NAME"
echo "    端口: $PORT"
echo "    分配 GPU: $GPU_ID"
echo "    显存限制: ${GPU_MEM_UTILIZATION:-默认}"
echo "    最大 LoRA Rank: $MAX_LORA_RANK"

CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_CMD -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL_PATH" \
    --port $PORT \
    --trust-remote-code \
    --enable-lora \
    --lora-modules ${LORA_NAME}="$LORA_PATH" \
    --max-loras 1 \
    --max-lora-rank $MAX_LORA_RANK \
    $MEM_ARG \
    --disable-log-requests #\
    --max-model-len 100000
    #--log-config=logging.yaml

echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 重写器 vLLM 服务已退出。"