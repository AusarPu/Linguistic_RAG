#!/bin/bash

# --- 配置: 重写器服务 ---
BASE_MODEL="/home/pushihao/model-dir"
LORA_PATH="/home/pushihao/RAG/linguistic_ai"
LORA_NAME="rewriter_lora" # 与后续 Python 代码中使用的名称一致
PORT=8001
GPU_ID=1 # 分配给重写器的 GPU (如果只有一块 GPU, 请设为 0)
GPU_MEM_UTILIZATION="" # 例如: "0.45" (如果与生成器共享 GPU)
MAX_LORA_RANK=32 # 根据你的 LoRA adapter 调整

MEM_ARG=""
if [ ! -z "$GPU_MEM_UTILIZATION" ]; then
  MEM_ARG="--gpu-memory-utilization $GPU_MEM_UTILIZATION"
fi

echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 启动 vLLM 服务器 (重写器)..."
echo "    基础模型: $BASE_MODEL"
echo "    LoRA 路径: $LORA_PATH"
echo "    LoRA 名称: $LORA_NAME"
echo "    端口: $PORT"
echo "    GPU: $GPU_ID"
echo "    显存限制: ${GPU_MEM_UTILIZATION:-默认}"
echo "    最大 LoRA Rank: $MAX_LORA_RANK"

CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --port $PORT \
    --trust-remote-code \
    --enable-lora \
    --lora-modules ${LORA_NAME}="$LORA_PATH" \
    --max-loras 1 \
    --max-lora-rank $MAX_LORA_RANK \
    $MEM_ARG \
    --disable-log-requests \

echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 重写器 vLLM 服务已退出。"