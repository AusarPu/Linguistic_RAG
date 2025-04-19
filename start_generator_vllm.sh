#!/bin/bash

# --- 配置: 生成器服务 ---
BASE_MODEL="/home/pushihao/model-dir"
PORT=8000
GPU_ID=0 # 分配给生成器的 GPU
GPU_MEM_UTILIZATION="" # 例如: "0.9" (90%) 或留空使用默认值

MEM_ARG=""
if [ ! -z "$GPU_MEM_UTILIZATION" ]; then
  MEM_ARG="--gpu-memory-utilization $GPU_MEM_UTILIZATION"
fi

echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 启动 vLLM 服务器 (生成器)..."
echo "    模型: $BASE_MODEL"
echo "    端口: $PORT"
echo "    GPU: $GPU_ID"
echo "    显存限制: ${GPU_MEM_UTILIZATION:-默认}"

CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --port $PORT \
    --trust-remote-code \
    $MEM_ARG \
    --disable-log-requests \

echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 生成器 vLLM 服务已退出。"