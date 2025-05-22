#!/bin/bash

# --- 配置 ---
# 你可以从你的 config.py 或环境变量中读取这些值
MODEL_NAME_OR_PATH="/home/pushihao/RAG/models/BAAI/bge-reranker-v2-m3" # 或者你本地的路径
PORT=8002 # 使用一个与你其他VLLM服务不同的端口
GPU_ID="2" # 根据需要指定GPU
# TENSOR_PARALLEL_SIZE=1 # 根据你的GPU数量调整
# MAX_MODEL_LEN=8192 # BGE-M3 支持较长序列，但嵌入任务通常不需要这么长，可以根据需要调整

# 构建参数
VLLM_ARGS="${MODEL_NAME_OR_PATH}"
VLLM_ARGS="${VLLM_ARGS} --port ${PORT}"
VLLM_ARGS="${VLLM_ARGS} --trust-remote-code" # BGE-M3 通常需要
# VLLM_ARGS="${VLLM_ARGS} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}"
# VLLM_ARGS="${VLLM_ARGS} --max-model-len ${MAX_MODEL_LEN}"
VLLM_ARGS="${VLLM_ARGS} --disable-log-requests" # 可选，减少日志噪音
VLLM_ARGS="${VLLM_ARGS} --dtype auto"

# 设置CUDA设备 (如果需要指定特定GPU)
export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 启动 BGE VLLM reranker 服务..."
echo "    模型: ${MODEL_NAME_OR_PATH}"
echo "    端口: ${PORT}"
echo "    分配 GPU: ${GPU_ID:-默认}"

# 启动 VLLM OpenAI 兼容 API 服务器
vllm serve ${VLLM_ARGS}

# 使用 eval 来正确处理带引号的参数 (另一种方式)
# eval "python -m vllm.entrypoints.openai.api_server ${VLLM_ARGS}"


echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] BGE VLLM reranker 服务已退出。"