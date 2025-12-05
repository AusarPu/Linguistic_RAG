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

# 默认并发设置（可通过环境变量覆盖）；若未显式设置 CPU 线程控制，则默认 32
TOKENIZER_POOL_SIZE="${TOKENIZER_POOL_SIZE:-32}"
MAX_PARALLEL_LOADING_WORKERS="${MAX_PARALLEL_LOADING_WORKERS:-32}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
if [[ -z "${RAYON_NUM_THREADS}" ]]; then export RAYON_NUM_THREADS="$TOKENIZER_POOL_SIZE"; fi
if [[ -z "${OMP_NUM_THREADS}" ]]; then export OMP_NUM_THREADS="$TOKENIZER_POOL_SIZE"; fi
if [[ -z "${MKL_NUM_THREADS}" ]]; then export MKL_NUM_THREADS="$TOKENIZER_POOL_SIZE"; fi

REWRITER_LOG="$LOG_DIR/vllm_rewriter.log"
FILTERED_LOG="$LOG_DIR/vllm_filtered.log"
FILTER_PROCESS_LOG="$LOG_DIR/filter_process.log"

EMBEDDING_LOG="$LOG_DIR/vllm_embedding.log"
PID_DIR="$PROJECT_ROOT/pids" # 定义 PID_DIR
REWRITER_PID_FILE="$PID_DIR/vllm_rewriter.pid"
FILTER_PID_FILE="$PID_DIR/filter.pid"

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
EMBEDDING_TENSOR_PARALLEL_SIZE=$(read_config VLLM_EMBEDDING_TENSOR_PARALLEL_SIZE)
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
    
    # 停止日志过滤器
    if [ -f "$FILTER_PID_FILE" ]; then
        FILTER_PID=$(cat "$FILTER_PID_FILE")
        echo "    停止日志过滤器 (PID: $FILTER_PID)..."
        kill "$FILTER_PID" &> /dev/null || echo "    日志过滤器进程 $FILTER_PID 可能已停止。"
        rm -f "$FILTER_PID_FILE"
    fi
    
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
    echo "    分词池大小: ${TOKENIZER_POOL_SIZE}"
    echo "    加载并行数: ${MAX_PARALLEL_LOADING_WORKERS}"
    echo "    CPU线程(RAYON): ${RAYON_NUM_THREADS}"

    # 使用 bash 数组来安全地构建命令
    REWRITER_CMD_ARRAY=(
        vllm serve "$REWRITER_BASE_MODEL_PATH"
        --port "$REWRITER_PORT"
        --trust-remote-code
        --disable-log-requests
        --enforce-eager
        --max-model-len 10240
        --tensor-parallel-size "$REWRITER_TENSOR_PARALLEL_SIZE"
        --max_num_seqs 1024
        --kv-cache-dtype fp8
        --max-parallel-loading-workers "$MAX_PARALLEL_LOADING_WORKERS"
        --reasoning-parser deepseek_r1 \
        #--quantization fp8
        #--rope-scaling '{"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}' \
    )

    # 有条件地添加内存参数
    if [ ! -z "$REWRITER_GPU_MEM_UTILIZATION" ] && [ "$REWRITER_GPU_MEM_UTILIZATION" != "None" ]; then
      REWRITER_CMD_ARRAY+=(--gpu-memory-utilization "$REWRITER_GPU_MEM_UTILIZATION")
    fi
    
    # 使用 nohup 和正确的变量展开来启动服务
    (export CUDA_VISIBLE_DEVICES=${REWRITER_GPU_ID}; nohup "${REWRITER_CMD_ARRAY[@]}" > "$REWRITER_LOG" 2>&1 & echo $! > "$REWRITER_PID_FILE")
    echo "    Rewriter 服务 PID: $(cat "$REWRITER_PID_FILE")，日志: $REWRITER_LOG"
    echo "    等待 Rewriter 服务启动 ..."
    sleep 60
    
    # 启动日志过滤器
    echo ">>> 启动 vLLM 日志过滤器..."
    nohup $PYTHON_CMD "$PROJECT_ROOT/filter_vllm_logs.py" \
        --input "$REWRITER_LOG" \
        --output "$FILTERED_LOG" \
        --max-lines 100 \
        --interval 30 \
        > "$FILTER_PROCESS_LOG" 2>&1 &
    FILTER_PID=$!
    echo $FILTER_PID > "$FILTER_PID_FILE"
    echo "    日志过滤器 PID: $FILTER_PID，过滤后日志: $FILTERED_LOG"
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
    echo "    张量并行数: ${EMBEDDING_TENSOR_PARALLEL_SIZE}"
    echo "    加载并行数: ${MAX_PARALLEL_LOADING_WORKERS}"
    echo "    CPU线程(RAYON): ${RAYON_NUM_THREADS}"

    # 使用 bash 数组来安全地构建命令
    EMBEDDING_CMD_ARRAY=(
        vllm serve "$EMBEDDING_MODEL_PATH"
        --port "$EMBEDDING_PORT"
        --trust-remote-code
        --disable-log-requests
        --enforce-eager
        --max-model-len 2048
        --max_num_seqs 2048
        --kv-cache-dtype fp8
        --tensor-parallel-size "$EMBEDDING_TENSOR_PARALLEL_SIZE"
        --max-parallel-loading-workers "$MAX_PARALLEL_LOADING_WORKERS"
    )

    # 有条件地添加内存参数
    if [ ! -z "$EMBEDDING_MEM_UTILIZATION" ] && [ "$EMBEDDING_MEM_UTILIZATION" != "None" ]; then
      EMBEDDING_CMD_ARRAY+=(--gpu-memory-utilization "$EMBEDDING_MEM_UTILIZATION")
    fi
    
    # 使用 nohup 和正确的变量展开来启动服务
    (export CUDA_VISIBLE_DEVICES=${EMBEDDING_GPU_ID}; nohup "${EMBEDDING_CMD_ARRAY[@]}" > "$EMBEDDING_LOG" 2>&1 & echo $! > "$EMBEDDING_PID_FILE")
    echo "    Embedding 服务 PID: $(cat "$EMBEDDING_PID_FILE")，日志: $EMBEDDING_LOG"
echo "    等待 Embedding 服务启动 ..."
    sleep 60
fi

## --- 启动 Reranker vLLM 服务（在 Embedding 之后 60s 启动） ---
RERANKER_MODEL_PATH=$(read_config RERANKER_MODEL_NAME_FOR_API)
RERANKER_PORT=$(read_config VLLM_RERANKER_PORT)
RERANKER_GPU_ID=$(read_config VLLM_RERANKER_GPU_ID)
RERANKER_MEM_UTILIZATION=$(read_config VLLM_RERANKER_MEM_UTILIZATION)
RERANKER_TENSOR_PARALLEL_SIZE=$(read_config VLLM_RERANKER_TENSOR_PARALLEL_SIZE)
RERANKER_LOG="$LOG_DIR/vllm_reranker.log"
RERANKER_PID_FILE="$PID_DIR/vllm_reranker.pid"

if [ -z "$RERANKER_MODEL_PATH" ] || [ -z "$RERANKER_PORT" ]; then
    echo "错误: Reranker 服务配置不完整 (模型路径或端口缺失)，跳过启动。" >&2
else
    echo ">>> 正在后台启动 Reranker vLLM 服务..."
    echo "    模型路径: $RERANKER_MODEL_PATH"
    echo "    端口: $RERANKER_PORT"
    echo "    分配 GPU: ${RERANKER_GPU_ID:-默认所有可见GPU}"
    echo "    显存限制: ${RERANKER_MEM_UTILIZATION:-默认}"
    echo "    张量并行数: ${RERANKER_TENSOR_PARALLEL_SIZE}"

    RERANKER_CMD_ARRAY=(
        vllm serve "$RERANKER_MODEL_PATH" \
        --port "$RERANKER_PORT" \
        --trust-remote-code \
        --disable-log-requests \
        --enforce-eager \
        --max-model-len 550 \
        --tensor-parallel-size "$RERANKER_TENSOR_PARALLEL_SIZE" \
        --max_num_seqs 2048 \
        --max-parallel-loading-workers "$MAX_PARALLEL_LOADING_WORKERS"\
        --kv-cache-dtype fp8
    )

    if [ ! -z "$RERANKER_MEM_UTILIZATION" ] && [ "$RERANKER_MEM_UTILIZATION" != "None" ]; then
      RERANKER_CMD_ARRAY+=(--gpu-memory-utilization "$RERANKER_MEM_UTILIZATION")
    fi

    (export CUDA_VISIBLE_DEVICES=${RERANKER_GPU_ID}; nohup "${RERANKER_CMD_ARRAY[@]}" > "$RERANKER_LOG" 2>&1 & echo $! > "$RERANKER_PID_FILE")
    echo "    Reranker 服务 PID: $(cat "$RERANKER_PID_FILE")，日志: $RERANKER_LOG"
fi

echo ">>> vLLM 服务启动完成。"
echo ">>> 如需启动前端，请运行: ./start_frontend.sh"

# 保持脚本运行，等待信号
echo ">>> 按 Ctrl+C 停止所有服务..."
while true; do
    sleep 1
done