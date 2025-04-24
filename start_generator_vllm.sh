#!/bin/bash

# --- 读取配置 ---
# 获取脚本所在目录的上级目录作为项目根目录 (假设脚本在 script/ 或类似子目录)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( dirname "$SCRIPT_DIR" )

# 设置 Python 命令和配置模块路径
PYTHON_CMD="python" # 或者你环境中的 python 版本
CONFIG_MODULE="script.config" # 相对于项目根目录的模块路径

# 定义一个函数从 config.py 读取变量值
read_config() {
    local var_name=$1
    # 使用 python -c 执行代码，将项目根目录添加到 sys.path
    # 使用 stderr 重定向来抑制潜在的 Python 警告/错误，只获取 print 输出
    local value=$($PYTHON_CMD -c "import sys; sys.path.insert(0, '$PROJECT_ROOT'); from $CONFIG_MODULE import $var_name; print($var_name)" 2>/dev/null)
    # 检查是否成功读取 (如果变量不存在或导入失败，value 可能为空)
    if [ -z "$value" ]; then
        echo "Error: Could not read $var_name from $CONFIG_MODULE" >&2
        exit 1
    fi
    echo "$value"
}

# 读取需要的配置项
echo ">>> 读取生成器配置从 $CONFIG_MODULE..."
BASE_MODEL_PATH=$(read_config VLLM_BASE_MODEL_LOCAL_PATH)
PORT=$(read_config VLLM_GENERATOR_PORT)
GPU_ID=$(read_config VLLM_GENERATOR_GPU_ID)
GPU_MEM_UTILIZATION=$(read_config VLLM_GENERATOR_MEM_UTILIZATION)
echo "    配置读取完成。"

# --- 准备 vLLM 启动参数 ---
MEM_ARG=""
# 检查 GPU_MEM_UTILIZATION 是否非空且不为 "None" (Python 返回 None 字符串)
if [ ! -z "$GPU_MEM_UTILIZATION" ] && [ "$GPU_MEM_UTILIZATION" != "None" ]; then
  MEM_ARG="--gpu-memory-utilization $GPU_MEM_UTILIZATION"
fi

# --- 启动 vLLM ---
echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 启动 vLLM 服务器 (生成器)..."
echo "    基础模型: $BASE_MODEL_PATH"
echo "    端口: $PORT"
echo "    分配 GPU: $GPU_ID"
echo "    显存限制: ${GPU_MEM_UTILIZATION:-默认}"

CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_CMD -m vllm.entrypoints.openai.api_server \
    --quantization fp8 \
    --model "$BASE_MODEL_PATH" \
    --port $PORT \
    --trust-remote-code \
    $MEM_ARG \
    --disable-log-requests \
    --max_model_len 30000
    #--log-config=logging.yaml # 可选

echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 生成器 vLLM 服务已退出。"