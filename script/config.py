# script/config.py (集中配置版)

import logging
import os
import sys

# --- 日志配置 (保持不变) ---
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = '%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
# --------------------------

# --- 基础路径 ---
# 获取项目根目录 (假设 config.py 在 script/ 子目录下)
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ----------------

# --- RAG 应用配置 (保持不变) ---
TOP_K_INITIAL_RETRIEVAL = 100
CHUNK_SIZE = 500
OVERLAP = 200
MIN_CHUNK_LENGTH = 200
MAX_HISTORY = 5
RETRIEVAL_STRATEGY = "threshold"
HYBRID_SIMILARITY_THRESHOLD = 0.75 # 根据你的测试调整
MAX_THRESHOLD_RESULTS = 4
MAX_AGGREGATED_RESULTS = 15
DENSE_WEIGHT = 0.99
# -----------------------------

# --- 模型本地路径配置 ---
# vLLM 使用的基础模型 (Generator 和 Rewriter 都用这个) 的本地路径
# !! 需要确保这个路径在新设备上是正确的 !!
VLLM_BASE_MODEL_LOCAL_PATH = "/home/pushihao/RAG/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" #<--- 修改为你 base_llm 下载的实际路径
VLLM_REWRITE_MODEL_LOCAL_PATH = "/home/pushihao/RAG/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# Rewriter 使用的 LoRA adapter 的本地路径
# !! 需要确保这个路径在新设备上是正确的 !!
VLLM_REWRITER_LORA_LOCAL_PATH = "/home/pushihao/RAG/linguistic_ai" #<--- 修改为你 rewriter_lora 下载或存放的实际路径

# 嵌入模型本地路径
# !! 需要确保这个路径在新设备上是正确的 !!
EMBEDDING_MODEL_PATH = "/home/pushihao/RAG/models/BAAI/bge-large-zh-v1.5" #<--- 修改为你 embedding 下载的实际路径

# 知识库和处理数据路径
KNOWLEDGE_BASE_DIR = os.path.join(PROJECT_ROOT_DIR, "knowledge_base/") # 使用相对项目根目录的路径
KNOWLEDGE_FILE_PATTERN = "*.txt"
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "processed_knowledge_base/") # 使用相对项目根目录的路径
# --------------------------

# --- vLLM 服务配置 ---
# 生成器服务配置
VLLM_GENERATOR_HOST = "localhost" # vLLM 监听的主机名 (通常 localhost 即可，因为 Gradio 和 vLLM 在同一容器/机器)
VLLM_GENERATOR_PORT = 8000        # vLLM 生成器监听的端口
VLLM_GENERATOR_GPU_ID = 0         # 分配给生成器的 GPU ID
VLLM_GENERATOR_MEM_UTILIZATION = 0.9 # GPU 显存使用率 (例如 0.9 for 90%)

# 重写器服务配置
VLLM_REWRITER_HOST = "localhost"
VLLM_REWRITER_PORT = 8001         # vLLM 重写器监听的端口
VLLM_REWRITER_GPU_ID = 1          # 分配给重写器的 GPU ID (如果只有一块 GPU, 设为 0)
VLLM_REWRITER_MEM_UTILIZATION = 0.9 # 如果独占 GPU 可设高，共享则需调低 (例如 0.45)

# 重写器 LoRA 配置
REWRITER_LORA_NAME = "rewriter_lora" # 在 vLLM 中标识 LoRA 的名称
VLLM_MAX_LORA_RANK = 32           # 支持的最大 LoRA Rank

# API 端点 (根据上面配置自动生成)
VLLM_GENERATOR_API_BASE_URL = f"http://{VLLM_GENERATOR_HOST}:{VLLM_GENERATOR_PORT}/v1"
VLLM_REWRITER_API_BASE_URL = f"http://{VLLM_REWRITER_HOST}:{VLLM_REWRITER_PORT}/v1"

# vLLM 使用的模型标识符 (通常就是基础模型路径，用于 API 请求中的 'model' 字段)
# 注意：对于带 LoRA 的请求，实际 model 字段可能需要拼接 Lora 名称，这在 query_rewriter.py 中处理
VLLM_GENERATOR_MODEL_ID_FOR_API = VLLM_BASE_MODEL_LOCAL_PATH
VLLM_REWRITER_MODEL_ID_FOR_API = VLLM_REWRITE_MODEL_LOCAL_PATH # API 请求中 model 字段通常仍是基础模型
# -------------------------

# --- Prompt 文件路径 (使用绝对路径或相对于 config.py 的路径) ---
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATOR_SYSTEM_PROMPT_FILE = os.path.join(_CONFIG_DIR, "../prompts/generator_system_prompt.txt")
REWRITER_INSTRUCTION_FILE = os.path.join(_CONFIG_DIR, "../prompts/rewriter_instruction.txt")
# -------------------------

# --- 生成参数配置 ---
GENERATION_CONFIG = { # 用于生成器 vLLM API
    "max_tokens": 5000,
    "temperature": 0.9,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "stop": None,
}
REWRITER_GENERATION_CONFIG = { # 用于重写器 vLLM API
    "max_tokens": 1000,
    "temperature": 0.2,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "stop": None,
}
# -----------------

# --- 日志配置函数 (方便在其他地方统一设置) ---
def setup_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        # filename='rag_chat.log', # 可以取消注释将日志写入文件
        # filemode='a',
        handlers=[logging.StreamHandler(sys.stdout)] # 直接输出到 stdout
    )
    # 可以设置特定库的日志级别，例如减少 VLLM 自身日志
    # logging.getLogger("vllm").setLevel(logging.WARNING)

# 在模块导入时可以先配置一次 (如果需要)
# setup_logging()
# -----------------------------------------