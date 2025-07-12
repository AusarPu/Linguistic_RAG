# script/config.py (集中配置版)

import logging
import os
import sys

# --- 日志配置 (保持不变) ---
LOG_LEVEL = logging.WARNING
LOG_FORMAT = '%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
# --------------------------

# --- 基础路径 ---
# 获取项目根目录 (假设 config.py 在 script/ 子目录下)
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ----------------
# 检索参数
MAX_HISTORY = 10
DENSE_CHUNK_RETRIEVAL_TOP_K = 1000
DENSE_QUESTION_RETRIEVAL_TOP_K = 5 # 可以与上面不同
SPARSE_KEYWORD_RETRIEVAL_TOP_K = 1000
DENSE_CHUNK_THRESHOLD = 0.7
DENSE_QUESTION_THRESHOLD = 0.6
SPARSE_KEYWORD_THRESHOLD = 0.6
# -----------------------------

# --- 模型本地路径配置 (保持不变) ---
VLLM_BASE_MODEL_LOCAL_PATH = "./models/Qwen/Qwen3-30B-A3B-FP8"
VLLM_REWRITE_MODEL_LOCAL_PATH = "./models/Qwen/Qwen3-30B-A3B-FP8"
EMBEDDING_MODEL_PATH = "./models/BAAI/bge-large-zh-v1.5"
VLLM_RERANKER_MODEL_PATH = "./models/BAAI/bge-reranker-v2-m3"
EMBEDDING_MODEL_PATH = "./models/Qwen/Qwen3-Embedding-8B"
VLLM_REWRITER_LORA_LOCAL_PATH = ""

# --- 知识库和处理数据路径 (你已有的，确保ALL_QUESTION_TEXTS_SAVE_PATH已添加) ---
KNOWLEDGE_BASE_DIR = os.path.join(PROJECT_ROOT_DIR, "knowledge_base/")
KNOWLEDGE_FILE_PATTERN = "*.txt"
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "processed_knowledge_base/")

ENHANCED_CHUNKS_JSON_PATH = os.path.join(PROCESSED_DATA_DIR,"enhanced_knowledge_base_chunks_llm.json")
FAISS_INDEX_CHUNKS_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, "faiss_index_chunks_ip.idx")
INDEXED_CHUNKS_METADATA_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, "indexed_chunks_metadata.json")
PHRASE_SPARSE_WEIGHTS_MAP_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, "phrase_sparse_weights_map.pkl")
FAISS_INDEX_QUESTIONS_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, "faiss_index_questions_ip.idx")
QUESTION_INDEX_TO_CHUNK_ID_MAP_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, "question_index_to_chunk_id_map.json")
ALL_QUESTION_TEXTS_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, "all_question_texts.json") # 确保这个已存在

# --- vLLM 服务配置 ---
# 生成器服务配置
VLLM_GENERATOR_HOST = "localhost" # vLLM 监听的主机名 (通常 localhost 即可，因为 Gradio 和 vLLM 在同一容器/机器)
VLLM_GENERATOR_PORT = 8001        # vLLM 生成器监听的端口
VLLM_GENERATOR_GPU_ID = 0         # 分配给生成器的 GPU ID
VLLM_GENERATOR_MEM_UTILIZATION = 0.9 # GPU 显存使用率 (例如 0.9 for 90%)

# 重写器服务配置
VLLM_REWRITER_HOST = "localhost"
VLLM_REWRITER_PORT = 8001         # vLLM 重写器监听的端口
VLLM_REWRITER_GPU_ID = "0,1"         # 分配给重写器的 GPU ID (如果只有一块 GPU, 设为 0)
VLLM_REWRITER_MEM_UTILIZATION = 0.90 # 如果独占 GPU 可设高，共享则需调低 (例如 0.45)
VLLM_REWRITER_TENSOR_PARALLEL_SIZE = 2 # 新增：Rewriter的张量并行数

# 重写器 LoRA 配置
REWRITER_LORA_NAME = "rewriter_lora" # 在 vLLM 中标识 LoRA 的名称
VLLM_MAX_LORA_RANK = 32           # 支持的最大 LoRA Rank

# --- 新增：Reranker VLLM 服务配置 ---
VLLM_RERANKER_HOST = "localhost"  # 假设 Reranker 服务也部署在本地
VLLM_RERANKER_PORT = 8002       # 为 Reranker 分配一个新的端口，确保不冲突
VLLM_RERANKER_GPU_ID = 1      # 分配给 Reranker 的 GPU ID (注意：如果GPU资源紧张，需要合理分配)
VLLM_RERANKER_MEM_UTILIZATION = 0.1 # 如果与其他模型共享GPU，可能需要较低的显存占用

# --- 新增：Embedding VLLM 服务配置 ---
VLLM_EMBEDDING_HOST = "localhost"  # Embedding 服务部署在本地
VLLM_EMBEDDING_PORT = 8850       # 为 Embedding 分配端口 8850
VLLM_EMBEDDING_GPU_ID = 2        # 分配给 Embedding 的 GPU ID
VLLM_EMBEDDING_MEM_UTILIZATION = 0.3 # Embedding 模型通常显存占用较少

# --- API 端点 (根据上面配置自动生成) ---
GENERATOR_API_URL = f"http://{VLLM_GENERATOR_HOST}:{VLLM_GENERATOR_PORT}/v1/chat/completions"
REWRITER_API_URL = f"http://{VLLM_REWRITER_HOST}:{VLLM_REWRITER_PORT}/v1/chat/completions"
RERANKER_API_URL = f"http://{VLLM_RERANKER_HOST}:{VLLM_RERANKER_PORT}/v2/rerank"
EMBEDDING_API_URL = f"http://{VLLM_EMBEDDING_HOST}:{VLLM_EMBEDDING_PORT}/v1/embeddings"

# --- vLLM 使用的模型标识符 (用于 API 请求中的 'model' 字段) ---
GENERATOR_MODEL_NAME_FOR_API = VLLM_BASE_MODEL_LOCAL_PATH # 你已有的
REWRITER_MODEL_NAME_FOR_API = VLLM_REWRITE_MODEL_LOCAL_PATH # 你已有的
RERANKER_MODEL_NAME_FOR_API = VLLM_RERANKER_MODEL_PATH # 用于发送给 Reranker API 的模型名
EMBEDDING_MODEL_NAME_FOR_API = EMBEDDING_MODEL_PATH # 用于发送给 Embedding API 的模型名

# --- Prompt 文件路径 (使用绝对路径或相对于 config.py 的路径) ---
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATOR_SYSTEM_PROMPT_FILE = os.path.join(_CONFIG_DIR, "../prompts/generator_system_prompt.txt")
REWRITER_INSTRUCTION_FILE = os.path.join(_CONFIG_DIR, "../prompts/rewriter_instruction.txt")
# -------------------------

# --- 生成参数配置 ---
GENERATION_CONFIG = { # 用于生成器 vLLM API
    "max_tokens": 40960,
    "temperature": 0.9,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "stop": None,
}
REWRITER_GENERATION_CONFIG = { # 用于重写器 vLLM API
    "max_tokens": 40960,
    "temperature": 0.2,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "stop": None,
}

# 新增：专门为 RAG 流程中答案生成步骤的配置 (可以基于 GENERATION_CONFIG 修改)
GENERATOR_RAG_CONFIG = {
    "max_tokens": 10000,       # RAG回答通常不需要像通用聊天那么长
    "temperature": 0.7,       # 可以略微降低温度，使其更忠实于上下文
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "stop": None
    # 其他你认为适合RAG场景的参数
}
# -----------------

# --- RAG 流程控制参数 (新增) ---
RERANKER_TOP_N_INPUT_MAX = 100       # 多路召回后，最多取多少个候选块送给 Reranker
RERANKER_BATCH_SIZE = 1000            # 调用 Reranker API 时的批处理大小 (如果API支持批量输入passage)
GENERATOR_CONTEXT_TOP_N = 10         # Reranker 精排后，选取多少个最优块作为最终上下文

# --- VLLM 请求超时配置 (新增或统一) ---
VLLM_REQUEST_TIMEOUT = 60.0                 # 通用请求超时 (例如用于 Rewriter, Reranker)
VLLM_REQUEST_TIMEOUT_GENERATION = 300.0     # 为生成答案设置更长的超时时间

# --- 块优化专用超时配置 ---
VLLM_REQUEST_TIMEOUT_SINGLE = 60*3          # 超时B：单个块优化超时
VLLM_REQUEST_TIMEOUT_TOTAL = 3600*8         # 超时A：整体流程超时(8小时)
OPTIMIZATION_BATCH_SIZE = 250                # 分批处理大小

# --- 日志配置函数 (方便在其他地方统一设置) ---
def setup_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        # filename='rag_chat.log', # 可以取消注释将日志写入文件
        # filemode='w',
        handlers=[logging.StreamHandler(sys.stdout)] # 直接输出到 stdout
    )
    # 可以设置特定库的日志级别，例如减少 VLLM 自身日志
    # logging.getLogger("vllm").setLevel(logging.WARNING)

