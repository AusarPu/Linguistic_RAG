# script/config.py (集中配置版)

import logging
import os
import sys

# --- 日志配置 (保持不变) ---
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
# --------------------------

# --- 基础路径 ---
# 获取项目根目录 (假设 config.py 在 script/ 子目录下)
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ----------------
# 检索参数
MAX_HISTORY = 10
DENSE_CHUNK_RETRIEVAL_TOP_K = 10
DENSE_QUESTION_RETRIEVAL_TOP_K = 5 # 可以与上面不同
SPARSE_KEYWORD_RETRIEVAL_TOP_K = 5
DENSE_CHUNK_THRESHOLD = 0.5
DENSE_QUESTION_THRESHOLD = 0.5
SPARSE_KEYWORD_THRESHOLD = 0.5
RERANKER_SCORE_THRESHOLD = 0.5
FINAL_CONTEXT_TOP_K = 5

BM25_TOKENIZER_LANG = "auto"
BM25_TOKENIZER_SOURCE = "hf"
EN_STOPWORDS_FILE = ""

# BM25和语义搜索融合参数
RRF_K = 60 
# -----------------------------

# --- 模型本地路径配置 (保持不变) ---
VLLM_BASE_MODEL_LOCAL_PATH = VLLM_REWRITE_MODEL_LOCAL_PATH = "./models/Qwen/Qwen3-30B-A3B-FP8"
EMBEDDING_MODEL_PATH = "./models/Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL_NAME_FOR_API = "./models/Qwen/Qwen3-Reranker-0.6B"
VLLM_REWRITER_LORA_LOCAL_PATH = ""              

# --- Tokenizer 并行配置 ---
# 控制 Hugging Face fast tokenizer 使用的 CPU 线程数（通过 RAYON 线程池）
TOKENIZER_CPU_THREADS = 32

# --- 知识库和处理数据路径 ---
KNOWLEDGE_BASE_DIR = os.path.join(PROJECT_ROOT_DIR, "knowledge_base/")
KNOWLEDGE_FILE_PATTERN = "*.txt"
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "processed_knowledge_base/")

ENHANCED_CHUNKS_JSON_PATH = os.path.join(PROCESSED_DATA_DIR,"enhanced_knowledge_base_chunks_llm.json")
FAISS_INDEX_CHUNKS_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, "faiss_index_chunks_ip.idx")
INDEXED_CHUNKS_METADATA_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, "indexed_chunks_metadata.json")
PHRASE_SPARSE_WEIGHTS_MAP_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, "phrase_sparse_weights_map.pkl")
PHRASE_DENSE_EMBEDDINGS_MAP_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, "phrase_dense_embeddings_map.pkl")
BM25_INDEX_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, "bm25_index.pkl")
CHUNK_BM25_INDEX_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, "chunk_bm25_index.pkl")
FAISS_INDEX_QUESTIONS_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, "faiss_index_questions_ip.idx")
QUESTION_INDEX_TO_CHUNK_ID_MAP_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, "question_index_to_chunk_id_map.json")
ALL_QUESTION_TEXTS_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, "all_question_texts.json")

# --- vLLM 服务配置 ---
# 生成器服务配置
GPU_ID = "4,5"

VLLM_GENERATOR_HOST = "localhost" # vLLM 监听的主机名 (通常 localhost 即可，因为 Gradio 和 vLLM 在同一容器/机器)
VLLM_GENERATOR_PORT = 8001        # vLLM 生成器监听的端口
VLLM_GENERATOR_GPU_ID = GPU_ID        # 分配给生成器的 GPU ID
VLLM_GENERATOR_MEM_UTILIZATION = 0.4 # GPU 显存使用率 (例如 0.9 for 90%)

# 重写器服务配置
VLLM_REWRITER_HOST = "localhost"
VLLM_REWRITER_PORT = 8001         # vLLM 重写器监听的端口
VLLM_REWRITER_GPU_ID = GPU_ID        # 分配给重写器的 GPU ID (如果只有一块 GPU, 设为 0)
VLLM_REWRITER_MEM_UTILIZATION = 0.5 # 如果独占 GPU 可设高，共享则需调低 (例如 0.45)
VLLM_REWRITER_TENSOR_PARALLEL_SIZE = 2 # 新增：Rewriter的张量并行数

# 重写器 LoRA 配置
REWRITER_LORA_NAME = "rewriter_lora" # 在 vLLM 中标识 LoRA 的名称
VLLM_MAX_LORA_RANK = 32           # 支持的最大 LoRA Rank

# --- Embedding VLLM 服务配置 ---
VLLM_EMBEDDING_HOST = "localhost"  # Embedding 服务部署在本地
VLLM_EMBEDDING_PORT = 8850       # 为 Embedding 分配端口 8850
VLLM_EMBEDDING_GPU_ID = GPU_ID        # 分配给 Embedding 的 GPU ID
VLLM_EMBEDDING_MEM_UTILIZATION = 0.05
VLLM_EMBEDDING_TENSOR_PARALLEL_SIZE = 2

VLLM_RERANKER_HOST = "localhost"
VLLM_RERANKER_PORT = 8860
VLLM_RERANKER_GPU_ID = GPU_ID
VLLM_RERANKER_MEM_UTILIZATION = 0.3
VLLM_RERANKER_TENSOR_PARALLEL_SIZE = 2

# --- API 端点 (根据上面配置自动生成) ---
GENERATOR_API_URL = f"http://{VLLM_GENERATOR_HOST}:{VLLM_GENERATOR_PORT}/v1/chat/completions"
REWRITER_API_URL = f"http://{VLLM_REWRITER_HOST}:{VLLM_REWRITER_PORT}/v1/chat/completions"

EMBEDDING_API_URL = f"http://{VLLM_EMBEDDING_HOST}:{VLLM_EMBEDDING_PORT}/v1/embeddings"
RERANKER_API_URL = f"http://{VLLM_RERANKER_HOST}:{VLLM_RERANKER_PORT}/v1/chat/completions"

# --- vLLM 使用的模型标识符 (用于 API 请求中的 'model' 字段) ---
GENERATOR_MODEL_NAME_FOR_API = VLLM_BASE_MODEL_LOCAL_PATH 
REWRITER_MODEL_NAME_FOR_API = VLLM_REWRITE_MODEL_LOCAL_PATH 

EMBEDDING_MODEL_NAME_FOR_API = EMBEDDING_MODEL_PATH

API_PLATFORM_BASE_URL = "https://api.deepseek.com/v1"
API_PLATFORM_API_KEY_FILE = "/home/pushihao/RAG/script/api_keys/deepseek_api.txt"
API_PLATFORM_GENERATOR_MODEL = "deepseek-chat"
USE_API_PLATFORM_FOR_RAGAS = False

EVALUATION_LLM_MODEL_LOCAL_PATH = "/home/pushihao/RAG/models/openai/gpt-oss-120b"
EVALUATION_LLM_HOST = "localhost"
EVALUATION_LLM_PORT = 8003
EVALUATION_LLM_API_URL = f"http://{EVALUATION_LLM_HOST}:{EVALUATION_LLM_PORT}/v1/chat/completions"

def read_api_platform_key():
    return open(API_PLATFORM_API_KEY_FILE, "r", encoding="utf-8").read().strip()

def get_ragas_llm_base_url():
    return API_PLATFORM_BASE_URL if USE_API_PLATFORM_FOR_RAGAS else GENERATOR_API_URL.rsplit("/chat/completions", 1)[0]

def get_ragas_llm_api_key():
    return read_api_platform_key() if USE_API_PLATFORM_FOR_RAGAS else "-"

def get_ragas_llm_model():
    return API_PLATFORM_GENERATOR_MODEL if USE_API_PLATFORM_FOR_RAGAS else GENERATOR_MODEL_NAME_FOR_API

# --- Prompt 文件路径 (使用绝对路径或相对于 config.py 的路径) ---
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATOR_SYSTEM_PROMPT_FILE = os.path.join(_CONFIG_DIR, "../prompts/generator_system_prompt_eval.txt")
REWRITER_INSTRUCTION_FILE = os.path.join(_CONFIG_DIR, "../prompts/rewriter_instruction.txt")
USEFUL_JUDGER_INSTRUCTION_FILE = os.path.join(_CONFIG_DIR, "../prompts/useful_judge_v2.txt")
# 新增：块优化与元数据生成提示词文件路径
METADATA_PROMPT_ZH_FILE = os.path.join(_CONFIG_DIR, "../prompts/metadata_prompt_zh.txt")
METADATA_PROMPT_EN_FILE = os.path.join(_CONFIG_DIR, "../prompts/metadata_prompt_en.txt")
OPTIMIZATION_PROMPT_ZH_FILE = os.path.join(_CONFIG_DIR, "../prompts/chunk_optimization_prompt_zh.txt")
OPTIMIZATION_PROMPT_EN_FILE = os.path.join(_CONFIG_DIR, "../prompts/chunk_optimization_prompt_en.txt")
# -------------------------

# --- 生成参数配置 ---
GENERATION_CONFIG = { # 用于生成器 vLLM API
    "max_tokens": 4096,
    "temperature": 0.6,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "stop": None,
    "chat_template_kwargs": {"enable_thinking": True}
}
REWRITER_GENERATION_CONFIG = { # 用于重写器 vLLM API
    "max_tokens": 8192,
    "temperature": 0.6,
    "stop": None,
    "chat_template_kwargs": {"enable_thinking": False}
}
USEFULNESS_GENERATION_CONFIG = { # 用于有用性判断 vLLM API
    "max_tokens": 4096,
    "temperature": 0.6,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "stop": None,
    "chat_template_kwargs": {"enable_thinking": False}
}

RERANKER_GENERATION_CONFIG = {
    "max_tokens": 1,
    "temperature": 0,
    "top_p": 1.0,
    "logprobs": True,
    "top_logprobs": 20,
    "stop": None,
    "chat_template_kwargs": {"enable_thinking": False}
}

# -----------------

# --- VLLM 请求超时配置 (新增或统一) ---
VLLM_REQUEST_TIMEOUT = 60*20                 # 通用请求超时 (例如用于 Rewriter, Embedding)
VLLM_REQUEST_TIMEOUT_GENERATION = 60*20     # 为生成答案设置更长的超时时间

# --- 块优化专用超时配置 ---
VLLM_REQUEST_TIMEOUT_SINGLE = 60*30          # 超时B：单个块优化超时
VLLM_REQUEST_TIMEOUT_TOTAL = 3600*24         # 超时A：整体流程超时(8小时)
OPTIMIZATION_BATCH_SIZE = 5000                # 分批处理大小

# --- 并发控制配置 ---
MAX_CONCURRENT_REQUESTS = 1000                # 最大文本块并发请求
METADATA_MAX_CONCURRENT_REQUESTS = 1000       # 元数据生成的最大并发请求数
USEFULNESS_MAX_CONCURRENT_REQUESTS = 1000      # 有用性判断最大并发请求数

# --- 评估并发与输出限制 (Ragas 评估专用) ---
# 说明：用于在评估阶段（Ragas）控制客户端并发与单次评判的最大生成长度。
EVALUATION_CONCURRENCY_LIMIT = 10           # 评判请求的客户端并发上限（信号量）
EVALUATION_MAX_TOKENS = 8192                   # 单次评判的最大生成 tokens，用于限制长输出

RERANKER_CONCURRENCY_LIMIT = 2048
RERANKER_CONNECTOR_LIMIT = 2048
RERANKER_CONNECTOR_LIMIT_PER_HOST = 2048
RERANKER_KEEPALIVE_TIMEOUT = 20
RERANKER_FORCE_CLOSE = False
RERANKER_ENABLE_CLEANUP_CLOSED = True
RERANKER_CLIENT_TIMEOUT_TOTAL = 180
RERANKER_CLIENT_TIMEOUT_CONNECT = 15
RERANKER_CLIENT_TIMEOUT_SOCK_READ = 180


# --- 有用性判断软保留策略 ---
# 在多跳或不确定场景，避免过度过滤导致证据链断裂
SOFT_KEEP_MIN_CHUNKS = 5                      # 至少保留的上下文块数（含判定为useful者）
SOFT_KEEP_RATIO = 0.3                         # 至少保留原候选的比例（含判定为useful者）

# --- 日志配置函数 (方便在其他地方统一设置) ---
def setup_logging():
    """
    统一日志输出策略：
    - 仅输出重试与警告信息；常规 info 不再显示。
    - 显式开启 ragas/retry 的日志通道，便于观察 tenacity 重试。

    注意：tenacity 的重试日志在 ragas.run_config 中以 DEBUG 等级输出，
    因此这里为对应 logger 单独配置 handler 与等级，避免被全局 WARNING 屏蔽。
    """
    # 根日志：只输出 WARNING 及以上
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True  # 强制覆盖之前的 basicConfig 设置
    )

    # 常规库的 info 全部屏蔽，只保留 WARNING
    logging.getLogger("ragas.executor").setLevel(logging.WARNING)
    logging.getLogger("vllm").setLevel(logging.WARNING)

    # # 为重试相关 logger 单独打开 DEBUG handler（否则不会显示 tenacity 的重试）
    # def _ensure_retry_logger(logger_name: str):
    #     lg = logging.getLogger(logger_name)
    #     lg.setLevel(logging.DEBUG)
    #     lg.propagate = False
    #     # 避免重复添加 handler
    #     if not lg.handlers:
    #         h = logging.StreamHandler(sys.stdout)
    #         h.setLevel(logging.DEBUG)
    #         h.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    #         lg.addHandler(h)

    # # ragas.run_config.add_retry 使用的命名（同步）
    # _ensure_retry_logger("ragas.retry.embed_documents")
    # _ensure_retry_logger("ragas.retry.aembed_documents")
    # _ensure_retry_logger("ragas.retry.agenerate_text")

    # # ragas.run_config.add_async_retry 使用的命名（异步，函数名会体现在方括号内）
    # _ensure_retry_logger("TENACITYRetry")
    # _ensure_retry_logger("TENACITYRetry[agenerate_text]")
    # _ensure_retry_logger("TENACITYRetry[aembed_documents]")
    # _ensure_retry_logger("tenacity")
