import logging

# ---- 日志配置 ----
LOG_LEVEL = logging.DEBUG # 设置全局日志级别 (可以是 DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
# -----------------

# ---- RAG 配置 ----
TOP_K_INITIAL_RETRIEVAL = 500 # Hybrid 模式下，初始从向量数据库检索的候选数量
CHUNK_SIZE = 250
OVERLAP = 20
MIN_CHUNK_LENGTH = 150
MAX_HISTORY = 10

# 检索策略配置 (Hybrid 模式下)
RETRIEVAL_STRATEGY = "threshold" # 或者 "top_k"，用于区分最终筛选策略
HYBRID_SIMILARITY_THRESHOLD = 0.6 # 混合相似度阈值
MAX_THRESHOLD_RESULTS = 4 # 阈值模式下，最多返回的结果数量
MAX_AGGREGATED_RESULTS = 30 # RAG 最终送入 LLM 的总块数上限

# ---- 模型路径 (本地) ----
# BASE_MODEL_PATH = "/home/pushihao/model-dir" # 不再需要，由 vLLM 加载
# ADAPTER_PATH = "/home/pushihao/RAG/linguistic_ai" # 如果 adapter 在 vLLM 加载时合并，则不再需要
# REWRITER_MODEL_PATH = "/home/pushihao/model-dir" # 不再需要，由 vLLM 加载
EMBEDDING_MODEL_PATH = "/home/pushihao/RAG/sentence_similarity" # 嵌入模型仍然本地加载
PROCESSED_DATA_DIR = "/home/pushihao/RAG/processed_knowledge_base" # 处理后的知识库数据目录

# ---- vLLM 服务配置 ----
# 生成器服务的 API 地址
VLLM_GENERATOR_API_BASE_URL = "http://localhost:8000/v1" # 指向 8000 端口
# 重写器服务的 API 地址
VLLM_REWRITER_API_BASE_URL = "http://localhost:8001/v1"  # 指向 8001 端口

# vLLM 中使用的生成模型的标识符 (基础模型路径)
VLLM_GENERATOR_MODEL = "/home/pushihao/model-dir"
# vLLM 中使用的重写模型的基础模型标识符
VLLM_REWRITER_MODEL = "/home/pushihao/model-dir" # 仍然是基础模型路径


# 在启动脚本中为重写器 LoRA 定义的名称
REWRITER_LORA_NAME = "rewriter_lora"

# ---- 知识库源文件 ----
KNOWLEDGE_BASE_DIR = "/home/pushihao/RAG/knowledge_base/"
KNOWLEDGE_FILE_PATTERN = "*.txt"

# ---- Prompt 文件路径 ----
# 注意：路径相对于调用脚本的位置，对于 Gradio 应用 (app_gradio.py) 和 命令行应用 (main.py) 可能需要调整
# 推荐使用绝对路径或基于 __file__ 的相对路径来确保一致性
import os
_SCRIPT_DIR = os.path.dirname(__file__)
GENERATOR_SYSTEM_PROMPT_FILE = os.path.join(_SCRIPT_DIR, "../prompts/generator_system_prompt.txt")
REWRITER_INSTRUCTION_FILE = os.path.join(_SCRIPT_DIR, "../prompts/rewriter_instruction.txt")
# -------------------------

# ---- 生成参数 (用于 vLLM API 调用) ----
# 这些参数现在将用于构造发送给 vLLM API 的请求体
GENERATION_CONFIG = {
    "max_tokens": 5000, # 在 vLLM API 中通常用 max_tokens
    "temperature": 0.9,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "stop": None, # 可以设置停止词，例如 ["\nUser:", "</s>"]
    # "do_sample" 在 OpenAI API 风格中通常由 temperature 控制 (temperature > 0 意味着采样)
    # "stream": True, # 这个参数在调用 API 时直接设置，不放在这里
}
# -----------------

# ---- (可选) 重写模型的生成参数 ----
# 如果需要为重写任务使用不同的参数，可以单独定义
REWRITER_GENERATION_CONFIG = {
    "max_tokens": 1024, # 重写查询通常较短
    "temperature": 0.2, # 可能需要更确定性的输出
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "stop": None, # 例子：假设重写器生成的查询以空行分隔
}
# -----------------