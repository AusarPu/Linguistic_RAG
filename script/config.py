import logging # 导入 logging

# ---- 日志配置 ----
LOG_LEVEL = logging.DEBUG # 设置全局日志级别 (可以是 DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
# -----------------

# ---- RAG 配置 ----
TOP_K_INITIAL_RETRIEVAL = 5000 # Hybrid 模式下，初始从向量数据库检索的候选数量 (可以比最终需要的多)
CHUNK_SIZE = 400
OVERLAP = 20
MIN_CHUNK_LENGTH = 150
MAX_HISTORY = 10

# 检索策略配置 (Hybrid 模式下)
RETRIEVAL_STRATEGY = "threshold" # 或者 "top_k"，用于区分最终筛选策略
# 设置混合相似度阈值 (需要根据你的数据实验调整，0到1之间，越高越严格)
HYBRID_SIMILARITY_THRESHOLD = 0.55
# 设置阈值模式下，最多返回的结果数量 (防止上下文过长)
MAX_THRESHOLD_RESULTS = 4
# 控制 RAG 最终送入 LLM 的总块数上限，防止上下文超长
MAX_AGGREGATED_RESULTS = 30

# ---- 模型路径 ----
BASE_MODEL_PATH = "/home/pushihao/model-dir"
ADAPTER_PATH = "/home/pushihao/RAG/linguistic_ai"
# REWRITER_MODEL_PATH = "/home/pushihao/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
REWRITER_MODEL_PATH = "/home/pushihao/model-dir"
EMBEDDING_MODEL_PATH = "/home/pushihao/RAG/sentence_similarity"
PROCESSED_DATA_DIR = "/home/pushihao/RAG/processed_knowledge_base"

# ---- 知识库源文件 ----
# 指定包含知识库源文件的目录
KNOWLEDGE_BASE_DIR = "/home/pushihao/RAG/knowledge_base/" # <--- 修改成你的目录路径
# 指定在该目录中要读取的文件匹配模式 (例如：所有 .txt 文件)
KNOWLEDGE_FILE_PATTERN = "*.txt"

# ---- Prompt 文件路径 ----
GENERATOR_SYSTEM_PROMPT_FILE = "../prompts/generator_system_prompt.txt"
REWRITER_INSTRUCTION_FILE = "../prompts/rewriter_instruction.txt"
# -------------------------

# ---- 生成参数 ----
GENERATION_CONFIG = {
    "max_new_tokens": 4096,
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "repetition_penalty": 1.1
}
# -----------------