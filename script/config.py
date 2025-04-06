# RAG 配置
KNOWLEDGE_BASE_FILE = "/knowledge_base/test.txt"
TOP_K = 10
CHUNK_SIZE = 256
OVERLAP = 50
MIN_CHUNK_LENGTH = 128
SEARCH_MODE = "hybrid"
# === 新增：阈值检索相关配置 ===
# 设置检索模式为阈值模式
RETRIEVAL_STRATEGY = "threshold" # 或者 "top_k"，用于区分策略

# 设置混合相似度阈值 (需要根据你的数据实验调整，0到1之间，越高越严格)
HYBRID_SIMILARITY_THRESHOLD = 0.5 # <--- 示例值，请调整！

# 设置阈值模式下，最多返回的结果数量 (防止上下文过长)
MAX_THRESHOLD_RESULTS = 15 # <--- 示例值，请调整！
# ============================

# 模型路径
BASE_MODEL_PATH = "/home/pushihao/model-dir"
ADAPTER_PATH = "/home/pushihao/RAG/linguistic_ai"
# REWRITER_MODEL_PATH = "/home/pushihao/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
REWRITER_MODEL_PATH = "/home/pushihao/model-dir"
EMBEDDING_MODEL_PATH = "/home/pushihao/RAG/sentence_similarity"
PROCESSED_DATA_DIR = "/home/pushihao/RAG/processed_knowledge_base"

# 指定包含知识库源文件的目录
KNOWLEDGE_BASE_DIR = "/home/pushihao/RAG/knowledge_base/" # <--- 修改成你的目录路径
# 指定在该目录中要读取的文件匹配模式 (例如：所有 .txt 文件)
KNOWLEDGE_FILE_PATTERN = "*.txt"

# 生成参数
GENERATION_CONFIG = {
    "max_new_tokens": 4096,
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "repetition_penalty": 1.1
}
