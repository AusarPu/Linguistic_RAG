# RAG 配置
KNOWLEDGE_BASE_FILE = "/knowledge_base/test.txt"
TOP_K = 10
CHUNK_SIZE = 256
OVERLAP = 50
MIN_CHUNK_LENGTH = 128
SIMILARITY_THRESHOLD = 0.8
SEARCH_MODE = "hybrid"

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
