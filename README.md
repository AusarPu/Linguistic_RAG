# RAG 语言学知识库问答系统

一个基于检索增强生成（RAG）技术的语言学知识库问答系统，支持多路召回、查询重写、重排序等高级功能。

## 🌟 项目特色

> **注意**: 本项目专为语言学研究设计，包含丰富的语言学知识库和专业术语处理能力。

- **多路召回策略**: 结合全文稠密、关键词稀疏、预生成问题三种检索方式
- **智能查询重写**: 基于对话历史优化用户查询，支持复杂查询分解
- **高效重排序**: 使用BGE重排序模型提升检索精度
- **流式生成**: 支持实时流式回答生成
- **Web界面**: 基于Gradio的友好用户界面，运行在8848端口
- **灵活部署**: vLLM后端完全兼容OpenAI API，支持本地部署或外部API
- **上下文优化**: 智能优化文本块连贯性，提升检索质量

## 🏗️ 系统架构

```
用户查询 → 查询重写 → 多路召回 → 重排序 → 生成回答
    ↓           ↓         ↓        ↓        ↓
  Web UI → Query Rewriter → Knowledge Base → Reranker → Generator
```

## 📁 项目结构

```
RAG/
├── knowledge_base/          # 原始知识库文档
├── processed_knowledge_base/ # 处理后的知识库数据
├── script/                  # 核心脚本
│   ├── config.py           # 配置文件
│   ├── knowledge_base.py   # 知识库管理
│   ├── rag_pipeline.py     # RAG流程
│   ├── query_rewriter.py   # 查询重写
│   └── vllm_clients.py     # vLLM客户端
├── preprocess/             # 数据预处理
│   ├── preprocess_documents.py    # 文档预处理和分块
│   ├── build_core_indexes.py      # 构建FAISS索引
│   ├── generate_summary_question.py # 生成问题摘要
│   └── optimize_chunk_b_via_vllm.py # 基于上下文的文本块优化

├── web_ui/                 # Web界面
│   ├── app.py             # 主应用
│   ├── ui_components.py   # UI组件
│   └── event_handlers.py  # 事件处理
├── prompts/               # 提示词模板
├── requirements.txt       # 依赖包
└── start_server.sh       # 启动脚本
```

## 🚀 快速开始

### 环境要求

- Python 3.12+ (当前测试环境: Python 3.12.9)
- CUDA 12.0+ (GPU推荐)
- 32GB+ GPU显存 (本地部署LLM时)
- 1-2GB GPU显存 (仅使用BGE嵌入模型，LLM使用外部API时)

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd RAG

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 模型准备

#### 本地部署模式 (需要32GB+ GPU显存)

在 `script/config.py` 中配置本地模型路径：

```python
# 基础模型路径
VLLM_BASE_MODEL_LOCAL_PATH = "/path/to/Qwen3-30B-A3B-FP8"
VLLM_REWRITE_MODEL_LOCAL_PATH = "/path/to/Qwen3-30B-A3B-FP8"
EMBEDDING_MODEL_PATH = "/path/to/bge-large-zh-v1.5"
VLLM_RERANKER_MODEL_PATH = "/path/to/bge-reranker-v2-m3"
```

#### 外部API模式 (仅需1-2GB GPU显存)

如果使用外部LLM API服务，只需配置嵌入模型：

```python
# 仅需要嵌入和重排序模型
EMBEDDING_MODEL_PATH = "/path/to/bge-large-zh-v1.5"
VLLM_RERANKER_MODEL_PATH = "/path/to/bge-reranker-v2-m3"

# API配置
GENERATOR_API_URL = "https://api.openai.com/v1/chat/completions"
REWRITER_API_URL = "https://api.openai.com/v1/chat/completions"
```

### 数据预处理

1. **准备知识库文档**
   - 将文档放入 `knowledge_base/` 目录
   - 支持包含页码标记 `≦ N ≧` 的TXT格式

2. **运行预处理**
   ```bash
   # 文档预处理和分块
   python preprocess/preprocess_documents.py
   
   # 基于上下文优化文本块（可选，提升文本连贯性）
   python preprocess/optimize_chunk_b_via_vllm.py
   
   # 构建FAISS向量索引
   python preprocess/build_core_indexes.py
   
   # 生成问题摘要（可选，用于问题匹配检索）
   python preprocess/generate_summary_question.py
   ```

### 启动服务

```bash
# 启动所有服务（推荐）
bash start_server.sh

# 或手动启动Web界面
python web_ui/app.py
```

## 🔧 配置说明

### 核心配置 (`script/config.py`)

```python
# 检索参数
DENSE_CHUNK_RETRIEVAL_TOP_K = 1000      # 稠密检索召回数量
DENSE_QUESTION_RETRIEVAL_TOP_K = 5       # 问题检索召回数量
SPARSE_KEYWORD_RETRIEVAL_TOP_K = 1000    # 关键词检索召回数量

# 阈值设置
DENSE_CHUNK_THRESHOLD = 0.7              # 稠密检索阈值
DENSE_QUESTION_THRESHOLD = 0.6           # 问题检索阈值
SPARSE_KEYWORD_THRESHOLD = 0.6           # 关键词检索阈值

# GPU配置
VLLM_GENERATOR_GPU_ID = 0                # 生成器GPU
VLLM_REWRITER_GPU_ID = "0,1"             # 重写器GPU（支持多卡）
VLLM_RERANKER_GPU_ID = 1                 # 重排序器GPU
```

### 服务端口配置

- **生成器服务**: `localhost:8001`
- **重写器服务**: `localhost:8001` 
- **重排序服务**: `localhost:8002`
- **Web界面**: `localhost:8848`

## 📊 功能模块

### 1. 知识库管理 (`knowledge_base.py`)

- **多策略检索**: 全文稠密、关键词稀疏、预生成问题三种检索方式
- **FAISS向量索引**: 高效的相似度搜索
- **元数据管理**: 文档来源、页码等信息追踪
- **阈值控制**: 可配置的相似度阈值过滤

### 2. 查询重写 (`query_rewriter.py`)

- **上下文理解**: 基于对话历史理解用户真实意图
- **查询分解**: 将复杂查询分解为多个可独立检索的子查询
- **语言学优化**: 针对语言学领域的专业查询优化
- **JSON格式输出**: 结构化的查询重写结果

### 3. RAG流程 (`rag_pipeline.py`)

- **异步流式处理**: 实时响应用户交互
- **多路并行召回**: 同时执行三种检索策略
- **智能重排序**: BGE重排序模型提升结果质量
- **上下文感知生成**: 基于检索结果的智能回答生成

### 4. 数据预处理 (`preprocess/`)

- **文档解析**: 支持带页码标记的OCR文档
- **智能分块**: RecursiveCharacterTextSplitter分块策略
- **上下文优化**: `optimize_chunk_b_via_vllm.py`基于上下文优化文本块连贯性
- **索引构建**: 自动化的向量索引和元数据构建

### 5. Web界面 (`web_ui/`)

- **实时对话**: 支持流式回答显示
- **多列布局**: 对话区、检索结果区、处理状态区
- **引用追踪**: 清晰显示答案来源和片段信息
- **状态监控**: 实时显示RAG流程各阶段状态

## 🧪 测试

```bash
# 运行测试脚本
python test.py
```

## 📝 使用示例

### API调用示例

```python
from script.rag_pipeline import execute_rag_flow
from script.knowledge_base import KnowledgeBase

# 初始化知识库
kb = KnowledgeBase()

# 执行RAG查询
async for event in execute_rag_flow(
    user_query="什么是音韵学？",
    chat_history_openai=[],
    kb_instance=kb
):
    print(event)
```

### Web界面使用

1. 访问 `http://localhost:8848`
2. 在输入框中输入问题
3. 查看实时生成的回答和引用来源
4. 支持多轮对话和上下文理解

## 🔍 检索策略详解

本系统采用三种互补的检索策略，确保全面覆盖用户查询需求：

### 1. 全文稠密检索 (Dense Retrieval)
- **技术**: 使用BGE-large-zh-v1.5嵌入模型
- **原理**: 基于语义相似度的向量匹配
- **优势**: 理解查询的深层语义，适合概念性和描述性查询
- **应用场景**: "什么是音韵学"、"语言学的发展历程"

### 2. 关键词稀疏检索 (Sparse Retrieval)
- **技术**: 基于TF-IDF的关键词匹配
- **原理**: 精确匹配重要术语和专有名词
- **优势**: 补充稠密检索的不足，处理专业术语
- **应用场景**: "乔姆斯基"、"转换生成语法"、"音位变体"

### 3. 预生成问题检索 (Question Matching)
- **技术**: 预先生成的问题-答案对匹配
- **原理**: 直接匹配用户问题意图
- **优势**: 提供精确的答案定位和快速响应
- **应用场景**: 常见问题的直接匹配

## 🛠️ 开发指南

### 添加新的检索策略

1. 在 `knowledge_base.py` 中实现新的检索方法
2. 在 `rag_pipeline.py` 中集成新策略
3. 更新配置文件中的相关参数

### 自定义提示词

- 编辑 `prompts/generator_system_prompt.txt` 修改生成器提示
- 编辑 `prompts/rewriter_instruction.txt` 修改重写器提示

### 扩展知识库

1. 将新文档放入 `knowledge_base/` 目录
2. 运行预处理脚本重建索引
3. 重启服务加载新数据

## 📈 性能优化

### GPU内存优化
- 调整 `MEM_UTILIZATION` 参数
- 使用模型量化（FP8）
- 合理分配多GPU资源

### 检索优化
- 调整各检索策略的TOP_K值
- 优化相似度阈值
- 使用更高效的索引结构

### 生成优化
- 调整生成参数（temperature, top_p等）
- 优化提示词模板
- 使用更大的上下文窗口

## 🐛 常见问题

### Q: 启动时提示模型路径不存在
A: 检查 `config.py` 中的模型路径配置，确保模型文件已下载到指定位置。

### Q: GPU显存不足
A: 降低 `MEM_UTILIZATION` 参数值，或使用更小的模型。

### Q: 检索结果不准确
A: 调整检索阈值参数，或重新训练嵌入模型。

### Q: Web界面无法访问
A: 检查防火墙设置，确保8848端口开放。

### Q: 如何使用外部LLM API？
A: vLLM后端完全兼容OpenAI API格式，可以轻松切换到外部API服务，大幅降低GPU显存需求。

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 本项目专为语言学研究设计，包含专业的语言学知识库和术语。使用前请确保已正确配置相关模型和数据。