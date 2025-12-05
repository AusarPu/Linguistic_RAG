# 数据集索引构建工具

这个工具用于从增强后的数据集构建独立的知识库索引，每个数据集对应一个知识库文件夹。

## 功能特性

- 支持多个数据集：hotpotqa, ms_marco, natural_questions, triviaqa
- 为每个数据集创建独立的知识库索引
- 支持测试模式，可以限制处理的数据量
- 自动验证输入数据格式
- 详细的日志输出和错误处理

## 索引类型

每个数据集会构建以下索引：

1. **稠密向量索引 (Faiss)** - 基于文本块的语义向量
2. **文本块BM25索引** - 基于jieba分词的关键词检索
3. **关键词短语稠密向量映射** - 关键词的语义向量
4. **关键词短语BM25索引** - 关键词的关键词检索
5. **预生成问题稠密向量索引** - 问题的语义向量

## 使用方法

### 基本用法

```bash
# 构建所有数据集的索引
python build_dataset_indexes.py

# 测试模式（只处理前10个条目）
python build_dataset_indexes.py --test

# 指定测试条目数量
python build_dataset_indexes.py --test --test-limit 50

# 只处理特定数据集
python build_dataset_indexes.py --datasets hotpotqa ms_marco

# 只处理单个数据集
python build_dataset_indexes.py --single natural_questions
```

### 参数说明

- `--test`: 启用测试模式，只处理少量数据用于测试
- `--test-limit N`: 测试模式下限制处理的条目数量（默认：10）
- `--datasets`: 指定要处理的数据集列表
- `--single`: 只处理单个数据集

## 输入输出

### 输入
- 增强数据集文件：`/home/pushihao/RAG/Reports/experiments/datasets/enhanced/{dataset_name}_enhanced.json`

### 输出
- 知识库索引：`/home/pushihao/RAG/Reports/experiments/datasets/knowledge_bases/{dataset_name}/`

每个知识库文件夹包含：
- `dense_embeddings_chunks.npy` - 文本块稠密向量
- `faiss_index_chunks_ip.idx` - 文本块Faiss索引
- `indexed_chunks_metadata.json` - 索引块元数据
- `chunk_bm25_index.pkl` - 文本块BM25索引
- `phrase_dense_embeddings_map.pkl` - 关键词短语向量映射
- `phrase_bm25_index.pkl` - 关键词短语BM25索引
- `dense_embeddings_questions.npy` - 问题稠密向量
- `faiss_index_questions_ip.idx` - 问题Faiss索引
- `question_index_to_chunk_id_map.json` - 问题到块ID的映射
- `all_question_texts.json` - 所有问题文本列表

## 数据格式要求

增强数据集文件应包含以下字段：
- `text`: 文本内容
- `chunk_id`: 块ID
- `keyword_summaries`: 关键词摘要列表
- `generated_questions`: 生成的问题列表
- `is_meaningful`: 是否有意义的标记

## 依赖要求

- faiss-cpu 或 faiss-gpu
- numpy
- jieba
- rank-bm25
- tqdm
- requests

## 注意事项

1. 确保嵌入API服务正在运行
2. 构建索引需要较长时间，建议先使用测试模式验证
3. 确保有足够的磁盘空间存储索引文件
4. 如果API调用失败，脚本会自动重试