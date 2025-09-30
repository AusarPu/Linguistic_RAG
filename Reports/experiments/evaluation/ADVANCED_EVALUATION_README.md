# 高级RAG评估脚本

## 概述

`advanced_evaluation.py` 是一个用于评估RAG系统性能的高级脚本，提供两个核心评估功能：

1. **答案正确性评估** - 使用vLLM后端评估系统回答是否正确回答了问题
2. **检索准确性评估** - 检查original_id是否在retrieved_chunk_ids中

## 功能特性

### 1. 答案正确性评估
- 调用8001端口的vLLM后端进行智能评估
- 使用专业的评估提示词模板
- 比较system_answer和ground_truth_answer
- 提供详细的评估解释和信心度评分

### 2. 检索准确性评估
- 检查original_id是否包含在retrieved_chunk_ids中
- 统计检索成功率
- 提供详细的检索信息

### 3. 输出格式
生成的JSON文件包含原始数据的所有字段，并新增两个评估结果字段：

```json
{
  // ... 原始字段 ...
  "answer_correctness": {
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "explanation": "详细解释",
    "status": "success/error/parse_error/api_error"
  },
  "retrieval_accuracy": {
    "is_retrieved": true/false,
    "original_id": "原始ID",
    "retrieved_chunk_ids": ["检索到的块ID列表"],
    "total_retrieved": 检索块数量
  }
}
```

## 使用方法

### 基本用法
```bash
python advanced_evaluation.py <input_file> <output_file>
```

### 测试模式（限制处理数量）
```bash
python advanced_evaluation.py <input_file> <output_file> --limit 10
```

### 示例
```bash
# 评估完整结果文件
python advanced_evaluation.py \
  /path/to/evaluation_results.json \
  /path/to/advanced_evaluation_results.json

# 测试模式，仅处理前5个结果
python advanced_evaluation.py \
  /path/to/sample_results.json \
  /path/to/advanced_sample_results.json \
  --limit 5
```

## 输入文件格式

脚本支持处理包含以下字段的JSON文件：
- `question`: 问题文本
- `system_answer`: 系统回答
- `ground_truth_answer`: 标准答案
- `original_id`: 原始文档ID
- `retrieved_chunk_ids`: 检索到的文本块ID列表

## 输出统计信息

脚本运行完成后会显示统计信息：
- 总结果数
- 答案正确数和正确率
- 检索成功数和成功率

## 依赖要求

- Python 3.7+
- aiohttp
- asyncio
- 运行在8001端口的vLLM服务

## 配置

### vLLM API配置
```python
VLLM_API_URL = "http://localhost:8001/v1/chat/completions"
EVALUATION_MODEL = "./models/Qwen/Qwen3-30B-A3B-FP8"
```

### 批处理配置
- 默认批处理大小：10个结果项
- 请求超时：300秒
- 连接池限制：100个连接

## 注意事项

1. 确保vLLM服务正在8001端口运行
2. 模型名称需要与实际部署的模型匹配
3. 大文件处理可能需要较长时间
4. 建议先使用--limit参数进行小规模测试

## 错误处理

脚本包含完善的错误处理机制：
- API请求失败时会记录错误状态
- JSON解析错误时会保留原始响应
- 网络异常时会继续处理其他项目
- 所有错误都会在最终结果中标记

## 性能优化

- 使用异步并发处理提高效率
- 批量处理避免内存溢出
- 连接池复用减少网络开销
- 详细的进度日志便于监控