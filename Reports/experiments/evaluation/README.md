# RAG系统数据集评估脚本

本目录包含用于评估RAG系统在多个数据集上性能的脚本。

## 脚本说明

### 1. evaluate_datasets.py ⭐
- **功能**: 主要评估脚本，支持问题级别的并发处理
- **特点**: 批量并发处理，显著提升性能，自动区分示例/完整模式
- **适用**: 所有评估场景，推荐使用

### 2. summarize_evaluation_results.py
- **功能**: 结果统计分析脚本
- **特点**: 分析生成的JSON文件，提供统计信息，支持新旧文件格式
- **适用**: 结果分析和报告生成

## 主要特性

### 🚀 性能优势
- **问题级别并发**: 同一批次内的问题并行处理
- **批量处理**: 可配置批处理大小，平衡性能和资源占用
- **数据集并发**: 可选择是否同时处理多个数据集
- **资源控制**: 通过批大小控制并发度，避免资源过载
- **智能模式**: 自动判断示例模式（≤100问题）或完整模式

### 🔧 技术实现
- **异步处理**: 使用asyncio实现高效并发
- **错误隔离**: 单个问题失败不影响其他问题
- **进度保存**: 每批次完成后自动保存结果
- **线程安全**: 使用锁保护共享资源
- **智能命名**: 根据问题数量自动选择文件名

### 📊 性能对比
以6个问题为例的测试结果：
- **旧串行版本**: 约120-150秒
- **新并发版本**: 约69秒 (批大小=2)
- **性能提升**: 约50%+

## 使用方法

### 基础用法
```bash
# 示例模式（自动检测，≤100问题）
python evaluate_datasets.py --dataset hotpotqa --batch-size 3 --max-questions 50

# 完整模式（处理所有问题）
python evaluate_datasets.py --dataset hotpotqa --batch-size 3
```

### 高级用法
```bash
# 处理所有数据集，批大小为3
python evaluate_datasets.py --dataset all --batch-size 3 --max-questions 100

# 启用数据集级别并发（需要更多资源）
python evaluate_datasets.py --dataset all --batch-size 2 --dataset-concurrent

# 完整评估（处理所有问题）
python evaluate_datasets.py --dataset hotpotqa --batch-size 5
```

### 参数说明
- `--dataset`: 要评估的数据集 (hotpotqa/ms_marco/natural_questions/triviaqa/all)
- `--batch-size`: 并发批处理大小 (默认: 3)
- `--max-questions`: 限制处理的问题数量 (默认: 50，用于示例模式)
- `--dataset-concurrent`: 启用数据集级别并发

## 输出格式和命名

### 文件命名规则
- **示例结果**: `sample_results.json` (≤100问题)
- **完整结果**: `evaluation_results.json` (>100问题或无限制)

### JSON格式
```json
{
  "question": "原始问题",
  "retrieved_chunk_ids": ["检索到的知识库内容ID列表"],
  "system_answer": "系统生成的回答（不包含思考过程）",
  "original_id": "原始数据集中的问题ID",
  "ground_truth_answer": "标准答案"
}
```

## 输出路径
- hotpotqa: `/home/pushihao/RAG/Reports/experiments/rag_evaluation_results/hotpotqa/`
- ms_marco: `/home/pushihao/RAG/Reports/experiments/rag_evaluation_results/ms_marco/`
- natural_questions: `/home/pushihao/RAG/Reports/experiments/rag_evaluation_results/natural_questions/`
- triviaqa: `/home/pushihao/RAG/Reports/experiments/rag_evaluation_results/triviaqa/`

## 性能调优建议

### 批大小选择
- **小批量 (2-3)**: 适合资源有限的环境，稳定性好
- **中批量 (4-6)**: 平衡性能和资源占用，推荐设置
- **大批量 (7+)**: 需要充足的GPU/API资源，最高性能

### 资源监控
- 监控GPU显存使用率
- 观察API响应时间
- 注意系统内存占用

### 故障处理
- 单个问题失败不会影响整个批次
- 每批次完成后自动保存结果
- 支持中断后继续处理

## 注意事项

1. **资源要求**: 并发处理需要更多GPU/API资源
2. **稳定性**: 批大小过大可能导致资源竞争
3. **调试**: 出现问题时可以降低批大小
4. **兼容性**: 确保vLLM服务正常运行且支持并发请求
5. **文件兼容**: 统计脚本支持新旧文件格式，确保向后兼容

## 结果分析
```bash
# 查看评估结果统计
python summarize_evaluation_results.py
```

这将显示各数据集的：
- 总问题数和成功率
- 平均检索块数
- 平均回答长度
- 示例问题预览
- 自动识别新旧文件格式

## 更新说明

### v2.0 更新内容
- ✅ 移除了串行版本脚本
- ✅ 统一使用并发版本作为主脚本
- ✅ 智能文件命名（示例/完整模式自动区分）
- ✅ 向后兼容旧文件格式
- ✅ 更简洁的命令行接口