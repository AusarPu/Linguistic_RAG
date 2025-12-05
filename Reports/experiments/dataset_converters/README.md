# 数据集格式转换器

本目录包含将四个数据集转换为统一格式的脚本。

## 统一格式说明

所有数据集将被转换为包含以下四个字段的JSON格式：

```json
{
    "id": "样本唯一标识符",
    "question": "问题文本",
    "answer": "答案文本", 
    "context": "上下文信息"
}
```

## 转换脚本

### 单独运行

每个数据集都有独立的转换脚本：

1. **HotpotQA**: `convert_hotpotqa.py`
   - 将复杂的多文档context结构合并为单一字符串
   - 保留文档标题和句子编号信息

2. **MS MARCO**: `convert_msmarco.py`
   - 提取第一个答案作为标准答案
   - 将多个passages合并，标记选中状态
   - 包含原始URL信息

3. **Natural Questions**: `convert_natural_questions.py`
   - 直接映射，结构已经接近统一格式

4. **TriviaQA**: `convert_triviaqa.py`
   - 直接映射，结构已经接近统一格式

### 批量运行

使用 `convert_all.py` 可以一次性转换所有数据集：

```bash
cd /home/pushihao/RAG/Reports/experiments/dataset_converters
python convert_all.py
```

## 输入输出路径

### 输入路径
- HotpotQA: `/home/pushihao/RAG/Reports/experiments/datasets/hotpotqa/`
- MS MARCO: `/home/pushihao/RAG/Reports/experiments/datasets/ms_marco/`
- Natural Questions: `/home/pushihao/RAG/Reports/experiments/datasets/natural_questions/`
- TriviaQA: `/home/pushihao/RAG/Reports/experiments/datasets/triviaqa/`

### 输出路径
转换后的数据将保存在：
`/home/pushihao/RAG/Reports/experiments/dataset_converters/converted/`

每个数据集的输出结构：
```
converted/
├── hotpotqa/
│   ├── train_converted.json
│   └── validation_converted.json
├── ms_marco/
│   ├── train_converted.json
│   └── validation_converted.json
├── natural_questions/
│   ├── train_converted.json
│   └── validation_converted.json
└── triviaqa/
    ├── train_converted.json
    └── validation_converted.json
```

## 使用示例

### 转换单个数据集

```bash
# 转换HotpotQA
python convert_hotpotqa.py

# 转换MS MARCO
python convert_msmarco.py

# 转换Natural Questions
python convert_natural_questions.py

# 转换TriviaQA
python convert_triviaqa.py
```

### 转换所有数据集

```bash
python convert_all.py
```

## 转换细节

### HotpotQA转换
- 将`context`中的多个文档合并为单一字符串
- 每个文档以"Document N: 标题"开头
- 句子按编号列出
- 文档间用空行分隔

### MS MARCO转换
- 优先使用`answers`字段的第一个答案
- 如果`answers`为空，则使用`wellFormedAnswers`
- 将所有passages合并，标记[SELECTED]或[CANDIDATE]
- 包含原始URL信息

### Natural Questions & TriviaQA转换
- 这两个数据集结构已经接近统一格式
- 主要进行字段映射和数据清理

## 错误处理

- 脚本会跳过JSON解析失败的行
- 处理异常时会输出警告信息
- 继续处理剩余数据，不会因单个样本错误而中断

## 注意事项

1. 确保输入数据集文件存在
2. 输出目录会自动创建
3. 转换过程中会显示进度信息
4. 建议在转换前备份原始数据