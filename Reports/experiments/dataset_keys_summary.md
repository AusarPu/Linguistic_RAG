# 数据集Key结构说明

## 数据集Key对比表格

| 数据集 | 主要Keys | 数据类型 | 说明 |
|--------|----------|----------|------|
| **HotpotQA** | `id`, `question`, `answer`, `type`, `level`, `supporting_facts`, `context` | 多跳问答 | - `id`: 样本唯一标识符<br>- `question`: 问题文本<br>- `answer`: 答案文本<br>- `type`: 问题类型(comparison/bridge)<br>- `level`: 难度等级(easy/medium/hard)<br>- `supporting_facts`: 支持事实，包含title和sent_id<br>- `context`: 上下文信息，包含title和sentences |
| **MS MARCO** | `query_id`, `query`, `query_type`, `answers`, `passages`, `wellFormedAnswers` | 阅读理解 | - `query_id`: 查询唯一标识符<br>- `query`: 查询问题文本<br>- `query_type`: 查询类型(DESCRIPTION等)<br>- `answers`: 答案列表<br>- `passages`: 段落信息，包含is_selected、passage_text、url<br>- `wellFormedAnswers`: 格式化答案(可能为空) |
| **Natural Questions** | `id`, `question`, `answer`, `context` | 自然问题问答 | - `id`: 样本唯一标识符<br>- `question`: 自然语言问题<br>- `answer`: 答案文本<br>- `context`: 相关上下文信息 |
| **TriviaQA** | `id`, `question`, `answer`, `context` | 知识问答 | - `id`: 样本唯一标识符<br>- `question`: 问题文本<br>- `answer`: 答案文本<br>- `context`: 上下文信息 |

## 详细Key说明

### HotpotQA数据集
- **特点**: 多跳推理问答，需要从多个文档中推理得出答案
- **核心字段**:
  - `supporting_facts`: 关键支持信息，格式为`{'title': [...], 'sent_id': [...]}`
  - `context`: 复杂的上下文结构，包含多个文档的标题和句子
  - `type`: 问题分类(comparison比较型, bridge桥接型)
  - `level`: 三个难度等级(easy, medium, hard)

### MS MARCO数据集
- **特点**: 基于真实搜索查询的阅读理解数据集
- **核心字段**:
  - `passages`: 包含多个候选段落，每个段落有选择标记(`is_selected`)
  - `query_type`: 查询分类(如DESCRIPTION描述型)
  - `url`: 每个段落对应的原始网页链接

### Natural Questions数据集
- **特点**: 基于真实用户搜索查询的问答数据集
- **结构**: 相对简单，主要包含问题、答案和上下文三元组

### TriviaQA数据集
- **特点**: 基于知识问答的数据集
- **结构**: 类似Natural Questions，包含基本的问答对和上下文信息

## 数据集用途对比

| 数据集 | 主要任务 | 复杂度 | 推理类型 |
|--------|----------|--------|----------|
| HotpotQA | 多跳推理问答 | 高 | 多步推理 |
| MS MARCO | 段落检索+阅读理解 | 中 | 信息检索 |
| Natural Questions | 开放域问答 | 中 | 单步推理 |
| TriviaQA | 知识问答 | 中 | 事实查询 |

## 注意事项

1. **HotpotQA**的`context`字段结构最为复杂，包含多个文档的完整信息
2. **MS MARCO**独有`passages`字段，支持多候选段落的检索任务
3. 所有数据集都包含基本的`id`、`question`、`answer`字段
4. **HotpotQA**额外提供了问题分类和难度标注，便于细粒度评估
5. **MS MARCO**提供了原始网页URL，支持溯源验证