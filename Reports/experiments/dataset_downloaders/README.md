# 数据集下载器

这个目录包含了从原始 `dataset_preparation.py` 拆分出来的独立数据集下载器，每个数据集都有自己的下载脚本，并集成了看门狗测速功能。

## 功能特点

- **独立下载器**: 每个数据集都有独立的下载脚本
- **看门狗测速**: 监控下载速度，低于阈值时自动重试
- **环境配置**: 自动配置 Hugging Face 镜像和缓存路径
- **错误处理**: 完善的异常处理和重试机制
- **数据保存**: 下载数据到缓存并保存到指定目录 (`Reports/experiments/datasets`)

## 文件结构

```
dataset_downloaders/
├── __init__.py                      # 包初始化文件
├── base_downloader.py               # 基础下载器类（包含看门狗功能）
├── download_natural_questions.py    # Natural Questions 下载器
├── download_hotpotqa.py             # HotpotQA 下载器
├── download_triviaqa.py             # TriviaQA 下载器
├── download_msmarco.py              # MS MARCO 下载器
├── download_all.py                  # 统一下载脚本
└── README.md                        # 说明文档
```

## 使用方法

### 1. 下载单个数据集

```bash
# 下载 Natural Questions
python download_natural_questions.py

# 下载 HotpotQA
python download_hotpotqa.py

# 下载 TriviaQA
python download_triviaqa.py

# 下载 MS MARCO
python download_msmarco.py
```

### 2. 使用统一下载脚本

```bash
# 列出所有可用数据集
python download_all.py --list

# 下载所有数据集
python download_all.py --all

# 下载指定数据集
python download_all.py --dataset natural_questions
python download_all.py --dataset hotpotqa
python download_all.py --dataset triviaqa
python download_all.py --dataset msmarco
```

### 3. 在代码中使用

```python
from dataset_downloaders import NaturalQuestionsDownloader

# 创建下载器实例
downloader = NaturalQuestionsDownloader()

# 开始下载
success = downloader.download()

if success:
    print("下载成功")
else:
    print("下载失败")
```

## 环境配置

下载器会自动设置以下环境变量：

- `HF_ENDPOINT`: `https://hf-mirror.com` (使用国内镜像)
- `HUGGINGFACE_HUB_CACHE`: `/tmp/huggingface_cache`
- `HF_DATASETS_CACHE`: `/tmp/hf_datasets_cache`

## 数据存储

下载的数据集会保存到以下位置：

- **输出目录**: `/home/pushihao/RAG/Reports/experiments/datasets/`
- **数据格式**: 每个数据集保存为独立的JSON文件
  - `train.json`: 训练集数据
  - `validation.json`: 验证集数据
  - `test.json`: 测试集数据（如果没有验证集）

## 看门狗功能

每个下载器都集成了看门狗功能，具有以下特点：

- **速度监控**: 实时监控下载速度
- **阈值检测**: 当速度低于 1 MiB/s 持续 60 秒时触发重试
- **自动重试**: 最多重试 3 次
- **进程管理**: 自动终止慢速进程并重新启动

## 支持的数据集

| 数据集 | 数据源 | 配置 | 说明 |
|--------|--------|------|------|
| Natural Questions | `google-research-datasets/natural_questions` | 默认 | Google 自然问题数据集 |
| HotpotQA | `hotpot_qa` | `fullwiki` | 多跳推理问答数据集 |
| TriviaQA | `mandarjoshi/trivia_qa` | `rc` 或 `unfiltered` | 知识问答数据集 |
| MS MARCO | `microsoft/ms_marco` | `v2.1`, `v1.1` 或默认 | 微软问答数据集 |

## 注意事项

1. **网络要求**: 需要稳定的网络连接下载大型数据集
2. **存储空间**: 确保 `/tmp` 目录有足够的存储空间
3. **权限要求**: 需要对缓存目录的读写权限
4. **依赖包**: 需要安装 `datasets`, `huggingface_hub` 等依赖包

## 故障排除

### 下载速度慢
- 看门狗会自动检测并重试
- 可以手动调整 `min_speed_bytes` 参数

### 缓存空间不足
- 清理 `/tmp/huggingface_cache` 和 `/tmp/hf_datasets_cache` 目录
- 或修改环境变量指向其他目录

### 网络连接问题
- 检查网络连接
- 确认可以访问 `https://hf-mirror.com`

### 权限问题
- 确保对缓存目录有读写权限
- 必要时使用 `sudo` 运行