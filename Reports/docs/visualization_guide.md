# RAG系统实验结果可视化指导

## 概述
本文档详细说明了RAG系统实验论文所需的所有图表类型、数据来源、绘图库选择和实现方法。

## 图表分类

### 一、系统架构与设计图表

#### 1. RAG系统架构图
- **图形类型**: 流程图/架构图
- **用途**: 展示RAG系统的整体架构和各组件关系
- **绘图库**: `matplotlib` + `networkx` 或 `graphviz`
- **数据来源**: 无需实验数据，基于系统设计绘制
- **关键元素**: 
  - 查询重写模块
  - 多路检索模块（BM25、Dense、问题相似性）
  - 有用性判断模块
  - 软保留策略
  - 生成模块

#### 2. 实验设计总览图
- **图形类型**: 树状图或层次结构图
- **用途**: 展示整个消融实验的设计逻辑和层次关系
- **绘图库**: `matplotlib` + `networkx` 或 `graphviz`
- **数据来源**: 实验配置信息
- **内容**: 
  - 完整系统（final_result_2）作为根节点
  - 各个消融实验作为分支
  - 清晰标注每个实验移除的组件

### 二、消融实验对比图表

#### 3. 消融实验性能对比图
- **图形类型**: 分组柱状图
- **用途**: 展示不同组件对系统性能的影响
- **绘图库**: `matplotlib` 或 `seaborn`
- **数据来源**: 
  ```
  final_result_2_preprocess_think/evaluation_summary.txt (完整系统-原版)
  final_result_8_preprocess_think_usefulness_v2/evaluation_summary.txt (完整系统-优化版)
  final_result_3_preprocess_think_no_rewriter/evaluation_summary.txt (无查询重写)
  final_result_4_preprocess_think_no_usefulness/evaluation_summary.txt (无有用性判断)
  final_result_5_preprocess_think_no_dense_chunks/evaluation_summary.txt (无密集块检索)
  final_result_6_preprocess_think_no_dense_keywords/evaluation_summary.txt (无密集关键词)
  final_result_7_preprocess_think_no_dense_questions/evaluation_summary.txt (无密集问题)
  ```
- **指标**: Accuracy, F1-Score, BLEU, ROUGE等

#### 3.1. 版本对比图表（新增）
- **图形类型**: 并排柱状图和改进仪表盘
- **用途**: 对比原版和优化版完整系统的性能差异
- **绘图库**: `matplotlib` 或 `seaborn`
- **数据来源**: 
  ```
  final_result_2_preprocess_think/evaluation_summary.txt (完整系统-原版)
  final_result_8_preprocess_think_usefulness_v2/evaluation_summary.txt (完整系统-优化版)
  ```
- **生成的图表**:
  - `version_comparison_en.png`: 总体性能对比
  - `dataset_version_comparison_en.png`: 各数据集性能对比
  - `improvement_summary_dashboard_en.png`: 改进总结仪表盘
- **关键指标**: 总体准确率、F1分数、检索成功率的改进幅度

#### 4. 组件贡献度总结图
- **图形类型**: 堆叠柱状图或瀑布图
- **用途**: 量化展示每个组件对整体性能的贡献
- **绘图库**: `matplotlib` 或 `plotly`
- **数据处理**: 
  - 以完整系统为基准(100%)
  - 计算移除各组件后的性能下降幅度
  - 展示累积效应

### 三、数据集对比图表

#### 5. 数据集性能对比图
- **图形类型**: 分组柱状图
- **用途**: 展示在不同数据集上的性能表现
- **绘图库**: `matplotlib` 或 `seaborn`
- **数据来源**: 各final_result文件夹中的4个数据集结果
  - hotpotqa/advanced_sample_results.json
  - ms_marco/advanced_sample_results.json
  - natural_questions/advanced_sample_results.json
  - triviaqa/advanced_sample_results.json

#### 6. 实验结论矩阵图
- **图形类型**: 热力图矩阵
- **用途**: 展示不同数据集×不同配置的性能表现全景
- **绘图库**: `seaborn`
- **维度**:
  - X轴: 6种实验配置
  - Y轴: 4个数据集
  - 颜色深度: 性能指标值

### 四、性能趋势与分布图表

#### 7. 性能指标趋势图
- **图形类型**: 折线图
- **用途**: 展示不同配置下各项指标的变化趋势
- **绘图库**: `matplotlib`
- **数据来源**: 从各evaluation_summary.txt中提取指标

#### 8. 检索效果分布图
- **图形类型**: 箱线图或小提琴图
- **用途**: 展示检索质量分布和软保留策略效果
- **绘图库**: `matplotlib` 或 `seaborn`
- **数据来源**: advanced_sample_results.json中的详细结果

### 五、总结性图表

#### 9. 核心发现汇总仪表盘
- **图形类型**: 综合仪表盘 (Dashboard)
- **用途**: 一张图展示所有关键指标和主要结论
- **绘图库**: `matplotlib` 子图组合或 `plotly`
- **布局**:
  - 左上：最佳配置的性能指标
  - 右上：各组件重要性排序
  - 左下：数据集适应性对比
  - 右下：关键数值摘要

#### 10. 综合性能雷达图
- **图形类型**: 多维雷达图
- **用途**: 多角度对比完整系统与简化版本
- **绘图库**: `matplotlib`
- **维度**:
  - 准确性、召回率、F1分数
  - 不同数据集的适应性

#### 11. 最优策略推荐流程图
- **图形类型**: 决策树或流程图
- **用途**: 基于实验结果给出不同场景下的最优配置建议
- **绘图库**: `graphviz` 或 `matplotlib`
- **内容**:
  - 根据数据集特点推荐配置
  - 根据性能要求推荐配置

## 数据处理流程

### 数据提取步骤
1. **解析evaluation_summary.txt文件**
   ```python
   import re
   import json
   
   def parse_summary_file(file_path):
       # 提取总体指标
       pass
   ```

2. **解析advanced_sample_results.json文件**
   ```python
   def parse_detailed_results(file_path):
       # 提取样本级详细结果
       pass
   ```

3. **数据标准化处理**
   ```python
   def normalize_metrics(data):
       # 统一指标名称和格式
       pass
   ```

### 数据文件结构
```
Reports/experiments/datasets/
├── final_result_2_preprocess_think/
│   ├── evaluation_summary.txt
│   ├── hotpotqa/advanced_sample_results.json
│   ├── ms_marco/advanced_sample_results.json
│   ├── natural_questions/advanced_sample_results.json
│   └── triviaqa/advanced_sample_results.json
├── final_result_3_preprocess_think_no_rewriter/
│   └── ... (同样结构)
├── final_result_4_preprocess_think_no_usefulness/
│   └── ... (同样结构)
├── final_result_5_preprocess_think_no_dense_chunks/
│   └── ... (同样结构)
├── final_result_6_preprocess_think_no_dense_keywords/
│   └── ... (同样结构)
└── final_result_7_preprocess_think_no_dense_questions/
    └── ... (同样结构)
```

## 技术栈要求

### 必需的Python库
```python
# 基础数据处理
import pandas as pd
import numpy as np
import json
import re

# 绘图库
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# 网络图和流程图
import networkx as nx
import graphviz

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

### 推荐的绘图配置
```python
# 设置绘图风格
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 图片保存配置
save_config = {
    'dpi': 300,
    'bbox_inches': 'tight',
    'format': 'png'
}
```

## 实现优先级

### 高优先级（核心图表）
1. RAG系统架构图
2. 消融实验性能对比图
3. 数据集性能对比图
4. 核心发现汇总仪表盘

### 中优先级（分析图表）
5. 组件贡献度总结图
6. 实验结论矩阵图
7. 综合性能雷达图

### 低优先级（补充图表）
8. 实验设计总览图
9. 性能指标趋势图
10. 检索效果分布图
11. 最优策略推荐流程图

## 输出规范

### 文件命名规范
- 架构图: `rag_system_architecture.png`
- 消融实验: `ablation_study_comparison.png`
- 数据集对比: `dataset_performance_comparison.png`
- 汇总仪表盘: `summary_dashboard.png`
- 组件贡献: `component_contribution.png`
- 矩阵热力图: `performance_heatmap.png`
- 雷达图: `comprehensive_radar_chart.png`

### 图片规格
- 分辨率: 300 DPI
- 格式: PNG (论文用) / SVG (可编辑)
- 尺寸: 适合论文双栏布局

## 注意事项

1. **数据一致性**: 确保所有图表使用相同的指标定义和计算方法
2. **颜色方案**: 使用色盲友好的颜色方案
3. **字体大小**: 确保在论文中缩放后仍然清晰可读
4. **图例说明**: 每个图表都要有清晰的图例和标题
5. **数据标注**: 在关键数据点添加数值标注
6. **统计显著性**: 在对比图中标注统计显著性检验结果

## 后续扩展

根据论文审稿意见，可能需要补充的图表：
- 计算复杂度对比图
- 响应时间分析图
- 资源消耗对比图
- 错误案例分析图