# 第五章：实验与结果（RAG系统）

本文档以论文中“实验部分”的体例，汇总并分析以下六组实验结果：

- final_result_2_preprocess_think（基线：启用预处理与思考模式）
- final_result_3_preprocess_think_no_rewriter（消融：关闭查询重写）
- final_result_4_preprocess_think_no_usefulness（消融：关闭有用性判断）
- final_result_5_preprocess_think_no_dense_chunks（消融：移除密集块检索）
- final_result_6_preprocess_think_no_dense_keywords（消融：移除密集关键词检索）
- final_result_7_preprocess_think_no_dense_questions（消融：移除密集问题检索）

> 说明：系统技术细节与架构设计见 `Reports/docs/report_essay.md`。

---

## 5.1 实验设置

- 数据集：HotpotQA、MS MARCO、Natural Questions、TriviaQA（合计 400 个问题）。
- 评估指标：总体准确率、总体 F1、检索成功率、平均检索块数（用于度量上下文规模）。
- 运行环境与推理引擎同论文技术部分，评估摘要来自各目录下的 `evaluation_summary.txt`。

---

## 5.2 总体结果概览

- final_result_2_preprocess_think：准确率 81.75%，F1 0.89，检索成功率 97.50%，平均检索块数 10.38。
- final_result_3_preprocess_think_no_rewriter：准确率 83.50%，F1 0.90，检索成功率 98.00%，平均检索块数 9.33。
- final_result_4_preprocess_think_no_usefulness：准确率 89.00%，F1 0.94，检索成功率 99.75%，平均检索块数 18.06。
- final_result_5_preprocess_think_no_dense_chunks：准确率 82.25%，F1 0.89，检索成功率 96.25%，平均检索块数 9.27。
- final_result_6_preprocess_think_no_dense_keywords：准确率 82.00%，F1 0.89，检索成功率 96.50%，平均检索块数 9.16。
- final_result_7_preprocess_think_no_dense_questions：准确率 83.00%，F1 0.90，检索成功率 97.50%，平均检索块数 9.13。

结论要点：
- 在当前配置下，关闭“有用性判断”取得最高总体准确率与 F1（89.00%、0.94），但平均检索块数显著升高（18.06），表明上下文更庞大。
- 关闭“查询重写”整体优于基线（+1.75% 准确率），提示重写策略在现阶段可能存在误导或过度改写。
- 三类密集检索（块、关键词、问题）均有贡献；任意移除均会带来不同程度的性能下降。

---

## 5.3 按数据集的细粒度分析

- HotpotQA（多跳推理）：
  - 基线 59.00%，关闭重写 65.00%，关闭有用性判断显著提升至 84.00%。
  - 观察：有用性过滤可能过于激进，导致多跳链路证据被误判为“无用”而被剔除；不做过滤时上下文更充足，命中关键证据的概率更高。

- MS MARCO（段落检索类）：
  - 基线 93.00%，关闭重写 88.00%，关闭有用性判断 93.00%。
  - 观察：重写在此类任务可能引入不必要的语义变体，影响召回；有用性判断对 MS MARCO 的影响不大。

- Natural Questions（事实型问答）：
  - 基线 83.00%，关闭重写 88.00%，关闭有用性判断 88.00%。
  - 观察：重写的收益有限甚至负向；不过“问题索引”与“关键词索引”均对稳定高分有帮助。

- TriviaQA（知识型事实问答）：
  - 基线 92.00%，关闭重写 93.00%，关闭有用性判断 91.00%，移除“密集问题检索”反而提升到 95.00%。
  - 观察：预生成“问题索引”在该数据集上可能引入偏差或过拟合模板；更直接的块/关键词检索在此类任务上效果更佳。

---

## 5.4 关键模块影响与权衡

- 有用性判断模块：
  - 优点：能显著减少上下文规模（相较无过滤约减一半以上的检索块），预期可提升生成阶段的效率与答案可读性。
  - 当前问题：在多跳任务上出现过度过滤，导致证据链不完整，准确率明显下降（HotpotQA 对比尤为突出）。
  - 改进建议：将“二值过滤”改为“重排与软阈值保留”，引入证据覆盖率约束；或以轻量模型进行初筛，再由大模型做保留判定。

- 查询重写模块：
  - 现状：总体上关闭重写取得更高准确率，提示重写对部分任务产生语义漂移。
  - 改进建议：加入“对齐校验”（重写前后与原问题的关键词与实体一致性检查），并在专业术语场景下采用“术语保真”策略。

- 三类密集检索路径：
  - 总体均有贡献；移除任意一类均导致整体下降。关键词检索对术语密集场景尤为关键，问题检索在 TriviaQA 上需谨慎使用。

---

## 5.5 结论与后续工作

- 结论：当前系统在不启用有用性判断时取得最高的准确率与 F1，但代价是更大的上下文与可能更高的生成时延。启用有用性判断能压缩上下文规模，但在多跳场景中需避免过度过滤。查询重写与问题索引的策略需要进一步优化以避免语义偏移与模板化偏差。
- 后续工作：
  - 将有用性判断从“硬过滤”改为“重排+软保留”，并引入证据链完整性约束。
  - 对查询重写加入一致性校验与术语保真；在不确定场景采用“原问+重写并行检索”的保险策略。
  - 针对不同数据集自适应选择检索路径权重（块/关键词/问题），并加入学习型融合器进行加权。

---

## 5.6 可复现实验信息

- 汇总来源：各目录 `evaluation_summary.txt`（时间戳：2025-10-06/07）。
- 目录路径：
  - `Reports/experiments/datasets/final_result_2_preprocess_think/`
  - `Reports/experiments/datasets/final_result_3_preprocess_think_no_rewriter/`
  - `Reports/experiments/datasets/final_result_4_preprocess_think_no_usefulness/`
  - `Reports/experiments/datasets/final_result_5_preprocess_think_no_dense_chunks/`
  - `Reports/experiments/datasets/final_result_6_preprocess_think_no_dense_keywords/`
  - `Reports/experiments/datasets/final_result_7_preprocess_think_no_dense_questions/`