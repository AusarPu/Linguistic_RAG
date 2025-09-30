#!/usr/bin/env python3
"""
TriviaQA数据集格式转换脚本
将TriviaQA数据集转换为统一格式：{id, question, answer, context}
"""

import json
import os
from pathlib import Path


def _gather_triviaqa_context(sample) -> str:
    """从TriviaQA原始样本中汇总上下文文本。
    优先使用 entity_pages.wiki_context 与 search_results.search_context，标题作为补充。
    """
    texts = []

    entity_pages = sample.get("entity_pages") or {}
    if isinstance(entity_pages, dict):
        wiki_ctx = entity_pages.get("wiki_context") or []
        if isinstance(wiki_ctx, list):
            texts.extend(wiki_ctx)
        titles = entity_pages.get("title") or []
        if isinstance(titles, list):
            texts.extend(titles)

    search_results = sample.get("search_results") or {}
    if isinstance(search_results, dict):
        search_ctx = search_results.get("search_context") or []
        if isinstance(search_ctx, list):
            texts.extend(search_ctx)
        sr_titles = search_results.get("title") or []
        if isinstance(sr_titles, list):
            texts.extend(sr_titles)

    # 过滤非字符串与空白
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    # 去重保序
    seen = set()
    uniq = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return "\n\n".join(uniq)


def convert_triviaqa_sample(sample):
    """
    转换单个TriviaQA样本为统一格式
    
    Args:
        sample: TriviaQA原始样本
        
    Returns:
        dict: 统一格式的样本
    """
    # TriviaQA 原始：有 question / question_id，无直接 context
    # context 从 entity_pages.wiki_context / search_results.search_context 汇总
    # id 使用 question_id（或回退 id）
    # answer 兼容 answer / answer_text / answers 数组
    ans = sample.get('answer') or sample.get('answer_text') or ""
    answers_list = sample.get('answers')
    if not ans and isinstance(answers_list, list) and answers_list:
        ans = answers_list[0] if isinstance(answers_list[0], str) else ""

    context_text = _gather_triviaqa_context(sample)

    return {
        "id": sample.get('id') or sample.get('question_id', ''),
        "question": sample.get('question', ''),
        "answer": ans,
        "context": context_text
    }


def convert_triviaqa_dataset(input_file, output_file, max_samples=None):
    """
    转换整个TriviaQA数据集文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        max_samples: 最大样本数量，None表示不限制
    """
    print(f"正在转换 {input_file} -> {output_file}")
    if max_samples:
        print(f"限制样本数量: {max_samples}")
    
    # 如果需要限制样本数量，需要按question_id分组采样
    if max_samples:
        from collections import defaultdict
        
        # 先读取所有样本并按question_id分组
        question_groups = defaultdict(list)
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    question_id = sample.get('question_id', f'unknown_{line_num}')
                    question_groups[question_id].append(sample)
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON解析失败: {e}")
                    continue
                except Exception as e:
                    print(f"警告: 第{line_num}行处理失败: {e}")
                    continue
        
        print(f"读取了 {len(question_groups)} 个唯一问题ID")
        
        # 按question_id顺序选择，确保每个选中的question_id的所有样本都被包含
        selected_samples = []
        current_count = 0
        
        for question_id, samples in list(question_groups.items()):
            if current_count + len(samples) <= max_samples:
                selected_samples.extend(samples)
                current_count += len(samples)
            else:
                # 如果加上这个问题的所有样本会超过限制，就停止
                break
        
        print(f"选择了 {len(selected_samples)} 个样本（来自 {len(set(s.get('question_id') for s in selected_samples))} 个问题ID）")
        
        # 转换选中的样本
        converted_samples = []
        for sample in selected_samples:
            try:
                converted_sample = convert_triviaqa_sample(sample)
                converted_samples.append(converted_sample)
            except Exception as e:
                print(f"警告: 样本处理失败: {e}")
                continue
    else:
        # 原有逻辑：处理所有样本
        converted_samples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    converted_sample = convert_triviaqa_sample(sample)
                    converted_samples.append(converted_sample)
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON解析失败: {e}")
                    continue
                except Exception as e:
                    print(f"警告: 第{line_num}行处理失败: {e}")
                    continue
    
    # 写入转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in converted_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"转换完成: {len(converted_samples)} 个样本")


def main():
    """
    主函数：转换TriviaQA数据集
    """
    # 设置路径
    base_dir = Path("/home/pushihao/RAG/Reports/experiments")
    input_dir = base_dir / "datasets" / "triviaqa"
    output_dir = base_dir / "dataset_converters" / "converted" / "triviaqa"
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换训练集和验证集
    datasets = {
        "train.json": "train_converted.json",
        "validation.json": "validation_converted_1k.json"  # 验证集限制1000个样本
    }
    
    for input_name, output_name in datasets.items():
        input_file = input_dir / input_name
        output_file = output_dir / output_name
        
        if input_file.exists():
            # 验证集限制1000个样本
            max_samples = 1000 if input_name == "validation.json" else None
            convert_triviaqa_dataset(input_file, output_file, max_samples=max_samples)
        else:
            print(f"警告: 输入文件不存在: {input_file}")
    
    print("TriviaQA数据集转换完成！")


if __name__ == "__main__":
    main()