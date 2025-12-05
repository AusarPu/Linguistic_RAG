#!/usr/bin/env python3
"""
TriviaQA数据集格式转换脚本
将TriviaQA数据集转换为统一格式：{id, question, answer, context}
"""

import argparse
import json
import os
from pathlib import Path
import sys

# 导入统一配置
sys.path.append(str(Path(__file__).parent))
from convert_all import get_output_dir, get_output_filename


def _gather_triviaqa_context(sample) -> str:
    """从TriviaQA原始样本中汇总上下文文本。
    优先使用 entity_pages.wiki_context 与 search_results.search_context，标题作为补充。
    """
    texts = []

    entity_pages = sample.get("entity_pages") or {}
    if isinstance(entity_pages, dict):
        wiki_ctx = entity_pages.get("wiki_context") or []
        if isinstance(wiki_ctx, list):
            # 确保只处理字符串类型的数据
            texts.extend([item for item in wiki_ctx if isinstance(item, str)])
        titles = entity_pages.get("title") or []
        if isinstance(titles, list):
            # 确保只处理字符串类型的数据
            texts.extend([item for item in titles if isinstance(item, str)])

    search_results = sample.get("search_results") or {}
    if isinstance(search_results, dict):
        search_ctx = search_results.get("search_context") or []
        if isinstance(search_ctx, list):
            # 确保只处理字符串类型的数据
            texts.extend([item for item in search_ctx if isinstance(item, str)])
        sr_titles = search_results.get("title") or []
        if isinstance(sr_titles, list):
            # 确保只处理字符串类型的数据
            texts.extend([item for item in sr_titles if isinstance(item, str)])

    # 过滤空白字符串
    texts = [t.strip() for t in texts if t.strip()]
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
    
    # 处理 answer 字段可能是字典的情况
    if isinstance(ans, dict):
        # 如果 answer 是字典，尝试获取 value 字段
        ans = ans.get('value', '') or ans.get('text', '') or ""
    
    answers_list = sample.get('answers')
    if not ans and isinstance(answers_list, list) and answers_list:
        first_answer = answers_list[0]
        if isinstance(first_answer, str):
            ans = first_answer
        elif isinstance(first_answer, dict):
            ans = first_answer.get('value', '') or first_answer.get('text', '') or ""
        else:
            ans = ""

    context_text = _gather_triviaqa_context(sample)

    return {
        "id": sample.get('id') or sample.get('question_id', ''),
        "question": sample.get('question', ''),
        "answer": ans,
        "context": context_text
    }


def convert_triviaqa_dataset(input_file, output_file, max_samples=None, filter_no_answer=True):
    """
    转换整个TriviaQA数据集文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        max_samples: 最大样本数量，None表示不限制
        filter_no_answer: 是否过滤没有答案的数据
    """
    print(f"正在转换 {input_file} -> {output_file}")
    if max_samples:
        print(f"限制样本数量: {max_samples}")
    if filter_no_answer:
        print("过滤模式: 丢弃没有答案的数据")
    
    # TriviaQA 验证集为 NDJSON，每行一个样本，改用逐行流式处理
    from streaming_processor import process_data_streaming
    
    converted_samples, stats = process_data_streaming(
        input_file, 
        convert_triviaqa_sample, 
        max_samples, 
        filter_no_answer
    )
    
    # 输出统计信息
    print(f"处理了 {stats.get('total_processed', 0)} 个样本")
    print(f"成功转换 {stats.get('converted_count', 0)} 个样本")
    if filter_no_answer:
        print(f"过滤掉 {stats.get('filtered_count', 0)} 个没有答案的样本")
    if stats.get('shortage'):
        print(f"注意: 数据不足，缺少 {stats['shortage']} 个样本")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_samples, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成: {len(converted_samples)} 个样本")


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="转换TriviaQA数据集")
    parser.add_argument('--max-samples', type=int, default=None, help='最大样本数量')
    parser.add_argument('--filter-no-answer', action='store_true', help='过滤没有答案的数据')
    return parser.parse_args()


def main():
    """
    主函数：转换TriviaQA数据集
    """
    args = parse_args()
    
    # 设置路径
    base_dir = Path("/home/pushihao/RAG/Reports/experiments")
    input_dir = base_dir / "datasets" / "triviaqa"
    
    # 使用统一配置的输出目录
    output_dir = Path(get_output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换训练集
    input_file = input_dir / "train.json"
    
    # 使用统一配置的文件名
    output_file = output_dir / get_output_filename("triviaqa")
    
    if input_file.exists():
        convert_triviaqa_dataset(
            input_file, 
            output_file, 
            max_samples=args.max_samples,
            filter_no_answer=args.filter_no_answer
        )
    else:
        print(f"警告: 输入文件不存在: {input_file}")
    
    print("TriviaQA训练集转换完成！")


if __name__ == "__main__":
    main()