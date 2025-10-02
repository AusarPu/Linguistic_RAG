#!/usr/bin/env python3
"""
Natural Questions数据集格式转换脚本
将Natural Questions数据集转换为统一格式：{id, question, answer, context}
"""

import argparse
import json
import os
from pathlib import Path
import re
import html
import sys

# 导入统一配置
sys.path.append(str(Path(__file__).parent))
from convert_all import get_output_dir, get_output_filename

try:
    from bs4 import BeautifulSoup  # 可选依赖
except ImportError:
    BeautifulSoup = None


def _html_to_text(html_str: str) -> str:
    """将HTML内容转换为纯文本。
    优先使用BeautifulSoup（如已安装），否则回退到正则去标签与HTML实体反解。
    """
    if not isinstance(html_str, str):
        return ""
    text = ""
    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html_str, "lxml")
            text = soup.get_text(" ")
        except Exception:
            text = re.sub(r"<[^>]+>", " ", html_str)
    else:
        text = re.sub(r"<[^>]+>", " ", html_str)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_answer(sample):
    """Extract answer from annotations."""
    annotations = sample.get("annotations", {})
    
    # Natural Questions has multiple annotators, so annotations are lists
    short_answers = annotations.get("short_answers", [])
    long_answers = annotations.get("long_answer", [])
    
    # Try to find a valid short answer from any annotator
    for short_answer in short_answers:
        if isinstance(short_answer, dict) and "text" in short_answer:
            texts = short_answer["text"]
            if texts and len(texts) > 0 and texts[0]:  # Check if text list is not empty
                return texts[0]
    
    # If no short answer, try to extract from long answer
    for long_answer in long_answers:
        if isinstance(long_answer, dict) and long_answer.get("candidate_index", -1) >= 0:
            # Get the text from document using byte positions
            document_html = sample["document"]["html"]
            start_byte = long_answer.get("start_byte", 0)
            end_byte = long_answer.get("end_byte", len(document_html))
            if start_byte >= 0 and end_byte > start_byte:
                answer_html = document_html[start_byte:end_byte]
                answer_text = _html_to_text(answer_html)
                # Return first 200 characters to avoid too long answers
                return answer_text[:200].strip()
    
    return ""

def convert_natural_questions_sample(sample):
    """
    转换单个Natural Questions样本为统一格式
    
    Args:
        sample: Natural Questions原始样本
        
    Returns:
        dict: 统一格式的样本
    """
    sample_id = sample.get("id", "")
    
    # Extract question text
    question_data = sample.get("question", {})
    question_text = question_data.get("text", "") if isinstance(question_data, dict) else ""
    
    # Extract answer
    answer_text = _extract_answer(sample)
    
    # Extract HTML content from document
    document = sample.get("document")
    doc_html = ""
    if isinstance(document, dict):
        doc_html = document.get("html", "") or ""
    context_text = _html_to_text(doc_html)

    return {
        "id": sample_id,
        "question": question_text,
        "answer": answer_text,
        "context": context_text
    }


def convert_natural_questions_dataset(input_file, output_file, max_samples=None, filter_no_answer=True):
    """
    转换整个Natural Questions数据集文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        max_samples: 最大样本数量，None表示不限制
        filter_no_answer: 是否过滤没有答案的数据
    """
    import random
    
    print(f"正在转换 {input_file} -> {output_file}")
    if max_samples:
        print(f"限制样本数量: {max_samples}")
    if filter_no_answer:
        print("过滤模式: 丢弃没有答案的数据")
    
    # 首先读取所有样本
    all_samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                all_samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析失败: {e}")
                continue
            except Exception as e:
                print(f"警告: 第{line_num}行处理失败: {e}")
                continue
    
    print(f"总共读取 {len(all_samples)} 个样本")
    
    # 转换样本并过滤
    converted_samples = []
    processed_count = 0
    filtered_count = 0
    
    for sample in all_samples:
        try:
            converted_sample = convert_natural_questions_sample(sample)
            
            # 检查是否需要过滤没有答案的数据
            if filter_no_answer and (not converted_sample["answer"] or converted_sample["answer"].strip() == ""):
                filtered_count += 1
                continue
                
            converted_samples.append(converted_sample)
            processed_count += 1
            
            # 如果达到最大样本数，停止处理
            if max_samples and processed_count >= max_samples:
                break
                
        except Exception as e:
            print(f"警告: 样本处理失败: {e}")
            continue
    
    print(f"成功转换 {len(converted_samples)} 个样本")
    if filter_no_answer:
        print(f"过滤掉 {filtered_count} 个没有答案的样本")
    
    # 写入转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_samples, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成: {len(converted_samples)} 个样本")


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="转换Natural Questions数据集")
    parser.add_argument('--max-samples', type=int, default=None, help='最大样本数量')
    parser.add_argument('--filter-no-answer', action='store_true', help='过滤没有答案的数据')
    return parser.parse_args()


def main():
    """
    主函数：转换Natural Questions数据集
    """
    args = parse_args()
    
    # 设置路径
    base_dir = Path("/home/pushihao/RAG/Reports/experiments")
    input_dir = base_dir / "datasets" / "natural_questions"
    
    # 使用统一配置的输出目录
    output_dir = Path(get_output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换验证集
    input_file = input_dir / "validation.json"
    
    # 使用统一配置的文件名
    output_file = output_dir / get_output_filename("natural_questions")
    
    if input_file.exists():
        convert_natural_questions_dataset(
            input_file, 
            output_file, 
            max_samples=args.max_samples,
            filter_no_answer=args.filter_no_answer
        )
    else:
        print(f"警告: 输入文件不存在: {input_file}")
    
    print("Natural Questions验证集转换完成！")


if __name__ == "__main__":
    main()