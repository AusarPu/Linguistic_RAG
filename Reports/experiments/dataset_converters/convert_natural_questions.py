#!/usr/bin/env python3
"""
Natural Questions数据集格式转换脚本
将Natural Questions数据集转换为统一格式：{id, question, answer, context}
"""

import json
import os
from pathlib import Path
import re
import html
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


def convert_natural_questions_sample(sample):
    """
    转换单个Natural Questions样本为统一格式
    
    Args:
        sample: Natural Questions原始样本
        
    Returns:
        dict: 统一格式的样本
    """
    # Natural Questions 原始结构：{"id": "...", "document": {"html": "..."}}
    document = sample.get("document")
    doc_html = ""
    if isinstance(document, dict):
        doc_html = document.get("html", "") or ""
    context_text = _html_to_text(doc_html)

    return {
        "id": sample.get('id', ''),
        "question": "",            # 原始文件不含问题
        "answer": "",              # 原始文件不含答案
        "context": context_text
    }


def convert_natural_questions_dataset(input_file, output_file):
    """
    转换整个Natural Questions数据集文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    print(f"正在转换 {input_file} -> {output_file}")
    
    converted_samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                converted_sample = convert_natural_questions_sample(sample)
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
    主函数：转换Natural Questions数据集
    """
    # 设置路径
    base_dir = Path("/home/pushihao/RAG/Reports/experiments")
    input_dir = base_dir / "datasets" / "natural_questions"
    output_dir = base_dir / "dataset_converters" / "converted" / "natural_questions"
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换训练集和验证集
    datasets = {
        "train.json": "train_converted.json",
        "validation.json": "validation_converted.json"
    }
    
    for input_name, output_name in datasets.items():
        input_file = input_dir / input_name
        output_file = output_dir / output_name
        
        if input_file.exists():
            convert_natural_questions_dataset(input_file, output_file)
        else:
            print(f"警告: 输入文件不存在: {input_file}")
    
    print("Natural Questions数据集转换完成！")


if __name__ == "__main__":
    main()