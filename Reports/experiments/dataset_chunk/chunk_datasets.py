#!/usr/bin/env python3
"""
数据集分块脚本

该脚本用于对 converted 目录中的数据集文件进行分块处理，
使用 preprocess_documents.py 中的分块功能，
输出到 datasets/chunked 目录中，文件名格式为 {dataset}_chunked.json
"""

import os
import json
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# 导入 preprocess_documents.py 中的分块功能
from preprocess.preprocess_documents import generate_document_chunks_langchain


def load_dataset_file(file_path: str) -> list:
    """
    加载数据集文件
    
    Args:
        file_path: 数据集文件路径
        
    Returns:
        数据集内容列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return []


def extract_text_from_dataset_item(item: dict) -> str:
    """
    从数据集项目中提取文本内容
    
    Args:
        item: 数据集中的单个项目
        
    Returns:
        提取的文本内容
    """
    # 根据数据集格式提取文本
    text_parts = []
    
    # 添加问题
    text_parts.append(f"Question: {item['question']}")
    
    # 添加答案
    text_parts.append(f"Answer: {item['answer']}")
    
    # 添加上下文
    text_parts.append(f"Context: {item['context']}")
    
    return "\n\n".join(text_parts)


def chunk_dataset(dataset_name: str, dataset_items: list, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 100,
                 min_chunk_length: int = 50) -> list:
    """
    对数据集进行分块处理
    
    Args:
        dataset_name: 数据集名称
        dataset_items: 数据集项目列表
        chunk_size: 分块大小
        chunk_overlap: 分块重叠
        min_chunk_length: 最小分块长度
        
    Returns:
        分块后的数据列表
    """
    all_chunks = []
    
    # 自定义分隔符，适合问答数据集
    separators = ["\n\nDocument ", "\n\nQuestion:", "\n\nAnswer:", "\n\nContext:", 
                 "\n\n", "\n", "。", "！", "？", "，", "、", ". ", "! ", "? ", ", ", " ", ""]
    
    for idx, item in enumerate(dataset_items):
        # 提取文本内容
        full_text = extract_text_from_dataset_item(item)
        
        if not full_text.strip():
            continue
            
        # 使用 preprocess_documents.py 中的分块功能
        chunks = generate_document_chunks_langchain(
            full_document_text=full_text,
            doc_name=f"{dataset_name}_item_{idx}",
            char_chunk_size=chunk_size,
            char_overlap=chunk_overlap,
            char_min_chunk_length=min_chunk_length,
            separators=separators
        )
        
        # 为每个分块添加原始数据集信息
        for chunk in chunks:
            chunk['original_item_id'] = item.get('id', f"item_{idx}")
            chunk['dataset_name'] = dataset_name
            chunk['question'] = item.get('question', '')
            chunk['answer'] = item.get('answer', '')
            
        all_chunks.extend(chunks)
    
    return all_chunks


def process_all_datasets(converted_dir: str, output_dir: str):
    """
    处理所有数据集文件
    
    Args:
        converted_dir: 转换后数据集目录
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有 JSON 文件
    converted_path = Path(converted_dir)
    json_files = list(converted_path.glob("*.json"))
    
    if not json_files:
        print(f"在 {converted_dir} 中未找到 JSON 文件")
        return
    
    print(f"找到 {len(json_files)} 个数据集文件")
    
    for json_file in json_files:
        # 提取数据集名称（去掉 _validation_kb_chunks.json 后缀）
        dataset_name = json_file.stem.replace('_validation_kb_chunks', '')
        
        print(f"\n正在处理数据集: {dataset_name}")
        
        # 加载数据集
        dataset_items = load_dataset_file(str(json_file))
        
        if not dataset_items:
            print(f"数据集 {dataset_name} 为空或加载失败，跳过")
            continue
        
        print(f"数据集 {dataset_name} 包含 {len(dataset_items)} 个项目")
        
        # 进行分块处理
        chunks = chunk_dataset(dataset_name, dataset_items)
        
        print(f"数据集 {dataset_name} 生成了 {len(chunks)} 个分块")
        
        # 保存分块结果
        output_file = os.path.join(output_dir, f"{dataset_name}_chunked.json")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            print(f"分块结果已保存到: {output_file}")
        except Exception as e:
            print(f"保存文件 {output_file} 时出错: {e}")


def main():
    """主函数"""
    # 设置路径
    base_dir = Path(__file__).parent.parent
    converted_dir = base_dir / "datasets" / "converted"
    output_dir = base_dir / "datasets" / "chunked"
    
    print("=== 数据集分块处理 ===")
    print(f"输入目录: {converted_dir}")
    print(f"输出目录: {output_dir}")
    
    # 检查输入目录是否存在
    if not converted_dir.exists():
        print(f"错误: 输入目录 {converted_dir} 不存在")
        return
    
    # 处理所有数据集
    process_all_datasets(str(converted_dir), str(output_dir))
    
    print("\n=== 处理完成 ===")


if __name__ == "__main__":
    main()