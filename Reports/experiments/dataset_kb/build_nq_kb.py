#!/usr/bin/env python3
"""
为 Natural Questions 构建独立的知识库块文件
输入: /home/pushihao/RAG/Reports/experiments/dataset_converters/converted/natural_questions/validation_converted.json
输出: /home/pushihao/RAG/Reports/experiments/dataset_kb/natural_questions/validation_kb_chunks.json
只处理验证集的前10000条数据
"""
from pathlib import Path
from kb_utils import process_ndjson_or_list_json

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
MIN_CHUNK_LEN = 10
SEPARATORS = ["\n\n","。", "！", "？", "，", "、", ". ", "! ", "? ", ", ", " ", ""]


def main():
    base = Path("/home/pushihao/RAG/Reports/experiments")
    input_dir = base / "dataset_converters" / "converted" / "natural_questions"
    output_dir = base / "dataset_kb" / "natural_questions"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 只处理验证集，使用1k版本
    input_file = input_dir / "validation_converted_1k.json"
    output_file = output_dir / "validation_kb_chunks_1k.json"
    
    if input_file.exists():
        process_ndjson_or_list_json(
            input_file=input_file,
            dataset_name="natural_questions",
            split_name="validation",
            output_file=output_file,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            min_chunk_len=MIN_CHUNK_LEN,
            separators=SEPARATORS,
        )
    else:
        print(f"警告: 输入文件不存在: {input_file}")


if __name__ == "__main__":
    main()