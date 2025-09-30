#!/usr/bin/env python3
"""
为 HotpotQA 构建独立的知识库块文件
输入: /home/pushihao/RAG/Reports/experiments/dataset_converters/converted/hotpotqa/{train,validation}_converted.json
输出: /home/pushihao/RAG/Reports/experiments/dataset_kb/hotpotqa/{train,validation}_kb_chunks.json
"""
from pathlib import Path
from kb_utils import process_ndjson_or_list_json

# 统一的切分参数，可按需调整
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
MIN_CHUNK_LEN = 10
SEPARATORS = ["\n\n","。", "！", "？", "，", "、", ". ", "! ", "? ", ", ", " ", ""]


def main():
    base = Path("/home/pushihao/RAG/Reports/experiments")
    input_dir = base / "dataset_converters" / "converted" / "hotpotqa"
    output_dir = base / "dataset_kb" / "hotpotqa"
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = {
        "train_converted.json": "train_kb_chunks.json",
        "validation_converted_1k.json": "validation_kb_chunks_1k.json",
    }

    for inp, outp in pairs.items():
        process_ndjson_or_list_json(
            input_file=input_dir / inp,
            dataset_name="hotpotqa",
            split_name=Path(outp).stem.split("_")[0],
            output_file=output_dir / outp,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            min_chunk_len=MIN_CHUNK_LEN,
            separators=SEPARATORS,
        )


if __name__ == "__main__":
    main()