import os
import re
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from script.config import KNOWLEDGE_BASE_DIR,PROCESSED_DATA_DIR


# -----------------------------------------------------------------------------
# 函数一：解析包含页码标记的 TXT 文档为结构化页面列表
# (这个函数与我们上一轮讨论并你确认可用的版本一致)
# -----------------------------------------------------------------------------
def parse_txt_to_structured_pages(full_document_text: str, doc_name: str) -> list:
    """
    将包含 ≦ N ≧ 页码标记的文档字符串解析为一个结构化的页面对象列表。
    每个对象代表一页及其文本内容。
    """
    pages_data = []
    page_marker_pattern_for_split = r"(≦\s*[^≧]+\s*≧)"
    page_number_extract_pattern = re.compile(r"≦\s*([^≧]+?)\s*≧")
    parts = re.split(page_marker_pattern_for_split, full_document_text)

    if not parts:
        return []

    if len(parts) == 1 and not page_number_extract_pattern.search(parts[0]):
        document_text_stripped = parts[0].strip()
        if document_text_stripped:
            pages_data.append({
                "doc_name": doc_name,
                "page_number": 1,
                "text": document_text_stripped
            })
        return pages_data

    current_part_index = 0
    # 处理第一个标记之前的部分 (如果有文本)
    if current_part_index < len(parts) and \
            not page_number_extract_pattern.fullmatch(parts[current_part_index].strip()):
        text_before_first_marker = parts[current_part_index].strip()
        if text_before_first_marker:
            pages_data.append({
                "doc_name": doc_name,
                "page_number": 0,  # 将第一个标记前的文本视为第0页或"前言"
                "text": text_before_first_marker
            })
        current_part_index += 1

    # 处理标记和对应的页面文本
    while current_part_index < len(parts) - 1:
        marker_candidate = parts[current_part_index].strip()
        page_text_candidate = parts[current_part_index + 1].strip()  # 标记后的文本部分
        marker_match = page_number_extract_pattern.fullmatch(marker_candidate)

        if marker_match:
            page_identifier = marker_match.group(1).strip()
            
            # 处理空页码的情况
            if not page_identifier:
                page_identifier = "未命名页面"
            
            # 尝试将页码转换为整数，如果失败则保持原始字符串
            try:
                page_num = int(page_identifier)
            except ValueError:
                # 如果不是数字，则保持原始字符串作为页码标识
                page_num = page_identifier
            
            if page_text_candidate:  # 确保页面文本不为空
                pages_data.append({
                    "doc_name": doc_name,
                    "page_number": page_num,
                    "text": page_text_candidate
                })
        current_part_index += 2  # 跳过当前标记和当前文本，移向下一个标记

    return pages_data


# -----------------------------------------------------------------------------
# 函数二：使用 Langchain 对单个文档进行按页解析和细粒度切块
# -----------------------------------------------------------------------------
def generate_document_chunks_langchain(
        full_document_text: str,
        doc_name: str,
        char_chunk_size: int,
        char_overlap: int,
        char_min_chunk_length: int,
        separators: list = None  # 允许自定义分隔符
) -> list:
    """
    处理单个完整文档：
    1. 将文档文本按页码标记解析成多个页面。
    2. 使用 Langchain 的 RecursiveCharacterTextSplitter 将每一页的文本切分成更小的块。
    3. 返回包含完整元数据的最终文本块列表。
    """

    structured_pages = parse_txt_to_structured_pages(full_document_text, doc_name)
    all_final_chunks_with_metadata = []

    if separators is None:
        # 默认分隔符列表，你可以根据你的文本特性调整
        separators = ["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=char_chunk_size,
        chunk_overlap=char_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=separators
    )

    for page_info in structured_pages:
        page_text = page_info['text']
        page_num = page_info['page_number']

        if not page_text.strip():
            continue

        chunks_from_this_page_texts = text_splitter.split_text(page_text)

        for i, chunk_text_content in enumerate(chunks_from_this_page_texts):
            if len(chunk_text_content.strip()) >= char_min_chunk_length:  # 确保块在去除首尾空格后仍满足最小长度
                # 处理页码标识符，确保chunk_id的有效性
                page_id_safe = str(page_num).replace(" ", "_").replace("/", "_").replace("\\", "_")
                all_final_chunks_with_metadata.append({
                    "doc_name": doc_name,
                    "page_number": page_num,
                    "chunk_id": f"{doc_name}_p{page_id_safe}_c{i + 1}",  # 创建一个唯一的块 ID
                    "text": chunk_text_content.strip()  # 存储去除首尾空格的文本
                })

    return all_final_chunks_with_metadata


# -----------------------------------------------------------------------------
# 主逻辑：处理知识库目录下的所有 TXT 文件
# -----------------------------------------------------------------------------
def process_knowledge_base(
        knowledge_base_dir: str,
        output_json_path: str,
        char_chunk_size: int,
        char_overlap: int,
        char_min_chunk_length: int,
        langchain_separators: list = None
):
    """
    遍历知识库目录中的所有 .txt 文件，使用 Langchain 进行切分，
    并将所有块及其元数据保存到单个JSON文件中。
    """
    all_documents_chunks = []

    if not os.path.isdir(knowledge_base_dir):
        print(f"错误：知识库目录 '{knowledge_base_dir}' 不存在或不是一个目录。")
        return

    print(f"开始处理知识库目录: '{knowledge_base_dir}'")
    print(
        f"切分参数: Chunk Size (chars)={char_chunk_size}, Overlap (chars)={char_overlap}, Min Length (chars)={char_min_chunk_length}")
    if langchain_separators:
        print(f"Langchain 分隔符: {langchain_separators}")
    else:
        print(f"Langchain 分隔符: 使用默认列表")

    for filename in os.listdir(knowledge_base_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(knowledge_base_dir, filename)
            print(f"\n正在处理文件: {filename} ...")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    document_content = f.read()

                if not document_content.strip():
                    print(f"  文件 '{filename}' 为空或只包含空白字符，已跳过。")
                    continue

                chunks_for_this_doc = generate_document_chunks_langchain(
                    document_content,
                    filename,  # 使用文件名作为 doc_name
                    char_chunk_size,
                    char_overlap,
                    char_min_chunk_length,
                    separators=langchain_separators
                )
                all_documents_chunks.extend(chunks_for_this_doc)
                print(f"  文件 '{filename}' 处理完毕，生成 {len(chunks_for_this_doc)} 个文本块。")

            except Exception as e:
                print(f"  处理文件 '{filename}' 时发生错误: {e}")

    print(f"\n知识库处理完成。总共生成 {len(all_documents_chunks)} 个文本块。")

    # 保存到 JSON 文件
    try:
        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(all_documents_chunks, outfile, ensure_ascii=False, indent=2)
        print(f"所有文本块已保存到: '{output_json_path}'")
    except Exception as e:
        print(f"保存到 JSON 文件 '{output_json_path}' 时发生错误: {e}")


if __name__ == '__main__':

    KNOWLEDGE_BASE_DIRECTORY = KNOWLEDGE_BASE_DIR

    # 处理后输出的 JSON 文件路径
    OUTPUT_JSON_FILE = PROCESSED_DATA_DIR
    os.makedirs(OUTPUT_JSON_FILE, exist_ok=True)

    # 文本切分参数 (基于字符)
    # 你之前提到："我之前的设置是200~400字，重叠50字"
    TARGET_CHAR_CHUNK_SIZE = 2000  # 你可以调整在 200-400 之间
    TARGET_CHAR_OVERLAP = 0
    MIN_CHAR_CHUNK_LENGTH = 10  # 设定一个合适的最小块长度，避免过小的碎块

    # Langchain RecursiveCharacterTextSplitter 的分隔符
    # 你可以根据你的 OCR 文本特性调整这个列表及其顺序
    # None 表示使用 generate_document_chunks_langchain 中的默认列表
    LANGCHAIN_SEPARATORS = ["\n\n","。", "！", "？", "，", "、", ". ", "! ", "? ", ", ", " ", ""]
    # LANGCHAIN_SEPARATORS = None # 使用函数内默认值

    # --- 执行处理 ---
    process_knowledge_base(
        KNOWLEDGE_BASE_DIRECTORY,
        OUTPUT_JSON_FILE+"processed_knowledge_base_chunks.json",
        TARGET_CHAR_CHUNK_SIZE,
        TARGET_CHAR_OVERLAP,
        MIN_CHAR_CHUNK_LENGTH,
        langchain_separators=LANGCHAIN_SEPARATORS
    )

    print("\n--- 运行完毕 ---")
    print(f"如果一切顺利，你应该能在 '{OUTPUT_JSON_FILE}' 找到处理好的数据。")
    print("这个 JSON 文件中的每个条目都是一个文本块及其元数据，可用于后续的嵌入和LLM增强。")