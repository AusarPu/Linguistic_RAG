import re
import json  # 用于在 __main__ 中美化打印输出
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 函数一：解析包含页码标记的 TXT 文档为结构化页面列表 (保持不变)
def parse_txt_to_structured_pages(full_document_text: str, doc_name: str) -> list:
    """
    将包含 ≦ N ≧ 页码标记的文档字符串解析为一个结构化的页面对象列表。
    每个对象代表一页及其文本内容。
    """
    pages_data = []
    page_marker_pattern_for_split = r"(≦\s*\d+\s*≧)"
    page_number_extract_pattern = re.compile(r"≦\s*(\d+)\s*≧")
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
    if parts[current_part_index] and not page_number_extract_pattern.fullmatch(parts[current_part_index].strip()):
        text_before_first_marker = parts[current_part_index].strip()
        if text_before_first_marker:
            pages_data.append({
                "doc_name": doc_name,
                "page_number": 0,
                "text": text_before_first_marker
            })
        current_part_index += 1

    while current_part_index < len(parts) - 1:
        marker_candidate = parts[current_part_index].strip()
        page_text_candidate = parts[current_part_index + 1].strip()
        marker_match = page_number_extract_pattern.fullmatch(marker_candidate)

        if marker_match:
            page_num = int(marker_match.group(1))
            if page_text_candidate:
                pages_data.append({
                    "doc_name": doc_name,
                    "page_number": page_num,
                    "text": page_text_candidate
                })
        current_part_index += 2

    return pages_data


# 函数二：新的编排函数，使用 Langchain 进行细粒度切分
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
        # 针对 OCR 文本和中文特性，可以调整分隔符的优先级和内容
        # \n\n 通常是段落分隔
        # \n 可能是 OCR 带来的换行，也可能是段内换行
        # 。！？ 是中文句尾标点
        # ，、 是中文逗号和顿号
        # " " 空格
        # "" 最后按字符切分
        separators = ["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""]

        # 初始化 Langchain 文本切分器
    # 我们按字符长度切分，所以 length_function=len
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=char_chunk_size,
        chunk_overlap=char_overlap,
        length_function=len,
        is_separator_regex=False,  # 分隔符不是正则表达式
        separators=separators
    )

    for page_info in structured_pages:
        page_text = page_info['text']
        page_num = page_info['page_number']

        if not page_text.strip():  # 跳过没有实际文本内容的页面
            continue

        # 使用 Langchain 切分器处理这一页的文本
        chunks_from_this_page_texts = text_splitter.split_text(page_text)

        for i, chunk_text_content in enumerate(chunks_from_this_page_texts):
            # 应用最小长度过滤
            if len(chunk_text_content) >= char_min_chunk_length:
                all_final_chunks_with_metadata.append({
                    "doc_name": doc_name,
                    "page_number": page_num,
                    "chunk_id": f"{doc_name}_p{page_num}_c{i + 1}",
                    "text": chunk_text_content
                })

    return all_final_chunks_with_metadata

