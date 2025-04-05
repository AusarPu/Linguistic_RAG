import re

def split_text(text, chunk_size, overlap, min_chunk_length):
    """智能文本分块函数"""
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        if current_length + len(para) <= chunk_size:
            current_chunk.append(para)
            current_length += len(para)
        else:
            if current_chunk:
                chunk = '\n'.join(current_chunk)
                if len(chunk) >= min_chunk_length:
                    chunks.append(chunk)
                current_chunk = []
                current_length = 0

            if len(para) > chunk_size:
                words = para.split()
                sub_chunk = []
                sub_length = 0
                for word in words:
                    if sub_length + len(word) > chunk_size:
                        chunks.append(' '.join(sub_chunk))
                        sub_chunk = sub_chunk[-overlap // 20:]
                        sub_length = sum(len(w) for w in sub_chunk)
                    sub_chunk.append(word)
                    sub_length += len(word) + 1
                if sub_chunk:
                    chunks.append(' '.join(sub_chunk))
            else:
                current_chunk = [para]
                current_length = len(para)

    if current_chunk:
        chunk = '\n'.join(current_chunk)
        if len(chunk) >= min_chunk_length:
            chunks.append(chunk)

    if len(chunks) > 1:
        for i in range(len(chunks) - 1, 0, -1):
            if len(chunks[i]) + len(chunks[i - 1]) < chunk_size * 1.2:
                chunks[i - 1] = chunks[i - 1] + '\n' + chunks[i]
                del chunks[i]

    return chunks
