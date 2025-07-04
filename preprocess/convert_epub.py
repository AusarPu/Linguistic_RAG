import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from script.config import *

def chap2text(chap):
    """从Ebooklib的章节对象中提取纯文本。"""
    output = ''
    soup = BeautifulSoup(chap.get_body_content(), 'html.parser')
    
    # 查找并提取章节标题
    title_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    title = ''
    for tag in title_tags:
        title_node = soup.find(tag)
        if title_node:
            title = title_node.get_text().strip()
            # 从soup中移除标题节点，避免重复输出
            title_node.decompose() 
            break
    
    # 只处理有标题的章节，无标题的章节丢弃
    if not title:
        return ''
        
    # 在章节内容前加上新的结构化标记
    output += f'≦{title}≧\n\n'
    
    # 提取所有段落文本
    text = soup.find_all('p')
    for p in text:
        output += f'{p.get_text().strip()}\n'
        
    return output

def epub_to_structured_text(epub_path):
    """
    读取一个EPUB文件，并将其转换为带有结构化标记的TXT文件。
    """
    try:
        book = epub.read_epub(epub_path)
        chapters_text = []
        
        # 遍历书籍的所有项目，找到HTML章节
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                chapters_text.append(chap2text(item))

        return "".join(chapters_text)
        
    except Exception as e:
        print(f"处理文件 {epub_path} 时出错: {e}")
        return None

def main():
    """
    主函数，处理当前目录下所有的EPUB文件。
    """

    print(f"正在扫描目录: {KNOWLEDGE_BASE_DIR}")
    
    successful_conversions = []
    failed_conversions = []
    
    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        if filename.lower().endswith('.epub'):
            epub_filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
            print(f"  正在处理: {filename}...")
            
            structured_text = epub_to_structured_text(epub_filepath)
            
            if structured_text:
                # 创建新的txt文件名
                txt_filename = os.path.splitext(filename)[0] + '.txt'
                txt_filepath = os.path.join(KNOWLEDGE_BASE_DIR, txt_filename)
                
                try:
                    with open(txt_filepath, 'w', encoding='utf-8') as f:
                        f.write(structured_text)
                    print(f"  成功转换并保存为: {txt_filename}")
                    
                    # 删除原始epub文件
                    os.remove(epub_filepath)
                    print(f"  已删除原始文件: {filename}")
                    successful_conversions.append(filename)
                    
                except Exception as e:
                    print(f"  保存文件时出错: {e}")
                    failed_conversions.append(filename)
            else:
                failed_conversions.append(filename)
    
    # 转换完成后的总结
    print("\n=== 转换完成 ===")
    if successful_conversions:
        print(f"成功转换 {len(successful_conversions)} 个文件:")
        for file in successful_conversions:
            print(f"  ✓ {file}")
    
    if failed_conversions:
        print(f"\n转换失败 {len(failed_conversions)} 个文件，请检查:")
        for file in failed_conversions:
            print(f"  ✗ {file}")
    
    if not successful_conversions and not failed_conversions:
        print("未找到任何epub文件。")

if __name__ == '__main__':
    main()