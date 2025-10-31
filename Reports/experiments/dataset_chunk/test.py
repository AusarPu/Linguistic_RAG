#!/usr/bin/env python3
"""
vLLM Tokenizer 切分测试脚本
使用运行在8001端口的vLLM服务进行文本tokenizer切分
"""

import requests
import json
from typing import List, Optional
import time


class VLLMTokenizer:
    """使用vLLM API进行tokenizer操作的类"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.tokenize_url = f"{base_url}/tokenize"
        self.detokenize_url = f"{base_url}/detokenize"
    
    def tokenize(self, text: str) -> List[int]:
        """
        将文本转换为token IDs
        
        Args:
            text: 输入文本
            
        Returns:
            token IDs列表
        """
        try:
            payload = {
                "prompt": text,
                "add_special_tokens": False
            }
            
            response = requests.post(
                self.tokenize_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("tokens", [])
            else:
                print(f"Tokenize请求失败: {response.status_code}, {response.text}")
                return []
                
        except Exception as e:
            print(f"Tokenize请求异常: {e}")
            return []
    
    def detokenize(self, token_ids: List[int]) -> str:
        """
        将token IDs转换回文本
        
        Args:
            token_ids: token IDs列表
            
        Returns:
            解码后的文本
        """
        try:
            payload = {
                "tokens": token_ids
            }
            
            response = requests.post(
                self.detokenize_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("prompt", "")
            else:
                print(f"Detokenize请求失败: {response.status_code}, {response.text}")
                return ""
                
        except Exception as e:
            print(f"Detokenize请求异常: {e}")
            return ""


def chunk_text_by_tokens(text: str, chunk_size: int, tokenizer: VLLMTokenizer, overlap: int = 0) -> List[str]:
    """
    按照token长度对文本进行分块
    
    Args:
        text: 输入文本
        chunk_size: 每个分块的token数量
        tokenizer: VLLMTokenizer实例
        overlap: 分块之间的重叠token数量
        
    Returns:
        分块后的文本列表
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")
    
    if overlap >= chunk_size:
        raise ValueError("overlap必须小于chunk_size")
    
    # 获取完整文本的tokens
    tokens = tokenizer.tokenize(text)
    
    if not tokens:
        print("无法获取tokens，返回原文本")
        return [text]
    
    print(f"原文本token数量: {len(tokens)}")
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(tokens):
        # 计算当前分块的结束位置
        end_idx = min(start_idx + chunk_size, len(tokens))
        
        # 提取当前分块的tokens
        chunk_tokens = tokens[start_idx:end_idx]
        
        # 将tokens转换回文本
        chunk_text = tokenizer.detokenize(chunk_tokens)
        
        if chunk_text.strip():  # 只添加非空分块
            chunks.append(chunk_text)
        
        # 计算下一个分块的起始位置（考虑重叠）
        if end_idx >= len(tokens):
            break
        
        start_idx = end_idx - overlap
    
    return chunks


def test_vllm_connection(tokenizer: VLLMTokenizer) -> bool:
    """测试vLLM服务连接"""
    try:
        test_text = "Hello, world!"
        tokens = tokenizer.tokenize(test_text)
        if tokens:
            decoded = tokenizer.detokenize(tokens)
            print(f"连接测试成功!")
            print(f"测试文本: '{test_text}'")
            print(f"Token数量: {len(tokens)}")
            print(f"解码结果: '{decoded}'")
            return True
        else:
            print("连接测试失败: 无法获取tokens")
            return False
    except Exception as e:
        print(f"连接测试异常: {e}")
        return False


def main():
    """主函数 - 演示tokenizer切分功能"""
    print("=== vLLM Tokenizer 切分测试 ===\n")
    
    # 初始化tokenizer
    tokenizer = VLLMTokenizer()
    
    # 测试连接
    print("1. 测试vLLM服务连接...")
    if not test_vllm_connection(tokenizer):
        print("❌ 无法连接到vLLM服务，请确保服务在8001端口运行")
        return
    
    print("\n" + "="*50 + "\n")
    
    # 测试文本
    test_text = """
    人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质，
    并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、
    语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，
    应用领域也不断扩大。可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
    人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考、
    也可能超过人的智能。
    """
    
    print("2. 原始测试文本:")
    print(f"'{test_text.strip()}'")
    print(f"文本长度: {len(test_text.strip())} 字符\n")
    
    # 测试不同的分块大小
    chunk_sizes = [50, 100, 150]
    
    for chunk_size in chunk_sizes:
        print(f"3. 使用chunk_size={chunk_size}进行分块:")
        print("-" * 40)
        
        start_time = time.time()
        chunks = chunk_text_by_tokens(test_text.strip(), chunk_size, tokenizer)
        end_time = time.time()
        
        print(f"分块数量: {len(chunks)}")
        print(f"处理时间: {end_time - start_time:.2f}秒")
        
        for i, chunk in enumerate(chunks, 1):
            # 计算每个分块的token数量
            chunk_tokens = tokenizer.tokenize(chunk)
            print(f"\n分块 {i} (tokens: {len(chunk_tokens)}):")
            print(f"'{chunk[:100]}{'...' if len(chunk) > 100 else ''}'")
        
        print("\n" + "="*50 + "\n")
    
    # 测试带重叠的分块
    print("4. 测试带重叠的分块 (chunk_size=100, overlap=20):")
    print("-" * 40)
    
    chunks_with_overlap = chunk_text_by_tokens(test_text.strip(), 100, tokenizer, overlap=20)
    print(f"带重叠分块数量: {len(chunks_with_overlap)}")
    
    for i, chunk in enumerate(chunks_with_overlap, 1):
        chunk_tokens = tokenizer.tokenize(chunk)
        print(f"\n重叠分块 {i} (tokens: {len(chunk_tokens)}):")
        print(f"'{chunk[:80]}{'...' if len(chunk) > 80 else ''}'")


if __name__ == "__main__":
    main()