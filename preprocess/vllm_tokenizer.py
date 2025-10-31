#!/usr/bin/env python3
"""
vLLM Tokenizer 模块
提供基于vLLM API的tokenizer功能，用于文本切分
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


# 全局tokenizer实例
_global_tokenizer = None


def get_tokenizer() -> VLLMTokenizer:
    """获取全局tokenizer实例"""
    global _global_tokenizer
    if _global_tokenizer is None:
        _global_tokenizer = VLLMTokenizer()
    return _global_tokenizer


def token_length_function(text: str) -> int:
    """
    基于token的长度函数，用于替换langchain中的len函数
    
    Args:
        text: 输入文本
        
    Returns:
        文本的token数量
    """
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(text)
    return len(tokens)


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


def test_vllm_connection(tokenizer: VLLMTokenizer = None) -> bool:
    """测试vLLM服务连接"""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    try:
        test_text = "Hello, world!"
        tokens = tokenizer.tokenize(test_text)
        if tokens:
            decoded = tokenizer.detokenize(tokens)
            print(f"vLLM连接测试成功!")
            print(f"测试文本: '{test_text}'")
            print(f"Token数量: {len(tokens)}")
            print(f"解码结果: '{decoded}'")
            return True
        else:
            print("vLLM连接测试失败: 无法获取tokens")
            return False
    except Exception as e:
        print(f"vLLM连接测试异常: {e}")
        return False