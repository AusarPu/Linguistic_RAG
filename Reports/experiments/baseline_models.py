#!/usr/bin/env python3
"""
基线模型实现
实现用于对比实验的各种基线模型
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
import requests
from script.vllm_clients import EmbeddingAPIClient, call_generator_vllm_stream
from script.knowledge_base import KnowledgeBase
# 创建一个简单的配置类
class RAGConfig:
    def __init__(self):
        # 从config_rag导入必要的配置
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
        from script import config_rag
        
        self.GENERATOR_BASE_URL = config_rag.GENERATOR_API_URL
        self.GENERATOR_MODEL_NAME = config_rag.GENERATOR_MODEL_NAME_FOR_API
        self.EMBEDDING_API_URL = config_rag.EMBEDDING_API_URL
        self.EMBEDDING_MODEL_NAME = config_rag.EMBEDDING_MODEL_NAME_FOR_API
        self.KNOWLEDGE_BASE_DIR = config_rag.KNOWLEDGE_BASE_DIR
        self.PROCESSED_DATA_DIR = config_rag.PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseRAGModel(ABC):
    """
    RAG模型基类
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回文档数量
        
        Returns:
            检索到的文档列表
        """
        pass
    
    async def generate_answer(self, query: str, context: str) -> str:
        """
        生成答案
        
        Args:
            query: 用户问题
            context: 检索到的上下文
        
        Returns:
            生成的答案
        """
        prompt = f"""基于以下上下文回答问题。如果上下文中没有相关信息，请说明无法回答。

上下文：
{context}

问题：{query}

答案："""
        
        try:
            # 使用call_generator_vllm_stream函数生成答案
            full_response = ""
            async for chunk in call_generator_vllm_stream(
                user_content_for_generator=prompt,
                generation_config={
                    "max_tokens": 512,
                    "temperature": 0.1
                }
            ):
                if chunk.get("type") == "content_delta":
                    full_response += chunk.get("text", "")
                elif chunk.get("type") == "error":
                    logger.error(f"生成过程中出错: {chunk.get('message')}")
                    return "抱歉，无法生成答案。"
            
            return full_response.strip()
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return "抱歉，无法生成答案。"
    
    async def answer_question(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        完整的问答流程
        
        Args:
            query: 用户问题
            top_k: 检索文档数量
        
        Returns:
            包含答案和检索信息的字典
        """
        # 检索
        retrieved_docs = await self.retrieve(query, top_k)
        
        # 构建上下文
        context = "\n\n".join([doc['content'] for doc in retrieved_docs])
        
        # 生成答案
        answer = await self.generate_answer(query, context)
        
        return {
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'context': context,
            'num_retrieved': len(retrieved_docs)
        }

class BasicRAG(BaseRAGModel):
    """
    基础RAG模型：简单的向量检索 + LLM生成
    """
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.knowledge_base = KnowledgeBase()
        logger.info("基础RAG模型初始化完成")
    
    async def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        使用密集向量检索
        """
        try:
            # 只使用密集块检索
            results = self.knowledge_base.search_dense_chunks(query, top_k)
            return results
        except Exception as e:
            logger.error(f"基础RAG检索失败: {e}")
            return []

class HybridRAG(BaseRAGModel):
    """
    传统混合检索RAG：BM25 + 单一向量检索的组合
    """
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.knowledge_base = KnowledgeBase()
        logger.info("传统混合检索RAG模型初始化完成")
    
    async def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        结合BM25和向量检索
        """
        try:
            # 分别进行密集检索和稀疏检索
            dense_results = self.knowledge_base.search_dense_chunks(query, top_k // 2)
            sparse_results = self.knowledge_base.search_dense_keywords(query, top_k // 2)
            
            # 简单合并（去重）
            all_results = dense_results + sparse_results
            seen_ids = set()
            unique_results = []
            
            for result in all_results:
                if result['id'] not in seen_ids:
                    seen_ids.add(result['id'])
                    unique_results.append(result)
            
            return unique_results[:top_k]
        except Exception as e:
            logger.error(f"传统混合检索失败: {e}")
            return []

class RerankerRAG(BaseRAGModel):
    """
    基于重排序的RAG：检索 + Cross-Encoder重排序
    """
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.knowledge_base = KnowledgeBase()
        
        # 初始化Cross-Encoder重排序模型
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("重排序模型加载成功")
        except Exception as e:
            logger.warning(f"重排序模型加载失败: {e}，将使用简单排序")
            self.reranker = None
        
        logger.info("重排序RAG模型初始化完成")
    
    async def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        检索后使用Cross-Encoder重排序
        """
        try:
            # 先检索更多候选文档
            candidate_k = min(top_k * 3, 50)  # 检索3倍数量的候选文档
            
            # 使用混合检索获取候选文档
            dense_results = self.knowledge_base.search_dense_chunks(query, candidate_k // 2)
            sparse_results = self.knowledge_base.search_dense_keywords(query, candidate_k // 2)
            
            # 合并去重
            all_results = dense_results + sparse_results
            seen_ids = set()
            candidates = []
            
            for result in all_results:
                if result['id'] not in seen_ids:
                    seen_ids.add(result['id'])
                    candidates.append(result)
            
            if not candidates:
                return []
            
            # 如果有重排序模型，进行重排序
            if self.reranker is not None:
                try:
                    # 准备重排序输入
                    pairs = [(query, doc['content']) for doc in candidates]
                    
                    # 计算重排序分数
                    scores = self.reranker.predict(pairs)
                    
                    # 按分数排序
                    scored_docs = list(zip(candidates, scores))
                    scored_docs.sort(key=lambda x: x[1], reverse=True)
                    
                    # 返回top-k结果
                    reranked_results = [doc for doc, score in scored_docs[:top_k]]
                    
                    logger.info(f"重排序完成，从{len(candidates)}个候选文档中选出{len(reranked_results)}个")
                    return reranked_results
                    
                except Exception as e:
                    logger.warning(f"重排序失败: {e}，使用原始排序")
            
            # 如果重排序失败，返回原始结果
            return candidates[:top_k]
            
        except Exception as e:
            logger.error(f"重排序RAG检索失败: {e}")
            return []

class MultiPathRAGWithoutFilter(BaseRAGModel):
    """
    多路径检索但不进行有用性判断的RAG（用于消融研究）
    """
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.knowledge_base = KnowledgeBase()
        logger.info("多路径RAG（无过滤）模型初始化完成")
    
    async def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        使用完整的多路径检索但不进行过滤
        """
        try:
            # 执行三路径并行检索
            dense_chunk_task = asyncio.create_task(asyncio.to_thread(self.knowledge_base.search_dense_chunks, query, top_k))
            dense_question_task = asyncio.create_task(asyncio.to_thread(self.knowledge_base.search_dense_questions, query, top_k // 2))
            sparse_task = asyncio.create_task(asyncio.to_thread(self.knowledge_base.search_dense_keywords, query, top_k // 2))
            
            # 等待所有检索完成
            dense_chunk_results, dense_question_results, sparse_results = await asyncio.gather(
                dense_chunk_task, dense_question_task, sparse_task
            )
            
            # 合并所有结果并去重
            all_results = dense_chunk_results + dense_question_results + sparse_results
            seen_ids = set()
            unique_results = []
            
            for result in all_results:
                if result['id'] not in seen_ids:
                    seen_ids.add(result['id'])
                    unique_results.append(result)
            
            return unique_results[:top_k]
            
        except Exception as e:
            logger.error(f"多路径检索（无过滤）失败: {e}")
            return []

class FullRAGSystem(BaseRAGModel):
    """
    完整的RAG系统（我们的方法）：多路径检索 + 有用性判断
    """
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.knowledge_base = KnowledgeBase()
        
        # 有用性判断提示模板
        self.useful_judge_prompt_template = """请判断以下知识块是否对回答问题有用。

问题：{question}

知识块：
{content}

请回答 "useful" 或 "useless"："""
        
        logger.info("完整RAG系统初始化完成")
    
    async def judge_usefulness(self, question: str, content: str) -> bool:
        """
        判断知识块的有用性
        
        Args:
            question: 用户问题
            content: 知识块内容
        
        Returns:
            是否有用
        """
        try:
            prompt = self.useful_judge_prompt_template.format(
                question=question,
                content=content
            )
            
            # 使用call_generator_vllm_stream函数进行有用性判断
            full_response = ""
            async for chunk in call_generator_vllm_stream(
                user_content_for_generator=prompt,
                generation_config={
                    "max_tokens": 10,
                    "temperature": 0.0
                }
            ):
                if chunk.get("type") == "content_delta":
                    full_response += chunk.get("text", "")
            
            response = full_response.strip()
            
            # 解析判断结果
            result = response.strip().lower()
            return "useful" in result
            
        except Exception as e:
            logger.warning(f"有用性判断失败: {e}，默认认为有用")
            return True
    
    async def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        多路径检索 + 有用性判断
        """
        try:
            # 执行三路径并行检索
            dense_chunk_task = asyncio.create_task(asyncio.to_thread(self.knowledge_base.search_dense_chunks, query, top_k))
            dense_question_task = asyncio.create_task(asyncio.to_thread(self.knowledge_base.search_dense_questions, query, top_k // 2))
            sparse_task = asyncio.create_task(asyncio.to_thread(self.knowledge_base.search_dense_keywords, query, top_k // 2))
            
            # 等待所有检索完成
            dense_chunk_results, dense_question_results, sparse_results = await asyncio.gather(
                dense_chunk_task, dense_question_task, sparse_task
            )
            
            # 合并所有结果并去重
            all_results = dense_chunk_results + dense_question_results + sparse_results
            seen_ids = set()
            candidates = []
            
            for result in all_results:
                if result['id'] not in seen_ids:
                    seen_ids.add(result['id'])
                    candidates.append(result)
            
            if not candidates:
                return []
            
            # 对每个候选文档进行有用性判断
            useful_docs = []
            judgment_tasks = []
            
            for doc in candidates:
                task = self.judge_usefulness(query, doc['content'])
                judgment_tasks.append((doc, task))
            
            # 并行执行有用性判断
            for doc, task in judgment_tasks:
                try:
                    is_useful = await task
                    if is_useful:
                        useful_docs.append(doc)
                except Exception as e:
                    logger.warning(f"判断文档{doc['id']}有用性失败: {e}，保留该文档")
                    useful_docs.append(doc)
            
            logger.info(f"有用性判断完成：从{len(candidates)}个候选文档中筛选出{len(useful_docs)}个有用文档")
            
            return useful_docs[:top_k]
            
        except Exception as e:
            logger.error(f"完整RAG系统检索失败: {e}")
            return []

class ModelFactory:
    """
    模型工厂类
    """
    
    @staticmethod
    def create_model(model_type: str, config: RAGConfig) -> BaseRAGModel:
        """
        创建指定类型的模型
        
        Args:
            model_type: 模型类型
            config: 配置对象
        
        Returns:
            模型实例
        """
        model_map = {
            'basic_rag': BasicRAG,
            'hybrid_rag': HybridRAG,
            'reranker_rag': RerankerRAG,
            'multipath_no_filter': MultiPathRAGWithoutFilter,
            'full_rag': FullRAGSystem
        }
        
        if model_type not in model_map:
            raise ValueError(f"未知的模型类型: {model_type}")
        
        return model_map[model_type](config)
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        获取可用的模型类型列表
        """
        return ['basic_rag', 'hybrid_rag', 'reranker_rag', 'multipath_no_filter', 'full_rag']

if __name__ == "__main__":
    # 测试基线模型
    async def test_models():
        config = RAGConfig()
        
        # 测试所有模型
        for model_type in ModelFactory.get_available_models():
            print(f"\n=== 测试 {model_type} ===")
            try:
                model = ModelFactory.create_model(model_type, config)
                result = await model.answer_question("什么是机器学习？", top_k=5)
                print(f"检索到 {result['num_retrieved']} 个文档")
                print(f"答案: {result['answer'][:100]}...")
            except Exception as e:
                print(f"测试失败: {e}")
    
    # 运行测试
    asyncio.run(test_models())