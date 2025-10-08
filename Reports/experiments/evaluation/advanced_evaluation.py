#!/usr/bin/env python3
"""
高级评估脚本
用于评估RAG系统的答案正确性和检索准确性
"""

import os
import sys
import json
import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from script import config_rag as config

# 设置日志
config.setup_logging()
logger = logging.getLogger(__name__)

# vLLM API配置
VLLM_API_URL = "http://localhost:8001/v1/chat/completions"
EVALUATION_MODEL = config.VLLM_BASE_MODEL_LOCAL_PATH

# 答案评估提示词模板
ANSWER_EVALUATION_PROMPT = """你是一个专业的问答评估专家。请评估系统回答是否正确回答了用户问题。

**评估标准：**
1. 系统回答是否直接回答了问题
2. 系统回答的内容是否与标准答案一致或相符
3. 系统回答是否包含了关键信息
4. 即使表述不同，但意思相同也算正确

**问题：** {question}

**系统回答：** {system_answer}

**标准答案：** {ground_truth_answer}

**请严格按照以下JSON格式回答：**
{{
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "explanation": "详细解释评估理由"
}}

注意：
- 如果系统回答是"未能生成有效回答"、"抱歉，我没有找到"等无效回答，则判定为错误
- 如果系统回答包含正确信息但表述方式不同，仍可判定为正确
- confidence表示你对这个判断的信心程度（0.0-1.0）
"""

class AdvancedEvaluator:
    """高级评估器"""
    
    def __init__(self):
        self.session = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        timeout = aiohttp.ClientTimeout(total=300)  # 5分钟超时
        connector = aiohttp.TCPConnector(limit=100)
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def evaluate_answer_correctness(self, question: str, system_answer: str, ground_truth_answer: str, max_retries: int = 5) -> Dict[str, Any]:
        """
        使用vLLM评估答案正确性，带重试机制
        
        Args:
            question: 问题
            system_answer: 系统回答
            ground_truth_answer: 标准答案
            max_retries: 最大重试次数
            
        Returns:
            评估结果字典
        """
        # 构建评估提示
        prompt = ANSWER_EVALUATION_PROMPT.format(
            question=question,
            system_answer=system_answer,
            ground_truth_answer=ground_truth_answer
        )
        
        # 构建请求数据
        request_data = {
            "model": EVALUATION_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 20480,
            "stream": False
        }
        
        last_error = None
        
        # 重试机制
        for attempt in range(max_retries):
            try:
                # 发送请求
                async with self.session.post(VLLM_API_URL, json=request_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # 检查响应结构
                        if "choices" not in result or len(result["choices"]) == 0:
                            logger.warning(f"vLLM API响应格式异常 (尝试 {attempt + 1}/{max_retries}): 缺少choices字段")
                            last_error = "API响应格式异常: 缺少choices字段"
                            continue
                        
                        choice = result["choices"][0]
                        if "message" not in choice:
                            logger.warning(f"vLLM API响应格式异常 (尝试 {attempt + 1}/{max_retries}): 缺少message字段")
                            last_error = "API响应格式异常: 缺少message字段"
                            continue
                        
                        content = choice["message"].get("content")
                        
                        # 检查content是否为None或空
                        if content is None:
                            logger.warning(f"vLLM API返回的content为None (尝试 {attempt + 1}/{max_retries})")
                            last_error = "API返回内容为None"
                            # 如果不是最后一次尝试，等待一下再重试
                            if attempt < max_retries - 1:
                                await asyncio.sleep(1)
                            continue
                        
                        if not content.strip():
                            logger.warning(f"vLLM API返回的content为空字符串 (尝试 {attempt + 1}/{max_retries})")
                            last_error = "API返回内容为空字符串"
                            if attempt < max_retries - 1:
                                await asyncio.sleep(1)
                            continue
                        
                        content = content.strip()
                        
                        # 尝试解析JSON响应
                        try:
                            evaluation_result = json.loads(content)
                            
                            # 验证必需字段
                            required_fields = ["is_correct", "confidence", "explanation"]
                            missing_fields = [field for field in required_fields if field not in evaluation_result]
                            
                            if missing_fields:
                                logger.warning(f"评估结果缺少必需字段 (尝试 {attempt + 1}/{max_retries}): {missing_fields}")
                                last_error = f"评估结果缺少字段: {missing_fields}"
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(1)
                                continue
                            
                            # 成功解析，返回结果
                            logger.debug(f"评估成功 (尝试 {attempt + 1}/{max_retries})")
                            return {
                                "is_correct": evaluation_result.get("is_correct", False),
                                "confidence": evaluation_result.get("confidence", 0.0),
                                "explanation": evaluation_result.get("explanation", ""),
                                "status": "success"
                            }
                            
                        except json.JSONDecodeError as e:
                            logger.warning(f"无法解析评估结果JSON (尝试 {attempt + 1}/{max_retries}): {content[:200]}...")
                            last_error = f"JSON解析错误: {str(e)}"
                            if attempt < max_retries - 1:
                                await asyncio.sleep(1)
                            continue
                    
                    else:
                        # HTTP状态码错误
                        error_text = await response.text()
                        logger.warning(f"vLLM API请求失败 (尝试 {attempt + 1}/{max_retries}): 状态码 {response.status}, 响应: {error_text[:200]}...")
                        last_error = f"HTTP {response.status}: {error_text[:100]}"
                        
                        # 对于某些错误码，不需要重试
                        if response.status in [400, 401, 403, 404]:
                            break
                        
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)  # HTTP错误等待更长时间
                        continue
                        
            except asyncio.TimeoutError:
                logger.warning(f"vLLM API请求超时 (尝试 {attempt + 1}/{max_retries})")
                last_error = "请求超时"
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                continue
                
            except Exception as e:
                logger.warning(f"vLLM API请求发生异常 (尝试 {attempt + 1}/{max_retries}): {e}")
                last_error = f"请求异常: {str(e)}"
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                continue
        
        # 所有重试都失败了
        logger.error(f"vLLM API调用失败，已重试 {max_retries} 次。最后错误: {last_error}")
        return {
            "is_correct": False,
            "confidence": 0.0,
            "explanation": f"API调用失败 (重试{max_retries}次): {last_error}",
            "status": "api_error"
        }
    
    def check_retrieval_accuracy(self, original_id: str, retrieved_chunk_ids: List[str]) -> Dict[str, Any]:
        """
        检查检索准确性（original_id是否在retrieved_chunk_ids中）
        
        Args:
            original_id: 原始文档ID
            retrieved_chunk_ids: 检索到的文本块ID列表
            
        Returns:
            检索准确性结果
        """
        try:
            # 检查original_id是否在retrieved_chunk_ids中
            is_retrieved = False
            
            for chunk_id in retrieved_chunk_ids:
                # 检查chunk_id是否包含original_id
                if original_id in chunk_id:
                    is_retrieved = True
                    break
            
            return {
                "is_retrieved": is_retrieved,
                "original_id": original_id,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "total_retrieved": len(retrieved_chunk_ids)
            }
            
        except Exception as e:
            logger.error(f"检查检索准确性时发生错误: {e}")
            return {
                "is_retrieved": False,
                "original_id": original_id,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "total_retrieved": len(retrieved_chunk_ids),
                "error": str(e)
            }
    
    async def evaluate_single_result(self, result_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个结果项
        
        Args:
            result_item: 单个评估结果项
            
        Returns:
            增强后的结果项
        """
        # 复制原始数据
        enhanced_result = result_item.copy()
        
        # 1. 评估答案正确性
        answer_evaluation = await self.evaluate_answer_correctness(
            question=result_item.get("question", ""),
            system_answer=result_item.get("system_answer", ""),
            ground_truth_answer=result_item.get("ground_truth_answer", "")
        )
        enhanced_result["answer_correctness"] = answer_evaluation
        
        # 2. 检查检索准确性
        retrieval_evaluation = self.check_retrieval_accuracy(
            original_id=result_item.get("original_id", ""),
            retrieved_chunk_ids=result_item.get("retrieved_chunk_ids", [])
        )
        enhanced_result["retrieval_accuracy"] = retrieval_evaluation
        
        return enhanced_result

async def evaluate_results_file(input_file: str, output_file: str, limit: Optional[int] = None):
    """
    评估结果文件，带改进的错误处理和统计信息
    
    Args:
        input_file: 输入的评估结果文件
        output_file: 输出的增强评估结果文件
        limit: 限制处理的条目数量（用于测试）
    """
    logger.warning(f"开始评估结果文件: {input_file}")
    logger.warning(f"输出文件: {output_file}")
    
    # 加载输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        logger.error(f"输入文件不存在: {input_file}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"输入文件JSON格式错误: {e}")
        return
    except Exception as e:
        logger.error(f"读取输入文件时发生错误: {e}")
        return
    
    if not isinstance(results, list):
        logger.error("输入文件格式错误：应该是JSON数组")
        return
    
    logger.warning(f"成功加载 {len(results)} 个结果项")
    
    # 限制处理数量（用于测试）
    if limit and limit > 0:
        results = results[:limit]
        logger.warning(f"测试模式：仅处理前 {limit} 个结果项")
    
    # 初始化统计信息
    stats = {
        "total_results": len(results),
        "correct_answers": 0,
        "successful_retrievals": 0,
        "api_errors": 0,
        "parse_errors": 0,
        "other_errors": 0,
        "processing_errors": 0
    }
    
    # 创建评估器
    async with AdvancedEvaluator() as evaluator:
        enhanced_results = []
        
        # 批量处理
        batch_size = 100  # 减少批次大小以提高稳定性
        total_batches = (len(results) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(results))
            batch_results = results[start_idx:end_idx]
            
            logger.warning(f"处理批次 {batch_idx + 1}/{total_batches} ({start_idx + 1}-{end_idx})")
            
            # 并发处理当前批次
            batch_tasks = [
                evaluator.evaluate_single_result(result_item)
                for result_item in batch_results
            ]
            batch_results_enhanced = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 处理并发结果
            for i, enhanced_result in enumerate(batch_results_enhanced):
                if isinstance(enhanced_result, Exception):
                    logger.error(f"处理第 {start_idx + i + 1} 个结果时发生错误: {enhanced_result}")
                    stats["processing_errors"] += 1
                    # 创建错误项
                    error_result = batch_results[i].copy()
                    error_result["answer_correctness"] = {
                        "is_correct": False,
                        "confidence": 0.0,
                        "explanation": f"处理错误: {str(enhanced_result)}",
                        "status": "processing_error"
                    }
                    error_result["retrieval_accuracy"] = {
                        "is_retrieved": False,
                        "explanation": "由于处理错误无法评估检索准确性"
                    }
                    enhanced_results.append(error_result)
                else:
                    enhanced_results.append(enhanced_result)
                    
                    # 更新统计信息
                    if enhanced_result.get("answer_correctness", {}).get("is_correct", False):
                        stats["correct_answers"] += 1
                    
                    if enhanced_result.get("retrieval_accuracy", {}).get("is_retrieved", False):
                        stats["successful_retrievals"] += 1
                    
                    # 统计错误类型
                    answer_status = enhanced_result.get("answer_correctness", {}).get("status", "unknown")
                    if answer_status == "api_error":
                        stats["api_errors"] += 1
                    elif answer_status == "parse_error":
                        stats["parse_errors"] += 1
                    elif answer_status not in ["success"]:
                        stats["other_errors"] += 1
            
            # 每5个批次后保存中间结果（防止长时间运行后丢失数据）
            if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                try:
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(enhanced_results, f, ensure_ascii=False, indent=2)
                    logger.debug(f"已保存中间结果到: {output_file}")
                except Exception as e:
                    logger.error(f"保存中间结果时发生错误: {e}")
    
    # 保存最终结果
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_results, f, ensure_ascii=False, indent=2)
        logger.warning(f"评估完成！结果已保存到: {output_file}")
    except Exception as e:
        logger.error(f"保存最终结果时发生错误: {e}")
        return
    
    # 输出详细统计信息
    logger.warning("统计信息:")
    logger.warning(f"  总结果数: {stats['total_results']}")
    logger.warning(f"  答案正确数: {stats['correct_answers']} ({stats['correct_answers']/stats['total_results']*100:.1f}%)")
    logger.warning(f"  检索成功数: {stats['successful_retrievals']} ({stats['successful_retrievals']/stats['total_results']*100:.1f}%)")
    
    if stats['api_errors'] > 0:
        logger.warning(f"  API错误数: {stats['api_errors']} ({stats['api_errors']/stats['total_results']*100:.1f}%)")
    if stats['parse_errors'] > 0:
        logger.warning(f"  解析错误数: {stats['parse_errors']} ({stats['parse_errors']/stats['total_results']*100:.1f}%)")
    if stats['other_errors'] > 0:
        logger.warning(f"  其他错误数: {stats['other_errors']} ({stats['other_errors']/stats['total_results']*100:.1f}%)")
    if stats['processing_errors'] > 0:
        logger.warning(f"  处理错误数: {stats['processing_errors']} ({stats['processing_errors']/stats['total_results']*100:.1f}%)")
    
    total_errors = stats['api_errors'] + stats['parse_errors'] + stats['other_errors'] + stats['processing_errors']
    if total_errors > 0:
        logger.warning(f"  总错误数: {total_errors} ({total_errors/stats['total_results']*100:.1f}%)")
    else:
        logger.warning("  无错误发生")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="高级RAG评估脚本")
    parser.add_argument("input_file", help="输入的评估结果文件路径")
    parser.add_argument("output_file", help="输出的增强评估结果文件路径")
    parser.add_argument("--limit", type=int, help="限制处理的结果数量（用于测试）")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_file):
        logger.error(f"输入文件不存在: {args.input_file}")
        return
    
    # 运行评估
    asyncio.run(evaluate_results_file(args.input_file, args.output_file, args.limit))

if __name__ == "__main__":
    main()