import logging
import os
import shutil
import sys
import time
import asyncio
import aiohttp  # 用于异步HTTP请求
import json
from typing import Optional, Dict, Any
from collections import defaultdict

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script import config

# --- 配置日志 ---
config.setup_logging()
logger = logging.getLogger(__name__)

# --- VLLM 服务和模型配置 (从 config.py 导入) ---
VLLM_OPTIMIZER_API_URL = config.GENERATOR_API_URL
OPTIMIZER_MODEL_NAME = config.GENERATOR_MODEL_NAME_FOR_API
OPTIMIZER_GENERATION_CONFIG = config.GENERATOR_RAG_CONFIG
# 使用config.py中的新超时配置
VLLM_REQUEST_TIMEOUT_SINGLE = config.VLLM_REQUEST_TIMEOUT_SINGLE
VLLM_REQUEST_TIMEOUT_TOTAL = config.VLLM_REQUEST_TIMEOUT_TOTAL
OPTIMIZATION_BATCH_SIZE = config.OPTIMIZATION_BATCH_SIZE

# --- 元数据生成配置 ---
VLLM_METADATA_API_URL = config.GENERATOR_API_URL
METADATA_MODEL_NAME = config.GENERATOR_MODEL_NAME_FOR_API
METADATA_GENERATION_CONFIG = {
    "temperature": 0.4,  # 对于信息提取和遵循指令，较低的温度可能更好
    "max_tokens": 4096,  # 需要足够容纳关键词、问题和JSON结构
    # "response_format": {"type": "json_object"} # 如果VLLM和模型支持，强烈建议使用
}
METADATA_BATCH_SIZE = config.OPTIMIZATION_BATCH_SIZE

# --- LLM提示词模板：为单个块生成元数据并判断意义 ---
METADATA_GENERATION_PROMPT_TEMPLATE = """你是一位专业的知识分析和信息提取专家。
我将提供一段文本（一个文本块）及其标识符。请你基于这段文本内容完成以下任务：

1.  **评估内容意义**：
    * 首先，请判断该文本块是否包含实质性的、有意义的、适合用于构建问答知识库的信息。
    * 如果文本块内容非常稀疏、几乎没有信息量（例如，可能只是页眉、页脚、孤立的图表编号、空白内容，或非常零碎以至于无法理解的片段），请在返回结果中指明。

2.  **关键词摘要提取 (如果文本块有意义)**：
    * 提取或生成若干（例如，1到5个，不必强求数量，质量优先）最能概括该文本块核心内容的"关键词摘要"或"标签"。
    * 每个关键词摘要应该简短精炼，方便作为标签查找，或者其他能准确捕捉核心概念的短语组合。
    * 关键词应该是全局通用的，不应该出现类似``这本书``，``本书作者``这类，而应该明确这本书是什么，作者是什么，便于以后查找
    * 可以参考文本块标识符中的信息来生成更准确的关键词
    * 如果文本块评估为无实质意义，则关键词摘要列表应为空。

3.  **预生成相关问题 (如果文本块有意义)**：
    * 根据该文本块的内容，生成若干（例如，1到3个，不必强求数量，质量优先）用户最有可能提出的、并且该文本块能够清晰、直接回答的高质量问题。
    * 生成的问题应该是全局通用的，不应该出现类似``这本书``，``本书作者``这类，而应该明确这本书是什么，作者是什么，便于以后查找
    * 可以参考文本块标识符中的信息来生成更准确的问题
    * 如果文本块评估为无实质意义，则生成问题列表应为空。

请将你的结果以严格的JSON格式返回，包含以下字段：
- "is_meaningful": 布尔值 (true 表示有意义，false 表示无实质意义应考虑丢弃)。
- "reason_if_not_meaningful": 字符串，如果 "is_meaningful" 为 false，请简要说明原因 (例如："仅包含页眉", "内容过短且无信息")。如果 "is_meaningful" 为 true，此字段可为空字符串。
- "keyword_summaries": 一个包含所提取/生成的关键词摘要字符串的列表。如果 "is_meaningful" 为 false，此列表必须为空。
- "generated_questions": 一个包含所生成的问句字符串的列表。如果 "is_meaningful" 为 false，此列表必须为空。

确保JSON格式正确，所有字符串值内部的特殊字符（如换行符、双引号）都已正确转义。不要包含任何其他解释或对话。

文本块标识符：{chunk_id}

提供的文本块如下：
```
{text_chunk_content}
```
"""

# --- 优化中心块B的提示词模板 ---
# 这个提示词要求LLM只返回包含优化后块B文本的JSON对象
CHUNK_OPTIMIZATION_PROMPT_TEMPLATE = """
你是一位专业的文本编辑。你的任务是基于其前文（块A）和后文（块C）来优化中间的文本块（块B）。
目标是确保块B在语义上连贯、完整，并且其与块A或块C的边界处没有句子被不自然地切断。
请只进行必要的最小调整，例如，如果一个句子明显在块A的末尾和块B的开头之间被切断，你可以将属于块B的句子片段从块A的末尾移入块B的开头，或者将属于块A的句子片段从块B的开头移回块A的末尾，以确保块B以一个完整的句子开始和结束（如果上下文允许）。
同样地，处理块B和块C之间的边界。
不要添加任何新的、源于外部的信息，也不要对块B的内容进行实质性的重写或总结。优化后的块B应忠于原文的意义和风格，并且长度应与原始块B大致相似。
但是注意，你可以改写你认为的由OCR识别错误的部分，并进行格式优化，把格式不清楚的部分，都用markdown格式转写一遍

块A (前文内容，如果块B是文档的第一个块，则此部分内容为 "None" 或非常简短):
```
{text_chunk_a}
```

块B (需要优化的当前块，这是你的主要操作对象):
```
{text_chunk_b}
```

块C (后文内容，如果块B是文档的最后一个块，则此部分内容为 "None" 或非常简短):
```
{text_chunk_c}
```

请严格按照以下JSON格式返回优化后的块B的文本。
确保 "optimized_chunk_B_text" 字段的值是一个符合JSON规范的字符串，这意味着字符串内部的特殊字符（如换行符、双引号、反斜杠等）都需要被正确转义（例如，换行符应表示为 \\n，双引号应表示为 \\"，反斜杠应表示为 \\\\）。
{{
  "optimized_chunk_B_text": "这里是优化后的块B的文本内容..."
}}
"""


async def generate_metadata_for_chunk_via_vllm(text_chunk_content, chunk_id=None, max_retries=3):
    """
    异步调用VLLM API为单个文本块生成元数据（关键词摘要和相关问题）。
    
    Args:
        text_chunk_content (str): 文本块内容
        max_retries (int): 最大重试次数，默认为3
    
    Returns:
        dict: 包含状态和数据的字典
            成功时: {'status': 'success', 'data': {...}}
            失败时: {'status': 'failed', 'reason': '...', 'content': '...'}
    """
    if not text_chunk_content or not text_chunk_content.strip():
        return {
            'status': 'success',
            'data': {
                'is_meaningful': False,
                'reason_if_not_meaningful': '文本块为空或仅包含空白字符',
                'keyword_summaries': [],
                'generated_questions': []
            }
        }
    
    # 构建请求数据
    prompt = METADATA_GENERATION_PROMPT_TEMPLATE.format(
        chunk_id=chunk_id or "未知",
        text_chunk_content=text_chunk_content
    )
    request_data = {
        "model": METADATA_MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        **METADATA_GENERATION_CONFIG
    }
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            # 使用单次请求超时
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=VLLM_REQUEST_TIMEOUT_SINGLE)) as session:
                async with session.post(VLLM_METADATA_API_URL, json=request_data) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        
                        # 提取LLM生成的内容
                        if 'choices' in response_data and len(response_data['choices']) > 0:
                            content = response_data['choices'][0]['message']['content'].strip()
                            
                            try:
                                # 尝试解析JSON
                                metadata = json.loads(content)
                                
                                # 验证必需字段
                                required_fields = ['is_meaningful', 'reason_if_not_meaningful', 'keyword_summaries', 'generated_questions']
                                if all(field in metadata for field in required_fields):
                                    return {'status': 'success', 'data': metadata}
                                else:
                                    missing_fields = [field for field in required_fields if field not in metadata]
                                    raise ValueError(f"响应缺少必需字段: {missing_fields}")
                                    
                            except json.JSONDecodeError as e:
                                raise ValueError(f"无法解析JSON响应: {e}, 内容: {content[:200]}...")
                        else:
                            raise ValueError("API响应格式异常：缺少choices字段")
                    else:
                        error_text = await response.text()
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"HTTP {response.status}: {error_text[:200]}"
                        )
                        
        except asyncio.TimeoutError as e:
            last_exception = f"请求超时 (尝试 {attempt + 1}/{max_retries}): {str(e)}"
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                await asyncio.sleep(wait_time)
                continue
        except aiohttp.ClientResponseError as e:
            last_exception = f"HTTP错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}"
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
                continue
        except Exception as e:
            last_exception = f"其他错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}"
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
                continue
    
    # 所有重试都失败了
    logging.error(f"元数据生成失败，已重试{max_retries}次: {last_exception}")
    return {
        'status': 'failed',
        'reason': last_exception,
        'content': text_chunk_content[:200] + '...' if len(text_chunk_content) > 200 else text_chunk_content
    }


async def optimize_chunk_b_via_vllm(
        text_a: Optional[str],
        text_b: str,
        text_c: Optional[str],
        session: aiohttp.ClientSession,  # 传入 session 以复用连接
        request_id: str = "N/A"  # 用于日志追踪
) -> Dict[str, Any]:
    """
    使用 VLLM API 异步地优化中心块 B。
    这是你在上一步测试并使其工作的函数。
    """
    func_start_time = time.time()
    # logger.info(f"[{func_start_time:.3f}] [ReqID: {request_id}] 开始优化块B (首50字符): '{text_b[:50]}...'") # 日志移到调用处

    formatted_text_a = text_a if text_a is not None else "N/A (此为文档首个待优化块的上下文)"
    formatted_text_c = text_c if text_c is not None else "N/A (此为文档末尾待优化块的上下文)"

    final_prompt = CHUNK_OPTIMIZATION_PROMPT_TEMPLATE.format(
        text_chunk_a=formatted_text_a,
        text_chunk_b=text_b,
        text_chunk_c=formatted_text_c
    )
    api_messages = [{"role": "user", "content": final_prompt}]
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = {
        "model": OPTIMIZER_MODEL_NAME,
        "messages": api_messages,
        **{k: v for k, v in OPTIMIZER_GENERATION_CONFIG.items() if v is not None},
        "stream": False,
    }

    # --- 实际的 aiohttp API 调用逻辑 ---
    try:
        async with session.post(VLLM_OPTIMIZER_API_URL, headers=headers, json=payload) as response:
            response.raise_for_status()
            response_data = await response.json()

            if not response_data.get("choices") or len(response_data["choices"]) == 0:
                logger.warning(f"[{time.time():.3f}] [ReqID: {request_id}] API响应的 'choices' 字段无效或为空。")
                return {"status": "api_error", "reason": "empty_choices"}

            message_content = response_data["choices"][0].get("message", {}).get("content")
            if not message_content:
                logger.warning(f"[{time.time():.3f}] [ReqID: {request_id}] API响应的 message content 为空。")
                return {"status": "api_error", "reason": "empty_content"}

            try:
                if message_content.startswith("```json"):
                    message_content = message_content.strip("```json").strip("`").strip()
                message_content = message_content.strip("\n")
                parsed_json_output = json.loads(message_content)
                optimized_text = parsed_json_output.get("optimized_chunk_B_text")

                if optimized_text is not None and isinstance(optimized_text, str):
                    return {"status": "success", "text": optimized_text.strip()}
                else:
                    logger.warning(
                        f"[{time.time():.3f}] [ReqID: {request_id}] API响应JSON中 'optimized_chunk_B_text' 缺失或非字符串。Parsed: {parsed_json_output}")
                    return {"status": "api_error", "reason": "missing_key"}

            except json.JSONDecodeError as e:
                # logger.error(...) # 日志记录移到调用处处理
                return {"status": "json_decode_error", "content": message_content}

    except Exception as e:  # 捕获所有可能的 aiohttp 异常和超时等
        logger.error(f"[{time.time():.3f}] [ReqID: {request_id}] 调用 VLLM API 时发生严重错误: {e}", exc_info=True)
        return {"status": "exception", "error": e}


# -----------------------------------------------------------------------------
# 主编排函数：加载初级块，异步使用LLM通过滑动窗口优化中心块
# -----------------------------------------------------------------------------
async def process_metadata_batch(chunks_batch):
    """
    批量处理文本块的元数据生成。
    
    Args:
        chunks_batch (list): 文本块列表
    
    Returns:
        tuple: (成功数量, 有意义数量, 无意义数量, 失败数量, 处理结果列表)
    """
    tasks = []
    for chunk in chunks_batch:
        text_content = chunk.get('text', '').strip()
        if not text_content:
            # 空文本块直接标记为无意义
            async def empty_result():
                return {
                    'status': 'success',
                    'data': {
                        'is_meaningful': False,
                        'reason_if_not_meaningful': '文本块为空',
                        'keyword_summaries': [],
                        'generated_questions': []
                    }
                }
            tasks.append(empty_result())
        else:
            chunk_id = chunk.get('chunk_id', '未知')
            tasks.append(generate_metadata_for_chunk_via_vllm(text_content, chunk_id))
    
    # 并发执行所有任务
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=VLLM_REQUEST_TIMEOUT_TOTAL)
    except asyncio.TimeoutError:
        logging.error(f"批次处理超时 ({VLLM_REQUEST_TIMEOUT_TOTAL}秒)")
        results = [{'status': 'failed', 'reason': '批次处理超时', 'content': ''} for _ in tasks]
    
    # 统计和处理结果
    success_count = 0
    meaningful_count = 0
    not_meaningful_count = 0
    failed_count = 0
    processed_chunks = []
    
    for i, (chunk, result) in enumerate(zip(chunks_batch, results)):
        try:
            if isinstance(result, Exception):
                # 处理异常情况
                logging.error(f"块 {chunk.get('chunk_id', i)} 处理异常: {result}")
                failed_count += 1
                continue
            
            if result['status'] == 'success':
                success_count += 1
                metadata = result['data']
                
                if metadata['is_meaningful']:
                    meaningful_count += 1
                    # 为有意义的块添加元数据
                    enhanced_chunk = chunk.copy()
                    enhanced_chunk.update({
                        'keyword_summaries': metadata['keyword_summaries'],
                        'generated_questions': metadata['generated_questions'],
                        'is_meaningful': True
                    })
                    processed_chunks.append(enhanced_chunk)
                else:
                    not_meaningful_count += 1
                    # 记录无意义的块但不添加到最终结果
                    logging.info(f"块 {chunk.get('chunk_id', i)} 被判定为无意义: {metadata.get('reason_if_not_meaningful', '未知原因')}")
            else:
                failed_count += 1
                logging.error(f"块 {chunk.get('chunk_id', i)} 元数据生成失败: {result.get('reason', '未知错误')}")
                
        except Exception as e:
            logging.error(f"处理块 {chunk.get('chunk_id', i)} 结果时发生异常: {e}")
            failed_count += 1
    
    return success_count, meaningful_count, not_meaningful_count, failed_count, processed_chunks


async def refine_all_chunks_with_llm(
        input_chunks_json_path: str,
        output_refined_chunks_json_path: str,
        limit: Optional[int] = None  # 新增 limit 参数
) -> None:
    """
    加载由前一阶段生成的文本块JSON文件，
    异步地使用滑动窗口和LLM来优化每个文档中的中心块，
    并将优化后的块保存到新的JSON文件中。
    此版本包含一个针对单个文档内块的重试队列，用于处理API解析错误。
    """
    try:
        with open(input_chunks_json_path, 'r', encoding='utf-8') as f:
            all_initial_chunks = json.load(f)
        logger.warning(f"成功从 '{input_chunks_json_path}' 加载 {len(all_initial_chunks)} 个初始文本块。")

        if limit is not None and limit > 0:
            logger.warning(f"--- 测试模式：仅处理前 {limit} 个文本块。 ---")
            all_initial_chunks = all_initial_chunks[:limit]
    except Exception as e:
        logger.error(f"错误：无法加载初始文本块文件 '{input_chunks_json_path}': {e}", exc_info=True)
        return

    chunks_by_doc = {}
    for chunk in all_initial_chunks:
        doc_name = chunk['doc_name']
        if doc_name not in chunks_by_doc:
            chunks_by_doc[doc_name] = []
        chunks_by_doc[doc_name].append(chunk)

    for doc_name in chunks_by_doc:
        chunks_by_doc[doc_name].sort(key=lambda c: (c['page_number'], int(c['chunk_id'].split('_c')[-1])))

    total_llm_calls = 0
    total_successful_optimizations = 0
    MAX_RETRIES = 4

    # --- 核心数据结构：用于存储所有优化结果，键为 chunk_id ---
    optimized_chunks_map = {}
    
    # 记录整体流程开始时间
    overall_start_time = time.time()
    logger.warning(f"开始整体优化流程，总体超时限制: {VLLM_REQUEST_TIMEOUT_TOTAL}秒 ({VLLM_REQUEST_TIMEOUT_TOTAL/3600:.1f}小时)")
    logger.warning(f"单个块优化超时限制: {VLLM_REQUEST_TIMEOUT_SINGLE}秒 ({VLLM_REQUEST_TIMEOUT_SINGLE/60:.1f}分钟)")
    logger.warning(f"分批处理大小: {OPTIMIZATION_BATCH_SIZE}")
    
    # 增量写入：创建临时输出文件
    temp_output_path = output_refined_chunks_json_path + ".tmp"
    processed_chunks_count = 0

    # 为单个请求创建超时配置
    single_timeout = aiohttp.ClientTimeout(total=VLLM_REQUEST_TIMEOUT_SINGLE)
    connector = aiohttp.TCPConnector(limit=1000)
    async with aiohttp.ClientSession(timeout=single_timeout, connector=connector) as session:
        all_tasks_info = []
        # --- 阶段1: 创建所有需要处理的任务 --- 
        for doc_name, original_doc_chunks in chunks_by_doc.items():
            if len(original_doc_chunks) < 3:
                continue
            # 只处理中间块
            for k in range(1, len(original_doc_chunks) - 1):
                all_tasks_info.append({
                    'doc_name': doc_name,
                    'chunk_a': original_doc_chunks[k - 1],
                    'chunk_b': original_doc_chunks[k],
                    'chunk_c': original_doc_chunks[k + 1],
                    'retry_count': 0
                })

        # --- 阶段2: 带重试逻辑的分批并发处理 ---
        tasks_to_process = all_tasks_info
        while tasks_to_process:
            # 检查总体超时
            elapsed_time = time.time() - overall_start_time
            if elapsed_time > VLLM_REQUEST_TIMEOUT_TOTAL:
                logger.warning(f"达到总体超时限制 {VLLM_REQUEST_TIMEOUT_TOTAL}秒，已处理时间: {elapsed_time:.2f}秒")
                logger.warning(f"剩余 {len(tasks_to_process)} 个任务将被跳过")
                break
            
            # 分批处理
            current_batch_info = tasks_to_process[:OPTIMIZATION_BATCH_SIZE]
            tasks_to_process = tasks_to_process[OPTIMIZATION_BATCH_SIZE:] # 剩余任务
            retry_tasks = [] # 当前批次的重试任务

            logger.warning(f"处理批次: {len(current_batch_info)} 个任务，剩余: {len(tasks_to_process)} 个任务")
            logger.warning(f"已用时间: {elapsed_time:.2f}秒，剩余时间: {VLLM_REQUEST_TIMEOUT_TOTAL - elapsed_time:.2f}秒")
            
            tasks = []
            for task_info in current_batch_info:
                request_id = f"{task_info['doc_name']}_{task_info['chunk_b']['chunk_id']}_retry{task_info['retry_count']}"
                logger.debug(f"  准备LLM优化任务 for chunk_id: {task_info['chunk_b']['chunk_id']} (尝试: {task_info['retry_count'] + 1})")
                
                # 记录单个任务开始时间
                task_start_time = time.time()
                task_info['start_time'] = task_start_time
                
                tasks.append(optimize_chunk_b_via_vllm(
                    task_info['chunk_a']['text'],
                    task_info['chunk_b']['text'],
                    task_info['chunk_c']['text'],
                    session,
                    request_id=request_id
                ))
                total_llm_calls += 1

            if not tasks:
                break

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result_or_exc in enumerate(results):
                task_info = current_batch_info[i]
                original_chunk_b = task_info['chunk_b']
                chunk_id = original_chunk_b['chunk_id']
                
                # 计算单个任务耗时
                task_duration = time.time() - task_info.get('start_time', time.time())
                
                if isinstance(result_or_exc, Exception):
                    logger.critical(f"  LLM优化块 '{chunk_id}' 时发生异常 (耗时: {task_duration:.2f}秒): {result_or_exc}")
                    continue # 异常情况不重试，直接跳过

                status = result_or_exc.get("status")
                if status == "success":
                    refined_chunk = dict(original_chunk_b)
                    refined_chunk['text'] = result_or_exc["text"]
                    optimized_chunks_map[chunk_id] = refined_chunk
                    total_successful_optimizations += 1
                    processed_chunks_count += 1
                    logger.info(f"  块 '{chunk_id}' 文本已由LLM更新 (耗时: {task_duration:.2f}秒)")
                elif status == "json_decode_error" and task_info['retry_count'] < MAX_RETRIES:
                    logger.error(f"  块 '{chunk_id}' 解析失败，将重试 (耗时: {task_duration:.2f}秒)。内容: '{result_or_exc.get('content', '')[:100]}...'")
                    task_info['retry_count'] += 1
                    retry_tasks.append(task_info)
                else:
                    if status == "json_decode_error":
                        logger.critical(f"  块 '{chunk_id}' 达到最大重试次数，放弃优化 (耗时: {task_duration:.2f}秒)")
                    else:
                        logger.error(f"  LLM优化块 '{chunk_id}' 未返回有效文本 (status: {status}, 耗时: {task_duration:.2f}秒), 保留原始文本")
            
            # 将重试任务添加到待处理队列的开头（优先处理重试）
            tasks_to_process = retry_tasks + tasks_to_process
            
            # 增量写入：每个批次完成后保存当前进度
            if processed_chunks_count > 0:
                try:
                    # 按原始顺序重建当前已处理的结果
                    current_refined_chunks_list = []
                    for chunk in all_initial_chunks:
                        chunk_id = chunk['chunk_id']
                        if chunk_id in optimized_chunks_map:
                            current_refined_chunks_list.append(optimized_chunks_map[chunk_id])
                        else:
                            current_refined_chunks_list.append(chunk)
                    
                    # 写入临时文件
                    with open(temp_output_path, 'w', encoding='utf-8') as temp_file:
                        json.dump(current_refined_chunks_list, temp_file, ensure_ascii=False, indent=2)
                    logger.debug(f"增量保存进度：已处理 {processed_chunks_count} 个块到临时文件")
                except Exception as e:
                    logger.error(f"增量写入临时文件失败: {e}")

    # --- 阶段3: 按原始顺序重建最终列表 --- 
    final_refined_chunks_list = []
    for chunk in all_initial_chunks:
        chunk_id = chunk['chunk_id']
        if chunk_id in optimized_chunks_map:
            final_refined_chunks_list.append(optimized_chunks_map[chunk_id])
        else:
            final_refined_chunks_list.append(chunk)

    # 计算总体处理时间
    total_elapsed_time = time.time() - overall_start_time
    
    logger.warning(
        f"\n所有文档LLM块优化处理完成。总共尝试优化 {total_llm_calls} 个块，成功优化 {total_successful_optimizations} 个。")
    logger.warning(f"最终总块数为: {len(final_refined_chunks_list)}")
    logger.warning(f"总体处理时间: {total_elapsed_time:.2f}秒 ({total_elapsed_time/60:.1f}分钟)")
    logger.warning(f"平均每个块处理时间: {total_elapsed_time/max(total_llm_calls, 1):.2f}秒")
    logger.warning(f"成功率: {total_successful_optimizations/max(total_llm_calls, 1)*100:.1f}%")

    try:
        # 最终写入：从临时文件移动到正式文件
        if os.path.exists(temp_output_path):
            shutil.move(temp_output_path, output_refined_chunks_json_path)
            logger.info(f"所有优化后的文本块已从临时文件移动到: '{output_refined_chunks_json_path}'")
        else:
            # 如果临时文件不存在，使用传统方式写入
            with open(output_refined_chunks_json_path, 'w', encoding='utf-8') as outfile:
                json.dump(final_refined_chunks_list, outfile, ensure_ascii=False, indent=2)
            logger.info(f"所有优化后的文本块已保存到: '{output_refined_chunks_json_path}'")
    except Exception as e:
        logger.critical(f"保存优化后的块到 JSON 文件 '{output_refined_chunks_json_path}' 时发生错误: {e}", exc_info=True)
        # 尝试清理临时文件
        try:
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
        except:
            pass


async def enhance_chunks_with_llm_metadata(input_chunks_json_path, output_chunks_json_path, test_limit=None):
    """
    主编排函数：为文本块生成元数据（关键词摘要和相关问题），并筛选有意义的块。
    
    Args:
        input_chunks_json_path (str): 输入的文本块JSON文件路径
        output_chunks_json_path (str): 输出的增强文本块JSON文件路径
        test_limit (int, optional): 测试模式下限制处理的块数量
    """
    import time
    import os
    
    # 加载初始文本块
    logging.warning(f"正在加载初始文本块: {input_chunks_json_path}")
    with open(input_chunks_json_path, 'r', encoding='utf-8') as f:
        initial_chunks = json.load(f)
    
    total_chunks = len(initial_chunks)
    logging.warning(f"加载了 {total_chunks} 个初始文本块")
    
    # 应用测试限制
    if test_limit and test_limit < total_chunks:
        initial_chunks = initial_chunks[:test_limit]
        logging.warning(f"测试模式：限制处理 {test_limit} 个文本块")
    
    # 设置增量写入的临时文件
    temp_output_path = output_chunks_json_path + '.tmp'
    
    # 初始化统计信息
    total_success = 0
    total_meaningful = 0
    total_not_meaningful = 0
    total_failed = 0
    all_enhanced_chunks = []
    
    start_time = time.time()
    
    # 分批处理
    for batch_start in range(0, len(initial_chunks), METADATA_BATCH_SIZE):
        batch_end = min(batch_start + METADATA_BATCH_SIZE, len(initial_chunks))
        current_batch = initial_chunks[batch_start:batch_end]
        
        logging.warning(f"处理批次 {batch_start//METADATA_BATCH_SIZE + 1}: 块 {batch_start+1}-{batch_end} (共 {len(current_batch)} 个)")
        
        # 处理当前批次
        batch_success, batch_meaningful, batch_not_meaningful, batch_failed, batch_enhanced = await process_metadata_batch(current_batch)
        
        # 更新统计信息
        total_success += batch_success
        total_meaningful += batch_meaningful
        total_not_meaningful += batch_not_meaningful
        total_failed += batch_failed
        
        # 将处理结果添加到总列表
        all_enhanced_chunks.extend(batch_enhanced)
        
        # 增量写入临时文件
        with open(temp_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_enhanced_chunks, f, ensure_ascii=False, indent=2)
        
        logging.warning(f"批次完成 - 成功: {batch_success}, 有意义: {batch_meaningful}, 无意义: {batch_not_meaningful}, 失败: {batch_failed}")
        
        # 检查是否达到处理上限
        if test_limit and len(all_enhanced_chunks) >= test_limit:
            logging.warning(f"已达到测试限制 {test_limit}，停止处理")
            break
    
    # 最终统计和保存
    end_time = time.time()
    processing_time = end_time - start_time
    
    logging.critical(f"元数据生成完成！")
    logging.critical(f"总处理时间: {processing_time:.2f} 秒")
    logging.critical(f"总块数: {len(initial_chunks)}, 成功: {total_success}, 有意义: {total_meaningful}, 无意义: {total_not_meaningful}, 失败: {total_failed}")
    logging.critical(f"最终保留的有意义块数: {len(all_enhanced_chunks)}")
    
    # 将临时文件移动到最终位置
    if os.path.exists(temp_output_path):
        os.rename(temp_output_path, output_chunks_json_path)
        logging.critical(f"结果已保存到: {output_chunks_json_path}")
    else:
        logging.error(f"临时文件不存在: {temp_output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="使用VLLM优化文本块或生成元数据")
    parser.add_argument("mode", nargs='?', choices=["optimize", "metadata", "pipeline"], default="pipeline", help="运行模式：optimize=优化文本块，metadata=生成元数据，pipeline=完整流水线（默认）")
    parser.add_argument("input_file", nargs='?', help="输入的JSON文件路径")
    parser.add_argument("output_file", nargs='?', help="输出的JSON文件路径")
    parser.add_argument("--test-limit", type=int, default=50, help="测试模式：限制处理的文档/块数量（默认：50）")
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 默认流水线模式：从processed_knowledge_base_chunks.json开始完整处理
    if args.mode == "pipeline":
        # 使用默认路径
        input_chunks_path = os.path.join(config.PROCESSED_DATA_DIR, "processed_knowledge_base_chunks.json")
        optimized_chunks_path = os.path.join(config.PROCESSED_DATA_DIR, "optimized_knowledge_base_chunks.json")
        enhanced_chunks_path = os.path.join(config.PROCESSED_DATA_DIR, "enhanced_knowledge_base_chunks_llm.json")
        
        # 检查输入文件是否存在
        if not os.path.exists(input_chunks_path):
            logging.critical(f"错误: 输入文件 '{input_chunks_path}' 未找到。请确保已运行生成初始块的脚本。")
            exit(1)
        
        logging.warning(f"=== 开始完整的LLM处理流水线 ===")
        logging.warning(f"输入文件: {input_chunks_path}")
        logging.warning(f"中间文件: {optimized_chunks_path}")
        logging.warning(f"最终文件: {enhanced_chunks_path}")
        
        # 第一步：优化文本块
        logging.warning(f"\n步骤1: 开始优化文本块，配置: 单次超时={VLLM_REQUEST_TIMEOUT_SINGLE}s, 总超时={VLLM_REQUEST_TIMEOUT_TOTAL}s, 批次大小={OPTIMIZATION_BATCH_SIZE}")
        asyncio.run(refine_all_chunks_with_llm(
            input_chunks_path, 
            optimized_chunks_path, 
            limit=args.test_limit
        ))
        
        # 第二步：生成元数据
        logging.warning(f"\n步骤2: 开始生成元数据，配置: 单次超时={VLLM_REQUEST_TIMEOUT_SINGLE}s, 总超时={VLLM_REQUEST_TIMEOUT_TOTAL}s, 批次大小={METADATA_BATCH_SIZE}")
        asyncio.run(enhance_chunks_with_llm_metadata(
            optimized_chunks_path, 
            enhanced_chunks_path, 
            test_limit=args.test_limit
        ))
        
        logging.warning(f"\n=== 完整流水线处理完成 ===")
        logging.warning(f"最终增强文件已保存到: {enhanced_chunks_path}")
        
    elif args.mode == "optimize":
        if not args.input_file or not args.output_file:
            logging.critical("错误: optimize模式需要指定input_file和output_file参数")
            exit(1)
        logging.warning(f"开始优化文本块，配置: 单次超时={VLLM_REQUEST_TIMEOUT_SINGLE}s, 总超时={VLLM_REQUEST_TIMEOUT_TOTAL}s, 批次大小={OPTIMIZATION_BATCH_SIZE}")
        # 运行优化函数
        asyncio.run(refine_all_chunks_with_llm(
            args.input_file, 
            args.output_file, 
            limit=args.test_limit
        ))
    elif args.mode == "metadata":
        if not args.input_file or not args.output_file:
            logging.critical("错误: metadata模式需要指定input_file和output_file参数")
            exit(1)
        logging.warning(f"开始生成元数据，配置: 单次超时={VLLM_REQUEST_TIMEOUT_SINGLE}s, 总超时={VLLM_REQUEST_TIMEOUT_TOTAL}s, 批次大小={METADATA_BATCH_SIZE}")
        # 运行元数据生成函数
        asyncio.run(enhance_chunks_with_llm_metadata(
            args.input_file, 
            args.output_file, 
            test_limit=args.test_limit
        ))

    logging.warning(f"\n--- LLM 处理流程运行完毕 ---")