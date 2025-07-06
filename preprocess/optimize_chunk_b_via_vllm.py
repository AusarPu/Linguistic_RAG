import logging
import os
import shutil
import sys
import time
import asyncio
import aiohttp  # 用于异步HTTP请求
import json
from typing import Optional, Dict, Any

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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="使用LLM优化文本块")
    parser.add_argument(
        '--test-limit', 
        type=int, 
        default=2000, 
        help='指定一个正整数N，将只处理输入文件中的前N个块用于测试。默认为0，表示处理所有块。'
    )
    args = parser.parse_args()

    # 使用 config.py 中定义的路径
    INITIAL_CHUNKS_JSON = os.path.join(config.PROCESSED_DATA_DIR, "processed_knowledge_base_chunks.json")
    
    # 根据是否为测试模式，决定输出文件名
    if args.test_limit > 0:
        output_filename = "enhanced_knowledge_base_chunks_llm_TEST.json"
    else:
        output_filename = "enhanced_knowledge_base_chunks_llm.json"
        
    OUTPUT_REFINED_JSON = os.path.join(config.PROCESSED_DATA_DIR, output_filename)

    # 确保输入文件存在
    if not os.path.exists(INITIAL_CHUNKS_JSON):
        logger.critical(f"错误: 输入文件 '{INITIAL_CHUNKS_JSON}' 未找到。请确保已运行生成初始块的脚本。")
    else:
        logger.warning(f"输入文件: {INITIAL_CHUNKS_JSON}")
        logger.warning(f"输出文件: {OUTPUT_REFINED_JSON}")
        
        # 运行主异步函数，并传入 limit 参数
        limit_value = args.test_limit if args.test_limit > 0 else None
        asyncio.run(refine_all_chunks_with_llm(INITIAL_CHUNKS_JSON, OUTPUT_REFINED_JSON, limit=limit_value))

        logger.warning(f"\n--- LLM 批量块优化流程运行完毕 --- (测试限制: {args.test_limit if args.test_limit > 0 else '无'})")
