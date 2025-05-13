import logging
import os
import time
import asyncio
import aiohttp  # 用于异步HTTP请求
import json
from typing import Optional, Dict, Any

# --- 配置日志 ---
# 你可以根据你的项目调整日志级别和格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- VLLM 服务和模型配置 (请根据你的实际部署修改这些值) ---
# 你的 VLLM 服务 API 端点 (通常是 chat/completions)
VLLM_OPTIMIZER_API_URL = "http://localhost:8001/v1/chat/completions"  # 示例URL，请修改
# 你在 VLLM 中部署的用于此优化任务的模型名称
OPTIMIZER_MODEL_NAME = "/home/pushihao/RAG/models/Qwen/Qwen3-30B-A3B-FP8"  # 例如 "Qwen/Qwen1.5-7B-Chat"
# LLM 生成参数配置
OPTIMIZER_GENERATION_CONFIG = {
    "temperature": 0.6,  # 对于编辑和遵循指令的任务，较低的温度通常更好
    "max_tokens": 5000,  # 需要足够大以容纳优化后的块B，可以基于你块的平均/最大长度设置
    "top_p": 0.95,      # 其他你可能想控制的参数
    # "stop": ["\n\n\n"] # 如果需要特定的停止符
}
# API 请求超时时间 (秒)
VLLM_REQUEST_TIMEOUT = 60*30

# --- 优化中心块B的提示词模板 ---
# 这个提示词要求LLM只返回包含优化后块B文本的JSON对象
CHUNK_OPTIMIZATION_PROMPT_TEMPLATE = """
你是一位专业的文本编辑。你的任务是基于其前文（块A）和后文（块C）来优化中间的文本块（块B）。
目标是确保块B在语义上连贯、完整，并且其与块A或块C的边界处没有句子被不自然地切断。
请只进行必要的最小调整，例如，如果一个句子明显在块A的末尾和块B的开头之间被切断，你可以将属于块B的句子片段从块A的末尾移入块B的开头，或者将属于块A的句子片段从块B的开头移回块A的末尾，以确保块B以一个完整的句子开始和结束（如果上下文允许）。
同样地，处理块B和块C之间的边界。
不要添加任何新的、源于外部的信息，也不要对块B的内容进行实质性的重写或总结。优化后的块B应忠于原文的意义和风格，并且长度应与原始块B大致相似。
但是你可以改写你认为的由OCR识别错误的部分，并进行格式优化

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
) -> Optional[str]:
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

    optimized_text_result = None
    # --- 实际的 aiohttp API 调用逻辑 ---
    # (这是你之前测试脚本中已替换模拟部分并使其工作的逻辑)
    try:
        # logger.info(f"[{time.time():.3f}] [ReqID: {request_id}] 发送优化请求到: {VLLM_OPTIMIZER_API_URL}")
        # logger.debug(f"[{time.time():.3f}] [ReqID: {request_id}] Payload: {json.dumps(payload, ensure_ascii=False)}")

        async with session.post(VLLM_OPTIMIZER_API_URL, headers=headers, json=payload) as response:
            response.raise_for_status()
            response_data = await response.json()
            # logger.debug(f"[{time.time():.3f}] [ReqID: {request_id}] VLLM API 响应: {json.dumps(response_data, ensure_ascii=False, indent=2)}")

            if response_data.get("choices") and len(response_data["choices"]) > 0:
                message_content = response_data["choices"][0].get("message", {}).get("content")
                if message_content:
                    try:
                        if message_content.startswith("```json"):
                            message_content = message_content.strip("```json").strip("`").strip()
                        message_content = message_content.strip("\n")
                        parsed_json_output = json.loads(message_content)
                        optimized_text = parsed_json_output.get("optimized_chunk_B_text")
                        if optimized_text is not None and isinstance(optimized_text, str):
                            optimized_text_result = optimized_text.strip()
                            # logger.info(f"[{time.time():.3f}] [ReqID: {request_id}] 成功解析优化文本。")
                        else:
                            logger.warning(
                                f"[{time.time():.3f}] [ReqID: {request_id}] API响应JSON中 'optimized_chunk_B_text' 缺失或非字符串。Parsed: {parsed_json_output}")
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"[{time.time():.3f}] [ReqID: {request_id}] 解析API返回的 message content 为JSON时失败: {e}. Content: '{message_content}'")
                else:
                    logger.warning(f"[{time.time():.3f}] [ReqID: {request_id}] API响应的 message content 为空。")
            else:
                logger.warning(f"[{time.time():.3f}] [ReqID: {request_id}] API响应的 'choices' 字段无效或为空。")
    except Exception as e:  # 捕获所有可能的 aiohttp 异常和超时等
        logger.error(f"[{time.time():.3f}] [ReqID: {request_id}] 调用 VLLM API 时发生错误: {e}",
                     exc_info=True)  # exc_info=True 会打印堆栈

    # log_status = "成功" if optimized_text_result is not None else "失败"
    # logger.info(f"[{time.time():.3f}] [ReqID: {request_id}] 异步优化块B {log_status} (耗时: {time.time() - func_start_time:.3f}s)。")
    return optimized_text_result


# -----------------------------------------------------------------------------
# 主编排函数：加载初级块，异步使用LLM通过滑动窗口优化中心块
# -----------------------------------------------------------------------------
async def refine_all_chunks_with_llm(
        input_chunks_json_path: str,
        output_refined_chunks_json_path: str
) -> None:
    """
    加载由前一阶段生成的文本块JSON文件，
    异步地使用滑动窗口和LLM来优化每个文档中的中心块，
    并将优化后的块保存到新的JSON文件中。
    """
    try:
        with open(input_chunks_json_path, 'r', encoding='utf-8') as f:
            all_initial_chunks = json.load(f)
        logger.info(f"成功从 '{input_chunks_json_path}' 加载 {len(all_initial_chunks)} 个初始文本块。")
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

    final_refined_chunks_list = []
    total_llm_calls = 0
    total_successful_optimizations = 0

    timeout = aiohttp.ClientTimeout(total=VLLM_REQUEST_TIMEOUT)
    connector = aiohttp.TCPConnector(limit=450)  # 你可以根据需要调整这个数字
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:  # 创建一个 session 供所有请求复用
        for doc_name, original_doc_chunks in chunks_by_doc.items():
            logger.info(f"\n开始处理文档 '{doc_name}' (共 {len(original_doc_chunks)} 个块)...")
            num_chunks_in_doc = len(original_doc_chunks)

            if num_chunks_in_doc < 3:  # 少于3个块，无法形成A-B-C窗口，直接保留
                logger.info(f"  文档 '{doc_name}' 块数量少于3，不进行LLM优化，直接使用原始块。")
                final_refined_chunks_list.extend(original_doc_chunks)
                continue

            # 第一个块直接保留
            final_refined_chunks_list.append(original_doc_chunks[0])

            # 准备异步任务列表来优化中间的块
            tasks = []
            # 中间块的索引范围是 1 到 num_chunks_in_doc - 2
            # （即 original_doc_chunks[1], ..., original_doc_chunks[num_chunks_in_doc-2]）
            for k in range(1, num_chunks_in_doc - 1):
                context_a_text = original_doc_chunks[k - 1]['text']
                chunk_b_to_optimize_text = original_doc_chunks[k]['text']
                context_c_text = original_doc_chunks[k + 1]['text']
                request_id = f"{doc_name}_p{original_doc_chunks[k]['page_number']}_c{original_doc_chunks[k]['chunk_id'].split('_c')[-1]}"

                logger.info(
                    f"  准备LLM优化任务 for chunk_id: {original_doc_chunks[k]['chunk_id']} (原始文本首20字: '{chunk_b_to_optimize_text[:20]}...')")
                tasks.append(
                    optimize_chunk_b_via_vllm(
                        context_a_text,
                        chunk_b_to_optimize_text,
                        context_c_text,
                        session,  # 传递 session
                        request_id=request_id
                    )
                )
                total_llm_calls += 1

            # 并发执行所有优化任务
            logger.info(f"  为文档 '{doc_name}' 创建了 {len(tasks)} 个LLM优化任务，开始并发执行...")
            optimized_b_texts_results = await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"  文档 '{doc_name}' 的所有LLM优化任务已完成。")

            # 处理优化结果
            for k_task_idx, optimized_text_or_exception in enumerate(optimized_b_texts_results):
                original_chunk_index_in_doc = k_task_idx + 1  # 因为我们从原始块列表的索引1开始创建任务

                refined_chunk_data = dict(original_doc_chunks[original_chunk_index_in_doc])  # 复制元数据

                if isinstance(optimized_text_or_exception, Exception):
                    logger.error(
                        f"  LLM优化块 '{original_doc_chunks[original_chunk_index_in_doc]['chunk_id']}' 时发生异常: {optimized_text_or_exception}")
                    # 保留原始文本
                elif optimized_text_or_exception is not None:
                    refined_chunk_data['text'] = optimized_text_or_exception
                    total_successful_optimizations += 1
                    logger.info(
                        f"  块 '{original_doc_chunks[original_chunk_index_in_doc]['chunk_id']}' 文本已由LLM更新。")
                else:
                    # LLM调用可能返回None（例如，JSON解析失败或API明确返回无优化）
                    logger.warning(
                        f"  LLM优化块 '{original_doc_chunks[original_chunk_index_in_doc]['chunk_id']}' 未返回有效文本，保留原始文本。")
                    # 保留原始文本 (refined_chunk_data['text'] 已经是原始的了)

                final_refined_chunks_list.append(refined_chunk_data)

            # 最后一个块直接保留
            final_refined_chunks_list.append(original_doc_chunks[num_chunks_in_doc - 1])
            logger.info(f"  文档 '{doc_name}' LLM优化流程结束。")

    logger.info(
        f"\n所有文档LLM块优化处理完成。总共尝试优化 {total_llm_calls} 个块，成功优化 {total_successful_optimizations} 个。")
    logger.info(f"最终总块数为: {len(final_refined_chunks_list)}")

    try:
        with open(output_refined_chunks_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(final_refined_chunks_list, outfile, ensure_ascii=False, indent=2)
        logger.info(f"所有优化后的文本块已保存到: '{output_refined_chunks_json_path}'")
    except Exception as e:
        logger.error(f"保存优化后的块到 JSON 文件 '{output_refined_chunks_json_path}' 时发生错误: {e}", exc_info=True)


if __name__ == '__main__':
    # --- 配置参数 ---
    INPUT_CHUNKS_JSON = "../processed_knowledge/processed_knowledge_base_chunks.json"
    OUTPUT_REFINED_JSON = "../processed_knowledge/refined_knowledge_base_chunks_llm.json"

    # 确保输入文件存在
    if not os.path.exists(INPUT_CHUNKS_JSON):
        logger.error(f"错误: 输入文件 '{INPUT_CHUNKS_JSON}' 未找到。请先运行之前的脚本生成初级文本块。")
    else:
        # 运行主异步函数
        # 在某些Windows环境下，如果遇到 aiohttp 的 DNS 解析问题，可能需要下面的策略
        # if os.name == 'nt':
        # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(refine_all_chunks_with_llm(INPUT_CHUNKS_JSON, OUTPUT_REFINED_JSON))

        logger.info("\n--- LLM 批量块优化流程运行完毕 ---")
