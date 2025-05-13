import json
import os
import asyncio
import aiohttp  # 用于异步HTTP请求
import time
import logging
from typing import List, Dict, Optional, Any

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- VLLM 服务和模型配置 (请根据你的实际部署修改这些值) ---
VLLM_METADATA_API_URL = "http://localhost:8001/v1/chat/completions"  # 修改为你的实际URL
METADATA_MODEL_NAME = "/home/pushihao/RAG/models/Qwen/Qwen3-30B-A3B-FP8"  # 修改为你的模型名
METADATA_GENERATION_CONFIG = {
    "temperature": 0.4,  # 对于信息提取和遵循指令，较低的温度可能更好
    "max_tokens": 4096,  # 需要足够容纳关键词、问题和JSON结构
    # "response_format": {"type": "json_object"} # 如果VLLM和模型支持，强烈建议使用
}
VLLM_REQUEST_TIMEOUT = 60*60*4  # 秒 - 根据需要调整

# --- LLM提示词模板：为单个块生成元数据并判断意义 ---
METADATA_GENERATION_PROMPT_TEMPLATE = """你是一位专业的知识分析和信息提取专家。
我将提供一段文本（一个文本块）。请你基于这段文本内容完成以下任务：

1.  **评估内容意义**：
    * 首先，请判断该文本块是否包含实质性的、有意义的、适合用于构建问答知识库的信息。
    * 如果文本块内容非常稀疏、几乎没有信息量（例如，可能只是页眉、页脚、孤立的图表编号、空白内容，或非常零碎以至于无法理解的片段），请在返回结果中指明。

2.  **关键词摘要提取 (如果文本块有意义)**：
    * 提取或生成若干（例如，1到5个，不必强求数量，质量优先）最能概括该文本块核心内容的“关键词摘要”或“标签”。
    * 每个关键词摘要应该简短精炼，例如 "主题A+主题B" 格式，或者其他能准确捕捉核心概念的短语组合。
    * 如果文本块评估为无实质意义，则关键词摘要列表应为空。

3.  **预生成相关问题 (如果文本块有意义)**：
    * 根据该文本块的内容，生成若干（例如，1到3个，不必强求数量，质量优先）用户最有可能提出的、并且该文本块能够清晰、直接回答的高质量问题。
    * 如果文本块评估为无实质意义，则生成问题列表应为空。

请将你的结果以严格的JSON格式返回，包含以下字段：
- "is_meaningful": 布尔值 (true 表示有意义，false 表示无实质意义应考虑丢弃)。
- "reason_if_not_meaningful": 字符串，如果 "is_meaningful" 为 false，请简要说明原因 (例如："仅包含页眉", "内容过短且无信息")。如果 "is_meaningful" 为 true，此字段可为空字符串。
- "keyword_summaries": 一个包含所提取/生成的关键词摘要字符串的列表。如果 "is_meaningful" 为 false，此列表必须为空。
- "generated_questions": 一个包含所生成的问句字符串的列表。如果 "is_meaningful" 为 false，此列表必须为空。

确保JSON格式正确，所有字符串值内部的特殊字符（如换行符、双引号）都已正确转义。不要包含任何其他解释或对话。

提供的文本块如下：
```
{text_chunk_content}
```
"""


# -----------------------------------------------------------------------------
# 异步函数：调用VLLM为单个块生成元数据
# -----------------------------------------------------------------------------
async def generate_metadata_for_chunk_via_vllm(
        chunk_text: str,
        session: aiohttp.ClientSession,
        request_id: str = "N/A"
) -> Optional[Dict[str, Any]]:
    """
    使用 VLLM API 异步地为单个文本块生成元数据。

    参数:
        chunk_text (str): 需要处理的文本块内容。
        session (aiohttp.ClientSession): 复用的aiohttp会话。
        request_id (str): 用于日志追踪的请求ID。

    返回:
        Optional[Dict[str, Any]]: 包含元数据的字典
        (例如 {'is_meaningful': True, 'keyword_summaries': [], 'generated_questions': []})，
        如果LLM调用失败或返回格式不正确则为None。
    """
    final_prompt = METADATA_GENERATION_PROMPT_TEMPLATE.format(text_chunk_content=chunk_text)
    api_messages = [{"role": "user", "content": final_prompt}]
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = {
        "model": METADATA_MODEL_NAME,
        "messages": api_messages,
        **{k: v for k, v in METADATA_GENERATION_CONFIG.items() if v is not None},
        "stream": False,
    }

    llm_response_data = None
    raw_llm_content = ""

    try:
        # logger.debug(f"[{time.time():.3f}] [ReqID: {request_id}] 发送元数据生成请求。")
        async with session.post(VLLM_METADATA_API_URL, headers=headers, json=payload) as response:
            response.raise_for_status()
            response_data_json = await response.json()

            if response_data_json.get("choices") and len(response_data_json["choices"]) > 0:
                message_content = response_data_json["choices"][0].get("message", {}).get("content")
                raw_llm_content = message_content
                if message_content:
                    message_content = message_content.strip("\n")
                    if message_content.strip().startswith("```json"):
                        message_content = message_content.strip("```json").strip("`").strip()
                    elif message_content.strip().startswith("```"):
                        message_content = message_content.strip("```").strip()

                    try:
                        llm_response_data = json.loads(message_content)
                        # 验证返回的结构是否符合预期
                        if not isinstance(llm_response_data, dict) or \
                                'is_meaningful' not in llm_response_data or \
                                not isinstance(llm_response_data['is_meaningful'], bool):
                            logger.warning(
                                f"[ReqID: {request_id}] LLM返回的JSON结构不符合预期 (缺少is_meaningful或类型错误)。Content: '{message_content}'")
                            llm_response_data = None  # 标记为无效响应
                        else:
                            # 确保其他字段存在且为列表（即使is_meaningful为false，也期望空列表）
                            if 'keyword_summaries' not in llm_response_data or not isinstance(
                                    llm_response_data['keyword_summaries'], list):
                                llm_response_data['keyword_summaries'] = []
                                logger.debug(f"[ReqID: {request_id}] 'keyword_summaries' 缺失或类型错误，已设为空列表。")
                            if 'generated_questions' not in llm_response_data or not isinstance(
                                    llm_response_data['generated_questions'], list):
                                llm_response_data['generated_questions'] = []
                                logger.debug(
                                    f"[ReqID: {request_id}] 'generated_questions' 缺失或类型错误，已设为空列表。")
                            if llm_response_data[
                                'is_meaningful'] is False and 'reason_if_not_meaningful' not in llm_response_data:
                                llm_response_data['reason_if_not_meaningful'] = "N/A"


                    except json.JSONDecodeError as e:
                        logger.error(
                            f"[ReqID: {request_id}] 解析LLM返回的元数据JSON时失败: {e}. Content: '{message_content}'")
                        llm_response_data = None
                else:
                    logger.warning(f"[ReqID: {request_id}] LLM API响应的 message content 为空。")
            else:
                logger.warning(f"[ReqID: {request_id}] LLM API响应的 'choices' 字段无效或为空。")
    except asyncio.TimeoutError:
        logger.error(
            f"[ReqID: {request_id}] 请求VLLM元数据生成API超时 (超过 {VLLM_REQUEST_TIMEOUT}s)。Chunk首20字: '{chunk_text[:20]}...'")
    except aiohttp.ClientResponseError as e:
        logger.error(f"[ReqID: {request_id}] VLLM元数据生成API HTTP错误: Status {e.status}, Message: {e.message}",
                     exc_info=False)
    except Exception as e:
        logger.error(
            f"[ReqID: {request_id}] 调用VLLM元数据生成API时发生一般错误: {e}. Chunk首20字: '{chunk_text[:20]}...'",
            exc_info=True)

    if llm_response_data is None:
        logger.warning(
            f"[ReqID: {request_id}] 未能从LLM获取有效的元数据。LLM原始返回(若有): '{raw_llm_content[:200]}...'")

    return llm_response_data


# -----------------------------------------------------------------------------
# 主编排函数：为所有块异步生成元数据并筛选
# -----------------------------------------------------------------------------
async def enhance_chunks_with_llm_metadata(
        input_chunks_json_path: str,
        output_enhanced_chunks_json_path: str,
        max_chunks_to_process_with_llm: Optional[int] = None
) -> None:
    """
    加载初级文本块，为每个块异步调用LLM生成元数据（关键词、问题、是否有意义），
    筛选掉无意义的块，并将增强后的块保存到新的JSON文件。
    """
    try:
        with open(input_chunks_json_path, 'r', encoding='utf-8') as f:
            all_initial_chunks = json.load(f)
        logger.info(f"成功从 '{input_chunks_json_path}' 加载 {len(all_initial_chunks)} 个初始文本块。")
    except Exception as e:
        logger.error(f"错误：无法加载初始文本块文件 '{input_chunks_json_path}': {e}", exc_info=True)
        return

    enhanced_and_filtered_chunks = []
    llm_calls_attempted = 0
    meaningful_chunks_count = 0
    not_meaningful_chunks_discarded = 0
    llm_call_failures = 0

    timeout_config = aiohttp.ClientTimeout(total=VLLM_REQUEST_TIMEOUT)
    connector = aiohttp.TCPConnector(limit=1000)
    async with aiohttp.ClientSession(timeout=timeout_config, connector=connector) as session:
        tasks = []
        chunks_being_processed_info = []  # 存储与任务对应的原始块信息

        for i, original_chunk_data in enumerate(all_initial_chunks):
            if max_chunks_to_process_with_llm is not None and \
                    llm_calls_attempted >= max_chunks_to_process_with_llm:
                logger.info(
                    f"已达到LLM处理上限 ({max_chunks_to_process_with_llm})。剩余 {len(all_initial_chunks) - i} 个块将不被处理。")
                # 将剩余未处理的块（如果策略是保留它们）直接加入结果，或简单地在此中断
                # 为了测试，我们这里直接中断，只处理限定数量的块
                break

            chunk_text = original_chunk_data.get('text', '')
            chunk_id = original_chunk_data.get('chunk_id', f'unknown_chunk_{i}')

            if not chunk_text.strip():
                logger.warning(f"块 '{chunk_id}' 文本为空，已跳过LLM处理。")
                # 决定是否要保留这个空块，通常不需要
                continue

            logger.info(f"准备LLM元数据生成任务 for chunk_id: {chunk_id} (尝试次数: {llm_calls_attempted + 1})")
            task = generate_metadata_for_chunk_via_vllm(
                chunk_text,
                session,
                request_id=chunk_id
            )
            tasks.append(task)
            chunks_being_processed_info.append(original_chunk_data)  # 保存原始块信息
            llm_calls_attempted += 1

        if tasks:
            logger.info(f"创建了 {len(tasks)} 个LLM元数据生成任务，开始并发执行...")
            start_gather_time = time.time()
            llm_results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"所有LLM元数据生成任务已完成 (耗时: {time.time() - start_gather_time:.2f}s)。")

            for i, result_or_exc in enumerate(llm_results_or_exceptions):
                original_chunk = chunks_being_processed_info[i]  # 获取对应的原始块信息

                if isinstance(result_or_exc, Exception):
                    logger.error(f"LLM处理块 '{original_chunk['chunk_id']}' 时发生异常: {result_or_exc}")
                    llm_call_failures += 1
                    # 决定如何处理失败的块：是保留原始块还是丢弃？
                    # 为安全起见，暂时保留原始块，不加新元数据
                    enhanced_and_filtered_chunks.append(original_chunk)
                    continue

                llm_generated_metadata = result_or_exc  # 这是解析后的字典或None

                if llm_generated_metadata and llm_generated_metadata.get('is_meaningful') is True:
                    new_chunk_data = dict(original_chunk)  # 复制原始元数据
                    new_chunk_data['is_meaningful'] = True
                    new_chunk_data['keyword_summaries'] = llm_generated_metadata.get('keyword_summaries', [])
                    new_chunk_data['generated_questions'] = llm_generated_metadata.get('generated_questions', [])
                    new_chunk_data['reason_if_not_meaningful'] = ""  # 清空或设为None
                    enhanced_and_filtered_chunks.append(new_chunk_data)
                    meaningful_chunks_count += 1
                    logger.info(f"块 '{original_chunk['chunk_id']}' 被判断为有意义，并已添加元数据。")
                elif llm_generated_metadata and llm_generated_metadata.get('is_meaningful') is False:
                    reason = llm_generated_metadata.get('reason_if_not_meaningful', '未提供原因')
                    logger.info(f"块 '{original_chunk['chunk_id']}' 被LLM判断为无意义 (原因: {reason})，将被丢弃。")
                    not_meaningful_chunks_discarded += 1
                else:
                    # LLM 调用可能成功返回，但内容无法解析或不符合预期结构 (is_meaningful 缺失等)
                    logger.warning(
                        f"LLM为块 '{original_chunk['chunk_id']}' 返回的元数据无效或不完整，保留原始块。LLM原始输出摘要: {str(llm_generated_metadata)[:100]}...")
                    llm_call_failures += 1
                    enhanced_and_filtered_chunks.append(original_chunk)  # 保留原始块

    logger.info(f"\n元数据生成流程结束。")
    logger.info(f"  总共尝试调用LLM处理 {llm_calls_attempted} 个块。")
    logger.info(f"  其中 {meaningful_chunks_count} 个块被判断为有意义并添加了元数据。")
    logger.info(f"  {not_meaningful_chunks_discarded} 个块因被判断为无意义而被丢弃。")
    logger.info(f"  {llm_call_failures} 次LLM调用失败或返回数据无效 (这些块被保留了原始信息)。")
    logger.info(f"  最终增强后的知识库包含 {len(enhanced_and_filtered_chunks)} 个块。")

    try:
        with open(output_enhanced_chunks_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(enhanced_and_filtered_chunks, outfile, ensure_ascii=False, indent=2)
        logger.info(f"所有增强并筛选后的文本块已保存到: '{output_enhanced_chunks_json_path}'")
    except Exception as e:
        logger.error(f"保存增强后的块到 JSON 文件 '{output_enhanced_chunks_json_path}' 时发生错误: {e}", exc_info=True)


if __name__ == '__main__':
    # --- 配置参数 ---
    INPUT_CHUNKS_JSON = "../processed_knowledge/refined_knowledge_base_chunks_llm.json"  # 你之前生成的包含初级块的JSON
    OUTPUT_ENHANCED_JSON = "../processed_knowledge/enhanced_knowledge_base_chunks_llm.json"  # LLM增强后输出的新JSON

    # --- !! 重要：进行小规模测试 !! ---
    # MAX_CHUNKS_TO_PROCESS_LLM = 100  # 例如，先测试处理20个块
    MAX_CHUNKS_TO_PROCESS_LLM = None # 设置为 None 来处理所有块

    if not os.path.exists(INPUT_CHUNKS_JSON):
        logger.error(f"错误: 输入文件 '{INPUT_CHUNKS_JSON}' 未找到。")
    else:
        logger.info(
            f"开始LLM块元数据增强流程，最多处理 {MAX_CHUNKS_TO_PROCESS_LLM if MAX_CHUNKS_TO_PROCESS_LLM is not None else '所有'} 个块。")
        logger.info(f"VLLM API URL: {VLLM_METADATA_API_URL}")
        logger.info(f"VLLM Model: {METADATA_MODEL_NAME}")
        logger.info(f"Request Timeout: {VLLM_REQUEST_TIMEOUT}s")

        asyncio.run(enhance_chunks_with_llm_metadata(
            INPUT_CHUNKS_JSON,
            OUTPUT_ENHANCED_JSON,
            max_chunks_to_process_with_llm=MAX_CHUNKS_TO_PROCESS_LLM
        ))

        logger.info(f"\n--- LLM 块元数据增强流程运行完毕 ---")
        logger.info(f"请检查输出文件: '{OUTPUT_ENHANCED_JSON}'")
