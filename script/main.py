# test/main.py

import threading
import logging
from queue import Queue
from typing import List, Dict, Optional # 添加类型注解

# 使用相对导入 (假设 main.py 与其他模块在同一目录下或上一级目录运行 `python -m test.main`)
# 如果直接运行 `python main.py`，请改回绝对导入
try:
    from .model_loader import load_models
    from .knowledge_base import KnowledgeBase
    from .streamer import CustomStreamer
    from .query_rewriter import generate_rewritten_query
    from .config import * # 导入所有配置
except ImportError:
    print("Error: Failed to import modules using relative paths. Trying absolute paths...")
    # 如果相对导入失败，尝试绝对导入（适用于直接运行脚本的情况）
    from model_loader import load_models
    from knowledge_base import KnowledgeBase
    from streamer import CustomStreamer
    from query_rewriter import generate_rewritten_query
    from config import *

# --- 配置 logging ---
# 注意：basicConfig 最好只调用一次。如果其他模块也配置了，可能会有冲突。
# 可以在一个单独的 setup_logging.py 中配置，或者确保只在这里配置一次。
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    filename='rag_chat.log', # 可选：将日志写入文件
    filemode='w' # 可选：追加模式
)
logger = logging.getLogger(__name__)
# --------------------

def load_prompt_from_file(file_path: str) -> Optional[str]:
    """从文件加载 Prompt 文本。"""
    logger.info(f"Loading prompt from: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompt = f.read()
        logger.info(f"Prompt loaded successfully from {file_path}.")
        return prompt
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {file_path}")
        return None
    except IOError as e:
        logger.error(f"Error reading prompt file {file_path}: {e}", exc_info=True)
        return None

def chat_loop():
    """主聊天循环函数。"""
    logger.info("Starting RAG Chat Application...")

    # --- 1. 加载 Prompts ---
    generator_system_prompt = load_prompt_from_file(GENERATOR_SYSTEM_PROMPT_FILE)
    rewriter_instruction_template = load_prompt_from_file(REWRITER_INSTRUCTION_FILE)

    if not generator_system_prompt or not rewriter_instruction_template:
        logger.critical("Failed to load essential prompt files. Exiting.")
        print("错误：无法加载必要的 Prompt 文件，请检查 config.py 中的路径设置。")
        return

    # --- 2. 加载模型 ---
    logger.info("Loading models...")
    try:
        # load_models 返回: generator_model, generator_tokenizer, rewriter_model, rewriter_tokenizer
        generator_model, generator_tokenizer, rewriter_model, rewriter_tokenizer = load_models()
        if generator_model is None or generator_tokenizer is None:
            # 如果核心生成模型加载失败，则无法继续
            logger.critical("Failed to load the core Generator model/tokenizer. Exiting.")
            print("错误：核心生成模型加载失败，无法启动。")
            return
        logger.info("Models loaded (Generator: OK, Rewriter: {}).".format("OK" if rewriter_model else "Failed/Skipped"))
    except Exception as e:
        logger.critical(f"An unexpected error occurred during model loading: {e}", exc_info=True)
        print(f"错误：模型加载过程中发生意外错误: {e}")
        return

    # --- 3. 初始化知识库 ---
    logger.info("Initializing Knowledge Base...")
    try:
        # 使用更新后的初始化，不再传入 mode
        kb = KnowledgeBase(
            knowledge_dir=KNOWLEDGE_BASE_DIR,
            knowledge_file_pattern=KNOWLEDGE_FILE_PATTERN
        )
        logger.info("Knowledge Base initialization complete.")
    except Exception as e:
         logger.critical(f"Failed to initialize Knowledge Base: {e}", exc_info=True)
         print(f"致命错误：知识库初始化失败: {e}")
         print("请确保 config.py 中的知识库目录和文件模式设置正确，且源文件存在。")
         return

    # --- 4. 初始化对话历史 ---
    messages: List[Dict[str, str]] = []

    # --- 5. 开始聊天循环 ---
    print("\n===================================")
    print(" RAG 对话系统已启动")
    print("===================================")
    print("输入 'q' 或 'exit' 退出。")

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["q", "exit"]:
                logger.info("User requested exit.")
                break
            if not user_input.strip():
                continue

            # --- 5.1 查询重写 (如果 Rewriter 模型可用) ---
            rewritten_query_str = user_input # 默认使用原始查询
            if rewriter_model and rewriter_tokenizer:
                logger.info("Performing query rewriting...")
                try:
                     # 调用重写函数，传入所需参数
                     rewritten_query_str = generate_rewritten_query(
                         model=rewriter_model,
                         tokenizer=rewriter_tokenizer,
                         messages=messages,
                         user_input=user_input,
                         rewriter_instruction_template=rewriter_instruction_template, # 传入模板
                         max_history=MAX_HISTORY,
                         generation_config=GENERATION_CONFIG # 可以为重写任务定义不同的生成参数
                     )
                     logger.info("Query rewriting finished.")
                except Exception as rewrite_err:
                     logger.error(f"Error during query rewriting: {rewrite_err}", exc_info=True)
                     logger.warning("Falling back to original query due to rewrite error.")
                     rewritten_query_str = user_input # 出错时回退
            else:
                logger.info("Skipping query rewriting (Rewriter model not available).")

            # --- 5.2 解析重写结果为查询词列表 ---
            # 假设重写器输出的是以换行符分隔的查询词
            search_terms = [term.strip() for term in rewritten_query_str.strip().split('\n') if term.strip()]
            if not search_terms:  # 如果输出为空或无效，则使用原始输入作为查询
                logger.warning("Rewritten query was empty or invalid, using original query as the only search term.")
                search_terms = [user_input]

            logger.info(f"Planned search terms ({len(search_terms)}): {search_terms}")
            # 使用 DEBUG 级别打印更详细的列表
            for i, term in enumerate(search_terms):
                 logger.debug(f"  Search Term {i+1}: '{term}'")

            # --- 5.3 循环检索、合并与去重 ---
            all_retrieved_data: List[Dict[str, str]] = []  # 存储最终合并、去重后的块字典列表
            processed_chunk_texts = set()  # 用于根据文本内容去重

            logger.info("--- Starting Multi-Query Retrieval & Aggregation ---")
            for i, term in enumerate(search_terms):
                logger.info(f"Retrieving for sub-query {i+1}/{len(search_terms)}: '{term[:50]}...'")
                try:
                    # retrieve_chunks 返回字典列表 [{"text": ..., "source": ...}, ...]
                    term_chunks_data = kb.retrieve_chunks(term)
                    logger.info(f"  Retrieved {len(term_chunks_data)} chunks for this term.")

                    # 合并与去重
                    added_count = 0
                    for chunk_data in term_chunks_data:
                        chunk_text = chunk_data.get("text")
                        # 只有当文本内容有效且之前未处理过时才添加
                        if chunk_text and chunk_text not in processed_chunk_texts:
                            all_retrieved_data.append(chunk_data)
                            processed_chunk_texts.add(chunk_text)
                            added_count += 1
                    if added_count > 0:
                        logger.debug(f"  Added {added_count} unique chunks to aggregation pool.")

                except Exception as retrieval_err:
                    logger.error(f"  Error retrieving chunks for term '{term}': {retrieval_err}", exc_info=True)
            logger.info("--- Multi-Query Retrieval Finished ---")
            logger.info(f"Total unique chunks aggregated before final filtering: {len(all_retrieved_data)}")

            # --- 5.4 (可选) 上下文重排序 ---
            # TODO: 在这里可以根据 all_retrieved_data 中每个 chunk 与原始 user_input 的相关性进行排序
            # 例如：计算 user_input embedding 与每个 chunk embedding 的相似度，然后排序
            # sorted_retrieved_data = rank_chunks_by_relevance(all_retrieved_data, user_input, kb.embedding_model)
            # logger.info("Re-ranked aggregated chunks based on relevance to original query.")
            # (暂时跳过，使用原始聚合顺序)
            sorted_retrieved_data = all_retrieved_data # 不重排序

            # --- 5.5 控制最终上下文长度 ---
            final_chunks_data: List[Dict[str, str]] = []
            if len(sorted_retrieved_data) > MAX_AGGREGATED_RESULTS:
                logger.warning(
                    f"Aggregated results ({len(sorted_retrieved_data)}) exceed limit ({MAX_AGGREGATED_RESULTS}). Truncating...")
                # 使用排序后的结果进行截断
                final_chunks_data = sorted_retrieved_data[:MAX_AGGREGATED_RESULTS]
            else:
                final_chunks_data = sorted_retrieved_data
            logger.info(f"Final number of chunks selected for LLM context: {len(final_chunks_data)}")

            # --- 5.6 格式化上下文并显示给用户 (可选) ---
            context_parts = []
            if final_chunks_data:
                print("\n" + "=" * 15 + " 参考知识库片段 " + "=" * 15) # 显示给用户
                for i, chunk_data in enumerate(final_chunks_data):
                    chunk_text = chunk_data.get("text", "内容缺失")
                    source_file = chunk_data.get("source", "未知来源")
                    # 格式化，用于显示和送入 LLM
                    formatted_chunk = f"【片段 {i + 1} | 来源文件: {source_file}】| 内容: {chunk_text}"
                    context_parts.append(formatted_chunk)
                    # 打印给用户看 (可以截断长内容)
                    print(f"片段 {i+1} (来源: {source_file}): {chunk_text[:200]}...") # 限制打印长度
                print("=" * 17 + " 参考内容结束 " + "=" * 17 + "\n")
                # 记录完整的上下文（可能很长，用 DEBUG）
                # logger.debug("Final context being sent to LLM:\n" + "\n---\n".join(context_parts))
            else:
                logger.info("No relevant chunks found in knowledge base for the query.")
                print("\n（本次回答未直接引用知识库中的具体文本片段）\n")
                # 可以在这里设置一个默认的 context 字符串告知 LLM 没有找到内容
                context = "根据规划的子主题，知识库中未找到相关内容。"

            # 组合最终的上下文文本
            context = "\n\n".join(context_parts) # 使用空行分隔不同的片段

            # --- 5.7 构建最终生成 Prompt ---
            # 使用从文件加载的系统 Prompt
            generation_messages = [{"role": "system", "content": generator_system_prompt}]
            # 添加历史消息
            generation_messages.extend(messages)
            # 添加当前用户输入和检索到的上下文
            # 注意 user content 的结构，结合了原始输入和知识库
            generation_messages.append({"role": "user", "content": f"User: {user_input}\n\n知识库：\n{context if context else '根据规划的子主题，知识库中未找到相关内容。'}"}) # 如果 context 为空，明确告知LLM

            try:
                # 使用 generator_tokenizer 应用模板
                prompt = generator_tokenizer.apply_chat_template(
                    generation_messages,
                    tokenize=False,
                    add_generation_prompt=True # 对于生成任务通常需要
                )
                # logger.debug(f"Final prompt for generation (first 200 chars):\n{prompt[:200]}...")

                # 使用 generator_tokenizer 进行编码
                inputs = generator_tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(generator_model.device)
            except Exception as e:
                 logger.error(f"Error creating generation prompt or tokenizing: {e}", exc_info=True)
                 print("错误：准备生成请求时出错，请稍后再试。")
                 continue # 跳过本次生成

            # --- 5.8 使用流式接口生成回复 ---
            result_queue = Queue()
            # streamer 需要使用 generator_tokenizer
            streamer = CustomStreamer(generator_tokenizer, result_queue)

            generation_kwargs = {
                **inputs,
                **GENERATION_CONFIG, # 使用配置中的生成参数
                "streamer": streamer,
                "eos_token_id": generator_tokenizer.eos_token_id,
                "pad_token_id": generator_tokenizer.pad_token_id if generator_tokenizer.pad_token_id is not None else generator_tokenizer.eos_token_id # 确保 pad_token 正确
            }

            # 在单独的线程中运行生成
            thread = threading.Thread(target=generator_model.generate, kwargs=generation_kwargs)
            thread.start()

            # --- 5.9 处理流式输出 ---
            print("Assistant: ", end="", flush=True) # 提示助手正在输出
            generated_text = ""
            try:
                 # 这里可以改进：不阻塞地处理队列，直到线程结束
                 # 但对于控制台输出，简单 join 也可以
                 thread.join() # 等待生成线程完成
                 generated_tokens = result_queue.get() # 从队列获取完整 token 列表
                 # 使用 generator_tokenizer 解码
                 model_reply = generator_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                 print() # 在流式输出后换行
                 logger.info(f"Generated reply length: {len(model_reply)}")
                 # logger.debug(f"Full model reply:\n{model_reply}")
                 generated_text = model_reply # 保存完整回复
            except Exception as e:
                 logger.error(f"Error during generation streaming or decoding: {e}", exc_info=True)
                 print("\n错误：处理模型回复时出错。")
                 # 即使出错，也尝试获取已生成的部分（如果 streamer 能处理）
                 # 或者直接跳过本次回复
                 continue


            # --- 5.10 更新对话历史 ---
            # 只保留核心对话内容，去除可能的前缀/思考过程等
            # (如果模型输出格式稳定，这里的处理可以简化)
            final_reply = generated_text.split("</think>")[-1].strip() # 假设可能存在 <think>

            # 添加用户输入和最终回复到历史记录
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": final_reply})

            # 维护历史长度
            if len(messages) > MAX_HISTORY * 2:
                 logger.debug(f"Trimming conversation history from {len(messages)} messages.")
                 messages = messages[-(MAX_HISTORY * 2):]
                 logger.debug(f"History trimmed to {len(messages)} messages.")

        except KeyboardInterrupt:
             logger.info("User interrupted (KeyboardInterrupt). Exiting.")
             print("\n检测到中断，正在退出...")
             break
        except Exception as e:
             logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
             print(f"\n发生意外错误: {e}。请尝试重新提问或重启程序。")
             # 可以选择是否重置 history 等
             # messages = [] # 重置历史

    logger.info("RAG Chat Application finished.")
    print("\n再见！")


if __name__ == "__main__":
    # 可以在这里添加命令行参数解析等
    chat_loop()