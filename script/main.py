import threading
from queue import Queue
from model_loader import load_models
from knowledge_base import KnowledgeBase
from streamer import CustomStreamer
from config import *
from query_rewriter import generate_rewritten_query

def chat_loop():
    print("Loading models...")
    generator_model, generator_tokenizer, rewriter_model, rewriter_tokenizer = load_models()
    print("Models loaded.")

    # === 修改：使用目录和模式初始化 KnowledgeBase ===
    print("Initializing Knowledge Base...")
    try:
        # 传入目录和文件模式
        kb = KnowledgeBase(
            similarity_mode=SEARCH_MODE,
            knowledge_dir=KNOWLEDGE_BASE_DIR,
            knowledge_file_pattern=KNOWLEDGE_FILE_PATTERN
        )
    except Exception as e:
         print(f"FATAL: Failed to initialize Knowledge Base: {e}")
         print("Please ensure the knowledge directory and pattern are set correctly in config.py.")
         return
    print("Knowledge Base initialization complete.")
    # =======================================================

    # 检查重写模型是否加载成功（如果 load_models 中没有抛出异常而是返回 None）
    if rewriter_model is None or rewriter_tokenizer is None:
        print("Warning: Rewriter model not loaded. Query rewriting will be skipped.")


    messages = []

    print("\n开始对话 (输入 'q' 或 'exit' 退出)")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["q", "exit"]:
            break

        # ================== 修改：使用重写模型 ==================
        # 只有在重写模型加载成功时才进行重写
        rewriter_output_str = user_input # 默认使用原始查询
        if rewriter_model and rewriter_tokenizer:
            try:
                 # 调用重写函数，传入重写模型和分词器
                 rewriter_output_str = generate_rewritten_query(
                     rewriter_model,        # <--- 使用重写模型
                     rewriter_tokenizer,    # <--- 使用重写分词器
                     messages,
                     user_input,
                     max_history=MAX_HISTORY
                 )
            except Exception as rewrite_err:
                 print(f"Error during query rewriting: {rewrite_err}")
                 print("Falling back to original query.")
                 rewriter_output_str = user_input # 出错时也回退
        else:
            print("Skipping query rewriting as rewriter model is not available.")
        # =======================================================

        # === 2. 解析规划结果为查询词列表 ===
        search_terms = [term.strip() for term in rewriter_output_str.strip().split('\n') if term.strip()]
        if not search_terms:  # 如果输出为空或无效，则使用原始输入作为查询
            search_terms = [user_input]
            print("Query planner returned empty or invalid output, using original query.")

        print(f"--- Planned Search Terms ({len(search_terms)}) ---")
        for term in search_terms: print(f"- '{term}'")
        print("-----------------------------")

        # === 3. 循环检索、合并与去重 ===
        all_retrieved_data = []  # 存储最终合并、去重后的块字典列表
        processed_chunk_texts = set()  # 用于根据文本内容去重

        print("--- Starting Multi-Query Retrieval & Aggregation ---")
        for term in search_terms:
            print(f"Retrieving for sub-query: '{term}'")
            try:
                # retrieve_chunks 返回字典列表 [{"text": ..., "source": ...}, ...]
                # 它内部已经处理了阈值和 MAX_THRESHOLD_RESULTS 限制 (针对单次查询)
                term_chunks_data = kb.retrieve_chunks(term)
                print(f"  Retrieved {len(term_chunks_data)} chunks for this term.")

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
                    print(f"  Added {added_count} unique chunks to aggregation.")

            except Exception as retrieval_err:
                print(f"  Error retrieving for term '{term}': {retrieval_err}")
        print("--- Multi-Query Retrieval Finished ---")
        print(f"Total unique chunks aggregated: {len(all_retrieved_data)}")

        # === 4. 控制总上下文长度 ===
        # 对合并去重后的结果应用最终的总数量限制
        final_chunks_data = []
        if len(all_retrieved_data) > MAX_AGGREGATED_RESULTS:
            print(
                f"Aggregated results ({len(all_retrieved_data)}) exceed limit ({MAX_AGGREGATED_RESULTS}). Truncating...")
            # 简单截断：取列表前面的部分。
            # 更好的方法：可以基于与原始 user_input 的相似度对 all_retrieved_data 进行重排序，再截断。
            final_chunks_data = all_retrieved_data[:MAX_AGGREGATED_RESULTS]
        else:
            final_chunks_data = all_retrieved_data
        print(f"Final number of chunks for context: {len(final_chunks_data)}")

        # === 5. 格式化最终上下文 ===
        context_parts = []
        if final_chunks_data:
            print("--- Final Context Sources ---")
            for i, chunk_data in enumerate(final_chunks_data):
                chunk_text = chunk_data.get("text", "内容缺失")
                source_file = chunk_data.get("source", "未知来源")
                # 使用你选择的格式
                context_parts.append(f"知识库来源 (文件: {source_file}):\n{chunk_text}")
                print(f"  Context {i + 1}: File='{source_file}'")
            print("--------------------------")
        else:
            print("--- No relevant context found after aggregation ---")

        context = "\n".join(context_parts)
        if not context:
            context = "根据规划的子主题，知识库中未找到相关内容。"  # 更具体的无结果提示

        print("--" * 25)  # 分隔符


        # ================== 修改：使用生成模型生成最终答案 ==================
        # 构建最终生成答案的 Prompt (逻辑不变，但确保使用 generator_tokenizer)
        sys_prompt = """
        你是一个语言学智能助手，请根据提供的知识库内容来回答问题。
        知识库内容会以 "知识库来源 (文件: 文件名):" 开头。
        在回答时，如果引用了知识库信息，请明确说明信息来源于哪个文件，例如“根据文件 '文件名.txt' 中的信息...” 或使用 “([来源: 文件名.txt])” 这样的标记。
        当所有知识库内容都与问题无关时，你的回答必须包括“知识库中未找到您要的答案！”这句话。
        回答需要考虑聊天历史。
        如果没有查询到答案，你可以根据自己的知识来回答，但是要注明是AI生成而非查询到的结果。"""

        generation_messages = [{"role": "system", "content": sys_prompt}]
        generation_messages.extend(messages)
        generation_messages.append({"role": "user", "content": f"User: {user_input}\n\n知识库：\n{context}"})

        # 使用 generator_tokenizer 应用模板
        prompt = generator_tokenizer.apply_chat_template(generation_messages, tokenize=False, add_generation_prompt=True)
        # 使用 generator_tokenizer 进行编码，并放到 generator_model 所在的设备
        inputs = generator_tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(generator_model.device)

        # 生成回复 (流式，使用 generator_model)
        result_queue = Queue()
        # streamer 需要使用 generator_tokenizer
        streamer = CustomStreamer(generator_tokenizer, result_queue)
        generation_kwargs = {
            **inputs,
            **GENERATION_CONFIG,
            "streamer": streamer,
            "eos_token_id": generator_tokenizer.eos_token_id, # 确保是 generator 的 token id
            "pad_token_id": generator_tokenizer.pad_token_id # 确保是 generator 的 token id
        }

        # target 指向 generator_model.generate
        thread = threading.Thread(target=generator_model.generate, kwargs=generation_kwargs)
        thread.start()

        print("Assistant: ", end="", flush=True)
        thread.join()

        generated_tokens = result_queue.get()
        # 使用 generator_tokenizer 解码
        model_reply = generator_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        # =====================================================================

        # 更新核心对话历史 (使用原始输入)
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": model_reply.split("</think>")[-1]})

        # 维护历史长度
        if len(messages) > MAX_HISTORY * 2:
             messages = messages[-(MAX_HISTORY * 2):]

if __name__ == "__main__":
    chat_loop()