import threading
import time
from queue import Queue
from model_loader import load_models # 函数现在返回4个值
from knowledge_base import KnowledgeBase
from streamer import CustomStreamer
from config import *
from query_rewriter import generate_rewritten_query

def chat_loop():
    # ================== 修改：接收两个模型 ==================
    # 加载两个模型和它们的分词器
    print("Loading models...")
    generator_model, generator_tokenizer, rewriter_model, rewriter_tokenizer = load_models()
    print("Models loaded.")
    # =======================================================

    # 检查重写模型是否加载成功（如果 load_models 中没有抛出异常而是返回 None）
    if rewriter_model is None or rewriter_tokenizer is None:
        print("Warning: Rewriter model not loaded. Query rewriting will be skipped.")

    print("Initializing Knowledge Base...")
    kb = KnowledgeBase(SEARCH_MODE)
    kb.create_index(KNOWLEDGE_BASE_FILE)
    print("Knowledge Base initialized.")

    messages = []
    MAX_HISTORY = 5

    print("\n开始对话 (输入 'q' 或 'exit' 退出)")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["q", "exit"]:
            break

        # ================== 修改：使用重写模型 ==================
        # 只有在重写模型加载成功时才进行重写
        rewritten_query = user_input # 默认使用原始查询
        if rewriter_model and rewriter_tokenizer:
            try:
                 # 调用重写函数，传入重写模型和分词器
                 rewritten_query = generate_rewritten_query(
                     rewriter_model,        # <--- 使用重写模型
                     rewriter_tokenizer,    # <--- 使用重写分词器
                     messages,
                     user_input,
                     max_history=MAX_HISTORY
                 )
            except Exception as rewrite_err:
                 print(f"Error during query rewriting: {rewrite_err}")
                 print("Falling back to original query.")
                 rewritten_query = user_input # 出错时也回退
        else:
            print("Skipping query rewriting as rewriter model is not available.")
        # =======================================================

        # RAG 检索 (使用最终确定的查询语句)
        relevant_chunks = kb.retrieve_chunks(rewritten_query, top_k=TOP_K)
        context = "\n".join([f"知识库来源{i + 1}: {chunk}" for i, chunk in enumerate(relevant_chunks)])
        print("--- Retrieved Context ---")
        print(context)
        print("-----------------------")
        print("--"*25)

        # ================== 修改：使用生成模型生成最终答案 ==================
        # 构建最终生成答案的 Prompt (逻辑不变，但确保使用 generator_tokenizer)
        sys_prompt = """
        你是一个语言学智能助手，请总结知识库的内容来回答问题，请列举知识库中的数据详细回答。
        在回答时你需要明确提到“根据来源{}”作为引用。
        注意！-----------------------------------------------------
        当所有知识库内容都与问题无关时，你的回答必须包括“知识库中未找到您要的答案！”这句话。
        ----------------------------------------------------------
        在回答的最后，（无论是否查询到结果）你可以根据自己的知识来回答，但是，注意！要注明是AI生成而非查询到的结果。"""

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