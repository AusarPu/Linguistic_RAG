# download_models.py
import logging
import os
import sys
try:
    from modelscope.hub.snapshot_download import snapshot_download
    # 尝试性导入，减少不必要的 modelscope 日志输出
    from modelscope.utils.logger import get_logger
    # 可以取消下面这行的注释来减少 modelscope 的日志等级
    # get_logger().setLevel(logging.WARNING)
except ImportError:
    print("错误：无法导入 modelscope 库。请确保已在虚拟环境中安装 modelscope。")
    print("运行: pip install \"modelscope[all]\"")
    sys.exit(1)

# 配置基础日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 用户配置 ---
# !!! 重要：请将下面的 model_id 替换为你要下载的真实 ModelScope 模型 ID !!!
# !!! 并根据需要修改 model_root_dir !!!

# 定义一个统一存放所有模型的根目录
MODEL_ROOT_DIR = "/home/pushihao/RAG/models" # <--- 修改为你希望存放模型的本地根目录

models_to_download = {
    # --- 基础 LLM (用于生成和重写的基础) ---
    "base_llm": {
        "model_id": "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF", # <--- !! 替换 !! 例如 'qwen/Qwen-7B-Chat' 或其他
        # "revision": "v1.0.0" # 可选：指定模型版本
    },
    # --- 嵌入模型 ---
    "embedding": {
        "model_id": "BAAI/bge-m3", # <--- !! 替换 !! 例如 'damo/nlp_gte_sentence-embedding_chinese-base' 或其他 BGE 模型
    },
}
# --- 结束用户配置 ---

def download_model(model_name, config):
    """下载单个模型"""
    model_id = config.get("model_id")
    revision = config.get("revision")

    # 检查 LoRA 是否需要特殊处理，如果 LoRA 不在 ModelScope，则跳过
    if model_name == "rewriter_lora" and model_id == "YourUsername/YourLoraModelID":
         logging.warning(f"跳过 '{model_name}'：请手动提供 LoRA 的 ModelScope ID 或使用其他方式下载。")
         # 如果 LoRA 需要用 git clone，可以在这里添加逻辑或提示用户
         # 例如： print(f"请手动 git clone <你的 LoRA 仓库地址> 到 {os.path.join(MODEL_ROOT_DIR, 'rewriter_lora')}")
         return None # 返回 None 表示未通过 snapshot_download 下载

    if not model_id:
        logging.error(f"跳过 '{model_name}': 配置中缺少 model_id。")
        return None

    logging.info(f"--- 开始下载模型: {model_name} ---")
    logging.info(f"    Model ID: {model_id}")
    logging.info(f"    根目录: {MODEL_ROOT_DIR}")
    if revision:
         logging.info(f"    版本: {revision}")

    # 创建根目录（如果不存在）
    os.makedirs(MODEL_ROOT_DIR, exist_ok=True)

    try:
        # snapshot_download 会在 cache_dir (我们设为 MODEL_ROOT_DIR) 下创建基于 model_id 的子目录结构
        # 它会返回实际下载到的快照目录的路径
        download_path = snapshot_download(
            model_id=model_id,
            cache_dir=MODEL_ROOT_DIR, # 指定下载的根目录
            revision=revision,
            local_files_only=False, # 确保从网络下载
            # allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "token*"] # 可选：只下载特定文件类型
        )
        logging.info(f"模型 '{model_name}' 下载成功，位于: {download_path}")
        return download_path # 返回实际路径

    except Exception as e:
        logging.error(f"下载模型 '{model_name}' ({model_id}) 失败: {e}", exc_info=True)
        return None

# --- 主下载循环 ---
if __name__ == "__main__":
    downloaded_paths = {}
    all_successful = True

    for name, config in models_to_download.items():
        actual_path = download_model(name, config)
        if actual_path:
            downloaded_paths[name] = actual_path
        elif name != "rewriter_lora": # 如果非 LoRA 下载失败，则标记失败
             all_successful = False
             # 如果 LoRA 是手动处理，上面 download_model 返回 None 不算失败

    print("-" * 40)
    if not downloaded_paths and all_successful:
         print("没有配置有效的模型 ID 进行下载（或者 LoRA 需要手动处理）。")
    elif all_successful:
        print("所有通过 ModelScope 配置的模型下载成功。")
    else:
        print("!!! 部分模型下载失败，请检查上面的日志。")

    print("\n下载完成后的模型路径:")
    for name, path in downloaded_paths.items():
        print(f"  {name}: {path}")

    print("\n重要提示:")
    print("1. 请根据上面打印的 '实际路径' 更新你的 `script/config.py` 文件中的模型路径配置。")
    print("   例如 `EMBEDDING_MODEL_PATH` 应指向 'embedding' 模型下载的路径。")
    print("2. 更新你的 vLLM 启动脚本 (`.sh` 文件)，确保 `--model` 参数指向 'base_llm' 下载的基础模型路径。")
    print("3. 如果 'rewriter_lora' 是手动下载的，请确保 `--lora-modules` 参数指向正确的 LoRA 文件路径。")
    print("-" * 40)