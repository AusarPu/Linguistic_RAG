# download_models.py (支持 Hugging Face 和 ModelScope, 使用镜像)
import logging
import os
import sys

# 确保导入 huggingface_hub
try:
    from huggingface_hub import snapshot_download as hf_snapshot_download
except ImportError:
    print("错误：无法导入 huggingface_hub 库。请确保已安装。")
    print("运行: pip install huggingface-hub (或 uv pip install huggingface-hub)")
    sys.exit(1)

# 可选导入 modelscope
try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    modelscope_available = True
except ImportError:
    modelscope_available = False
    print("信息：未找到 modelscope 库，将无法下载 'modelscope' 来源的模型。")


# 配置基础日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 用户配置 ---
# !!! 重要：请仔细检查并修改下面的配置项 !!!

# 1. 定义国内 Hugging Face 镜像地址 (如果需要)
#    常见的有: https://hf-mirror.com/ (或者你可以留空 "" 不使用镜像)
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

# 2. 定义所有模型存放的根目录
MODEL_ROOT_DIR = "/home/pushihao/RAG/models" # <--- 修改为你希望存放模型的本地根目录

# 3. 配置要下载的模型列表
models_to_download = {
    # --- 基础 LLM (用于生成器) ---
    # 假设生成器仍然使用之前从 ModelScope 下载的非量化模型
    "base_llm_generator": {
        "source": "modelscope",        # <--- 指定来源: "modelscope" 或 "huggingface"
        "model_id": "Qwen/QwQ-32B", # <--- !! 确认或替换为你生成器用的基础模型 ID !!
        "target_dir": os.path.join(MODEL_ROOT_DIR,"Qwen/QwQ-32B"),
        # "revision": "v1.0.0"       # 可选：指定版本
    },
    # --- 量化 LLM (用于重写器) ---
    "quantized_llm_rewriter": {
        "source": "modelscope",         # <--- 指定来源: "huggingface"
        "model_id": "Qwen/Qwen3-32B", # <--- 使用你找到的 HF ID
        # 为这个模型指定一个明确的本地存放目录 (会在 MODEL_ROOT_DIR 下创建)
        "target_dir": os.path.join(MODEL_ROOT_DIR,"Qwen/Qwen3-30B-A3B") #<--- 可以修改目录名
    },
    # --- 嵌入模型 ---
    "embedding": {
        "source": "modelscope",        # <--- 指定来源
        "model_id": "BAAI/bge-large-zh-v1.5", # <--- !! 确认或替换 !!
        "target_dir": os.path.join(MODEL_ROOT_DIR,"BAAI/bge-large-zh-v1.5")
    },
    # --- LoRA 适配器 (用于重写) ---
    # LoRA 通常体积较小，可以手动下载或用 git clone
    "rewriter_lora": {
        "source": "manual",           # <--- 标记为手动处理
        "model_id": "N/A",            # 无需 Hub ID
        # 指定 LoRA 文件最终应该存放的目录
        "target_dir": os.path.join(MODEL_ROOT_DIR, "rewriter_lora") #<--- 修改为你存放 LoRA 的目录
    }
}
# --- 结束用户配置 ---


def download_hf_model(model_name, config):
    """使用 huggingface_hub 从 Hugging Face Hub 下载模型"""
    repo_id = config.get("model_id")
    target_dir = config.get("target_dir") # 使用指定的本地目录
    revision = config.get("revision")

    if not repo_id or not target_dir:
        logging.error(f"跳过 Hugging Face 模型 '{model_name}': 缺少 repo_id 或 target_dir 配置。")
        return None

    logging.info(f"--- 开始下载 Hugging Face 模型: {model_name} ---")
    logging.info(f"    仓库 ID (Repo ID): {repo_id}")
    logging.info(f"    目标本地目录 (Target Directory): {target_dir}")
    if HF_MIRROR_ENDPOINT:
        logging.info(f"    使用镜像 (Using Mirror): {HF_MIRROR_ENDPOINT}")
    else:
        logging.info("    使用官方源 (Using Official Hub)")
    if revision: logging.info(f"    版本 (Revision): {revision}")

    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    try:
        # 使用 snapshot_download 下载整个仓库
        # local_dir 参数指定下载/链接到的确切目录
        # local_dir_use_symlinks=False 表示直接复制文件而非创建符号链接（更独立）
        hf_snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            endpoint=HF_MIRROR_ENDPOINT if HF_MIRROR_ENDPOINT else None, # 传入镜像地址
            revision=revision,
            # token=os.environ.get("HF_TOKEN"), # 如果是私有仓库，需要 Token
            # user_agent="my-rag-project" # 可选的用户代理
        )
        logging.info(f"Hugging Face 模型 '{model_name}' 下载成功到: {target_dir}")
        return target_dir # 返回指定的本地目录路径
    except Exception as e:
        logging.error(f"下载 Hugging Face 模型 '{model_name}' ({repo_id}) 失败: {e}", exc_info=True)
        return None


def download_ms_model(model_name, config):
    """使用 modelscope 从 ModelScope Hub 下载模型"""
    if not modelscope_available:
        logging.error(f"跳过 ModelScope 模型 '{model_name}': modelscope 库未安装。")
        return None

    model_id = config.get("model_id")
    revision = config.get("revision")
    # ModelScope 下载时，文件会放在 cache_dir 下以 model_id 命名的目录结构中
    cache_dir = MODEL_ROOT_DIR # 使用统一的根目录作为缓存目录

    if not model_id:
         logging.error(f"跳过 ModelScope 模型 '{model_name}': 缺少 model_id 配置。")
         return None

    logging.info(f"--- 开始下载 ModelScope 模型: {model_name} ---")
    logging.info(f"    模型 ID (Model ID): {model_id}")
    logging.info(f"    缓存根目录 (Cache Directory): {cache_dir}")
    if revision: logging.info(f"    版本 (Revision): {revision}")

    os.makedirs(cache_dir, exist_ok=True)

    try:
        # 调用 ModelScope 的下载函数
        download_path = ms_snapshot_download(
            model_id=model_id,
            cache_dir=cache_dir,
            revision=revision,
        )
        logging.info(f"ModelScope 模型 '{model_name}' 下载成功，位于: {download_path}")
        return download_path # 返回 ModelScope 实际创建的路径
    except Exception as e:
        logging.error(f"下载 ModelScope 模型 '{model_name}' ({model_id}) 失败: {e}", exc_info=True)
        return None


# --- 主下载循环 ---
if __name__ == "__main__":
    downloaded_paths = {}
    errors = False

    print("开始处理模型下载任务...")
    for name, config in models_to_download.items():
        source = config.get("source", "").lower()
        actual_path = None

        if source == "huggingface":
            actual_path = download_hf_model(name, config)
        elif source == "modelscope":
            actual_path = download_ms_model(name, config)
        elif source == "manual":
             logging.info(f"跳过 '{name}': 标记为手动处理。请确保文件已放置在预期路径。")
             actual_path = config.get("target_dir", os.path.join(MODEL_ROOT_DIR, name))
             if not os.path.exists(actual_path):
                  logging.warning(f"手动指定的路径 '{actual_path}' (用于 '{name}') 不存在!")
                  # 可以选择在这里标记错误
             downloaded_paths[name] = actual_path # 记录预期的路径
             continue # 继续下一个
        else:
             logging.warning(f"跳过 '{name}': 未知的来源类型 '{source}'。请在配置中指定 'huggingface', 'modelscope' 或 'manual'。")
             errors = True
             continue

        if actual_path:
            downloaded_paths[name] = actual_path
        else:
            # 如果下载函数返回 None，则标记为错误
            errors = True

    # --- 打印总结信息 ---
    print("-" * 40)
    if not errors and downloaded_paths:
        print("模型下载/路径检查完成。")
    elif errors:
        print("!!! 部分模型下载失败或配置有误，请检查上面的日志。")
    else:
         print("没有配置有效的模型进行下载或检查。")

    print("\n最终模型使用的参考路径:")
    # 打印所有处理过的模型的预期或实际路径
    for name, path in downloaded_paths.items():
        print(f"  [{name}]: {path}")

    print("\n重要提示:")
    print("1. 请根据上面打印的路径，仔细检查并更新你的 `script/config.py` 文件中的相关路径配置。")
    print("   - 例如 `VLLM_GENERATOR_MODEL_ID_FOR_API` 可能对应 'base_llm_generator' 的路径。")
    print("   - `VLLM_REWRITER_MODEL_ID_FOR_API` 现在应该对应 'quantized_llm_rewriter' 的**基础模型标识符**（API调用时可能仍用基础模型ID，vLLM内部知道加载的是量化版，需测试确认）。")
    print("   - `EMBEDDING_MODEL_PATH` 应指向 'embedding' 的路径。")
    print("   - `VLLM_REWRITER_LORA_LOCAL_PATH` 应指向 'rewriter_lora' 的路径。")
    print("2. 更新你的 vLLM 启动脚本 (`.sh` 文件):")
    print("   - `start_generator_vllm.sh`: 确保 `--model` 参数指向 'base_llm_generator' 下载到的**实际路径**。")
    print("   - `start_rewriter_vllm.sh`: 确保 `--model` 参数指向 'quantized_llm_rewriter' 下载到的**实际路径** (AWQ 模型目录)，并且**务必添加 `--quantization awq` 参数**。同时确保 `--lora-modules` 指向 'rewriter_lora' 的**实际路径**。")
    print("-" * 40)