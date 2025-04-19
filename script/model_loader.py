# script/model_loader.py

import logging
# import torch # 如果不再加载本地 Torch 模型，可能不再需要
# from modelscope import AutoTokenizer, AutoModelForCausalLM # 不再需要加载 LLM
# from peft import PeftModel # 如果不加载 adapter，不再需要
# from transformers import PreTrainedModel, PreTrainedTokenizerFast # 可能不再需要
# from typing import Tuple, Optional, Any # 根据返回值调整

# 配置文件导入 - 如果此文件不再加载任何模型，可能不再需要这些特定路径
# from config import (
#     BASE_MODEL_PATH,
#     ADAPTER_PATH,
#     REWRITER_MODEL_PATH
# )

logger = logging.getLogger(__name__)

# 函数签名保持不变，但返回值将是 None
# 或者可以修改签名，但这会立即在调用处引发错误，有助于强制修改
# def load_models() -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
def load_models():
    """
    加载 RAG 应用所需的模型。

    **重要提示**: 此函数当前版本**不再**负责加载大型语言模型（Generator 和 Rewriter LLM）。
    这些 LLM 现在应当由一个独立的 vLLM 服务器加载和提供服务。
    调用此函数将不会返回可用的 LLM 模型或分词器对象。
    如果未来需要在此处加载其他类型的本地模型（例如排序模型），相关逻辑可以加回。
    """
    logger.info("--- model_loader.py: load_models() ---")
    logger.warning("注意：LLM (Generator/Rewriter) 模型加载逻辑已被移除/注释掉。")
    logger.warning("请确保 LLM 已通过 vLLM 服务器运行。")
    logger.info("--------------------------------------")

    # ------------------------------------------------------------------
    # ----- 原本地加载 Generator 和 Rewriter LLM 的代码段 (已注释掉) -----
    # ------------------------------------------------------------------
    #
    # generator_model, generator_tokenizer = None, None
    # rewriter_model, rewriter_tokenizer = None, None
    #
    # # 1. 加载生成模型 (Generator Model)
    # logger.info("Attempting to load Generator Model locally (now deprecated)...")
    # logger.info(f"Original Base model path: {BASE_MODEL_PATH}")
    # logger.info(f"Original Adapter path: {ADAPTER_PATH}")
    # try:
    #     # --- 原加载 Generator 的代码 ---
    #     # generator_model = AutoModelForCausalLM.from_pretrained(...)
    #     # if ADAPTER_PATH and ADAPTER_PATH.strip():
    #     #     generator_model = PeftModel.from_pretrained(generator_model, ADAPTER_PATH)
    #     #     generator_model = generator_model.merge_and_unload()
    #     # generator_model.eval()
    #     # generator_tokenizer = AutoTokenizer.from_pretrained(...)
    #     # if generator_tokenizer.pad_token is None:
    #     #     ... set pad token ...
    #     logger.info("Original Generator Model/Tokenizer loading code skipped.")
    #
    # except Exception as e:
    #     logger.error(f"Error during (now deprecated) local Generator loading: {e}", exc_info=True)
    #     generator_model, generator_tokenizer = None, None # 标记加载失败
    #
    # # 2. 加载重写模型 (Rewriter Model)
    # # 仅在生成模型加载成功后尝试加载重写模型（可选逻辑）
    # # if generator_model and generator_tokenizer: # 或者其他判断条件
    # logger.info(f"Attempting to load Rewriter Model locally (now deprecated)...")
    # logger.info(f"Original Rewriter model path: {REWRITER_MODEL_PATH}")
    # try:
    #     # --- 原加载 Rewriter 的代码 ---
    #     # rewriter_model = AutoModelForCausalLM.from_pretrained(...)
    #     # ... adapter logic ...
    #     # rewriter_model.eval()
    #     # rewriter_tokenizer = AutoTokenizer.from_pretrained(...)
    #     # if rewriter_tokenizer.pad_token is None:
    #     #     ... set pad token ...
    #     logger.info("Original Rewriter Model/Tokenizer loading code skipped.")
    #
    # except Exception as e:
    #     logger.error(f"Error during (now deprecated) local Rewriter loading: {e}", exc_info=True)
    #     logger.warning("Proceeding without locally loaded rewriter model.")
    #     rewriter_model, rewriter_tokenizer = None, None # 标记加载失败
    # # else:
    # #      logger.warning("Skipping local Rewriter model loading because Generator model failed or was skipped.")
    #
    # ------------------------------------------------------------------
    # ------------- End of Commented Out LLM Loading Code --------------
    # ------------------------------------------------------------------

    # 因为 LLM 加载逻辑已移除，返回 None。
    # 调用此函数的代码（如 app_gradio.py）需要更新，不再依赖这些返回值。
    return None, None, None, None