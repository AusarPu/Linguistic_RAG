# test/model_loader.py

import logging
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
# 假设 config 在同一目录下或 Python 路径中
from config import ( # 使用相对导入
    BASE_MODEL_PATH,
    ADAPTER_PATH,
    REWRITER_MODEL_PATH
)
# from transformers import PreTrainedModel, PreTrainedTokenizerFast # 用于类型注解
from typing import Tuple, Optional # 用于类型注解

logger = logging.getLogger(__name__)

def load_models() -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer], Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
    """
    加载用于最终答案生成的模型和用于查询重写的模型。

    Returns:
        Tuple containing:
        - generator_model: Loaded generator model (or None if failed).
        - generator_tokenizer: Loaded generator tokenizer (or None if failed).
        - rewriter_model: Loaded rewriter model (or None if failed).
        - rewriter_tokenizer: Loaded rewriter tokenizer (or None if failed).
    """
    generator_model, generator_tokenizer = None, None
    rewriter_model, rewriter_tokenizer = None, None

    # 1. 加载生成模型 (Generator Model)
    logger.info("Loading Generator Model...")
    logger.info(f"Base model path: {BASE_MODEL_PATH}")
    logger.info(f"Adapter path: {ADAPTER_PATH}")
    try:
        generator_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map='cuda:0', # 使用 'auto'，注意显存！确保 transformers 版本支持
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, # 或根据模型选择 torch.float16
        )
        logger.info("Base generator model loaded.")

        # 如果 ADAPTER_PATH 有效，则加载并合并 Adapter
        if ADAPTER_PATH and ADAPTER_PATH.strip(): # 检查路径是否非空
             logger.info(f"Loading and merging adapter from {ADAPTER_PATH}...")
             # 加载 Adapter 前模型应在 CPU 或 Meta device，或者 from_pretrained 时 device_map='auto'
             generator_model = PeftModel.from_pretrained(generator_model, ADAPTER_PATH)
             generator_model = generator_model.merge_and_unload()
             logger.info("Adapter merged and unloaded.")
        else:
             logger.info("No adapter path provided or adapter path is empty, skipping adapter loading.")

        generator_model.eval() # 设置为评估模式

        generator_tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH,
            trust_remote_code=True,
            use_fast=True # 尽可能使用 Fast Tokenizer
        )
        # 设置 padding token
        if generator_tokenizer.pad_token is None:
            if generator_tokenizer.eos_token is not None:
                 logger.warning("Generator tokenizer missing pad_token, setting it to eos_token.")
                 generator_tokenizer.pad_token = generator_tokenizer.eos_token
            else:
                 logger.error("Generator tokenizer missing both pad_token and eos_token! Padding may fail.")
                 # 可以在这里添加一个默认的 pad_token，但这取决于模型
                 # generator_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                 # generator_model.resize_token_embeddings(len(generator_tokenizer))

        logger.info("Generator Model and Tokenizer loaded successfully.")

    except Exception as e:
        logger.error(f"Error loading Generator Model or Tokenizer: {e}", exc_info=True)
        # 根据需要决定是否继续，这里选择不加载生成模型则无法继续
        # return None, None, None, None # 或者抛出异常 raise e
        generator_model, generator_tokenizer = None, None # 标记加载失败


    # 2. 加载重写模型 (Rewriter Model)
    # 仅在生成模型加载成功后尝试加载重写模型（可选逻辑）
    if generator_model and generator_tokenizer:
        logger.info(f"Loading Rewriter Model from: {REWRITER_MODEL_PATH}...")
        try:
            # !!! 警告：显存管理 !!!
            # device_map='auto' 可能会将两个模型放到同一个 GPU 导致 OOM
            # 考虑手动指定 device_map，例如:
            # device_map_rewriter = {'': 1} # 如果有第二个 GPU
            # device_map_rewriter = {'': 'cpu'} # 如果显存不足，放到 CPU (会很慢)
            rewriter_model = AutoModelForCausalLM.from_pretrained(
                REWRITER_MODEL_PATH,
                device_map='cuda:1', # 同样需要注意显存！可能需要手动调整
                trust_remote_code=True,
                torch_dtype=torch.bfloat16, # 选择适合模型的类型
            )
            logger.info("Base rewriter model loaded.")
             # 如果 ADAPTER_PATH 有效，则加载并合并 Adapter
            if ADAPTER_PATH and ADAPTER_PATH.strip(): # 检查路径是否非空
                logger.info(f"Loading and merging adapter from {ADAPTER_PATH}...")
                 # 加载 Adapter 前模型应在 CPU 或 Meta device，或者 from_pretrained 时 device_map='auto'
                rewriter_model = PeftModel.from_pretrained(rewriter_model, ADAPTER_PATH)
                rewriter_model = rewriter_model.merge_and_unload()
                logger.info("Adapter merged and unloaded.")
            else:
                logger.info("No adapter path provided or adapter path is empty, skipping adapter loading.")
            rewriter_model.eval() # 设置为评估模式

            # 假设 tokenizer 和模型在同一路径
            rewriter_tokenizer_path = REWRITER_MODEL_PATH
            rewriter_tokenizer = AutoTokenizer.from_pretrained(
                rewriter_tokenizer_path,
                trust_remote_code=True,
                use_fast=True
            )
            # 设置 padding token
            if rewriter_tokenizer.pad_token is None:
                if rewriter_tokenizer.eos_token is not None:
                    logger.warning("Rewriter tokenizer missing pad_token, setting it to eos_token.")
                    rewriter_tokenizer.pad_token = rewriter_tokenizer.eos_token
                else:
                    logger.error("Rewriter tokenizer missing both pad_token and eos_token! Padding may fail.")

            logger.info("Rewriter Model and Tokenizer loaded successfully.")

        except Exception as e:
            logger.error(f"Error loading Rewriter Model or Tokenizer: {e}", exc_info=True)
            logger.warning("Proceeding without rewriter model functionality.")
            rewriter_model, rewriter_tokenizer = None, None # 标记加载失败
    else:
         logger.warning("Skipping Rewriter model loading because Generator model failed to load.")


    return generator_model, generator_tokenizer, rewriter_model, rewriter_tokenizer