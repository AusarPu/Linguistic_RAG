# test/model_loader.py

from modelscope import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from script.config import *

def load_models():
    """
    加载用于最终答案生成的模型和用于查询重写的模型。
    """
    print("Loading Generator Model...")
    # 1. 加载生成模型 (Generator Model) - 原来的逻辑
    generator_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map={'': 0}, # 注意：同时加载两个模型时，device_map='auto' 需要谨慎使用
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, # 或 torch.float16 等
    )
    # 如果需要加载 Adapter
    generator_model = PeftModel.from_pretrained(generator_model, ADAPTER_PATH)
    generator_model = generator_model.merge_and_unload()
    generator_model.eval()

    generator_tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
    )
    if generator_tokenizer.pad_token is None:
        generator_tokenizer.pad_token = generator_tokenizer.eos_token
    print("Generator Model loaded.")

    # 2. 加载重写模型 (Rewriter Model)
    print(f"Loading Rewriter Model from: {REWRITER_MODEL_PATH}...")
    rewriter_model = None
    rewriter_tokenizer = None
    try:
        # 尝试加载重写模型，同样使用 device_map='auto'
        # !!! 警告：如果两个模型都很大，显存可能不足 !!!
        # 你可能需要手动指定 device_map，例如将一个模型放在 CPU 或不同的 GPU
        # device_map={'': 'cpu'} 或 device_map={'': 0}, device_map={'': 1}
        rewriter_model = AutoModelForCausalLM.from_pretrained(
            REWRITER_MODEL_PATH,
            device_map={'': 1}, # 同样需要注意显存！
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, # 选择适合模型的类型
        )
        rewriter_model = PeftModel.from_pretrained(rewriter_model, ADAPTER_PATH)
        rewriter_model.eval()

        # 假设 tokenizer 和模型在同一路径，如果不是，请修改 config.py 并使用 REWRITER_TOKENIZER_PATH
        rewriter_tokenizer_path = REWRITER_MODEL_PATH
        rewriter_tokenizer = AutoTokenizer.from_pretrained(
            rewriter_tokenizer_path,
            trust_remote_code=True,
        )
        if rewriter_tokenizer.pad_token is None:
            rewriter_tokenizer.pad_token = rewriter_tokenizer.eos_token
        print("Rewriter Model loaded.")

    except Exception as e:
        print(f"Error loading Rewriter Model: {e}")
        print("Please check the REWRITER_MODEL_PATH in config.py and ensure sufficient resources (VRAM/RAM).")
        # 根据需要决定是否抛出异常或返回 None
        # raise e
        # 或者只加载了生成模型
        print("Proceeding without rewriter model...")


    # 返回两个模型和对应的 tokenizer
    return generator_model, generator_tokenizer, rewriter_model, rewriter_tokenizer