#!/usr/bin/env python3

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_datasets_cache'

from datasets import load_dataset

# 测试不同的数据集ID
dataset_ids_to_test = [
    # Natural Questions 可能的ID
    'natural_questions',
    'nq_open', 
    'google-research-datasets/natural_questions',
    'google/natural_questions',
    
    # HotpotQA 可能的ID
    'hotpot_qa',
    'hotpotqa',
    
    # TriviaQA 可能的ID
    'trivia_qa',
    'mandarjoshi/trivia_qa',
    
    # MS MARCO 可能的ID
    'ms_marco',
    'microsoft/ms_marco',
    
    # 一些已知存在的数据集作为对照
    'squad',
    'glue',
]

print("测试数据集ID可用性...\n")

for dataset_id in dataset_ids_to_test:
    print(f"测试: {dataset_id}")
    try:
        # 使用streaming=True来快速测试连接性，不下载完整数据集
        dataset = load_dataset(dataset_id, trust_remote_code=True, streaming=True)
        print(f"✅ {dataset_id} - 成功")
        
        # 尝试获取第一个样本来验证数据集结构
        try:
            first_item = next(iter(dataset['train']))
            print(f"   样本键: {list(first_item.keys())}")
        except:
            print(f"   无法获取样本结构")
            
    except Exception as e:
        print(f"❌ {dataset_id} - 失败: {str(e)[:100]}...")
    
    print()

print("测试完成")