#!/usr/bin/env python3
"""
流式数据处理模块
用于从头开始逐行处理数据文件，不需要将整个文件加载到内存中
"""

import json


def process_data_streaming(input_file, convert_func, target_count=None, filter_no_answer=True):
    """
    流式处理数据文件，从头开始逐行读取和处理
    
    Args:
        input_file: 输入文件路径
        convert_func: 转换函数，接受单个样本并返回转换后的样本
        target_count: 目标数量，None表示处理所有数据
        filter_no_answer: 是否过滤没有答案的数据
    
    Returns:
        tuple: (转换后的样本列表, 处理统计信息)
    """
    converted_samples = []
    processed_count = 0
    filtered_count = 0
    line_count = 0
    
    print(f"开始流式处理文件: {input_file}")
    if target_count:
        print(f"目标数量: {target_count}")
    if filter_no_answer:
        print("过滤模式: 丢弃没有答案的数据")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if not line:
                continue
                
            try:
                sample = json.loads(line)
                processed_count += 1
                
                # 转换样本
                converted_sample = convert_func(sample)
                
                # 检查是否需要过滤没有答案的数据
                if filter_no_answer and (not converted_sample["answer"] or converted_sample["answer"].strip() == ""):
                    filtered_count += 1
                    continue
                    
                converted_samples.append(converted_sample)
                
                # 如果达到目标数量，停止处理
                if target_count and len(converted_samples) >= target_count:
                    print(f"已达到目标数量 {target_count}，停止处理")
                    break
                    
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_count}行JSON解析失败: {e}")
                continue
            except Exception as e:
                print(f"警告: 第{line_count}行处理失败: {e}")
                continue
    
    # 返回统计信息
    stats = {
        "total_lines": line_count,
        "total_processed": processed_count,
        "converted_count": len(converted_samples),
        "filtered_count": filtered_count,
        "streaming": True
    }
    
    if target_count and len(converted_samples) < target_count:
        shortage = target_count - len(converted_samples)
        print(f"警告: 处理完文件后仍缺少 {shortage} 个有效样本")
        print(f"实际获得: {len(converted_samples)} 个样本")
        stats["shortage"] = shortage
    else:
        print(f"成功处理: {len(converted_samples)} 个样本")
    
    return converted_samples, stats


def process_json_array_streaming(input_file, convert_func, target_count=None, filter_no_answer=True):
    """
    流式处理JSON数组文件，适用于整个文件是一个JSON数组的情况
    
    Args:
        input_file: 输入文件路径
        convert_func: 转换函数，接受单个样本并返回转换后的样本
        target_count: 目标数量，None表示处理所有数据
        filter_no_answer: 是否过滤没有答案的数据
    
    Returns:
        tuple: (转换后的样本列表, 处理统计信息)
    """
    converted_samples = []
    processed_count = 0
    filtered_count = 0
    
    print(f"开始流式处理JSON数组文件: {input_file}")
    if target_count:
        print(f"目标数量: {target_count}")
    if filter_no_answer:
        print("过滤模式: 丢弃没有答案的数据")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # 读取整个JSON文件（这里仍需要读取完整文件，因为是JSON数组格式）
            data = json.load(f)
            
            # 处理不同的数据结构
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict) and "Data" in data:
                samples = data["Data"]
            else:
                samples = [data]
            
            print(f"文件包含 {len(samples)} 个样本")
            
            # 从头开始处理样本
            for sample in samples:
                try:
                    processed_count += 1
                    
                    # 转换样本
                    converted_sample = convert_func(sample)
                    
                    # 检查是否需要过滤没有答案的数据
                    if filter_no_answer and (not converted_sample["answer"] or converted_sample["answer"].strip() == ""):
                        filtered_count += 1
                        continue
                        
                    converted_samples.append(converted_sample)
                    
                    # 如果达到目标数量，停止处理
                    if target_count and len(converted_samples) >= target_count:
                        print(f"已达到目标数量 {target_count}，停止处理")
                        break
                        
                except Exception as e:
                    print(f"警告: 样本处理失败: {e}")
                    continue
                    
    except Exception as e:
        print(f"错误: 文件读取失败: {e}")
        return [], {"error": str(e)}
    
    # 返回统计信息
    stats = {
        "total_processed": processed_count,
        "converted_count": len(converted_samples),
        "filtered_count": filtered_count,
        "streaming": True
    }
    
    if target_count and len(converted_samples) < target_count:
        shortage = target_count - len(converted_samples)
        print(f"警告: 处理完文件后仍缺少 {shortage} 个有效样本")
        print(f"实际获得: {len(converted_samples)} 个样本")
        stats["shortage"] = shortage
    else:
        print(f"成功处理: {len(converted_samples)} 个样本")
    
    return converted_samples, stats