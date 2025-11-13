#!/usr/bin/env python3
"""
数据集批量转换脚本
运行所有数据集的格式转换，将它们转换为统一格式
支持命令行参数指定数据条数，自动过滤没有答案的数据
"""

import argparse
import subprocess
import sys
from pathlib import Path
import os

# 统一配置
DATASET_CONFIG = {
    "output_dir": "/home/pushihao/RAG/Reports/experiments/datasets/converted",
    "filename_template": "{dataset_name}_validation_kb_chunks.json",
    "datasets": {
        "hotpotqa": {
            "script": "convert_hotpotqa.py",
            "name": "hotpotqa"
        },
        "ms_marco": {
            "script": "convert_msmarco.py", 
            "name": "ms_marco"
        },
        "natural_questions": {
            "script": "convert_natural_questions.py",
            "name": "natural_questions"
        },
        "triviaqa": {
            "script": "convert_triviaqa.py",
            "name": "triviaqa"
        }
    }
}

def get_output_dir():
    """获取统一的输出目录
    支持通过环境变量 CONVERT_OUTPUT_DIR 覆盖默认输出目录，以便隔离不同实验运行。
    """
    env_dir = os.environ.get("CONVERT_OUTPUT_DIR")
    return env_dir if env_dir else DATASET_CONFIG["output_dir"]

def get_output_filename(dataset_name):
    """根据数据集名称生成输出文件名"""
    return DATASET_CONFIG["filename_template"].format(dataset_name=dataset_name)

def supplement_data_to_target(all_samples, convert_func, target_count, filter_no_answer=True):
    """
    补充数据到目标数量的通用函数
    
    Args:
        all_samples: 所有原始样本列表
        convert_func: 转换函数，接受单个样本并返回转换后的样本
        target_count: 目标数量
        filter_no_answer: 是否过滤没有答案的数据
    
    Returns:
        tuple: (转换后的样本列表, 处理统计信息)
    """
    if not target_count or target_count <= 0:
        # 如果没有指定目标数量，处理所有数据
        converted_samples = []
        filtered_count = 0
        
        for sample in all_samples:
            try:
                converted_sample = convert_func(sample)
                
                if filter_no_answer and (not converted_sample["answer"] or converted_sample["answer"].strip() == ""):
                    filtered_count += 1
                    continue
                    
                converted_samples.append(converted_sample)
                
            except Exception as e:
                print(f"警告: 样本处理失败: {e}")
                continue
        
        return converted_samples, {
            "total_processed": len(all_samples),
            "converted_count": len(converted_samples),
            "filtered_count": filtered_count,
            "supplemented": False
        }
    
    # 如果数据集本身就小于目标数量，直接处理所有数据
    if len(all_samples) <= target_count:
        print(f"数据集总量({len(all_samples)})小于或等于目标数量({target_count})，处理所有数据")
        return supplement_data_to_target(all_samples, convert_func, None, filter_no_answer)
    
    converted_samples = []
    processed_count = 0
    filtered_count = 0
    current_index = 0
    
    print(f"目标数量: {target_count}")
    print(f"数据集总量: {len(all_samples)}")
    
    # 循环处理直到达到目标数量或处理完所有数据
    while len(converted_samples) < target_count and current_index < len(all_samples):
        sample = all_samples[current_index]
        current_index += 1
        processed_count += 1
        
        try:
            converted_sample = convert_func(sample)
            
            # 检查是否需要过滤没有答案的数据
            if filter_no_answer and (not converted_sample["answer"] or converted_sample["answer"].strip() == ""):
                filtered_count += 1
                continue
                
            converted_samples.append(converted_sample)
            
        except Exception as e:
            print(f"警告: 样本处理失败: {e}")
            continue
    
    # 如果处理完所有数据仍未达到目标数量
    if len(converted_samples) < target_count:
        shortage = target_count - len(converted_samples)
        print(f"警告: 处理完所有数据后仍缺少 {shortage} 个有效样本")
        print(f"实际获得: {len(converted_samples)} 个样本")
    else:
        print(f"成功补充到目标数量: {len(converted_samples)} 个样本")
    
    return converted_samples, {
        "total_processed": processed_count,
        "converted_count": len(converted_samples),
        "filtered_count": filtered_count,
        "supplemented": True,
        "target_count": target_count
    }

def run_converter(script_name, dataset_name, max_samples=None, filter_no_answer=True):
    """
    运行单个转换脚本，只转换验证集
    
    Args:
        script_name: 转换脚本名称
        dataset_name: 数据集名称（用于输出文件命名）
        max_samples: 最大样本数量，None表示不限制
        filter_no_answer: 是否过滤没有答案的数据
    """
    script_path = Path(__file__).parent / script_name
    
    print(f"\n{'='*60}")
    print(f"运行转换脚本: {script_name}")
    print(f"数据集: {dataset_name} (仅验证集)")
    if max_samples:
        print(f"最大样本数: {max_samples}")
    if filter_no_answer:
        print("过滤模式: 丢弃没有答案的数据")
    print(f"{'='*60}")
    
    try:
        # 构建命令行参数
        cmd = [sys.executable, str(script_path)]
        if max_samples:
            cmd.extend(['--max-samples', str(max_samples)])
        if filter_no_answer:
            cmd.append('--filter-no-answer')
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print(f"错误输出: {result.stderr}")
        
        if result.returncode == 0:
            print(f"✅ {script_name} 转换成功")
        else:
            print(f"❌ {script_name} 转换失败，退出码: {result.returncode}")
            
    except Exception as e:
        print(f"❌ 运行 {script_name} 时发生异常: {e}")

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description="批量转换数据集格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python convert_all.py                    # 转换所有数据，不限制数量
  python convert_all.py --max-samples 1000 # 每个数据集最多转换1000条
  python convert_all.py --max-samples 10000 --no-filter  # 转换10000条，不过滤无答案数据
        """
    )
    
    parser.add_argument(
        '--max-samples', 
        type=int, 
        default=None,
        help='每个数据集的最大样本数量（默认：不限制）'
    )
    
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='不过滤没有答案的数据（默认：过滤）'
    )
    
    return parser.parse_args()


def main():
    """
    主函数：批量运行所有数据集转换脚本（仅验证集）
    """
    args = parse_args()
    
    print("开始批量转换数据集格式（仅验证集）...")
    print(f"目标格式: {{id, question, answer, context}}")
    
    if args.max_samples:
        print(f"每个数据集最大样本数: {args.max_samples}")
    else:
        print("样本数量: 不限制")
        
    filter_no_answer = not args.no_filter
    if filter_no_answer:
        print("数据过滤: 启用（丢弃没有答案的数据）")
    else:
        print("数据过滤: 禁用")
    
    # 使用统一配置中的数据集信息
    converters = [(config["script"], config["name"]) for config in DATASET_CONFIG["datasets"].values()]
    
    # 运行所有转换脚本
    for script_name, dataset_name in converters:
        run_converter(script_name, dataset_name, args.max_samples, filter_no_answer)
    
    print(f"\n{'='*60}")
    print("所有数据集转换完成！")
    print(f"转换后的验证集保存在: {get_output_dir()}")
    print(f"文件命名格式: {DATASET_CONFIG['filename_template']}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()