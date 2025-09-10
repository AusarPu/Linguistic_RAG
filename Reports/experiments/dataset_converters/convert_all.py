#!/usr/bin/env python3
"""
数据集批量转换脚本
运行所有数据集的格式转换，将它们转换为统一格式
"""

import subprocess
import sys
from pathlib import Path

def run_converter(script_name):
    """
    运行单个转换脚本
    
    Args:
        script_name: 转换脚本名称
    """
    script_path = Path(__file__).parent / script_name
    
    print(f"\n{'='*60}")
    print(f"运行转换脚本: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
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

def main():
    """
    主函数：批量运行所有数据集转换脚本
    """
    print("开始批量转换数据集格式...")
    print(f"目标格式: {{id, question, answer, context}}")
    
    # 定义所有转换脚本
    converters = [
        "convert_hotpotqa.py",
        "convert_msmarco.py", 
        "convert_natural_questions.py",
        "convert_triviaqa.py"
    ]
    
    # 运行所有转换脚本
    for converter in converters:
        run_converter(converter)
    
    print(f"\n{'='*60}")
    print("所有数据集转换完成！")
    print(f"转换后的数据保存在: /home/pushihao/RAG/Reports/experiments/datasets/converted/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()