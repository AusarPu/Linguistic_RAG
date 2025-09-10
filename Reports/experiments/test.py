#!/usr/bin/env python3
"""
数据集样本查看器
读取datasets目录中的所有数据集，输出每个数据集的训练集和验证集的前3个样本
"""
import json
from pathlib import Path

def load_json_samples(file_path, num_samples=3):
    """从JSON文件中加载指定数量的样本"""
    samples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue
        return samples
    except FileNotFoundError:
        return []
    except Exception as e:
        return []

def write_samples(f, samples, dataset_name, split_name):
    """将样本内容写入文件"""
    f.write(f"\n=== {dataset_name} - {split_name} (前3个样本) ===\n")
    
    if not samples:
        f.write("  没有找到样本\n")
        return
    
    for i, sample in enumerate(samples, 1):
        f.write(f"\n样本 {i}:\n")
        # 格式化输出样本内容
        for key, value in sample.items():
            # 如果值太长，截断显示
            if isinstance(value, str) and len(value) > 200:
                display_value = value[:200] + "..."
            else:
                display_value = value
            f.write(f"  {key}: {display_value}\n")

def main():
    """主函数"""
    datasets_dir = Path("/home/pushihao/RAG/Reports/experiments/datasets")
    output_file = Path("/home/pushihao/RAG/Reports/experiments/dataset_samples.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        if not datasets_dir.exists():
            f.write(f"数据集目录不存在: {datasets_dir}\n")
            return
        
        f.write("数据集样本查看器\n")
        f.write("=" * 50 + "\n")
        
        # 遍历所有数据集目录
        dataset_dirs = [d for d in datasets_dir.iterdir() if d.is_dir()]
        dataset_dirs.sort()  # 按名称排序
        
        if not dataset_dirs:
            f.write("没有找到数据集目录\n")
            return
        
        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name
            f.write(f"\n{'='*60}\n")
            f.write(f"数据集: {dataset_name}\n")
            f.write(f"{'='*60}\n")
            
            # 检查训练集
            train_file = dataset_dir / "train.json"
            if train_file.exists():
                train_samples = load_json_samples(train_file, 3)
                write_samples(f, train_samples, dataset_name, "训练集")
            else:
                f.write(f"\n=== {dataset_name} - 训练集 ===\n")
                f.write("  训练集文件不存在\n")
            
            # 检查验证集/测试集
            validation_file = dataset_dir / "validation.json"
            test_file = dataset_dir / "test.json"
            
            if validation_file.exists():
                validation_samples = load_json_samples(validation_file, 3)
                write_samples(f, validation_samples, dataset_name, "验证集")
            elif test_file.exists():
                test_samples = load_json_samples(test_file, 3)
                write_samples(f, test_samples, dataset_name, "测试集")
            else:
                f.write(f"\n=== {dataset_name} - 验证集/测试集 ===\n")
                f.write("  验证集/测试集文件不存在\n")
        
        f.write(f"\n{'='*60}\n")
        f.write("数据集样本查看完成\n")

if __name__ == "__main__":
    main()