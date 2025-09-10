#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集测试脚本
用于测试已下载数据集的问题-答案-上下文提取功能

使用方法:
1. 确保数据集已经下载并处理完成
2. 在Reports目录下运行: python test.py
3. 脚本会自动测试所有支持的数据集

支持的数据集:
- Natural Questions
- HotpotQA
- MS MARCO
- TriviaQA

测试内容:
- 验证数据集文件是否存在
- 检查必需字段完整性
- 提取问题-答案-上下文信息
- 显示数据集统计信息
"""

import json
import os
from typing import Dict, List, Any, Optional


class DatasetTester:
    """数据集测试类"""
    
    def __init__(self, datasets_dir: str = "./datasets"):
        """初始化测试器
        
        Args:
            datasets_dir: 数据集根目录路径
        """
        self.datasets_dir = datasets_dir
        self.supported_datasets = [
            "natural_questions",
            "hotpot_qa", 
            "ms_marco",
            "trivia_qa"
        ]
    
    def load_dataset(self, dataset_name: str, split: str = "train") -> Optional[List[Dict[str, Any]]]:
        """加载指定数据集
        
        Args:
            dataset_name: 数据集名称
            split: 数据集分割 (train/validation)
            
        Returns:
            数据集列表，如果加载失败返回None
        """
        if dataset_name not in self.supported_datasets:
            print(f"❌ 不支持的数据集: {dataset_name}")
            return None
            
        file_path = os.path.join(self.datasets_dir, dataset_name, f"{split}.json")
        
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ 成功加载 {dataset_name}/{split}.json，共 {len(data)} 条数据")
            return data
        except Exception as e:
            print(f"❌ 加载文件失败 {file_path}: {e}")
            return None
    
    def validate_sample(self, sample: Dict[str, Any], dataset_name: str) -> bool:
        """验证单个样本的数据完整性
        
        Args:
            sample: 数据样本
            dataset_name: 数据集名称
            
        Returns:
            验证是否通过
        """
        required_fields = ["id", "question", "answer", "context"]
        
        # 检查必需字段
        for field in required_fields:
            if field not in sample:
                print(f"❌ 缺少必需字段: {field}")
                return False
            # 对于context字段，允许为空（某些数据集可能没有上下文）
            if field == "context":
                continue
            if not sample[field] or (isinstance(sample[field], str) and sample[field].strip() == ""):
                print(f"❌ 字段为空: {field}")
                return False
        
        # 检查数据类型
        if not isinstance(sample["question"], str):
            print(f"❌ question字段类型错误: {type(sample['question'])}")
            return False
            
        if not isinstance(sample["answer"], str):
            print(f"❌ answer字段类型错误: {type(sample['answer'])}")
            return False
            
        if not isinstance(sample["context"], str):
            print(f"❌ context字段类型错误: {type(sample['context'])}")
            return False
        
        return True
    
    def extract_qa_context(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """提取问题-答案-上下文信息
        
        Args:
            sample: 数据样本
            
        Returns:
            包含question、answer、context的字典
        """
        return {
            "question": sample.get("question", "").strip(),
            "answer": sample.get("answer", "").strip(),
            "context": sample.get("context", "").strip(),
            "document_title": sample.get("document_title", "").strip()
        }
    
    def test_dataset(self, dataset_name: str, num_samples: int = 5) -> bool:
        """测试指定数据集
        
        Args:
            dataset_name: 数据集名称
            num_samples: 测试样本数量
            
        Returns:
            测试是否通过
        """
        print(f"\n🔍 测试数据集: {dataset_name}")
        print("=" * 50)
        
        # 测试训练集和验证集
        for split in ["train", "validation"]:
            print(f"\n📂 测试 {split} 分割:")
            data = self.load_dataset(dataset_name, split)
            
            if data is None:
                continue
                
            # 验证前几个样本
            test_samples = min(num_samples, len(data))
            valid_count = 0
            
            for i in range(test_samples):
                sample = data[i]
                print(f"\n样本 {i+1}:")
                
                if self.validate_sample(sample, dataset_name):
                    valid_count += 1
                    qa_context = self.extract_qa_context(sample)
                    
                    print(f"  ✅ ID: {sample['id']}")
                    print(f"  ✅ 问题: {qa_context['question'][:100]}...")
                    print(f"  ✅ 答案: {qa_context['answer'][:50]}...")
                    
                    if qa_context['context'].strip():
                        print(f"  ✅ 上下文长度: {len(qa_context['context'])} 字符")
                    else:
                        print(f"  ⚠️  上下文为空")
                        
                    if qa_context['document_title']:
                        print(f"  ✅ 文档标题: {qa_context['document_title'][:50]}...")
                else:
                    print(f"  ❌ 样本 {i+1} 验证失败")
            
            success_rate = valid_count / test_samples * 100
            print(f"\n📊 {split} 分割验证结果: {valid_count}/{test_samples} ({success_rate:.1f}%)")
            
            if success_rate < 100:
                return False
        
        return True
    
    def test_all_datasets(self, num_samples: int = 3) -> None:
        """测试所有支持的数据集
        
        Args:
            num_samples: 每个数据集测试的样本数量
        """
        print("🚀 开始测试所有数据集")
        print("=" * 60)
        
        results = {}
        
        for dataset_name in self.supported_datasets:
            try:
                results[dataset_name] = self.test_dataset(dataset_name, num_samples)
            except Exception as e:
                print(f"❌ 测试 {dataset_name} 时发生错误: {e}")
                results[dataset_name] = False
        
        # 输出总结
        print("\n" + "=" * 60)
        print("📋 测试结果总结:")
        print("=" * 60)
        
        passed = 0
        total = len(results)
        
        for dataset_name, success in results.items():
            status = "✅ 通过" if success else "❌ 失败"
            print(f"  {dataset_name:<20} {status}")
            if success:
                passed += 1
        
        print(f"\n🎯 总体结果: {passed}/{total} 个数据集测试通过 ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 所有数据集测试通过！数据集可以正常使用。")
        else:
            print("⚠️  部分数据集测试失败，请检查数据集完整性。")


def main():
    """主函数"""
    # 创建测试器实例
    tester = DatasetTester("./datasets")
    
    print("📚 数据集问题-答案-上下文提取测试")
    print("=" * 60)
    print("支持的数据集:")
    for i, dataset in enumerate(tester.supported_datasets, 1):
        print(f"  {i}. {dataset}")
    
    # 测试所有数据集
    tester.test_all_datasets(num_samples=3)
    
    print("\n" + "=" * 60)
    print("测试完成！")


if __name__ == "__main__":
    main()