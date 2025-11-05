#!/usr/bin/env python3
"""
数据集查询工具

该脚本用于根据ID和数据集名称查询数据集中的完整内容。
支持的数据集：hotpotqa, ms_marco, natural_questions, triviaqa

使用方法:
    python dataset_query.py <dataset_name> <id> [--file_type train|validation]
    
示例:
    python dataset_query.py natural_questions 5225754983651766092
    python dataset_query.py hotpotqa 5a8b57f25542995d1e6f1371 --file_type validation
"""

import json
import os
import sys
import argparse
from typing import Dict, Any, Optional


class DatasetQuery:
    """数据集查询类"""
    
    def __init__(self, base_path: str = "/home/pushihao/RAG/Reports/experiments/datasets"):
        """
        初始化数据集查询器
        
        Args:
            base_path: 数据集基础路径
        """
        self.base_path = base_path
        self.supported_datasets = ["hotpotqa", "ms_marco", "natural_questions", "triviaqa"]
    
    def validate_dataset(self, dataset_name: str) -> bool:
        """
        验证数据集名称是否支持
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            是否支持该数据集
        """
        return dataset_name.lower() in self.supported_datasets
    
    def get_dataset_path(self, dataset_name: str, file_type: str = "validation") -> str:
        """
        获取数据集文件路径
        
        Args:
            dataset_name: 数据集名称
            file_type: 文件类型 (train 或 validation)
            
        Returns:
            数据集文件完整路径
        """
        return os.path.join(self.base_path, dataset_name.lower(), f"{file_type}.json")

    def get_id_field(self, dataset_name: str) -> str:
        """
        返回不同数据集对应的主键字段名
        
        Args:
            dataset_name: 数据集名称
        
        Returns:
            主键字段名字符串
        """
        mapping = {
            "ms_marco": "query_id",
            "triviaqa": "question_id",
            "hotpotqa": "id",
            "natural_questions": "id",
        }
        return mapping.get(dataset_name.lower(), "id")
    
    def load_dataset(self, dataset_path: str) -> list:
        """
        加载数据集文件
        
        Args:
            dataset_path: 数据集文件路径
            
        Returns:
            数据集内容列表
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                # 尝试按行读取JSON（每行一个JSON对象）
                data = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"警告: 第{line_num}行JSON解析失败: {e}")
                            continue
                
                # 如果按行读取失败，尝试整体读取
                if not data:
                    f.seek(0)
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
                        if not isinstance(data, list):
                            data = [data]
                
                return data
                
        except Exception as e:
            raise Exception(f"加载数据集失败: {e}")
    
    def find_by_id(self, dataset_name: str, target_id: str, file_type: str = "validation") -> Optional[Dict[str, Any]]:
        """
        根据ID查找数据项
        
        Args:
            dataset_name: 数据集名称
            target_id: 目标ID
            file_type: 文件类型 (train 或 validation)
            
        Returns:
            找到的数据项，如果未找到则返回None
        """
        # 验证数据集名称
        if not self.validate_dataset(dataset_name):
            raise ValueError(f"不支持的数据集: {dataset_name}. 支持的数据集: {', '.join(self.supported_datasets)}")
        
        # 获取数据集路径
        dataset_path = self.get_dataset_path(dataset_name, file_type)
        
        # 加载数据集
        data = self.load_dataset(dataset_path)
        
        # 使用数据集对应的主键字段进行匹配
        id_field = self.get_id_field(dataset_name)
        for item in data:
            if str(item.get(id_field, "")) == str(target_id):
                return item
        
        return None
    
    def format_item_info(self, item: Dict[str, Any], dataset_name: str) -> str:
        """
        格式化数据项信息为字符串
        
        Args:
            item: 数据项
            dataset_name: 数据集名称
            
        Returns:
            格式化后的字符串
        """
        output = []
        output.append("=" * 80)
        output.append(f"数据集: {dataset_name.upper()}")
        id_field = self.get_id_field(dataset_name)
        output.append(f"ID: {item.get(id_field, 'N/A')}")
        output.append("=" * 80)
        
        # 根据不同数据集显示不同字段
        if dataset_name.lower() == "hotpotqa":
            output.append(f"问题: {item.get('question', 'N/A')}")
            output.append(f"答案: {item.get('answer', 'N/A')}")
            output.append(f"类型: {item.get('type', 'N/A')}")
            output.append(f"难度: {item.get('level', 'N/A')}")
            
            # 支持事实
            supporting_facts = item.get('supporting_facts', {})
            if supporting_facts:
                output.append("\n支持事实:")
                titles = supporting_facts.get('title', [])
                sent_ids = supporting_facts.get('sent_id', [])
                for title, sent_id in zip(titles, sent_ids):
                    output.append(f"  - {title} (句子ID: {sent_id})")
            
            # 上下文
            context = item.get('context', {})
            if context:
                output.append(f"\n上下文 (共{len(context.get('title', []))}个文档):")
                titles = context.get('title', [])
                sentences = context.get('sentences', [])
                for i, (title, sents) in enumerate(zip(titles, sentences)):
                    output.append(f"\n  文档 {i+1}: {title}")
                    for j, sent in enumerate(sents):
                        output.append(f"    [{j}] {sent}")
        
        elif dataset_name.lower() == "natural_questions":
            output.append(f"问题: {item.get('question', 'N/A')}")
            
            # 文档信息
            document = item.get('document', {})
            if document:
                output.append(f"\n文档标题: {document.get('title', 'N/A')}")
                output.append(f"文档URL: {document.get('url', 'N/A')}")
                
                # HTML内容（完整内容）
                html_content = document.get('html', '')
                if html_content:
                    output.append(f"\nHTML内容:")
                    output.append(html_content)
            
            # 注释信息
            annotations = item.get('annotations', [])
            if annotations:
                output.append(f"\n注释信息 (共{len(annotations)}条):")
                for i, annotation in enumerate(annotations):
                    if isinstance(annotation, dict):
                        output.append(f"  注释 {i+1}:")
                        output.append(f"    ID: {annotation.get('annotation_id', 'N/A')}")
                        
                        # 长答案候选
                        long_answer_candidates = annotation.get('long_answer_candidates', [])
                        output.append(f"    长答案候选: {len(long_answer_candidates)}")
                        
                        # 问题类型
                        output.append(f"    问题类型: {annotation.get('question_type', 'N/A')}")
                        
                        # 长答案
                        long_answer = annotation.get('long_answer', {})
                        if isinstance(long_answer, dict):
                            output.append(f"    长答案起始: {long_answer.get('start_byte', 'N/A')}")
                            output.append(f"    长答案结束: {long_answer.get('end_byte', 'N/A')}")
                        
                        # 短答案
                        short_answers = annotation.get('short_answers', [])
                        if short_answers:
                            output.append(f"    短答案数量: {len(short_answers)}")
                            for j, short_answer in enumerate(short_answers[:3]):  # 只显示前3个
                                if isinstance(short_answer, dict):
                                    output.append(f"      短答案{j+1}: 起始{short_answer.get('start_byte', 'N/A')}, 结束{short_answer.get('end_byte', 'N/A')}")
                    else:
                        output.append(f"  注释 {i+1}: {str(annotation)}")
        
        else:
            # 其他数据集的通用显示
            output.append("完整数据内容:")
            output.append(json.dumps(item, indent=2, ensure_ascii=False))
        
        return "\n".join(output)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="数据集查询工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python dataset_query.py natural_questions 5225754983651766092
  python dataset_query.py hotpotqa 5a8b57f25542995d1e6f1371 --file_type validation
  python dataset_query.py ms_marco 1048578 --file_type train
        """
    )
    
    parser.add_argument("dataset_name", help="数据集名称 (hotpotqa, ms_marco, natural_questions, triviaqa)")
    parser.add_argument("id", help="要查询的数据ID")
    parser.add_argument("--file_type", choices=["train", "validation"], default="validation", 
                       help="文件类型 (默认: validation)")
    parser.add_argument("--output", "-o", help="输出文件路径 (默认: 自动生成)")
    
    args = parser.parse_args()
    
    try:
        # 创建查询器
        query = DatasetQuery()
        
        # 查找数据项
        print(f"正在查询数据集 {args.dataset_name} 中ID为 {args.id} 的数据...")
        item = query.find_by_id(args.dataset_name, args.id, args.file_type)
        
        if item:
            # 格式化输出内容
            content = query.format_item_info(item, args.dataset_name)
            
            # 确定输出文件路径
            if args.output:
                output_file = args.output
            else:
                # 自动生成文件名
                output_file = f"{args.dataset_name}_{args.id}_{args.file_type}.txt"
            
            # 写入文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"查询结果已保存到: {output_file}")
            
        else:
            error_msg = []
            error_msg.append(f"未找到ID为 {args.id} 的数据项")
            error_msg.append(f"数据集: {args.dataset_name}")
            error_msg.append(f"文件类型: {args.file_type}")
            
            # 提供一些调试信息
            dataset_path = query.get_dataset_path(args.dataset_name, args.file_type)
            if os.path.exists(dataset_path):
                data = query.load_dataset(dataset_path)
                error_msg.append(f"数据集总条数: {len(data)}")
                if data:
                    error_msg.append("前3个ID示例:")
                    id_field = query.get_id_field(args.dataset_name)
                    for i, item in enumerate(data[:3]):
                        error_msg.append(f"  {i+1}. {item.get(id_field, 'N/A')}")
            else:
                error_msg.append(f"数据集文件不存在: {dataset_path}")
            
            # 输出错误信息到文件
            error_content = "\n".join(error_msg)
            error_file = f"error_{args.dataset_name}_{args.id}_{args.file_type}.txt"
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(error_content)
            
            print(f"错误信息已保存到: {error_file}")
    
    except Exception as e:
        error_msg = f"错误: {e}"
        print(error_msg)
        
        # 保存错误信息到文件
        error_file = f"error_{args.dataset_name if 'args' in locals() else 'unknown'}.txt"
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(error_msg)
        
        print(f"错误信息已保存到: {error_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()