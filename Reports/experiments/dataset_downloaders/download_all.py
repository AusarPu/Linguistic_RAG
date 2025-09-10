#!/usr/bin/env python3
"""
统一数据集下载脚本
可以下载所有数据集或指定的数据集
"""

import sys
import argparse
import logging
from typing import Dict, List

# 导入各个下载器
from download_natural_questions import NaturalQuestionsDownloader
from download_hotpotqa import HotpotQADownloader
from download_triviaqa import TriviaQADownloader
from download_msmarco import MSMarcoDownloader

logger = logging.getLogger(__name__)

class DatasetDownloadManager:
    """数据集下载管理器"""
    
    def __init__(self):
        self.downloaders = {
            'natural_questions': NaturalQuestionsDownloader,
            'hotpotqa': HotpotQADownloader,
            'triviaqa': TriviaQADownloader,
            'msmarco': MSMarcoDownloader
        }
    
    def get_available_datasets(self) -> List[str]:
        """获取可用的数据集列表"""
        return list(self.downloaders.keys())
    
    def download_dataset(self, dataset_name: str) -> bool:
        """下载指定数据集"""
        if dataset_name not in self.downloaders:
            logger.error(f"未知的数据集: {dataset_name}")
            logger.info(f"可用的数据集: {', '.join(self.get_available_datasets())}")
            return False
        
        downloader_class = self.downloaders[dataset_name]
        downloader = downloader_class()
        
        logger.info(f"开始下载数据集: {dataset_name}")
        success = downloader.download()
        
        if success:
            logger.info(f"数据集 {dataset_name} 下载成功")
        else:
            logger.error(f"数据集 {dataset_name} 下载失败")
        
        return success
    
    def download_all_datasets(self) -> Dict[str, bool]:
        """下载所有数据集"""
        results = {}
        
        for dataset_name in self.get_available_datasets():
            logger.info(f"\n{'='*50}")
            logger.info(f"开始下载数据集: {dataset_name}")
            logger.info(f"{'='*50}")
            
            try:
                success = self.download_dataset(dataset_name)
                results[dataset_name] = success
            except Exception as e:
                logger.error(f"下载数据集 {dataset_name} 时发生异常: {e}")
                results[dataset_name] = False
        
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据集下载工具')
    parser.add_argument('--dataset', '-d', type=str, 
                       help='指定要下载的数据集名称')
    parser.add_argument('--list', '-l', action='store_true',
                       help='列出所有可用的数据集')
    parser.add_argument('--all', '-a', action='store_true',
                       help='下载所有数据集')
    
    args = parser.parse_args()
    
    manager = DatasetDownloadManager()
    
    # 列出可用数据集
    if args.list:
        print("可用的数据集:")
        for dataset in manager.get_available_datasets():
            print(f"  - {dataset}")
        return 0
    
    # 下载所有数据集
    if args.all:
        print("开始下载所有数据集...")
        results = manager.download_all_datasets()
        
        print("\n" + "="*60)
        print("下载结果汇总:")
        print("="*60)
        
        success_count = 0
        for dataset_name, success in results.items():
            status = "✅ 成功" if success else "❌ 失败"
            print(f"{dataset_name}: {status}")
            if success:
                success_count += 1
        
        print(f"\n总计: {success_count}/{len(results)} 个数据集下载成功")
        return 0 if success_count == len(results) else 1
    
    # 下载指定数据集
    if args.dataset:
        success = manager.download_dataset(args.dataset)
        return 0 if success else 1
    
    # 没有指定参数，显示帮助
    parser.print_help()
    return 0

if __name__ == "__main__":
    sys.exit(main())