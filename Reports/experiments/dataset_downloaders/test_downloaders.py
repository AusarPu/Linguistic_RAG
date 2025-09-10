#!/usr/bin/env python3
"""
测试所有下载器的基本功能
仅测试初始化和配置，不进行实际下载
"""

import sys
import logging
from typing import Dict, Any

# 导入所有下载器
from download_natural_questions import NaturalQuestionsDownloader
from download_hotpotqa import HotpotQADownloader
from download_triviaqa import TriviaQADownloader
from download_msmarco import MSMarcoDownloader
from base_downloader import BaseDownloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_base_downloader():
    """测试基础下载器"""
    print("\n=== 测试基础下载器 ===")
    try:
        downloader = BaseDownloader()
        cache_size = downloader._get_combined_cache_size()
        print(f"✅ 基础下载器初始化成功")
        print(f"✅ 当前缓存大小: {cache_size / 1024 / 1024:.2f} MB")
        return True
    except Exception as e:
        print(f"❌ 基础下载器测试失败: {e}")
        return False

def test_downloader(downloader_class, name: str) -> Dict[str, Any]:
    """测试单个下载器"""
    print(f"\n=== 测试 {name} 下载器 ===")
    result = {
        'name': name,
        'success': False,
        'error': None,
        'dataset_name': None,
        'dataset_id': None
    }
    
    try:
        # 初始化下载器
        downloader = downloader_class()
        result['dataset_name'] = downloader.dataset_name
        result['dataset_id'] = getattr(downloader, 'dataset_id', 'N/A')
        
        print(f"✅ {name} 下载器初始化成功")
        print(f"   数据集名称: {result['dataset_name']}")
        print(f"   数据集ID: {result['dataset_id']}")
        print(f"   输出目录: {downloader.output_dir}")
        
        # 测试缓存大小计算
        cache_size = downloader._get_combined_cache_size()
        print(f"   当前缓存大小: {cache_size / 1024 / 1024:.2f} MB")
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        print(f"❌ {name} 下载器测试失败: {e}")
    
    return result

def main():
    """主测试函数"""
    print("开始测试所有数据集下载器...")
    
    # 测试基础下载器
    base_success = test_base_downloader()
    
    # 定义要测试的下载器
    downloaders = [
        (NaturalQuestionsDownloader, "Natural Questions"),
        (HotpotQADownloader, "HotpotQA"),
        (TriviaQADownloader, "TriviaQA"),
        (MSMarcoDownloader, "MS MARCO")
    ]
    
    # 测试所有下载器
    results = []
    for downloader_class, name in downloaders:
        result = test_downloader(downloader_class, name)
        results.append(result)
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总:")
    print("="*60)
    
    print(f"基础下载器: {'✅ 成功' if base_success else '❌ 失败'}")
    
    success_count = 0
    for result in results:
        status = "✅ 成功" if result['success'] else "❌ 失败"
        print(f"{result['name']}: {status}")
        if result['success']:
            success_count += 1
        elif result['error']:
            print(f"   错误: {result['error']}")
    
    total_success = base_success and (success_count == len(results))
    print(f"\n总计: {success_count}/{len(results)} 个下载器测试成功")
    print(f"整体测试: {'✅ 全部通过' if total_success else '❌ 存在失败'}")
    
    return 0 if total_success else 1

if __name__ == "__main__":
    sys.exit(main())