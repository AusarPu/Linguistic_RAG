#!/usr/bin/env python3
"""
TriviaQA 数据集下载器
仅下载数据集到缓存，不进行处理
"""

import logging
from datasets import load_dataset
from base_downloader import BaseDownloader

logger = logging.getLogger(__name__)

class TriviaQADownloader(BaseDownloader):
    """TriviaQA 数据集下载器"""
    
    def __init__(self):
        super().__init__()
        self.dataset_name = "TriviaQA"
        self.dataset_id = "mandarjoshi/trivia_qa"
        self.configs = ["rc", "unfiltered"]  # 尝试的配置列表
    
    def _download_raw(self):
        """仅下载 TriviaQA 原始数据到缓存"""
        logger.info(f"下载 {self.dataset_name} 原始数据...")
        
        # 尝试不同的配置
        for config in self.configs:
            try:
                logger.info(f"尝试使用配置: {config}")
                self.dataset = load_dataset(self.dataset_id, config, trust_remote_code=True)
                logger.info(f"{self.dataset_name} 数据集下载完成，使用配置: {config}")
                return True
            except Exception as e:
                logger.warning(f"使用配置 {config} 失败: {e}")
                continue
        
        # 所有配置都失败
        error_msg = f"所有配置都失败，无法下载 {self.dataset_name} 数据集"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    def _save_dataset(self):
        """保存数据集到输出目录"""
        if hasattr(self, 'dataset') and self.dataset:
            return self.save_dataset_to_output(self.dataset, self.dataset_name)
        return False
    
    def download(self) -> bool:
        """直接下载数据集，不使用看门狗机制"""
        logger.info(f"开始下载 {self.dataset_name} 数据集...")
        success = self.download_and_save_direct(self._download_raw, self._save_dataset)
        
        if success:
            logger.info(f"{self.dataset_name} 数据集下载成功")
        else:
            logger.error(f"{self.dataset_name} 数据集下载失败")
        
        return success

def main():
    """主函数"""
    downloader = TriviaQADownloader()
    success = downloader.download()
    
    if success:
        print(f"✅ {downloader.dataset_name} 数据集下载成功")
        return 0
    else:
        print(f"❌ {downloader.dataset_name} 数据集下载失败")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())