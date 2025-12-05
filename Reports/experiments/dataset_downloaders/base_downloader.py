#!/usr/bin/env python3
"""
基础数据集下载器
包含通用的看门狗功能和环境配置
"""

import os
import logging
import time
from collections import deque
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Tuple, Any

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_datasets_cache'

# 尝试设置huggingface_hub的配置
try:
    from huggingface_hub import configure_http_backend
    import requests
    configure_http_backend(backend_factory=lambda: requests.Session())
except ImportError:
    pass

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseDownloader:
    """基础下载器类，包含看门狗功能"""
    
    def __init__(self, cache_dir: str = "/tmp", output_dir: str = "/home/pushihao/RAG/Reports/experiments/datasets"):
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def _get_combined_cache_size(self) -> int:
        """计算缓存目录的累计大小（用于测速）"""
        total = 0
        for env_key in ("HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE"):
            d = os.environ.get(env_key)
            if d and os.path.isdir(d):
                for root, _, files in os.walk(d):
                    for fn in files:
                        fp = os.path.join(root, fn)
                        try:
                            total += os.path.getsize(fp)
                        except OSError:
                            pass
        return total

    def _watchdog_run(self, target: Callable, args: Tuple = (), 
                     min_speed_bytes: int = 1_048_576, window_seconds: int = 60, 
                     check_interval: float = 1.0, max_retries: int = 100) -> bool:
        """看门狗运行器：监控下载速度，低于阈值持续一段时间则中断并重试"""
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            logger.info(f"[Watchdog] 启动下载子进程 (第{attempt}次尝试)...")
            p = mp.Process(target=target, args=args)
            p.start()

            prev_size = self._get_combined_cache_size()
            speeds = deque(maxlen=window_seconds)
            aborted = False

            while p.is_alive():
                time.sleep(check_interval)
                cur_size = self._get_combined_cache_size()
                delta = max(0, cur_size - prev_size)
                prev_size = cur_size
                speeds.append(delta)

                if len(speeds) == window_seconds:
                    avg_speed = sum(speeds) / window_seconds
                    logger.info(f"[Watchdog] 最近{window_seconds}s平均速度: {avg_speed/1_048_576:.2f} MiB/s")
                    if avg_speed < min_speed_bytes:
                        logger.warning(f"[Watchdog] 速度低于阈值 ({avg_speed/1_048_576:.2f} < 1.00 MiB/s)，中断并重试...")
                        try:
                            p.terminate()
                            p.join(timeout=10)
                            if p.is_alive():
                                p.kill()
                                p.join(timeout=5)
                        finally:
                            aborted = True
                        break

            # 如果是因为低速而中断，直接重试
            if aborted:
                continue

            # 子进程自然结束
            p.join()
            if p.exitcode == 0:
                logger.info("[Watchdog] 子进程成功完成")
                return True
            else:
                logger.error(f"[Watchdog] 子进程异常退出，退出码: {p.exitcode}")
                if attempt < max_retries:
                    logger.info(f"[Watchdog] 准备第{attempt + 1}次重试...")
                    time.sleep(5)  # 等待5秒后重试

        logger.error(f"[Watchdog] 所有重试都失败了，放弃下载")
        return False

    def download_with_watchdog(self, download_func: Callable, *args, **kwargs) -> bool:
        """使用看门狗下载数据集"""
        return self._watchdog_run(download_func, args, **kwargs)
    
    def download_and_save_direct(self, download_func: Callable, save_func: Callable = None, *args, **kwargs) -> bool:
        """直接下载数据集并保存，不使用看门狗机制"""
        try:
            logger.info("[Direct Download] 开始直接下载数据集...")
            # 直接调用下载函数
            download_func(*args, **kwargs)
            logger.info("[Direct Download] 数据集下载完成")
            
            if save_func:
                # 执行保存操作
                logger.info("[Direct Download] 开始保存数据集到输出目录...")
                save_success = save_func()
                if save_success:
                    logger.info("[Direct Download] 数据集保存成功")
                else:
                    logger.warning("[Direct Download] 数据集保存失败，但下载成功")
                return True
            else:
                return True
                
        except Exception as e:
            logger.error(f"[Direct Download] 下载过程中发生异常: {e}")
            return False
    
    def download_and_save_with_watchdog(self, download_func: Callable, save_func: Callable = None, *args, **kwargs) -> bool:
        """使用看门狗下载数据集，然后在主进程中保存数据"""
        # 先用看门狗下载数据集到缓存
        download_success = self._watchdog_run(download_func, args, **kwargs)
        
        if download_success and save_func:
            # 下载成功后，在主进程中执行保存操作（不受看门狗监控）
            try:
                logger.info("[Main Process] 开始保存数据集到输出目录...")
                save_success = save_func()
                if save_success:
                    logger.info("[Main Process] 数据集保存成功")
                    return True
                else:
                    logger.warning("[Main Process] 数据集保存失败，但下载成功")
                    return True  # 下载成功就算成功
            except Exception as e:
                logger.error(f"[Main Process] 保存数据集时发生异常: {e}")
                return True  # 下载成功就算成功
        
        return download_success
    
    def save_dataset_to_output(self, dataset, dataset_name: str) -> bool:
        """将下载的数据集保存到输出目录"""
        try:
            import json
            
            # 创建数据集专用目录
            dataset_dir = self.output_dir / dataset_name.lower().replace(' ', '_')
            dataset_dir.mkdir(exist_ok=True, parents=True)
            
            # 保存训练集和验证集
            if 'train' in dataset:
                train_file = dataset_dir / 'train.json'
                with open(train_file, 'w', encoding='utf-8') as f:
                    for example in dataset['train']:
                        json.dump(example, f, ensure_ascii=False)
                        f.write('\n')
                logger.info(f"训练集已保存到: {train_file}")
            
            if 'validation' in dataset:
                val_file = dataset_dir / 'validation.json'
                with open(val_file, 'w', encoding='utf-8') as f:
                    for example in dataset['validation']:
                        json.dump(example, f, ensure_ascii=False)
                        f.write('\n')
                logger.info(f"验证集已保存到: {val_file}")
            
            # 如果没有validation，尝试test
            elif 'test' in dataset:
                test_file = dataset_dir / 'test.json'
                with open(test_file, 'w', encoding='utf-8') as f:
                    for example in dataset['test']:
                        json.dump(example, f, ensure_ascii=False)
                        f.write('\n')
                logger.info(f"测试集已保存到: {test_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"保存数据集失败: {e}")
            return False