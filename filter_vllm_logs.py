#!/usr/bin/env python3
"""
VLLM日志过滤器
过滤掉HTTP请求日志，只保留引擎性能统计和重要信息
支持自动日志轮转和清理
"""

import os
import re
import time
import threading
from pathlib import Path
from datetime import datetime

class VLLMLogFilter:
    def __init__(self, input_log_path, output_log_path, max_lines=1000, check_interval=10):
        """
        初始化日志过滤器
        
        Args:
            input_log_path: 原始VLLM日志文件路径
            output_log_path: 过滤后的日志文件路径
            max_lines: 最大保留行数，超过后会清理
            check_interval: 检查间隔（秒）
        """
        self.input_log_path = Path(input_log_path)
        self.output_log_path = Path(output_log_path)
        self.max_lines = max_lines
        self.check_interval = check_interval
        self.last_position = 0
        self.running = False
        
        # 定义要保留的日志模式
        self.keep_patterns = [
            r'Engine \d+: Avg prompt throughput:',  # 性能统计
            r'GPU KV cache usage:',                 # 缓存使用情况
            r'Prefix cache hit rate:',              # 缓存命中率
            r'Starting to load model',              # 模型加载
            r'Model loading took',                  # 模型加载完成
            r'Graph capturing finished',           # 图捕获完成
            r'Starting vLLM API server',           # 服务器启动
            r'Available routes are:',              # 路由信息
            r'ERROR',                              # 错误信息
            r'WARNING.*(?!127\.0\.0\.1)',          # 警告信息（排除HTTP相关）
            r'CRITICAL',                           # 严重错误
            r'Loading weights took',               # 权重加载
            r'init engine.*took.*seconds',         # 引擎初始化
        ]
        
        # 定义要过滤掉的模式
        self.filter_patterns = [
            r'127\.0\.0\.1.*POST /v1/chat/completions.*200 OK',  # HTTP请求日志
            r'127\.0\.0\.1.*POST /v1/completions.*200 OK',       # HTTP请求日志
            r'INFO:.*127\.0\.0\.1',                              # 所有来自127.0.0.1的INFO日志
        ]
        
        # 编译正则表达式
        self.keep_regex = [re.compile(pattern) for pattern in self.keep_patterns]
        self.filter_regex = [re.compile(pattern) for pattern in self.filter_patterns]
    
    def should_keep_line(self, line):
        """判断是否应该保留这一行日志"""
        # 首先检查是否应该被过滤掉
        for pattern in self.filter_regex:
            if pattern.search(line):
                return False
        
        # 然后检查是否匹配保留模式
        for pattern in self.keep_regex:
            if pattern.search(line):
                return True
        
        # 对于其他INFO级别的日志，如果不是HTTP相关的，也保留
        if 'INFO' in line and '127.0.0.1' not in line:
            return True
            
        return False
    
    def clean_log_file(self):
        """清理日志文件，保留最新的max_lines行"""
        if not self.output_log_path.exists():
            return
            
        try:
            with open(self.output_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) > self.max_lines:
                # 保留最新的max_lines行
                keep_lines = lines[-self.max_lines:]
                
                with open(self.output_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"# 日志已清理，保留最新 {self.max_lines} 行 - {datetime.now()}\n")
                    f.writelines(keep_lines)
                
                print(f"[{datetime.now()}] 日志已清理，从 {len(lines)} 行减少到 {len(keep_lines)} 行")
        
        except Exception as e:
            print(f"清理日志文件时出错: {e}")
    
    def process_new_lines(self):
        """处理新增的日志行"""
        if not self.input_log_path.exists():
            return
        
        try:
            with open(self.input_log_path, 'r', encoding='utf-8') as f:
                f.seek(self.last_position)
                new_lines = f.readlines()
                self.last_position = f.tell()
            
            if new_lines:
                filtered_lines = []
                for line in new_lines:
                    if self.should_keep_line(line.strip()):
                        filtered_lines.append(line)
                
                if filtered_lines:
                    # 追加到输出文件
                    with open(self.output_log_path, 'a', encoding='utf-8') as f:
                        f.writelines(filtered_lines)
        
        except Exception as e:
            print(f"处理日志时出错: {e}")
    
    def start(self):
        """启动日志过滤器"""
        self.running = True
        
        # 确保输出目录存在
        self.output_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果输出文件不存在，创建它
        if not self.output_log_path.exists():
            with open(self.output_log_path, 'w', encoding='utf-8') as f:
                f.write(f"# VLLM过滤日志开始 - {datetime.now()}\n")
        
        print(f"开始监控日志文件: {self.input_log_path}")
        print(f"过滤后的日志保存到: {self.output_log_path}")
        print(f"最大保留行数: {self.max_lines}")
        print(f"检查间隔: {self.check_interval}秒")
        
        # 获取当前文件大小作为起始位置
        if self.input_log_path.exists():
            self.last_position = self.input_log_path.stat().st_size
        
        while self.running:
            try:
                self.process_new_lines()
                self.clean_log_file()
                time.sleep(self.check_interval)
            except KeyboardInterrupt:
                print("\n收到停止信号，正在退出...")
                break
            except Exception as e:
                print(f"运行时错误: {e}")
                time.sleep(self.check_interval)
    
    def stop(self):
        """停止日志过滤器"""
        self.running = False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VLLM日志过滤器")
    parser.add_argument(
        '--input', 
        default='/home/pushihao/RAG/logs/vllm_rewriter.log',
        help='输入日志文件路径'
    )
    parser.add_argument(
        '--output', 
        default='/home/pushihao/RAG/logs/vllm_filtered.log',
        help='输出日志文件路径'
    )
    parser.add_argument(
        '--max-lines', 
        type=int, 
        default=1000,
        help='最大保留行数'
    )
    parser.add_argument(
        '--interval', 
        type=int, 
        default=10,
        help='检查间隔（秒）'
    )
    
    args = parser.parse_args()
    
    filter_tool = VLLMLogFilter(
        input_log_path=args.input,
        output_log_path=args.output,
        max_lines=args.max_lines,
        check_interval=args.interval
    )
    
    try:
        filter_tool.start()
    except KeyboardInterrupt:
        print("\n程序已停止")

if __name__ == '__main__':
    main()