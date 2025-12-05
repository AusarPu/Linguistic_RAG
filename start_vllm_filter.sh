#!/bin/bash

# VLLM日志过滤器启动脚本
# 用于在后台运行日志过滤器

cd /home/pushihao/RAG

# 检查是否已经有过滤器在运行
if pgrep -f "filter_vllm_logs.py" > /dev/null; then
    echo "日志过滤器已经在运行中"
    echo "当前运行的进程:"
    pgrep -f "filter_vllm_logs.py" -l
    exit 1
fi

echo "启动VLLM日志过滤器..."

# 后台运行日志过滤器
nohup python filter_vllm_logs.py \
    --input /home/pushihao/RAG/logs/vllm_rewriter.log \
    --output /home/pushihao/RAG/logs/vllm_filtered.log \
    --max-lines 500 \
    --interval 30 \
    > /home/pushihao/RAG/logs/filter_process.log 2>&1 &

# 获取进程ID
PID=$!
echo "日志过滤器已启动，进程ID: $PID"
echo "过滤后的日志文件: /home/pushihao/RAG/logs/vllm_filtered.log"
echo "过滤器进程日志: /home/pushihao/RAG/logs/filter_process.log"

# 保存PID到文件
echo $PID > /home/pushihao/RAG/logs/filter.pid

echo ""
echo "使用以下命令查看过滤后的日志:"
echo "  tail -f /home/pushihao/RAG/logs/vllm_filtered.log"
echo ""
echo "使用以下命令停止过滤器:"
echo "  bash /home/pushihao/RAG/stop_vllm_filter.sh"