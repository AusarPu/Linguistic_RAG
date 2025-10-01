#!/bin/bash

# VLLM日志过滤器停止脚本

echo "正在停止VLLM日志过滤器..."

# 从PID文件读取进程ID
PID_FILE="/home/pushihao/RAG/logs/filter.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "停止进程 $PID"
        kill "$PID"
        sleep 2
        
        # 如果进程还在运行，强制杀死
        if kill -0 "$PID" 2>/dev/null; then
            echo "强制停止进程 $PID"
            kill -9 "$PID"
        fi
        
        rm -f "$PID_FILE"
        echo "日志过滤器已停止"
    else
        echo "进程 $PID 不存在，清理PID文件"
        rm -f "$PID_FILE"
    fi
else
    echo "未找到PID文件，尝试通过进程名停止..."
    
    # 通过进程名查找并停止
    PIDS=$(pgrep -f "filter_vllm_logs.py")
    if [ -n "$PIDS" ]; then
        echo "找到进程: $PIDS"
        kill $PIDS
        sleep 2
        
        # 检查是否还有进程在运行
        REMAINING=$(pgrep -f "filter_vllm_logs.py")
        if [ -n "$REMAINING" ]; then
            echo "强制停止剩余进程: $REMAINING"
            kill -9 $REMAINING
        fi
        echo "日志过滤器已停止"
    else
        echo "未找到运行中的日志过滤器进程"
    fi
fi