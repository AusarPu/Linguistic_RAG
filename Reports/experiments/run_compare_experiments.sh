#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_SCRIPT="$SCRIPT_DIR/run_full_experiment.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    echo "用法: $0 <问题数> <实验名称>"
    echo "示例: $0 200 myexp"
}

if [ $# -lt 2 ]; then
    print_error "缺少必需参数"
    usage
    exit 1
fi

MAX_QUESTIONS="$1"
RUN_NAME="$2"

if ! [[ "$MAX_QUESTIONS" =~ ^[0-9]+$ ]]; then
    print_error "问题数必须是正整数: $MAX_QUESTIONS"
    exit 1
fi

if [ -z "$RUN_NAME" ]; then
    print_error "实验名称不能为空"
    exit 1
fi

if [ ! -f "$EXPERIMENT_SCRIPT" ]; then
    print_error "脚本不存在: $EXPERIMENT_SCRIPT"
    exit 1
fi

LOG_DIR="$SCRIPT_DIR/logs/$RUN_NAME"
mkdir -p "$LOG_DIR"

BASE_OPTS="-s converter -s chunk -s enhance -s index"

CMD_R1="DEFER_ADVANCED_EVAL=true bash \"$EXPERIMENT_SCRIPT\" $BASE_OPTS --no-usefulness-judger --no-dense-keywords --no-dense-questions \"$MAX_QUESTIONS\""
CMD_R4="DEFER_ADVANCED_EVAL=true bash \"$EXPERIMENT_SCRIPT\" $BASE_OPTS --no-usefulness-judger \"$MAX_QUESTIONS\""

print_info "启动 r1 实验: $CMD_R1"
eval "$CMD_R1" > "$LOG_DIR/r1.log" 2>&1 &
PID_R1=$!  

print_info "启动 r4 实验: $CMD_R4"
eval "$CMD_R4" > "$LOG_DIR/r4.log" 2>&1 &
PID_R4=$!

wait $PID_R1
STATUS_R1=$?
wait $PID_R4
STATUS_R4=$?

if [ $STATUS_R1 -ne 0 ] || [ $STATUS_R4 -ne 0 ]; then
    print_error "实验执行失败 r1_status=$STATUS_R1 r4_status=$STATUS_R4"
    print_info "r1 日志: $LOG_DIR/r1.log"
    print_info "r4 日志: $LOG_DIR/r4.log"
    exit 1
fi

RUN_DIR_R1=$(grep -E '运行目录:' "$LOG_DIR/r1.log" | tail -n 1 | sed -E 's/.*运行目录:\s*//' | sed -E 's/[[:space:]]+$//')
RUN_DIR_R4=$(grep -E '运行目录:' "$LOG_DIR/r4.log" | tail -n 1 | sed -E 's/.*运行目录:\s*//' | sed -E 's/[[:space:]]+$//')

if [ -z "$RUN_DIR_R1" ] || [ -z "$RUN_DIR_R4" ]; then
    print_error "无法解析运行目录"
    print_info "r1 日志: $LOG_DIR/r1.log"
    print_info "r4 日志: $LOG_DIR/r4.log"
    exit 1
fi

TARGET_BASE="$SCRIPT_DIR/datasets/runs/$RUN_NAME"
mkdir -p "$TARGET_BASE"

TARGET_R1="$TARGET_BASE/r1"
TARGET_R4="$TARGET_BASE/r4"

print_info "整理目录: $RUN_DIR_R1 -> $TARGET_R1"
rm -rf "$TARGET_R1"
mv "$RUN_DIR_R1" "$TARGET_R1"

print_info "整理目录: $RUN_DIR_R4 -> $TARGET_R4"
rm -rf "$TARGET_R4"
mv "$RUN_DIR_R4" "$TARGET_R4"

print_success "对比实验完成"
print_info "r1 结果目录: $TARGET_R1"
print_info "r4 结果目录: $TARGET_R4"
print_info "日志目录: $LOG_DIR"

exit 0
