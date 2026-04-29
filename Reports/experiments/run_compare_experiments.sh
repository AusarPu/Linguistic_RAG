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

CMD_R_DENSE="DEFER_ADVANCED_EVAL=true bash \"$EXPERIMENT_SCRIPT\" $BASE_OPTS --no-usefulness-judger --no-dense-keywords --no-dense-questions \"$MAX_QUESTIONS\""
CMD_R_DENSE_BM25="DEFER_ADVANCED_EVAL=true bash \"$EXPERIMENT_SCRIPT\" $BASE_OPTS --no-usefulness-judger --no-dense-keywords --no-dense-questions --use-bm25-chunks-only \"$MAX_QUESTIONS\""
CMD_R_FULL="DEFER_ADVANCED_EVAL=true bash \"$EXPERIMENT_SCRIPT\" $BASE_OPTS --no-usefulness-judger \"$MAX_QUESTIONS\""

print_info "启动 r_dense 实验: $CMD_R_DENSE"
eval "$CMD_R_DENSE" > "$LOG_DIR/r_dense.log" 2>&1 &
PID_DENSE=$!

print_info "启动 r_dense_bm25 实验: $CMD_R_DENSE_BM25"
eval "$CMD_R_DENSE_BM25" > "$LOG_DIR/r_dense_bm25.log" 2>&1 &
PID_BM25=$!

print_info "启动 r_full 实验: $CMD_R_FULL"
eval "$CMD_R_FULL" > "$LOG_DIR/r_full.log" 2>&1 &
PID_FULL=$!

wait $PID_DENSE
STATUS_DENSE=$?
wait $PID_BM25
STATUS_BM25=$?
wait $PID_FULL
STATUS_FULL=$?

if [ $STATUS_DENSE -ne 0 ] || [ $STATUS_BM25 -ne 0 ] || [ $STATUS_FULL -ne 0 ]; then
    print_error "实验执行失败 dense=$STATUS_DENSE bm25=$STATUS_BM25 full=$STATUS_FULL"
    print_info "r_dense 日志: $LOG_DIR/r_dense.log"
    print_info "r_dense_bm25 日志: $LOG_DIR/r_dense_bm25.log"
    print_info "r_full 日志: $LOG_DIR/r_full.log"
    exit 1
fi

RUN_DIR_DENSE=$(grep -E '运行目录:' "$LOG_DIR/r_dense.log" | tail -n 1 | sed -E 's/.*运行目录:\s*//' | sed -E 's/[[:space:]]+$//')
RUN_DIR_BM25=$(grep -E '运行目录:' "$LOG_DIR/r_dense_bm25.log" | tail -n 1 | sed -E 's/.*运行目录:\s*//' | sed -E 's/[[:space:]]+$//')
RUN_DIR_FULL=$(grep -E '运行目录:' "$LOG_DIR/r_full.log" | tail -n 1 | sed -E 's/.*运行目录:\s*//' | sed -E 's/[[:space:]]+$//')

if [ -z "$RUN_DIR_DENSE" ] || [ -z "$RUN_DIR_BM25" ] || [ -z "$RUN_DIR_FULL" ]; then
    print_error "无法解析运行目录"
    print_info "r_dense 日志: $LOG_DIR/r_dense.log"
    print_info "r_dense_bm25 日志: $LOG_DIR/r_dense_bm25.log"
    print_info "r_full 日志: $LOG_DIR/r_full.log"
    exit 1
fi

TARGET_BASE="$SCRIPT_DIR/datasets/runs/$RUN_NAME"
mkdir -p "$TARGET_BASE"

TARGET_DENSE="$TARGET_BASE/r_dense"
TARGET_BM25="$TARGET_BASE/r_dense_bm25"
TARGET_FULL="$TARGET_BASE/r_full"

print_info "整理目录: $RUN_DIR_DENSE -> $TARGET_DENSE"
rm -rf "$TARGET_DENSE"
mv "$RUN_DIR_DENSE" "$TARGET_DENSE"

print_info "整理目录: $RUN_DIR_BM25 -> $TARGET_BM25"
rm -rf "$TARGET_BM25"
mv "$RUN_DIR_BM25" "$TARGET_BM25"

print_info "整理目录: $RUN_DIR_FULL -> $TARGET_FULL"
rm -rf "$TARGET_FULL"
mv "$RUN_DIR_FULL" "$TARGET_FULL"

print_success "对比实验完成"
print_info "r_dense 结果目录: $TARGET_DENSE"
print_info "r_dense_bm25 结果目录: $TARGET_BM25"
print_info "r_full 结果目录: $TARGET_FULL"
print_info "日志目录: $LOG_DIR"

exit 0
