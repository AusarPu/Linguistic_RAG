#!/bin/bash

# 一键重算所有 runs 的 accuracy（acc）并重建汇总
# 位置：Reports/experiments/evaluation
# 说明：
# - 遍历 /home/pushihao/RAG/Reports/experiments/datasets/runs 下的所有一级子目录
# - 只要存在 advanced_evaluation_results/ragas_summary.csv 即纳入处理
# - 先备份该 run 的 ragas_summary.csv 与各数据集的 ragas_metrics.csv
# - 删除旧 ragas_summary.csv，随后调用 advanced_evaluation.py 重新计算 accuracy 并写回新汇总

set -euo pipefail

PROJECT_ROOT="/home/pushihao/RAG"
RUNS_ROOT="$PROJECT_ROOT/Reports/experiments/datasets/runs"
ADV_EVAL_SCRIPT="$PROJECT_ROOT/Reports/experiments/evaluation/advanced_evaluation.py"

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

timestamp() { date +"%Y%m%d-%H%M%S"; }

validate_env() {
  if [ ! -d "$RUNS_ROOT" ]; then
    print_error "runs 根目录不存在: $RUNS_ROOT"
    exit 1
  fi
  if [ ! -f "$ADV_EVAL_SCRIPT" ]; then
    print_error "advanced_evaluation.py 不存在: $ADV_EVAL_SCRIPT"
    exit 1
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    print_error "Python3 未安装或不在 PATH 中"
    exit 1
  fi
}

backup_file() {
  local file_path="$1"
  local ts="$2"
  local dir
  dir="$(dirname "$file_path")"
  local base
  base="$(basename "$file_path")"
  local name
  name="${base%.*}"
  local ext
  ext=".${base##*.}"
  local backup_path="$dir/${name}.backup-${ts}${ext}"
  cp "$file_path" "$backup_path"
  print_info "已备份: $file_path -> $backup_path"
}

process_run() {
  local run_dir="$1"
  local ts="$2"
  local adv_dir="$run_dir/advanced_evaluation_results"
  local summary_csv="$adv_dir/ragas_summary.csv"

  if [ ! -f "$summary_csv" ]; then
    print_warn "跳过（无 ragas_summary.csv）: $run_dir"
    return 0
  fi

  print_info "开始处理 run: $(basename "$run_dir")"

  backup_file "$summary_csv" "$ts"
  rm -f "$summary_csv"
  print_info "已删除旧汇总: $summary_csv"

  local dataset_count=0
  local metrics_backup_count=0
  for ds_dir in "$adv_dir"/*; do
    [ -d "$ds_dir" ] || continue
    local metrics_csv="$ds_dir/ragas_metrics.csv"
    if [ -f "$metrics_csv" ]; then
      backup_file "$metrics_csv" "$ts"
      metrics_backup_count=$((metrics_backup_count+1))
      dataset_count=$((dataset_count+1))
    fi
  done

  if [ $dataset_count -eq 0 ]; then
    print_warn "未发现任何数据集 ragas_metrics.csv，跳过重算: $run_dir"
    return 0
  fi

  local cmd
  cmd="python3 \"$ADV_EVAL_SCRIPT\" --attach-accuracy-dir \"$adv_dir\" --summary-csv \"$summary_csv\""
  print_info "执行命令: $cmd"
  if eval $cmd; then
    print_success "完成重算: $(basename "$run_dir") | 数据集数=$dataset_count, 备份数=$metrics_backup_count"
  else
    print_error "重算失败: $(basename "$run_dir")"
    return 1
  fi
}

main() {
  validate_env
  local ts
  ts="$(timestamp)"
  local total=0
  local processed=0
  local skipped=0
  local failed=0

  for run_dir in "$RUNS_ROOT"/*; do
    [ -d "$run_dir" ] || continue
    total=$((total+1))
    local adv_dir="$run_dir/advanced_evaluation_results"
    local summary_csv="$adv_dir/ragas_summary.csv"
    if [ -f "$summary_csv" ]; then
      if process_run "$run_dir" "$ts"; then
        processed=$((processed+1))
      else
        failed=$((failed+1))
      fi
    else
      skipped=$((skipped+1))
      print_warn "跳过（无汇总）: $(basename "$run_dir")"
    fi
  done

  print_info "统计：共发现 runs=$total, 已处理=$processed, 跳过=$skipped, 失败=$failed"
  if [ $failed -gt 0 ]; then
    exit 1
  fi
}

main "$@"