#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/script/config_rag.py"
RUN_COMPARE="$SCRIPT_DIR/run_compare_experiments.sh"
DEFAULT_JSON="$SCRIPT_DIR/params_grid.json"
RUNS_LIST_FILE="$SCRIPT_DIR/.grid_runs_$$.list"

if [ $# -lt 1 ]; then
  echo "缺少必需参数: 问题数"
  echo "用法: $0 <问题数> [--params-json <文件路径>] [--max-parallel N] [--import-grace-seconds S]"
  exit 1
fi

MAX_QUESTIONS="$1"
shift

PARAMS_JSON="$DEFAULT_JSON"
MAX_PARALLEL=5
IMPORT_GRACE=30

while [ $# -gt 0 ]; do
  case "$1" in
    --params-json)
      if [ $# -lt 2 ]; then echo "缺少 --params-json 的文件路径"; exit 1; fi
      PARAMS_JSON="$2"; shift 2 ;;
    --max-parallel)
      if [ $# -lt 2 ]; then echo "缺少 --max-parallel 的值"; exit 1; fi
      MAX_PARALLEL="$2"; shift 2 ;;
    --import-grace-seconds)
      if [ $# -lt 2 ]; then echo "缺少 --import-grace-seconds 的值"; exit 1; fi
      IMPORT_GRACE="$2"; shift 2 ;;
    *)
      echo "未知参数: $1"; exit 1 ;;
  esac
done

if ! [[ "$MAX_QUESTIONS" =~ ^[0-9]+$ ]]; then
  echo "问题数必须是正整数: $MAX_QUESTIONS"; exit 1
fi

if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [ "$MAX_PARALLEL" -lt 1 ]; then
  echo "--max-parallel 必须为正整数"; exit 1
fi

if ! [[ "$IMPORT_GRACE" =~ ^[0-9]+$ ]] || [ "$IMPORT_GRACE" -lt 1 ]; then
  echo "--import-grace-seconds 必须为正整数"; exit 1
fi

if [ ! -f "$RUN_COMPARE" ]; then
  echo "脚本不存在: $RUN_COMPARE"; exit 1
fi

if [ ! -f "$PARAMS_JSON" ]; then
  echo "参数JSON文件不存在: $PARAMS_JSON"; exit 1
fi

LOCK_DIR="$SCRIPT_DIR/.cfg_lock"
on_interrupt() {
  if [ -f "$CONFIG_FILE.bak" ]; then mv -f "$CONFIG_FILE.bak" "$CONFIG_FILE"; fi
  rmdir "$LOCK_DIR" 2>/dev/null || true
  kill -TERM -- -$$ >/dev/null 2>&1 || true
  exit 130
}
trap 'on_interrupt' INT TERM

# 记录本批次运行的组合名称列表
rm -f "$RUNS_LIST_FILE"

# 读取配置常量（从 Python 模块）
read_config() {
  local var_name="$1"
  python3 -c "import sys; sys.path.insert(0, '$PROJECT_ROOT'); from script.config_rag import $var_name; print($var_name)" 2>/dev/null
}

# 等待缓冲 & Kill 生成器 vLLM（释放 GPU 显存）
kill_generator_vllm() {
  local gen_port; gen_port=$(read_config VLLM_GENERATOR_PORT)
  sleep 10
  if [ -f "$PROJECT_ROOT/pids/vllm_rewriter.pid" ]; then
    local pid; pid=$(cat "$PROJECT_ROOT/pids/vllm_rewriter.pid")
    kill "$pid" 2>/dev/null || true
    rm -f "$PROJECT_ROOT/pids/vllm_rewriter.pid"
  fi
  pgrep -f "vllm serve .*--port $gen_port" | xargs -r kill 2>/dev/null || true
}

kill_reranker_vllm() {
  local reranker_port; reranker_port=$(read_config VLLM_RERANKER_PORT)
  sleep 10
  if [ -f "$PROJECT_ROOT/pids/vllm_reranker.pid" ]; then
    local pid; pid=$(cat "$PROJECT_ROOT/pids/vllm_reranker.pid")
    kill "$pid" 2>/dev/null || true
    rm -f "$PROJECT_ROOT/pids/vllm_reranker.pid"
  fi
  pgrep -f "vllm serve .*--port $reranker_port" | xargs -r kill 2>/dev/null || true
}

# 启动评估 LLM（gpt-oss-120b，端口 8003），等待 120s 就绪
start_eval_llm() {
  local eval_model; eval_model=$(read_config EVALUATION_LLM_MODEL_LOCAL_PATH)
  local eval_port; eval_port=$(read_config EVALUATION_LLM_PORT)
  local gen_gpu_ids; gen_gpu_ids=$(read_config VLLM_GENERATOR_GPU_ID)
  local eval_mem_util="0.7"
  local parallel_workers; parallel_workers=$(read_config VLLM_REWRITER_TENSOR_PARALLEL_SIZE)
  mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/pids"
  (export CUDA_VISIBLE_DEVICES=${gen_gpu_ids}; nohup vllm serve "$eval_model" \
    --port "$eval_port" \
    --trust-remote-code \
    --disable-log-requests \
    --enforce-eager \
    --gpu-memory-utilization "${eval_mem_util}" \
    --tensor-parallel-size "${parallel_workers}" \
    --max-model-len 8192 \
    --max_num_seqs 1024 \
    > "$PROJECT_ROOT/logs/vllm_eval_llm.log" 2>&1 & echo $! > "$PROJECT_ROOT/pids/vllm_eval_llm.pid")
  sleep 120
}

stop_eval_llm() {
  local eval_port; eval_port=$(read_config EVALUATION_LLM_PORT)
  if [ -f "$PROJECT_ROOT/pids/vllm_eval_llm.pid" ]; then
    local pid; pid=$(cat "$PROJECT_ROOT/pids/vllm_eval_llm.pid")
    kill "$pid" 2>/dev/null || true
    rm -f "$PROJECT_ROOT/pids/vllm_eval_llm.pid"
  fi
  pgrep -f "vllm serve .*--port $eval_port" | xargs -r kill 2>/dev/null || true
  sleep 10
}

stop_reranker_vllm() {
  local reranker_port; reranker_port=$(read_config VLLM_RERANKER_PORT)
  if [ -f "$PROJECT_ROOT/pids/vllm_reranker.pid" ]; then
    local pid; pid=$(cat "$PROJECT_ROOT/pids/vllm_reranker.pid")
    kill "$pid" 2>/dev/null || true
    rm -f "$PROJECT_ROOT/pids/vllm_reranker.pid"
  fi
  pgrep -f "vllm serve .*--port $reranker_port" | xargs -r kill 2>/dev/null || true
  sleep 10
}

stop_embedding_vllm() {
  local embedding_port; embedding_port=$(read_config VLLM_EMBEDDING_PORT)
  if [ -f "$PROJECT_ROOT/pids/vllm_embedding.pid" ]; then
    local pid; pid=$(cat "$PROJECT_ROOT/pids/vllm_embedding.pid")
    kill "$pid" 2>/dev/null || true
    rm -f "$PROJECT_ROOT/pids/vllm_embedding.pid"
  fi
  pgrep -f "vllm serve .*--port $embedding_port" | xargs -r kill 2>/dev/null || true
  sleep 10
}

start_generator_vllm() {
  nohup bash "$PROJECT_ROOT/start_server.sh" > "$PROJECT_ROOT/logs/start_server_restore.log" 2>&1 &
  sleep 60
}

update_py_const() {
  local name="$1"; local val="$2"
  if [[ "$val" =~ ^[0-9]+$ ]]; then
    sed -E -i "s/^([[:space:]]*$name[[:space:]]*=[[:space:]]*)[0-9]+(.*)$/\1$val\2/" "$CONFIG_FILE"
  else
    sed -E -i "s/^([[:space:]]*$name[[:space:]]*=[[:space:]]*)[0-9]*\.?[0-9]+(.*)$/\1$val\2/" "$CONFIG_FILE"
  fi
}

wait_for_eval_start() {
  local run_name="$1"
  local grace="$2"
  local extra=$((grace*3))
  local log_dir="$SCRIPT_DIR/logs/$run_name"
  local r_dense="$log_dir/r_dense.log"
  local r_bm25="$log_dir/r_dense_bm25.log"
  local r_full="$log_dir/r_full.log"

  local end1=$(( $(date +%s) + grace ))
  while [ $(date +%s) -le $end1 ]; do
    if [ -f "$r_dense" ] && [ -f "$r_bm25" ] && [ -f "$r_full" ]; then
      if grep -q "阶段5: RAG评估" "$r_dense" || grep -q "evaluate_datasets.py" "$r_dense"; then
        if grep -q "阶段5: RAG评估" "$r_bm25" || grep -q "evaluate_datasets.py" "$r_bm25"; then
          if grep -q "阶段5: RAG评估" "$r_full" || grep -q "evaluate_datasets.py" "$r_full"; then
          return 0
        fi
      fi
    fi
    sleep 1
  done

  local end2=$(( $(date +%s) + extra ))
  while [ $(date +%s) -le $end2 ]; do
    if [ -f "$r1" ] && [ -f "$r4" ]; then
    if [ -f "$r_dense" ] && [ -f "$r_bm25" ] && [ -f "$r_full" ]; then
      if grep -q "阶段5: RAG评估" "$r_dense" || grep -q "evaluate_datasets.py" "$r_dense"; then
        if grep -q "阶段5: RAG评估" "$r_bm25" || grep -q "evaluate_datasets.py" "$r_bm25"; then
          if grep -q "阶段5: RAG评估" "$r_full" || grep -q "evaluate_datasets.py" "$r_full"; then
        fi
      fi
    fi
    sleep 1
  done
  return 1
}

run_one_combo() {

run_one_combo() {
  local TC="$1"; local TQ="$2"; local TK="$3"; local CTP="$4"; local QTP="$5"; local KTP="$6"; local FTP="$7"; local RR="$8"
  local RUN_NAME="thr_chunk_${TC}_thr_question_${TQ}_thr_keyword_${TK}_topk_chunk_${CTP}_topk_question_${QTP}_topk_keyword_${KTP}_topk_final_${FTP}_thr_reranker_${RR}"
  echo "$RUN_NAME" >> "$RUNS_LIST_FILE"

  while ! mkdir "$LOCK_DIR" 2>/dev/null; do sleep 1; done
  cp "$CONFIG_FILE" "$CONFIG_FILE.bak"
  update_py_const DENSE_CHUNK_THRESHOLD "$TC"
  update_py_const DENSE_QUESTION_THRESHOLD "$TQ"
  update_py_const SPARSE_KEYWORD_THRESHOLD "$TK"
  update_py_const DENSE_CHUNK_RETRIEVAL_TOP_K "$CTP"
  update_py_const DENSE_QUESTION_RETRIEVAL_TOP_K "$QTP"
  update_py_const SPARSE_KEYWORD_RETRIEVAL_TOP_K "$KTP"
  update_py_const FINAL_CONTEXT_TOP_K "$FTP"
  update_py_const RERANKER_SCORE_THRESHOLD "$RR"

  bash "$RUN_COMPARE" "$MAX_QUESTIONS" "$RUN_NAME" &
  local child_pid=$!

  if wait_for_eval_start "$RUN_NAME" "$IMPORT_GRACE"; then
    mv -f "$CONFIG_FILE.bak" "$CONFIG_FILE"
    rmdir "$LOCK_DIR" || true
  else
    mv -f "$CONFIG_FILE.bak" "$CONFIG_FILE"
    rmdir "$LOCK_DIR" || true
  fi

  wait "$child_pid"
}

mapfile -t COMBOS < <(python3 - "$PARAMS_JSON" << 'PY'
import sys, json, itertools
path = sys.argv[1]
data = json.load(open(path, 'r', encoding='utf-8'))
topk_keys = [
    "DENSE_CHUNK_RETRIEVAL_TOP_K",
    "DENSE_QUESTION_RETRIEVAL_TOP_K",
    "SPARSE_KEYWORD_RETRIEVAL_TOP_K",
    "FINAL_CONTEXT_TOP_K",
]
for k in topk_keys:
    v = data.get(k)
    if not isinstance(v, list) or len(v) == 0:
        print(f"参数缺失或为空: {k}", file=sys.stderr)
        sys.exit(2)

unified = data.get("THRESHOLDS_ALL")
if isinstance(unified, list) and len(unified) > 0:
    thr_tuples = [(t, t, t) for t in unified]
else:
    thr_keys = [
        "DENSE_CHUNK_THRESHOLD",
        "DENSE_QUESTION_THRESHOLD",
        "SPARSE_KEYWORD_THRESHOLD",
    ]
    for k in thr_keys:
        v = data.get(k)
        if not isinstance(v, list) or len(v) == 0:
            print(f"参数缺失或为空: {k}", file=sys.stderr)
            sys.exit(2)
    thr_tuples = list(itertools.product(
        data["DENSE_CHUNK_THRESHOLD"],
        data["DENSE_QUESTION_THRESHOLD"],
        data["SPARSE_KEYWORD_THRESHOLD"],
    ))

vals_topk = [data[k] for k in topk_keys]
rr_vals = data.get("RERANKER_SCORE_THRESHOLD")
if not isinstance(rr_vals, list) or len(rr_vals) == 0:
    print("参数缺失或为空: RERANKER_SCORE_THRESHOLD", file=sys.stderr)
    sys.exit(2)
for comb in itertools.product(thr_tuples, *vals_topk, rr_vals):
    (tc, tq, tk), ctp, qtp, ktp, ftp, rr = comb
    print(f"{tc} {tq} {tk} {int(ctp)} {int(qtp)} {int(ktp)} {int(ftp)} {rr}")
PY
)

declare -a PIDS=()

for line in "${COMBOS[@]}"; do
  read -r TC TQ TK CTP QTP KTP FTP RR <<< "$line"
  run_one_combo "$TC" "$TQ" "$TK" "$CTP" "$QTP" "$KTP" "$FTP" "$RR" &
  PIDS+=("$!")

  while [ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]; do
    if wait -n 2>/dev/null; then
      for i in "${!PIDS[@]}"; do
        if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then unset 'PIDS[i]'; fi
      done
      PIDS=("${PIDS[@]}")
    else
      sleep 1
    fi
  done
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

# 所有组合的生成阶段已完成，切换到评估模式
kill_generator_vllm
kill_reranker_vllm
sleep 10
start_eval_llm

# 并行运行高级评估（不设并发上限）
declare -a ADV_PIDS=()
while read -r RUN_NAME; do
  [ -z "$RUN_NAME" ] && continue
  for variant in r_dense r_dense_bm25 r_full; do
    results_dir="$SCRIPT_DIR/datasets/runs/$RUN_NAME/$variant/rag_evaluation_results"
    output_dir="$SCRIPT_DIR/datasets/runs/$RUN_NAME/$variant/advanced_evaluation_results"
    if [ -d "$results_dir" ]; then
      RUN_RESULTS_DIR="$results_dir" RUN_ADV_OUTPUT_DIR="$output_dir" \
        bash "$SCRIPT_DIR/run_advanced_evaluation.sh" -a > "$SCRIPT_DIR/logs/$RUN_NAME/${variant}_advanced.log" 2>&1 &
      ADV_PIDS+=("$!")
    fi
  done
done < "$RUNS_LIST_FILE"

for pid in "${ADV_PIDS[@]}"; do
  wait "$pid"
done

stop_eval_llm
stop_reranker_vllm
stop_embedding_vllm
rm -f "$RUNS_LIST_FILE"
rm -f "$SCRIPT_DIR"/.grid_runs_*.list
