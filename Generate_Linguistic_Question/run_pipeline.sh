#!/usr/bin/env bash
set -euo pipefail

# 路径基于当前目录 (Generate_Linguistic_Question)
RAW_TXT="$(pwd)/datasets/Questions.txt"
RAW_JSON="$(pwd)/datasets/chunked_question_raw.json"
OPT_JSON="$(pwd)/datasets/chunked_question.json"

echo "[1/3] 单文件切分 -> $RAW_JSON"
python3 ../preprocess/preprocess_documents.py \
  --source-file "$RAW_TXT" \
  --chunk-size 1000 \
  --overlap 0 \
  --min-chunk-length 10 \
  --output "$RAW_JSON"

echo "[2/3] 检查 vLLM 服务 /v1/models 可用性"
curl -sSf http://localhost:8001/v1/models | jq '.data[].id' || {
  echo "vLLM服务不可用或未返回模型列表，请检查 start_server.sh"; exit 1; }

echo "[3/3] 运行优化器(全量) -> $OPT_JSON"
python3 ../preprocess/llm_chunk_processor.py optimize "$RAW_JSON" "$OPT_JSON" --test-limit 0

echo "完成：输出文件 -> $OPT_JSON"

# 可选：运行清洗流程（若存在 evaluation.csv）
EVAL_CSV="$(pwd)/datasets/evaluation.csv"
if [ -f "$EVAL_CSV" ]; then
  echo "[后处理] 审核并生成报告"
  python3 ./scripts/clean_evaluation.py \
    --input ./datasets/evaluation.csv \
    --report ./datasets/clean_report.json \
    --allowed-types VALID_RESULT \
    --scope page \
    --mode audit
  echo "[后处理] 清洗并写出结果（使用 vLLM）"
  python3 ./scripts/clean_evaluation.py \
    --input ./datasets/evaluation.csv \
    --output ./datasets/evaluation_clean.csv \
    --report ./datasets/clean_report.json \
    --invalid ./datasets/invalid_rows.csv \
    --scope page \
    --max-answer-chars 500 \
    --min-enum-members 2 \
    --mode fix \
    --allowed-types VALID_RESULT \
    --use-llm true \
    --llm-url http://localhost:8001 \
    --llm-endpoint /v1/chat/completions \
    --llm-model auto
fi