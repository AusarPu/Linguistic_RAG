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