# AGENTS.md — MARS (Multi-Aspect Retrieval System) RAG Project

## Project Overview
A RAG system with multi-path recall, Cross-Encoder reranking, and query rewriting for linguistics QA.
The paper describing the system is in `Reports/docs/essay.md`. All experiments and their code live under `Reports/experiments/`.

## Must-Run Commands

### Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_model.py   # downloads models to ./models/
```

### Start Services (vLLM backends, OpenAI-compatible API)
```bash
bash start_server.sh      # Rewriter/Generator (8001), Embedding (8850), Reranker (8860)
bash start_frontend.sh    # Gradio web UI (8080)
```
Services use GPUs 4,5 by default (configured in `script/config_rag.py`). All ports run vLLM `serve` with `--enforce-eager`.

### Preprocess Knowledge Base (order matters)
```bash
python preprocess/preprocess_documents.py      # chunk raw docs
python preprocess/optimize_chunk_b_via_vllm.py # optional: improve chunk coherence
python preprocess/build_core_indexes.py         # build FAISS + BM25 indexes
```

### Experiments (one-shot)
```bash
bash Reports/experiments/run_full_experiment.sh <num_samples>
bash Reports/experiments/run_compare_experiments.sh <num_samples> <run_name>
bash Reports/experiments/grid_search_run_compare.sh <num_samples> [--max-parallel N]
```
All experiment scripts must be run from the project root. They set `PYTHONPATH` internally.

### Run a Single RAG Query (streaming test)
```bash
python test.py   # requires generator service on 8001
```

## Architecture: What Every Agent Must Know

### Service Topology
| Service      | Port  | GPU IDs | Model                            |
|-------------|-------|---------|----------------------------------|
| Generator   | 8001  | 4,5     | Qwen3-30B-A3B-FP8 (tensor parallel 2) |
| Rewriter    | 8001  | 4,5     | Same vLLM instance as Generator  |
| Embedding   | 8850  | 4,5     | Qwen3-Embedding-0.6B (tensor parallel 2) |
| Reranker    | 8860  | 4,5     | Qwen3-Reranker-0.6B              |
| Eval LLM    | 8003  | 4,7     | GPT-OSS-120B                     |
| Gradio UI   | 8080  | —       | N/A                              |

All services use OpenAI-compatible `/v1/chat/completions` and `/v1/embeddings` endpoints.

### Retrieval Paths (multi-recall)
1. **Dense chunk retrieval** — Bi-Encoder FAISS (cosine/IP with L2-normalized embeddings)
2. **Sparse keyword retrieval** — BM25 (`rank-bm25` library) with jieba (zh) or snowballstemmer+NLTK (en)
3. **Pre-generated question matching** — FAISS index of pre-generated questions → mapped to chunks

Paths are configured in `script/config_rag.py` (thresholds, Top-K) and executed in `script/rag_pipeline.py:execute_rag_flow()`.

### Pipeline Flow
`user_query → Query Rewriter → multi-path recall → Cross-Encoder Reranker → top-K context → Generator`

The usefulness judger module (`useful_judger.py`) exists but is **disabled in all experiments** (`use_usefulness_judger=False`). The query rewriter is **always enabled** in experiments.

## Critical Gotchas

### sys.path / PYTHONPATH
All Python entrypoints must add the project root to `sys.path`. Shell scripts do this via `PYTHONPATH`. When writing new scripts, always include:
```python
import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### Embedding API requires `mode` parameter
The embedding service (`/v1/embeddings`) requires a `mode` field in the request payload: `"chunk"` for indexing or `"query"` for online retrieval. Omitting it returns HTTP 400.

### Config patching in experiments
Evaluation and index-building scripts dynamically override `script.config_rag` constants to point to dataset-specific directories. They then `importlib.reload(module)` to pick up changes. This pattern is intentional — do not refactor it away.

### Generator prompt file
The **active** generator system prompt is `prompts/generator_system_prompt_eval.txt`, not `generator_system_prompt.txt`. The config points to the eval version.

### BM25 English tokenization needs NLTK data
English BM25 uses `nltk.corpus.stopwords` + `snowballstemmer`. The `nltk` stopwords data must be downloaded before first use:
```python
import nltk; nltk.download('stopwords')
```
`build_core_indexes.py` does this automatically in English mode. `BM25_TOKENIZER_SOURCE` defaults to `"stem"`.

### Reranker model name in config is a path
`RERANKER_MODEL_NAME_FOR_API` in `config_rag.py` is a local filesystem path (e.g., `./models/Qwen/Qwen3-Reranker-0.6B`), not a short model name. The vLLM serve command uses this path directly.

### GPU memory is tight
All services share GPUs 4,5 by default. Generator mem utilization is 0.4, embedding is 0.05, reranker is 0.3. Tensor parallel size is 2 for all models. Starting services in parallel on the same GPUs will OOM — `start_server.sh` sequences them.

### Reports/docs/essay.md is the paper
The paper uses `\( ... \)` (not `$...$`) for inline math. Chapter/section numbering uses Chinese format ("一、二、三、") for top-level chapters. Images go in `Reports/docs/pics/`. References are at the end of `essay.md`.

## Directory Map
| Directory | Purpose |
|-----------|---------|
| `script/` | Core RAG pipeline library (imported by everything) |
| `preprocess/` | Document chunking, index building, tokenizer |
| `Gradio_UI/` | Web UI using Gradio |
| `Reports/experiments/` | Evaluation scripts, grid search, tools, shell scripts |
| `Reports/docs/` | Paper (essay.md), guidelines, refs, visualization guide |
| `Reports/experiments/evaluation/` | `evaluate_datasets.py`, `advanced_evaluation.py`, Ragas metrics |
| `Reports/experiments/dataset_index/` | Per-dataset index building |
| `Reports/experiments/tools/` | Smoke tests (BM25), visualization helpers |
| `Evaluate_Linguistic/` | Separate pipeline for linguistic benchmark evaluation |
| `Generate_Linguistic_Question/` | Synthetic linguistic question generation |
| `prompts/` | Prompt templates (system instructions for each module) |

## No Automated Testing / Linting
There is no CI, no pre-commit, no test framework, no linter config, and no `pyproject.toml`. The root `test.py` is a manual smoke test for the generator streaming client. When making changes, run `python test.py` to verify the generator service still works.
