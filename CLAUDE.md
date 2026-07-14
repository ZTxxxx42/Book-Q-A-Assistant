# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Book → Knowledge Graph: ingest a whole book (PDF/TXT/EPUB/MD), use **LightRAG** to extract entities/relations via an LLM, store the graph in **Neo4j**, vectors in **Milvus** (milvus-lite), and serve a hybrid-retrieval QA API (FastAPI + SSE streaming). Local **bge-m3** embeds and **bge-reranker-v2-m3** reranks; LLM generation goes to a local vLLM (Qwen2.5-7B) or any OpenAI-compatible endpoint (current `.env` uses 智谱 GLM-4-flash).

## Commands

Run everything from `book_knowledge_graph/`. **Use the miniforge `my_env` interpreter, not the root `.venv`** (root is Python 3.14 with no CUDA torch wheel — bge can only run CPU there):

```bash
# Start Neo4j (one-time, then leave running)
docker compose up -d

# FastAPI server (port 8010, not 8000 — 8000 is "ghost-occupied" on this machine)
D:/miniforge/envs/my_env/python.exe -m uvicorn src.api:app --port 8010 --host 127.0.0.1
# or: python serve.py   (runs on 8000 — prefer the explicit form above)

# CLI
python main.py ingest --file data/books/your_book.pdf [--max-chunks N]
python main.py query  --question "..." --mode hybrid
python main.py stats
python main.py cypher -c "MATCH (n) RETURN labels(n), count(*)"
python main.py documents list
python main.py entity edit <name> --description "..."
python main.py relation delete --source A --target B

# Scripts
python scripts/smoke_local_models.py   # component smoke test (bge / rerank / milvus / LightRAG build)
python scripts/check_model.py          # verify the .env LLM endpoint responds
python scripts/e2e_test.py             # ingest alice_en.txt (max 15 chunks) + stream a query
python scripts/ingest.py               # batch-ingest everything in data/books
python scripts/progress.py [log_path]  # tail an ingest log and render chunk-extraction progress
```

There is **no pytest suite** — verification is the smoke/e2e scripts above plus hitting the running API:
```bash
curl http://localhost:8010/health
curl -N -X POST http://localhost:8010/chat -H "Content-Type: application/json" \
  -d '{"question":"有哪些主要角色？","mode":"hybrid"}'
```

Query modes: `local` (specific entity/fact), `global` (cross-chapter), `hybrid` (default, recommended), `naive` (plain vector RAG, no graph).

## Architecture

The data flow is linear and lives in `src/`:

```
loader.py ──load_book/split_into_chunks──▶ graph_builder._build_rag() ──▶ LightRAG
                                              │        │           │
                                   Neo4JStorage  MilvusVector   KV(JSON) in working_dir
                                              │
                                              ▼
                                    query.ask / ask_stream / cypher_query
                                              │
                              api.py (FastAPI) ── /query /chat /ingest /stats /graph /documents /entities /relations /cypher
```

- `config.py` — single `Settings` dataclass; all values from env with defaults. `settings.ensure_dirs()` creates `working_dir` + `data/books`.
- `src/loader.py` — format-specific readers (`_read_pdf`/`_read_epub`/`_read_plain`) → `_normalize` → `split_into_chunks` (char-based, Chinese-friendly, no tokenizer dependency). `resolve_book_path` maps a bare filename to `data/books/`.
- `src/graph_builder.py` — **the integration core**. `_build_rag()` constructs the `LightRAG` instance with `Neo4JStorage` + `MilvusVectorDBStorage` + local bge embed/rerank funcs. `ingest_book()` loads, chunks, joins, and calls `rag.ainsert(full_text, file_paths=[fname])`. `_make_llm_func()` returns the LLM callable; it branches on `kwargs["stream"]` — streaming requests return an async generator (so LightRAG detects `is_streaming=True`), non-streaming (entity extraction) return a full string.
- `src/local_models.py` — module-level singletons (`_bge_m3`, `_reranker`) with async locks, lazy-loaded. `_resolve_model` downloads via ModelScope first (HF mirror lacks metadata headers), falls back to HF; passes `cache_dir=settings.model_cache_dir` so models land in-project. `_cuda_available()` honors `BGE_FORCE_CPU`. Rerank uses `sentence_transformers.CrossEncoder` (NOT `FlagEmbedding.FlagReranker` — see quirks), sigmoid-normalizes logits to (0,1).
- `src/query.py` — `ask` (non-stream), `ask_stream` (async generator, passes `conversation_history` for multi-turn), `cypher_query` (raw read-only Neo4j), `graph_stats`.
- `src/maintenance.py` — thin wrappers over LightRAG 1.5.x maintenance API (`aedit_entity`/`adelete_by_doc_id`/`aedit_relation`…); these sync vector store + Neo4j automatically. Each helper calls `_rag()` which builds + `initialize_storages()` a fresh LightRAG per call.
- `src/graph_view.py` — `get_subgraph`/`get_top_entities` for the frontend viz. Knows the Neo4j storage convention: node label = workspace name, entity name in `entity_id` property, edge type is `DIRECTED` with semantics in `description`/`weight`.
- `src/api.py` — FastAPI app. `/chat` is SSE streaming. `/ingest/async` + `/tasks/{id}` track long ingests in an in-process `_tasks` dict (no DB — lost on restart). `/cypher` allowlists only `MATCH/RETURN/WITH/CALL/UNWIND` prefixes.
- `main.py` — Click CLI mirroring the API surface.
- `static/index.html` — served at `/`, the graph viz UI.

## Critical environment quirks (these will bite you)

Documented in `docs/LOCAL_LLM_SETUP.md`; the non-obvious ones:

1. **`MILVUS_URI` must NOT be in `.env`.** pymilvus's legacy `connections` singleton parses `MILVUS_URI` at `import pymilvus` time; a file path (milvus-lite) triggers `ConnectionConfigException`. `_build_rag()` works around this by `import pymilvus` *before* `os.environ.setdefault("MILVUS_URI", …)`. Setting it in `.env` defeats this.
2. **`COSINE_THRESHOLD=1.0`** in `.env`. milvus-lite range-search keeps `distance <= radius` (L2 semantics) but COSINE similarity is "higher is better", so LightRAG's default 0.2 filter discards everything. `1.0` effectively disables the lower bound; `top_k` still limits count.
3. **`pymilvus==3.0.0`** is required (conda pulls 2.5.14, whose proto mismatches milvus-lite 3.0 → `AttributeError: function_score` → `aquery` returns None → `/chat` SSE crashes with `'async for' requires __aiter__`). Install with `--no-deps --ignore-installed`.
4. **numpy must be the pip wheel, not the conda build.** conda's numpy 2.5.1 puts BLAS DLLs in `my_env/Library/bin`, which isn't on PATH when calling `python.exe` directly (without `conda activate`) → large matmul crashes with `Windows fatal exception: code 0xc06d007f`. `pip install --no-deps --ignore-installed numpy==2.5.1` (PyPI wheel bundles scipy-openblas). Verify: `numpy.show_config()` `blas.name == scipy-openblas`.
5. **`FlagReranker` is broken on new transformers** (`tokenizer.prepare_for_model` removed). Use `sentence_transformers.CrossEncoder` — already done in `local_models.py`; don't "fix" it back to FlagReranker.
6. **`BGE_FORCE_CPU=1`** forces bge to CPU to avoid CUDA context conflicts with Ollama/vLLM sharing the GPU. Currently **commented out** in `.env` (GPU free, no Ollama) — bge-m3 runs fp16 on cuda:0, reranker on cuda:0 (~2.3GB VRAM). Re-enable only when sharing the GPU with a local vLLM/Ollama.
7. **conda packages lack dist-info** → `importlib.metadata.version()` returns None → transformers/sentence_transformers version checks throw `found=None`. Patched via `my_env/Lib/site-packages/sitecustomize.py` (environment-level, not in repo).
8. **Model cache lives in-project at `model_cache/`** (gitignored). `config.py` sets `MODELSCOPE_CACHE`/`HF_HOME`/`HUGGINGFACE_HUB_CACHE` to `settings.model_cache_dir` at import time (before any modelscope/HF import), and `_resolve_model` also passes `cache_dir=` explicitly. Layout: `model_cache/models/{owner}--{name}/snapshots/{rev}/` (modelscope), `model_cache/hf/` (HF fallback). The vLLM Docker container mounts `./model_cache` → `/root/.cache/modelscope` to reuse Qwen AWQ.
9. **Neo4j Community edition** can't create named databases, so LightRAG's `chunk-entity-relation` DB request logs "not found... Fallback to use the default database" — harmless, falls back to `neo4j` default DB.
10. **Protobuf gencode 5.27.2 vs runtime 6.31.1 warnings** come from pymilvus 3.0.0's bundled gencode lagging my_env's protobuf runtime — warnings only, function unaffected. Don't "fix" by downgrading protobuf (breaks other packages).

## Ports

| Service | Port |
|---------|------|
| vLLM (Docker container) | 8001 (container 8000 → host 8001) |
| FastAPI | 8010 |
| Neo4j Bolt / Browser | 7687 / 7474 |

## Re-ingesting

Before re-importing a book, clear `rag_storage/` (the working_dir) and the relevant Neo4j data to avoid duplicate entities. milvus-lite's `drop_collection` has a Windows rename race (`WinError 183`) — use `shutil.rmtree(db_dir)` instead of the collection drop API.
