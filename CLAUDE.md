# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Book → Knowledge Graph: ingest a whole book (PDF/TXT/EPUB/MD), use **LightRAG** to extract entities/relations via an LLM, store the graph in **Neo4j**, vectors in **Qdrant** (Docker container), and serve a hybrid-retrieval QA API (FastAPI + SSE streaming). **SiliconFlow** API provides bge-m3 embedding + bge-reranker-v2-m3 rerank. LLMs are **split by role** (LightRAG `role_llm_configs`): entity/keyword **extraction** → remote **GLM-4.7** (heavy work, keeps the local GPU free); final **answer generation** → local **Ollama** Qwen2.5-7B-Instruct (OpenAI-compatible, short streamed responses only).

> Why the split: running 7B extraction locally on this 30W laptop GPU caused 8-minute runaway generation that crashed the GPU driver. Extraction offloaded to GLM; the local Qwen only generates short final answers (verified stable). See `docs/TROUBLESHOOTING.md`.

> Migration note: the stack previously used local bge-m3/reranker + milvus-lite + remote GLM, then briefly NanoVectorDB (local JSON). It was simplified to API embed/rerank + Qdrant (Docker) + GLM-extract / Ollama-answer. See `docs/TROUBLESHOOTING.md` for the iteration history.

## Commands

Run everything from `book_knowledge_graph/`. Use the miniforge `my_env` interpreter:

```bash
# Prerequisites: Ollama running + Qwen pulled, Neo4j + Qdrant up
ollama pull qwen2.5:7b-instruct
# Ollama default port 11434 is in a Windows reserved range → use 21434
$env:OLLAMA_HOST="127.0.0.1:21434"   # PowerShell
ollama serve                       # leave running (OpenAI API at http://localhost:21434/v1)
docker compose up -d               # Neo4j + Qdrant

# FastAPI server (port 8010, not 8000 — 8000 is "ghost-occupied" on this machine)
D:/miniforge/envs/my_env/python.exe -m uvicorn src.api:app --port 8010 --host 127.0.0.1

# CLI
D:/miniforge/envs/my_env/python.exe main.py ingest --file alice_en.txt [--max-chunks N]
D:/miniforge/envs/my_env/python.exe main.py query  --question "..." --mode hybrid
D:/miniforge/envs/my_env/python.exe main.py stats
D:/miniforge/envs/my_env/python.exe main.py cypher -c "MATCH (n) RETURN labels(n), count(*)"

# Scripts
D:/miniforge/envs/my_env/python.exe scripts/smoke_remote_models.py  # smoke: SiliconFlow embed/rerank + Qdrant + Ollama + LightRAG build
D:/miniforge/envs/my_env/python.exe scripts/check_model.py           # verify the .env LLM endpoint responds
D:/miniforge/envs/my_env/python.exe scripts/e2e_test.py              # ingest alice_en.txt (max 15 chunks) + stream a query
```

There is **no pytest suite** — verification is the smoke script plus hitting the running API:
```bash
curl http://localhost:8010/health
curl -N -X POST http://localhost:8010/chat -H "Content-Type: application/json" \
  -d '{"question":"Who is the main character?","mode":"hybrid"}'
```

Query modes: `local` (specific entity/fact), `global` (cross-chapter), `hybrid` (default, recommended), `naive` (plain vector RAG, no graph).

**Per-book filtering**: `/query` and `/chat` accept an optional `book` (filename) field; CLI `query --book <name>`. When set, retrieval bypasses LightRAG and queries the Qdrant chunks collection directly with a `file_path` payload filter + rerank, then generates via the local Qwen with a strict "answer only from this book's context" prompt. Without `book`, the normal LightRAG hybrid retrieval runs cross-DB. See `docs/reports/phase2-multi-book.md`.

**Document upsert**: `POST /documents/upsert` (and `maintenance.upsert_document`) handles same-basename re-ingest — LightRAG's filename dedup blocks plain re-ingest of an changed file, so upsert deletes the existing doc_id first then re-ingests.

## Architecture

The data flow is linear and lives in `src/`:

```
loader.py ──load_book/split_into_chunks──▶ graph_builder._build_rag() ──▶ LightRAG
                                              │        │           │
                                   Neo4JStorage  QdrantVectorDB  KV(JSON) in working_dir
                                              │
                                              ▼
                                    query.ask / ask_stream / cypher_query
                                              │
                              api.py (FastAPI) ── /query /chat /ingest /stats /graph /documents /entities /relations /cypher
```

- `config.py` — single `Settings` dataclass; all values from env with defaults. `settings.ensure_dirs()` creates `working_dir` + `data/books`.
- `src/loader.py` — format-specific readers (`_read_pdf`/`_read_epub`/`_read_plain`) → `_normalize` → `split_into_chunks` (char-based, Chinese-friendly, no tokenizer dependency). `resolve_book_path` maps a bare filename to `data/books/`.
- `src/graph_builder.py` — **the integration core**. `_build_rag()` constructs the `LightRAG` instance with `Neo4JStorage` + `QdrantVectorDBStorage` + SiliconFlow embed/rerank funcs. It injects `QDRANT_URL`/`NO_PROXY` into the process env (LightRAG's internal `QdrantClient` reads them). LLM is split via `role_llm_configs={"query": ...}`: base `llm_model_func` = `_make_glm_func()` (GLM-4.7, used for `extract`/`keyword` roles), `query` role overridden with `_make_qwen_func()` (local Ollama Qwen, final answer). `_make_llm_func(api_key, base_url, model, streaming_enabled)` is the shared factory; it branches on `kwargs["stream"]` — streaming requests return an async generator (so LightRAG detects `is_streaming=True`), non-streaming (entity extraction) return a full string. `ingest_book()` loads, chunks, joins, and calls `rag.ainsert(full_text, file_paths=[fname])`.
- `src/remote_models.py` — `make_embedding_func()` wraps SiliconFlow `/v1/embeddings` (OpenAI-compatible, AsyncOpenAI) into a LightRAG `EmbeddingFunc` (returns float32 ndarray). `make_reranker_func()` calls SiliconFlow `/v1/rerank` (httpx, non-OpenAI) and returns `[{index, relevance_score}]` with sigmoid fallback for out-of-range scores. Both have tenacity retry on 429/5xx.
- `src/query.py` — `ask` (non-stream), `ask_stream` (async generator, passes `conversation_history` for multi-turn), `cypher_query` (raw read-only Neo4j), `graph_stats`.
- `src/maintenance.py` — thin wrappers over LightRAG 1.5.x maintenance API (`aedit_entity`/`adelete_by_doc_id`/`aedit_relation`…); these sync vector store + Neo4j automatically. Each helper calls `_rag()` which builds + `initialize_storages()` a fresh LightRAG per call.
- `src/graph_view.py` — `get_subgraph`/`get_top_entities` for the frontend viz. Knows the Neo4j storage convention: node label = workspace name, entity name in `entity_id` property, edge type is `DIRECTED` with semantics in `description`/`weight`.
- `src/api.py` — FastAPI app. `/chat` is SSE streaming. `/ingest/async` + `/tasks/{id}` track long ingests in an in-process `_tasks` dict (no DB — lost on restart). `/cypher` allowlists only `MATCH/RETURN/WITH/CALL/UNWIND` prefixes.
- `main.py` — Click CLI mirroring the API surface.
- `static/index.html` — served at `/`, the graph viz UI. Single-file (ECharts + vanilla JS, no build). Three sidebar tabs: **概览** (stats + clickable type legend from `node_counts_by_label` + node detail with description), **文档管理** (`/documents` list + `/ingest` + `/documents/upsert` + refresh + delete), **Cypher** (`/cypher` console with result table). Chat area has a book-scope dropdown (`/chat` with `book`), references rendered under each answer, toast replaces alert.

## Critical environment notes

See `docs/TROUBLESHOOTING.md` for the full iteration log. The non-obvious ones:

1. **`COSINE_THRESHOLD=0.2`** — LightRAG reads it → `cosine_better_than_threshold` → Qdrant's `score_threshold`. Qdrant `Distance.COSINE` returns **true cosine similarity** (verified: identical=1.0, orthogonal=0.0, range [-1,1]) and keeps `score >= threshold`. So `0.2` is a sensible floor. (Unrelated to the old milvus-lite `1.0` hack — that's gone.)
2. **Qdrant port is `16333`, not the default `6333`** — `6333`/`6334` fall in a Windows Hyper-V reserved port range and Docker bind fails. The compose maps host `16333/16334` → container `6333/6334`. `QDRANT_URL=http://localhost:16333`.
3. **`NO_PROXY=localhost,127.0.0.1` is force-set in `config.py`** — a system proxy (FlClash/Clash) is active on this machine; without the bypass, `qdrant-client`'s `requests` routes localhost through the proxy and Qdrant returns `502 Bad Gateway`. `config.py` injects this at import time so every entry point is covered (LightRAG builds its `QdrantClient` internally without `trust_env=False`).
4. **Neo4j Community edition** can't create named databases, so LightRAG's `chunk-entity-relation` DB request logs "not found... Fallback to use the default database" — harmless, falls back to `neo4j` default DB.
5. **Ollama concurrency**: default `OLLAMA_NUM_PARALLEL=1`, so `QUERY_LLM_MODEL_MAX_ASYNC=1`. Raising it requires starting Ollama with `OLLAMA_NUM_PARALLEL=N` and enough VRAM for N concurrent Qwen contexts.
6. **Local Qwen is answer-only**: do NOT route `extract`/`keyword` roles to the local Qwen — long extraction generation crashes this 30W laptop GPU (see TROUBLESHOOTING). Extraction stays on remote GLM.
7. **SiliconFlow rerank score range**: code assumes `relevance_score ∈ [0,1]` but sigmoid-normalizes anything outside (raw logit fallback). Verified in [0,1] via smoke.
8. **GLM 429**: `LLM_MODEL_MAX_ASYNC=2` + 6-retry backoff. If 429 persists, drop to 1.

## Ports

| Service | Port |
|---------|------|
| Ollama (OpenAI API) | 21434 (default 11434 is in a Windows reserved range) |
| FastAPI | 8010 |
| Neo4j Bolt / Browser | 7687 / 7474 |
| Qdrant HTTP / gRPC | 16333 / 16334 |

## Re-ingesting

Before re-importing a book, clear Qdrant collections + Neo4j to avoid duplicate entities:
```bash
# Qdrant: drop collections (re-created on next ingest)
curl -X DELETE http://localhost:16333/collections/lightrag_vdb_chunks_baai_bge_m3_1024d
curl -X DELETE http://localhost:16333/collections/lightrag_vdb_relationships_baai_bge_m3_1024d
# Neo4j:
python main.py cypher -c "MATCH (n) DETACH DELETE n"
```
`rag_storage/` (KV cache + doc_status JSON) can also be wiped for a fully clean slate.
