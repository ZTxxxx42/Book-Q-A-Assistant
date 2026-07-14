# 本地 LLM 部署：vLLM（Docker 推荐 / WSL2 备选）

本项目问答生成走本地 vLLM（Qwen2.5-7B-Instruct-AWQ 量化版，适配 RTX 4060 8GB）。
vLLM 在 Windows 原生不支持，两条路：**Docker Desktop（推荐，已有 WSL2 backend 即可）** 或 WSL2 内 pip 装。
bge-m3 / bge-reranker 在 Windows 原生 GPU 跑（miniforge `my_env`，Python 3.12 + CUDA torch），不依赖 WSL2。

## 端口约定

| 服务 | 端口 | 说明 |
|------|------|------|
| vLLM（Docker 容器） | 8001 | OpenAI 兼容 API（容器内 8000 → 宿主 8001） |
| FastAPI | 8010 | 本项目接口（避开 8000 幽灵占用） |
| Neo4j | 7687 | 已有 |

## 方案 A（推荐）：Docker 跑 vLLM

前提：Docker Desktop + WSL2 backend 已装（GPU 直通需 Windows 11 + NVIDIA 驱动，已满足）。

### A.1 拉镜像

```powershell
docker pull vllm/vllm-openai:latest
```

### A.2 启动 vLLM 容器（GPU + ModelScope 复用宿主缓存）

```powershell
docker run -d --gpus all --name book-vllm -p 8001:8000 `
  -v C:\Users\朱涛\.cache\modelscope:/root/.cache/modelscope `
  -e VLLM_USE_MODELSCOPE=True `
  vllm/vllm-openai:latest `
  --model Qwen/Qwen2.5-7B-Instruct-AWQ `
  --served-model-name Qwen2.5-7B-Instruct `
  --max-model-len 4096 `
  --gpu-memory-utilization 0.85 `
  --quantization awq
```

- 首次启动自动从 ModelScope 下载 AWQ 模型（~5GB），挂载宿主缓存后续复用。
- `--gpu-memory-utilization 0.85` 预留 ~1.2GB 给同卡运行的 bge-m3/reranker。
- 若 7B AWQ 仍 OOM：换 `Qwen/Qwen2.5-3B-Instruct`（去掉 `--quantization awq`，`--max-model-len 8192`）。
- 看日志：`docker logs -f book-vllm`，出现 `Uvicorn running on http://0.0.0.0:8000` 即就绪。

### A.3 验证

```bash
curl http://localhost:8001/v1/models   # 应返回 Qwen2.5-7B-Instruct
```

### A.4 常用运维

```powershell
docker stop book-vllm      # 停
docker start book-vllm     # 起（已创建后）
docker logs -f book-vllm   # 看日志
docker rm -f book-vllm     # 删除容器
```

---

## 方案 B（备选）：WSL2 内 pip 装 vLLM

Docker GPU 直通受阻时用此路。

### B.1 安装 WSL2 + Ubuntu

以**管理员**身份开 PowerShell：
```powershell
wsl --install -d Ubuntu
```
重启后设 Ubuntu 用户名密码。确认 WSL2（非 WSL1）：
```powershell
wsl -l -v   # VERSION 列应为 2
```

## 2. WSL 内装 CUDA toolkit

Windows 侧已装 NVIDIA 驱动即可，WSL 复用同一驱动，**不要**在 WSL 里装 Windows 驱动。
WSL 内装 CUDA 12.x toolkit（vLLM 需要）：
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvidia-smi   # 应显示 RTX 4060
```

## 3. WSL 内装 vLLM

```bash
sudo apt-get install -y python3.11 python3.11-venv
python3.11 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
pip install --upgrade pip
pip install vllm
```

## 4. 启动 vLLM 服务

国内用 hf-mirror 加速下载模型（Qwen2.5-7B-Instruct ~5GB）：
```bash
export HF_ENDPOINT=https://hf-mirror.com
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8001 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --served-model-name Qwen2.5-7B-Instruct
```
首次启动会自动下载模型。看到 `Uvicorn running on http://0.0.0.0:8001` 即就绪。

## 5. 验证（Windows 侧）

WSL2 默认把 localhost 转发到 Windows，直接在 Windows 测：
```bash
curl http://localhost:8001/v1/models
```
应返回含 `Qwen2.5-7B-Instruct` 的 JSON。

## 6. 配置 .env

项目根 `.env`（已 gitignore）：
```env
LLM_BINDING=openai
LLM_API_KEY=token-abc
LLM_BASE_URL=http://localhost:8001/v1
LLM_MODEL=Qwen2.5-7B-Instruct
LLM_STREAMING=true

EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
RERANK_MODEL=BAAI/bge-reranker-v2-m3
ENABLE_RERANK=true
COSINE_THRESHOLD=1.0
# 注意：MILVUS_URI 不要写进 .env（pymilvus Config 在 import 时解析会触发 ConnectionConfigException，
#   由 graph_builder._build_rag 在 import pymilvus 后用 os.environ.setdefault 设置）

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=bookgraph123
LANGUAGE=chinese
```

## 7. 本地 embedding/rerank 模型下载

bge-m3 / bge-reranker 在 Windows 原生跑，首次调用自动从 HuggingFace 下载。
国内加速，在启动 FastAPI 前设环境变量：
```bash
export HF_ENDPOINT=https://hf-mirror.com   # Windows PowerShell: $env:HF_ENDPOINT="https://hf-mirror.com"
```
模型约 2.6GB（bge-m3 ~2.2GB + bge-reranker-v2-m3 ~568MB）。

## 8. 启动验证

```bash
# Windows 侧
python -m uvicorn src.api:app --port 8010 --host 127.0.0.1
```
- `curl http://localhost:8010/health`
- `curl -N -X POST http://localhost:8010/chat -H "Content-Type: application/json" -d '{"question":"有哪些主要角色？","mode":"hybrid"}'`
  应逐 token 流出。

## 退路

若 WSL2+vLLM 部署受阻，临时把 `.env` 的 `LLM_BASE_URL` 改回智谱云端、`LLM_MODEL=glm-4-flash`，
其余链路（Milvus + rerank + 图）照常验证，不阻塞开发。当前 demo 即处于此状态：智谱 GLM 流式生成 + 本地 bge embed/rerank + Milvus + Neo4j，全链路验证通过。

## 实测坑（Windows + Python 3.14 + milvus-lite）

1. **CUDA torch 无 3.14 wheel**：`pip install torch --index-url .../cu121` 在 Python 3.14 下 no-op（无匹配 distribution）。bge-m3/reranker 暂跑 CPU（小数据可接受）。需 GPU 加速建议用 Python 3.11/3.12 venv。
2. **HF mirror 元数据头缺失**：`HF_ENDPOINT=https://hf-mirror.com` 下，新版 `huggingface_hub` 的 HEAD 响应缺 `X-Repo-Commit` 头，报 `FileMetadataError: Distant resource does not seem to be on huggingface.co`。改用 **ModelScope**（`src/local_models._resolve_model` 已自动走 modelscope）。
3. **pymilvus Config import 解析 MILVUS_URI**：pymilvus 的 legacy `connections` 单例在 `import pymilvus` 时读 `MILVUS_URI` env 并解析，文件路径（milvus-lite）触发 `ConnectionConfigException`。**解法**：先 `import pymilvus`（env 缺省），再设 `MILVUS_URI=文件路径`（milvus_impl 运行时用 `os.environ.get` 读，绕过 Config 单例）。`graph_builder._build_rag` 已如此处理。**不要**把 `MILVUS_URI` 写进 `.env`（会被 pymilvus Config 在 import 时读到）。
4. **milvus-lite range search radius 语义反转**：milvus-lite 的 range search 保留 `distance <= radius`（L2 语义），但 COSINE 相似度是越大越好。LightRAG 默认 `COSINE_THRESHOLD=0.2` 会过滤掉所有结果（相似度 0.4+ > 0.2 被丢弃）。**解法**：`.env` 设 `COSINE_THRESHOLD=1.0` 等效禁用下限过滤，`top_k` 仍限制数量。
5. **milvus-lite drop_collection Windows rename 竞态**：`drop_collection` 在 Windows 上 `manifest.json.tmp -> manifest.json` 报 WinError 183。清理时用 `shutil.rmtree(db_dir)` 代替。LightRAG 全新 ingest 不触发 drop，无影响；schema 迁移时可能遇到。
6. **FlagReranker 与新版 transformers 不兼容**：`FlagEmbedding.FlagReranker.compute_score` 调 `tokenizer.prepare_for_model`（新版 transformers 已移除）。改用 `sentence_transformers.CrossEncoder`（bge-reranker 兼容），sigmoid 归一化分数到 (0,1) 避免 `min_rerank_score=0` 过滤。

