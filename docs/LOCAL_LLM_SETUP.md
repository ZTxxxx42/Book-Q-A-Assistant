# 本地 LLM 部署：WSL2 + vLLM

本项目问答生成走本地 vLLM（Qwen2.5-7B-Instruct），vLLM 在 Windows 原生不支持，需在 WSL2 内运行。
bge-m3 / bge-reranker 在 Windows 原生跑，不依赖 WSL2。

## 端口约定

| 服务 | 端口 | 说明 |
|------|------|------|
| vLLM（WSL2 内） | 8001 | OpenAI 兼容 API |
| FastAPI | 8010 | 本项目接口（避开 8000 幽灵占用） |
| Neo4j | 7687 | 已有 |

## 1. 安装 WSL2 + Ubuntu

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
MILVUS_URI=rag_storage/milvus_lite.db

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
其余链路（Milvus + rerank + 图）照常验证，不阻塞开发。
