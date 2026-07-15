# 书籍知识图谱 app 镜像：FastAPI 服务。
# 嵌入/重排/抽取全走远程 API（SiliconFlow / GLM），答复走容器内 Ollama，
# 故无需 torch，镜像轻量。GPU 由 Ollama 容器独占，本镜像纯 CPU。
FROM python:3.11-slim

# curl 仅供容器内健康检查/调试；numpy/pandas/tiktoken 均有预编译 wheel，无需编译工具链。
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖，利用层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝应用代码（rag_storage / logs / data 由 compose 挂载卷，不入镜像）
COPY config.py serve.py main.py ./
COPY src ./src
COPY static ./static
COPY config.example.yaml ./

RUN mkdir -p data/books rag_storage logs

EXPOSE 8010

CMD ["python", "serve.py"]
