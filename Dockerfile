# 签名图章验证系统 - 单容器 Docker 镜像
# 包含前端(静态文件) + 后端(FastAPI + AI模型)

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖 (OpenCV 和 Git 需要)
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements.txt 并安装 Python 依赖
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# 预下载 CLIP 模型 (避免首次启动时下载)
RUN python -c "import clip; clip.load('ViT-B/32', device='cpu')" || echo "CLIP模型将在首次运行时下载"

# 复制后端代码和模型
COPY backend/ ./backend/

# 复制前端静态文件
COPY frontend/ ./frontend/

# 创建上传目录
RUN mkdir -p backend/uploads backend/models

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 启动命令
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
