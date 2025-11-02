#!/bin/bash
# Docker 分步构建脚本 - 可以在每一步暂停查看进度

set -e

echo "🐳 Docker 分步构建"
echo "=================="
echo ""

# 步骤 1: 拉取基础镜像
echo "📦 步骤 1/5: 拉取 Python 基础镜像..."
echo "这可能需要几分钟,请耐心等待..."
docker pull python:3.11-slim
echo "✅ 基础镜像拉取完成"
echo ""

# 步骤 2: 构建到系统依赖安装
echo "📦 步骤 2/5: 创建临时 Dockerfile (仅系统依赖)..."
cat > /tmp/Dockerfile.step1 << 'EOF'
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
EOF

docker build -t markus:step1 -f /tmp/Dockerfile.step1 .
echo "✅ 系统依赖安装完成"
echo ""

# 步骤 3: 配置 pip 并安装 Python 依赖
echo "📦 步骤 3/5: 安装 Python 依赖..."
echo "这是最耗时的步骤,需要安装 PyTorch, CLIP 等,可能需要 10-20 分钟..."
cat > /tmp/Dockerfile.step2 << 'EOF'
FROM markus:step1
WORKDIR /app
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt
EOF

docker build -t markus:step2 -f /tmp/Dockerfile.step2 .
echo "✅ Python 依赖安装完成"
echo ""

# 步骤 4: 复制代码
echo "📦 步骤 4/5: 复制项目代码..."
cat > /tmp/Dockerfile.step3 << 'EOF'
FROM markus:step2
WORKDIR /app
COPY backend/ ./backend/
COPY frontend/ ./frontend/
RUN mkdir -p backend/uploads backend/models
EOF

docker build -t markus:step3 -f /tmp/Dockerfile.step3 .
echo "✅ 代码复制完成"
echo ""

# 步骤 5: 最终镜像
echo "📦 步骤 5/5: 生成最终镜像..."
cat > /tmp/Dockerfile.final << 'EOF'
FROM markus:step3
WORKDIR /app
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

docker build -t markus:latest -f /tmp/Dockerfile.final .
echo "✅ 最终镜像构建完成"
echo ""

# 清理中间镜像
echo "🧹 清理中间镜像..."
docker rmi markus:step1 markus:step2 markus:step3 2>/dev/null || true

# 显示结果
echo ""
echo "🎉 构建完成！"
echo ""
echo "📊 镜像信息:"
docker images | grep markus
echo ""
echo "🚀 运行容器:"
echo "  docker run -d --name markus -p 8000:8000 markus:latest"
echo ""
echo "📝 查看日志:"
echo "  docker logs -f markus"
