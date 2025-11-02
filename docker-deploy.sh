#!/bin/bash

# 签名图章验证系统 - Docker 构建和运行脚本

set -e

echo "🐳 签名图章验证系统 - Docker 部署"
echo "=================================="
echo ""

# 配置
IMAGE_NAME="markus"
CONTAINER_NAME="markus"
PORT=8000

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker"
    echo "   安装指南: https://docs.docker.com/get-docker/"
    exit 1
fi

# 停止并删除旧容器
if docker ps -a | grep -q $CONTAINER_NAME; then
    echo "🛑 停止旧容器..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
fi

# 构建镜像
echo "🔨 构建 Docker 镜像..."
docker build -t $IMAGE_NAME:latest .

# 运行容器
echo "🚀 启动容器..."
docker run -d \
  --name $CONTAINER_NAME \
  --restart unless-stopped \
  -p $PORT:8000 \
  -v "$(pwd)/backend/uploads:/app/backend/uploads" \
  $IMAGE_NAME:latest

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 5

# 检查容器状态
if docker ps | grep -q $CONTAINER_NAME; then
    echo ""
    echo "✅ 部署成功！"
    echo ""
    echo "📊 服务信息:"
    echo "  🎨 前端界面: http://localhost:$PORT"
    echo "  📖 API 文档: http://localhost:$PORT/docs"
    echo "  🔌 API 接口: http://localhost:$PORT/api/verify"
    echo ""
    echo "📝 查看日志: docker logs -f $CONTAINER_NAME"
    echo "🛑 停止服务: docker stop $CONTAINER_NAME"
    echo "🔄 重启服务: docker restart $CONTAINER_NAME"
    echo ""
else
    echo "❌ 容器启动失败，请查看日志:"
    echo "   docker logs $CONTAINER_NAME"
    exit 1
fi
