#!/bin/bash
# 服务停止脚本 - 单容器架构

echo "🛑 停止所有服务..."

# 停止服务 (端口 8000)
if lsof -ti :8000 > /dev/null 2>&1; then
    lsof -ti :8000 | xargs kill -9
    echo "✅ 服务已停止 (端口 8000)"
else
    echo "ℹ️  服务未运行 (端口 8000)"
fi

# 清理旧的前端进程 (兼容旧版本)
if lsof -ti :3000 > /dev/null 2>&1; then
    lsof -ti :3000 | xargs kill -9
    echo "✅ 清理旧前端服务 (端口 3000)"
fi

echo "✨ 所有服务已停止"
