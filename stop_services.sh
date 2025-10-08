#!/bin/bash
# 服务停止脚本

echo "🛑 停止所有服务..."

# 停止后端
if lsof -ti :8000 > /dev/null 2>&1; then
    lsof -ti :8000 | xargs kill -9
    echo "✅ 后端服务已停止"
else
    echo "ℹ️  后端服务未运行"
fi

# 停止前端
if lsof -ti :3000 > /dev/null 2>&1; then
    lsof -ti :3000 | xargs kill -9
    echo "✅ 前端服务已停止"
else
    echo "ℹ️  前端服务未运行"
fi

echo "✨ 所有服务已停止"
