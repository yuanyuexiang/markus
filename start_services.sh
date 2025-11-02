#!/bin/bash
# 服务启动脚本 - 单容器架构

echo "🚀 启动签名图章验证系统..."

# 停止旧进程
echo "📌 清理旧进程..."
lsof -ti :8000 | xargs kill -9 2>/dev/null || true
lsof -ti :3000 | xargs kill -9 2>/dev/null || true
sleep 1

# 启动服务
echo "🔧 启动服务 (端口8000)..."
cd backend
nohup bash -c "source venv/bin/activate && python main.py" > backend.log 2>&1 &
cd ..

# 等待服务启动
echo "⏳ 等待服务初始化..."
sleep 3

# 等待端口监听
for i in {1..20}; do
    if lsof -ti :8000 > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# 检查服务
if lsof -ti :8000 > /dev/null 2>&1; then
    echo "✅ 服务启动成功 (PID: $(lsof -ti :8000))！"
    echo ""
    echo "🎉 服务启动完成！"
    echo ""
    echo "📊 服务信息:"
    echo "  🎨 前端界面: http://localhost:8000"
    echo "  📖 API文档: http://localhost:8000/docs"
    echo "  🔌 API接口: http://localhost:8000/api/verify"
    echo ""
    echo "📝 查看日志: tail -f backend/backend.log"
    echo "🛑 停止服务: ./stop_services.sh"
else
    echo "❌ 服务启动失败，查看 backend/backend.log"
    exit 1
fi
