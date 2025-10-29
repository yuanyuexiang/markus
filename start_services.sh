#!/bin/bash
# 服务启动脚本 - 前后端一键启动

echo "🚀 启动签名图章验证系统..."

# 停止旧进程
echo "📌 清理旧进程..."
lsof -ti :8000 | xargs kill -9 2>/dev/null || true
lsof -ti :3000 | xargs kill -9 2>/dev/null || true
sleep 1

# 启动后端
echo "🔧 启动后端服务 (端口8000)..."
cd backend
nohup bash -c "source venv/bin/activate && python main.py" > backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# 等待后端启动
echo "⏳ 等待后端初始化..."
sleep 3

# 等待端口监听（最多等待 20 秒）
for i in {1..20}; do
    if lsof -ti :8000 > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# 检查后端
if lsof -ti :8000 > /dev/null 2>&1; then
    echo "✅ 后端启动成功 (PID: $(lsof -ti :8000))！"
else
    echo "❌ 后端启动失败，查看 backend/backend.log"
    exit 1
fi

# 启动前端
echo "🎨 启动前端服务 (端口3000)..."
cd frontend
nohup python3 -m http.server 3000 > frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

sleep 2

# 检查前端
if lsof -ti :3000 > /dev/null 2>&1; then
    echo "✅ 前端启动成功 (PID: $(lsof -ti :3000))！"
else
    echo "❌ 前端启动失败，查看 frontend/frontend.log"
    exit 1
fi

echo ""
echo "🎉 所有服务启动完成！"
echo ""
echo "📊 服务信息:"
echo "  后端API: http://localhost:8000"
echo "  API文档: http://localhost:8000/docs"
echo "  前端界面: http://localhost:3000"
echo ""
echo "📝 查看日志:"
echo "  后端: tail -f backend/backend.log"
echo "  前端: tail -f frontend/frontend.log"
echo ""
echo "🛑 停止服务: ./stop_services.sh"
