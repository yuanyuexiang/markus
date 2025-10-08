# 🔧 CORS跨域问题 - 已解决

## 问题描述

```
请求网址: http://localhost:8000/api/verify
引荐来源网址政策: strict-origin-when-cross-origin
错误: 无法访问后端
```

## 问题原因

前端运行在 `http://localhost:3000`，后端运行在 `http://localhost:8000`，属于跨域请求。浏览器的同源策略会阻止这类请求。

## 解决方案

### ✅ 已配置 CORS

后端已经正确配置了CORS中间件：

```python
# backend/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 允许所有来源
    allow_credentials=True,       # 允许携带凭证
    allow_methods=["*"],          # 允许所有HTTP方法
    allow_headers=["*"],          # 允许所有请求头
)
```

### ✅ 验证CORS配置

运行测试脚本验证：

```bash
cd /Users/yuanyuexiang/Desktop/workspace/markus
./test_api.sh
```

预期输出：
```
✅ CORS配置正常
< access-control-allow-origin: http://localhost:3000
< access-control-allow-methods: DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT
< access-control-allow-credentials: true
```

### ✅ 前端已添加调试

前端JavaScript已添加详细日志：

```javascript
// 打开浏览器控制台 (F12) 查看
console.log('🚀 开始验证...');
console.log('📤 发送请求到后端...');
console.log('📥 收到响应，状态码:', response.status);
console.log('📊 验证结果:', result);
```

## 🚀 重启服务

### 方法1: 使用脚本

```bash
cd /Users/yuanyuexiang/Desktop/workspace/markus

# 停止所有服务
./stop.sh

# 启动所有服务
./start.sh
```

### 方法2: 手动启动

```bash
# 终端1 - 后端
cd /Users/yuanyuexiang/Desktop/workspace/markus/backend
venv/bin/python main.py

# 终端2 - 前端
cd /Users/yuanyuexiang/Desktop/workspace/markus/frontend
python3 -m http.server 3000
```

### 方法3: 后台运行

```bash
cd /Users/yuanyuexiang/Desktop/workspace/markus

# 后端
cd backend && venv/bin/python main.py > backend.log 2>&1 &

# 前端
cd frontend && python3 -m http.server 3000 > frontend.log 2>&1 &
```

## 🔍 排查步骤

### 1. 检查服务是否运行

```bash
# 检查后端 (应该显示进程)
lsof -i:8000

# 检查前端 (应该显示进程)
lsof -i:3000
```

### 2. 测试后端健康

```bash
curl http://localhost:8000
# 预期输出: {"message":"签名图章验证系统 API","status":"running"}
```

### 3. 测试CORS

```bash
curl -X OPTIONS http://localhost:8000/api/verify \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -v 2>&1 | grep -i "access-control"
```

### 4. 清除浏览器缓存

**Chrome/Edge**:
1. 打开开发者工具 (F12)
2. 右键点击刷新按钮
3. 选择"清空缓存并硬性重新加载"

**Safari**:
1. 开发 → 清空缓存
2. 或者按 Cmd+Option+E

**Firefox**:
1. Ctrl+Shift+Delete
2. 选择"缓存"
3. 点击"立即清除"

### 5. 查看浏览器控制台

1. 打开 http://localhost:3000
2. 按 F12 打开开发者工具
3. 切换到"控制台"标签
4. 上传图片并验证
5. 查看是否有错误信息

**正常情况应该看到**:
```
🚀 开始验证...
验证类型: signature
✅ 图片转换完成
📤 发送请求到后端...
📥 收到响应，状态码: 200
📊 验证结果: {success: true, ...}
```

**如果看到CORS错误**:
```
Access to fetch at 'http://localhost:8000/api/verify' from origin 
'http://localhost:3000' has been blocked by CORS policy
```

→ 说明后端没有启动或CORS配置有问题

## 💡 常见问题

### Q1: 提示"网络错误"
**原因**: 后端服务没有运行
**解决**: 
```bash
cd /Users/yuanyuexiang/Desktop/workspace/markus/backend
venv/bin/python main.py
```

### Q2: 提示"CORS policy blocked"
**原因**: 
- 后端CORS配置不正确
- 后端服务没有启动
- 端口被其他程序占用

**解决**:
1. 确认后端运行在8000端口
2. 确认前端运行在3000端口
3. 重启后端服务

### Q3: 验证按钮点击无反应
**原因**: 
- JavaScript缓存
- 没有裁剪图片

**解决**:
1. 清除浏览器缓存
2. 确保两张图片都已裁剪
3. 查看控制台是否有错误

### Q4: 提示"Failed to fetch"
**原因**: 后端地址错误或后端未启动

**解决**:
```bash
# 检查后端是否运行
curl http://localhost:8000

# 如果失败，启动后端
cd backend && venv/bin/python main.py
```

## 🎯 完整验证流程

```bash
# 1. 停止旧服务
lsof -ti:8000 | xargs kill -9
lsof -ti:3000 | xargs kill -9

# 2. 启动后端
cd /Users/yuanyuexiang/Desktop/workspace/markus/backend
venv/bin/python main.py &

# 3. 等待CLIP模型加载 (约5-10秒)
sleep 10

# 4. 启动前端
cd /Users/yuanyuexiang/Desktop/workspace/markus/frontend
python3 -m http.server 3000 &

# 5. 测试API
cd /Users/yuanyuexiang/Desktop/workspace/markus
./test_api.sh

# 6. 打开浏览器
# 访问 http://localhost:3000
# 按 Ctrl+Shift+R 强制刷新
```

## 📊 验证成功标志

### 后端日志
```
🔄 正在加载CLIP模型...
✅ CLIP模型加载完成
🚀 启动服务器: http://localhost:8000
INFO: Uvicorn running on http://0.0.0.0:8000
```

### 前端日志
```
Serving HTTP on :: port 3000
```

### 浏览器控制台
```
✅ 签名图章验证系统已加载 - 增强版
🚀 开始验证...
📊 验证结果: {success: true, similarity: 1.0, ...}
```

### API测试
```
✅ CORS配置正常
✅ 验证请求成功
相似度: 100%
```

## 🎉 问题已解决

当前状态:
- ✅ 后端已启动: http://localhost:8000
- ✅ 前端已启动: http://localhost:3000
- ✅ CORS已配置
- ✅ API测试通过
- ✅ 日志已添加

**现在可以正常使用系统了！**

访问: http://localhost:3000
按 Ctrl+Shift+R 强制刷新浏览器缓存

---

**最后更新**: 2024年10月7日
**版本**: v2.0
