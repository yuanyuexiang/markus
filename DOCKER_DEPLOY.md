# 签名图章验证系统 - Docker 部署指南

## 📦 单容器部署方案

本项目采用单容器架构,FastAPI 同时提供:
- 🎨 前端静态文件服务
- 🔌 后端 API 接口
- 🤖 三种 AI 算法 (SigNet, GNN, CLIP)

---

## 🚀 快速开始

### 本地测试

```bash
# 1. 构建镜像
docker build -t markus:latest .

# 2. 运行容器
docker run -d \
  --name markus \
  -p 8000:8000 \
  -v $(pwd)/backend/uploads:/app/backend/uploads \
  markus:latest

# 3. 访问服务
# 前端界面: http://localhost:8000
# API 文档: http://localhost:8000/docs
# API 接口: http://localhost:8000/api/verify

# 4. 查看日志
docker logs -f markus

# 5. 停止容器
docker stop markus
docker rm markus
```

---

## ☁️ 云端部署

### 方案 1: 阿里云 / 腾讯云服务器

```bash
# 1. 登录服务器
ssh user@your-server-ip

# 2. 安装 Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# 重新登录使 docker 权限生效

# 3. 克隆代码
git clone https://github.com/yuanyuexiang/markus.git
cd markus

# 4. 构建镜像
docker build -t markus:latest .

# 5. 运行容器
docker run -d \
  --name markus \
  --restart unless-stopped \
  -p 80:8000 \
  -v /data/markus/uploads:/app/backend/uploads \
  markus:latest

# 6. 配置防火墙
sudo ufw allow 80
sudo ufw allow 443

# 7. (可选) 配置 HTTPS
# 使用 Nginx 反向代理 + Let's Encrypt
```

### 方案 2: Docker Hub 部署

```bash
# 1. 登录 Docker Hub
docker login

# 2. 标记镜像
docker tag markus:latest yourusername/markus:latest

# 3. 推送镜像
docker push yourusername/markus:latest

# 4. 在服务器上拉取并运行
docker pull yourusername/markus:latest
docker run -d \
  --name markus \
  --restart unless-stopped \
  -p 80:8000 \
  -v /data/markus/uploads:/app/backend/uploads \
  yourusername/markus:latest
```

### 方案 3: Railway / Render 自动部署

1. 在项目根目录创建 `railway.toml`:
```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/"
healthcheckTimeout = 100
```

2. 推送到 GitHub,连接 Railway/Render 自动部署

---

## 🔧 配置说明

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| PORT | 8000 | 服务端口 |
| PYTHONUNBUFFERED | 1 | Python 输出不缓冲 |

### 数据持久化

推荐挂载以下目录:

```bash
-v /path/to/uploads:/app/backend/uploads  # 上传的图片
-v /path/to/models:/app/backend/models    # CLIP 模型缓存
```

### 资源要求

| 配置 | 最低要求 | 推荐配置 |
|------|----------|----------|
| CPU | 1核 | 2核+ |
| 内存 | 2GB | 4GB |
| 磁盘 | 5GB | 10GB |
| 网络 | 1Mbps | 10Mbps |

---

## 📊 性能优化

### 1. 预下载 CLIP 模型

在 Dockerfile 中已配置预下载,如果失败,首次启动会自动下载。

### 2. 使用 GPU (可选)

```bash
# 需要安装 NVIDIA Docker Runtime
docker run -d \
  --name markus \
  --gpus all \
  -p 8000:8000 \
  markus:latest
```

### 3. 调整 Worker 数量

```bash
# 修改启动命令 (在 Dockerfile 中)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

---

## 🔍 故障排查

### 问题 1: 容器启动失败

```bash
# 查看日志
docker logs markus

# 常见原因:
# - 端口被占用: 修改 -p 参数
# - 内存不足: 升级服务器配置
# - 模型下载失败: 检查网络连接
```

### 问题 2: 前端无法访问后端

```bash
# 检查 API 路径是否正确
# 前端使用相对路径 /api/verify
# 确保 FastAPI 路由正确挂载
```

### 问题 3: 内存占用过高

```bash
# 限制容器内存
docker run -d \
  --name markus \
  --memory="2g" \
  --memory-swap="2g" \
  -p 8000:8000 \
  markus:latest
```

---

## 🎯 更新部署

```bash
# 1. 拉取最新代码
git pull

# 2. 停止并删除旧容器
docker stop markus
docker rm markus

# 3. 重新构建镜像
docker build -t markus:latest .

# 4. 启动新容器
docker run -d \
  --name markus \
  --restart unless-stopped \
  -p 80:8000 \
  -v /data/markus/uploads:/app/backend/uploads \
  markus:latest
```

---

## 📝 监控和日志

### 查看实时日志
```bash
docker logs -f markus
```

### 查看资源使用
```bash
docker stats markus
```

### 进入容器调试
```bash
docker exec -it markus bash
```

---

## 🔒 安全建议

1. **生产环境**: 使用 Nginx 反向代理 + HTTPS
2. **API 限流**: 添加 rate limiting 中间件
3. **文件大小限制**: 已配置 10MB,可根据需要调整
4. **定期备份**: 备份 uploads 目录
5. **更新依赖**: 定期更新 Python 包

---

## 📞 技术支持

- GitHub: https://github.com/yuanyuexiang/markus
- Issues: https://github.com/yuanyuexiang/markus/issues
