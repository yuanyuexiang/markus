#!/bin/bash
# Docker 构建助手脚本

echo "🐳 签名图章验证系统 - Docker 构建助手"
echo "========================================"
echo ""
echo "请选择构建方式:"
echo "  1) 标准构建 (推荐,包含 CLIP 模型预下载)"
echo "  2) 快速构建 (跳过模型预下载,首次运行时下载)"
echo "  3) 测试网络连接"
echo "  4) 配置镜像加速器"
echo ""
read -p "请输入选择 [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "🔨 开始标准构建..."
        docker build -t markus:latest -f Dockerfile .
        ;;
    2)
        echo ""
        echo "⚡ 开始快速构建..."
        docker build -t markus:latest -f Dockerfile.fast .
        ;;
    3)
        echo ""
        echo "🔍 测试 Docker Hub 连接..."
        echo "正在拉取测试镜像 alpine:latest..."
        docker pull alpine:latest
        if [ $? -eq 0 ]; then
            echo "✅ 网络连接正常！"
            echo "可以尝试标准构建"
        else
            echo "❌ 网络连接失败！"
            echo "建议配置镜像加速器或使用快速构建"
        fi
        ;;
    4)
        echo ""
        echo "📝 镜像加速器配置指南:"
        echo ""
        echo "1. 打开 Docker Desktop"
        echo "2. 点击右上角 ⚙️ (Settings)"
        echo "3. 选择 Docker Engine"
        echo "4. 添加以下配置:"
        echo ""
        cat << 'EOF'
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://dockerproxy.com",
    "https://docker.nju.edu.cn"
  ]
}
EOF
        echo ""
        echo "5. 点击 Apply & Restart"
        echo "6. 等待 Docker 重启完成后,重新运行此脚本"
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac
