#!/bin/bash

echo "🧪 测试签名图章验证系统 API"
echo "================================"
echo ""

# 测试后端健康检查
echo "1️⃣ 测试后端健康状态..."
response=$(curl -s http://localhost:8000)
echo "响应: $response"
echo ""

# 测试CORS
echo "2️⃣ 测试CORS配置..."
cors_headers=$(curl -s -X OPTIONS http://localhost:8000/api/verify \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -v 2>&1 | grep -i "access-control-allow")

if [[ $cors_headers == *"access-control-allow-origin"* ]]; then
    echo "✅ CORS配置正常"
    echo "$cors_headers"
else
    echo "❌ CORS配置异常"
    echo "$cors_headers"
fi
echo ""

# 测试文件上传
echo "3️⃣ 测试文件上传和验证..."
if [ -f "test_images/signature_template.png" ] && [ -f "test_images/signature_real.png" ]; then
    echo "📤 上传测试图片..."
    
    response=$(curl -s -X POST http://localhost:8000/api/verify \
      -F "template_image=@test_images/signature_template.png" \
      -F "query_image=@test_images/signature_real.png" \
      -F "verification_type=signature")
    
    # 检查是否成功
    if [[ $response == *"success"* ]]; then
        echo "✅ 验证请求成功"
        echo ""
        echo "📊 验证结果:"
        echo "$response" | python3 -m json.tool
    else
        echo "❌ 验证请求失败"
        echo "$response"
    fi
else
    echo "⚠️  测试图片不存在，请先运行: python3 generate_test_images.py"
fi

echo ""
echo "================================"
echo "🎉 测试完成！"
