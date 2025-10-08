#!/bin/bash

echo "🧪 测试不同签名的相似度检测"
echo "================================"
echo ""

# 测试1: 相同签名（应该高相似度）
echo "📝 测试1: 相同签名 (signature_template vs signature_real)"
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@test_images/signature_template.png" \
  -F "query_image=@test_images/signature_real.png" \
  -F "verification_type=signature"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 测试2: 不同签名（应该低相似度）
echo "📝 测试2: 完全不同的签名 (signature_template vs signature_fake)"
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@test_images/signature_template.png" \
  -F "query_image=@test_images/signature_fake.png" \
  -F "verification_type=signature"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 测试3: 相同图章
echo "🔴 测试3: 相同图章 (seal_template vs seal_real)"
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@test_images/seal_template.png" \
  -F "query_image=@test_images/seal_real.png" \
  -F "verification_type=seal"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 测试4: 不同图章
echo "🔴 测试4: 不同图章 (seal_template vs seal_fake)"
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@test_images/seal_template.png" \
  -F "query_image=@test_images/seal_fake.png" \
  -F "verification_type=seal"

echo ""
echo "✅ 测试完成"
