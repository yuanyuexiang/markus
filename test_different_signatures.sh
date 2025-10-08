#!/bin/bash

echo "🧪 测试不同签名的相似度检测"
echo "================================"
echo ""

# 测试1: 相同签名（应该高相似度）
echo "📝 测试1: 相同签名 (signature_template vs signature_real)"
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@test_images/signature_template.png" \
  -F "query_image=@test_images/signature_real.png" \
  -F "verification_type=signature" | jq '{
    clip_sim: .clip_similarity,
    feature_sim: .feature_similarity,
    final_score: .final_score,
    is_authentic: .is_authentic,
    confidence: .confidence
  }'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 测试2: 不同签名（应该低相似度）
echo "📝 测试2: 完全不同的签名 (signature_template vs signature_fake)"
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@test_images/signature_template.png" \
  -F "query_image=@test_images/signature_fake.png" \
  -F "verification_type=signature" | jq '{
    CLIP相似度: .clip_similarity,
    特征点相似度: .feature_similarity,
    最终得分: .final_score,
    是否通过: .is_authentic,
    置信度: .confidence,
    建议: .recommendation
  }'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 测试3: 相同图章
echo "🔴 测试3: 相同图章 (seal_template vs seal_real)"
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@test_images/seal_template.png" \
  -F "query_image=@test_images/seal_real.png" \
  -F "verification_type=seal" | jq '{
    CLIP相似度: .clip_similarity,
    特征点相似度: .feature_similarity,
    最终得分: .final_score,
    是否通过: .is_authentic,
    置信度: .confidence,
    建议: .recommendation
  }'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 测试4: 不同图章
echo "🔴 测试4: 不同图章 (seal_template vs seal_fake)"
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@test_images/seal_template.png" \
  -F "query_image=@test_images/seal_fake.png" \
  -F "verification_type=seal" | jq '{
    CLIP相似度: .clip_similarity,
    特征点相似度: .feature_similarity,
    最终得分: .final_score,
    是否通过: .is_authentic,
    置信度: .confidence,
    建议: .recommendation
  }'

echo ""
echo "✅ 测试完成"
