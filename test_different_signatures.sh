#!/bin/bash

echo "🧪 回归测试: 签名(SigNet) + 印章(CLIP)"
echo "========================================="
echo ""

fmt='{
  type: .type,
  algorithm: .algorithm,
  similarity: .similarity,
  euclidean_distance: .euclidean_distance,
  final_score: .final_score,
  threshold: .threshold,
  is_authentic: .is_authentic,
  confidence: .confidence,
  degraded_mode: (.degraded_mode // false),
  recommendation: .recommendation
}'

echo "📝 测试1: 签名(同一人样本) signature_template vs signature_real"
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@test_images/signature_template.png" \
  -F "query_image=@test_images/signature_real.png" \
  -F "verification_type=signature" | jq "$fmt"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📝 测试2: 签名(不同人/伪造) signature_template vs signature_fake"
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@test_images/signature_template.png" \
  -F "query_image=@test_images/signature_fake.png" \
  -F "verification_type=signature" | jq "$fmt"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "🔴 测试3: 印章(相同) seal_template vs seal_real"
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@test_images/seal_template.png" \
  -F "query_image=@test_images/seal_real.png" \
  -F "verification_type=seal" | jq "$fmt"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "🔴 测试4: 印章(不同/伪造) seal_template vs seal_fake"
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@test_images/seal_template.png" \
  -F "query_image=@test_images/seal_fake.png" \
  -F "verification_type=seal" | jq "$fmt"

echo ""
echo "✅ 测试完成"
