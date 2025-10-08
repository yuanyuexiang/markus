#!/bin/bash
# 签名验证API测试脚本

echo "=== 测试同一签名不同裁剪 ==="
curl -X POST http://localhost:8000/api/verify \
  -F "template_image=@backend/uploaded_samples/signature_template_20251008_011305.png" \
  -F "query_image=@backend/uploaded_samples/signature_query_20251008_011305.png" \
  -F "verification_type=signature" \
  | jq '{similarity, euclidean_distance, ssim, confidence, recommendation}'

echo -e "\n=== 测试不同签名 ==="
curl -X POST http://localhost:8000/api/verify \
  -F "template_image=@backend/uploaded_samples/signature_template_20251008_011305.png" \
  -F "query_image=@backend/uploaded_samples/signature_query_20251008_013416.png" \
  -F "verification_type=signature" \
  | jq '{similarity, euclidean_distance, ssim, confidence, recommendation}'
