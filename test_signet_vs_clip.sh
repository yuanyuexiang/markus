#!/bin/bash

echo "======================================"
echo "SigNet vs CLIP 对比测试"
echo "======================================"

# 测试1: 不同签名(之前CLIP误判93.9%)
echo -e "\n📊 测试1: 不同签名 (CLIP误判案例)"
echo "Template: signature_template_20251007_033711.png"
echo "Query: signature_query_20251007_033711.png"

curl -s -X POST "http://localhost:8000/api/verify" \
  -F "template_image=@backend/uploaded_samples/signature_template_20251007_033711.png" \
  -F "query_image=@backend/uploaded_samples/signature_query_20251007_033711.png" \
  -F "verification_type=signature" | jq '{algorithm, similarity, euclidean_distance, is_authentic}'

# 测试2: 另一组样本
echo -e "\n📊 测试2: 不同签名 (另一组)"
echo "Template: signature_template_20251007_175643.png"
echo "Query: signature_query_20251007_175643.png"

curl -s -X POST "http://localhost:8000/api/verify" \
  -F "template_image=@backend/uploaded_samples/signature_template_20251007_175643.png" \
  -F "query_image=@backend/uploaded_samples/signature_query_20251007_175643.png" \
  -F "verification_type=signature" | jq '{algorithm, similarity, euclidean_distance, is_authentic}'

# 测试3: 完全相同
echo -e "\n📊 测试3: 相同签名 (完全一致)"
echo "Template & Query: signature_template_20251007_033711.png"

curl -s -X POST "http://localhost:8000/api/verify" \
  -F "template_image=@backend/uploaded_samples/signature_template_20251007_033711.png" \
  -F "query_image=@backend/uploaded_samples/signature_template_20251007_033711.png" \
  -F "verification_type=signature" | jq '{algorithm, similarity, euclidean_distance, is_authentic}'

echo -e "\n======================================"
