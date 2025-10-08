#!/bin/bash
# 对比签名与印章的相似度度量方式

echo "======================================================================"
echo "📊 相似度度量方式对比测试"
echo "======================================================================"
echo ""

echo "🖊️  测试1: 签名验证 (SigNet - 欧氏距离)"
echo "----------------------------------------------------------------------"
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@backend/uploaded_samples/signature_template_20251008_011305.png" \
  -F "query_image=@backend/uploaded_samples/signature_query_20251008_011305.png" \
  -F "verification_type=signature" | jq '{
    algorithm: .algorithm,
    similarity: .similarity,
    euclidean_distance: .euclidean_distance,
    ssim: .ssim,
    signet_pipeline: .signet_pipeline,
    confidence: .confidence
}'

echo ""
echo "🎨 测试2: 印章验证 (CLIP - 余弦相似度)"
echo "----------------------------------------------------------------------"
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@backend/uploaded_samples/seal_template_20251008_214858.png" \
  -F "query_image=@backend/uploaded_samples/seal_query_20251008_214858.png" \
  -F "verification_type=seal" | jq '{
    algorithm: .algorithm,
    similarity: .similarity,
    euclidean_distance: .euclidean_distance,
    ssim: .ssim,
    signet_pipeline: .signet_pipeline,
    confidence: .confidence
}'

echo ""
echo "======================================================================"
echo "📝 说明:"
echo "  - SigNet: 使用欧氏距离衡量签名笔迹差异"
echo "  - CLIP:   使用余弦相似度衡量图像语义相似性"
echo "  - N/A:    表示该度量方式不适用于当前算法"
echo "======================================================================"
