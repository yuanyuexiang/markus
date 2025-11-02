#!/bin/bash

echo "🔬 三种算法对比测试"
echo "======================="
echo ""

# 使用同一对清洁后的签名
TEMPLATE="backend/uploaded_samples/debug/template_cleaned_20251029_124648.png"
QUERY="backend/uploaded_samples/debug/query_cleaned_20251029_124648.png"

echo "📁 测试图像:"
echo "  模板: template_cleaned_20251029_124648.png"
echo "  查询: query_cleaned_20251029_124648.png"
echo ""

# 测试SigNet
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  SigNet (CNN深度学习方法)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -X POST http://localhost:8000/api/verify \
  -F "template_image=@$TEMPLATE" \
  -F "query_image=@$QUERY" \
  -F "verification_type=signature" \
  -F "algorithm=signet" \
  2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'  算法: {data.get(\"algorithm\")}')
print(f'  相似度: {data.get(\"similarity\"):.4f}')
print(f'  欧氏距离: {data.get(\"euclidean_distance\"):.4f}')
print(f'  SSIM: {data.get(\"ssim\"):.4f}')
print(f'  阈值: {data.get(\"threshold\"):.4f}')
print(f'  匹配结果: {\"✅ 相同\" if data.get(\"is_authentic\") else \"❌ 不同\"}')
"
echo ""

# 测试GNN
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  GNN (图神经网络方法)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -X POST http://localhost:8000/api/verify \
  -F "template_image=@$TEMPLATE" \
  -F "query_image=@$QUERY" \
  -F "verification_type=signature" \
  -F "algorithm=gnn" \
  2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'  算法: {data.get(\"algorithm\")}')
print(f'  相似度: {data.get(\"similarity\"):.4f}')
print(f'  GNN距离: {data.get(\"gnn_distance\"):.4f}')
print(f'  关键点-模板: {data.get(\"gnn_keypoints_template\")}')
print(f'  关键点-查询: {data.get(\"gnn_keypoints_query\")}')
print(f'  阈值: {data.get(\"threshold\"):.4f}')
print(f'  匹配结果: {\"✅ 相同\" if data.get(\"is_authentic\") else \"❌ 不同\"}')
"
echo ""

# 测试CLIP
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  CLIP (视觉Transformer方法)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -X POST http://localhost:8000/api/verify \
  -F "template_image=@$TEMPLATE" \
  -F "query_image=@$QUERY" \
  -F "verification_type=signature" \
  -F "algorithm=clip" \
  2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'  算法: {data.get(\"algorithm\")}')
print(f'  相似度: {data.get(\"similarity\"):.4f}')
print(f'  余弦相似度: {data.get(\"cosine_similarity\"):.4f}')
print(f'  欧氏距离: {data.get(\"euclidean_distance\"):.4f}')
print(f'  阈值: {data.get(\"threshold\"):.4f}')
print(f'  匹配结果: {\"✅ 相同\" if data.get(\"is_authentic\") else \"❌ 不同\"}')
"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 对比总结"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "测试的是同一对清洁后的签名图像"
echo "理想情况下,所有算法都应该判断为'相同'"
echo ""
echo "算法特点:"
echo "  • SigNet: 基于CNN的特征提取,适合一般签名验证"
echo "  • GNN: 基于图结构的关键点匹配,适合笔画复杂的签名"
echo "  • CLIP: 基于视觉语言模型,泛化能力强但可能精度略低"
echo ""
echo "✅ 测试完成!"
