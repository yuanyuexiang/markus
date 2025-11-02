#!/bin/bash

echo "=========================================="
echo "🧪 测试GNN后端集成"
echo "=========================================="

# 测试GNN算法
echo ""
echo "测试1: 使用GNN算法验证签名"
echo "----------------------------------------"

# 使用已有的测试图像
TEMPLATE="backend/uploaded_samples/signature_template_20251029_124648.png"
QUERY="backend/uploaded_samples/signature_query_20251029_124648.png"

if [ ! -f "$TEMPLATE" ]; then
    echo "⚠️ 找不到模板图像,使用debug目录的图像"
    TEMPLATE="backend/uploaded_samples/debug/template_cleaned_20251029_124648.png"
    QUERY="backend/uploaded_samples/debug/query_cleaned_20251029_124648.png"
fi

if [ ! -f "$TEMPLATE" ]; then
    echo "❌ 找不到测试图像,请先上传签名"
    exit 1
fi

echo "模板图像: $TEMPLATE"
echo "查询图像: $QUERY"
echo ""

# 测试GNN
echo "🕸️ 测试GNN算法..."
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@$TEMPLATE" \
  -F "query_image=@$QUERY" \
  -F "verification_type=signature" \
  -F "algorithm=gnn" | python3 -m json.tool | head -30

echo ""
echo "=========================================="
echo "测试2: 对比三种算法性能"
echo "----------------------------------------"

echo ""
echo "📊 SigNet算法:"
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@$TEMPLATE" \
  -F "query_image=@$QUERY" \
  -F "verification_type=signature" \
  -F "algorithm=signet" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"  相似度: {data['similarity']:.4f}\")
print(f\"  算法: {data['algorithm']}\")
print(f\"  耗时: {data['processing_time_ms']}ms\")
print(f\"  判断: {'✅ 真实' if data['is_authentic'] else '❌ 伪造'}\")
"

echo ""
echo "🕸️ GNN算法:"
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@$TEMPLATE" \
  -F "query_image=@$QUERY" \
  -F "verification_type=signature" \
  -F "algorithm=gnn" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"  相似度: {data['similarity']:.4f}\")
print(f\"  算法: {data['algorithm']}\")
print(f\"  耗时: {data['processing_time_ms']}ms\")
print(f\"  判断: {'✅ 真实' if data['is_authentic'] else '❌ 伪造'}\")
if 'gnn_keypoints_template' in data and data['gnn_keypoints_template']:
    print(f\"  关键点: T={data['gnn_keypoints_template']}, Q={data['gnn_keypoints_query']}\")
"

echo ""
echo "🎨 CLIP算法:"
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@$TEMPLATE" \
  -F "query_image=@$QUERY" \
  -F "verification_type=signature" \
  -F "algorithm=clip" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"  相似度: {data['similarity']:.4f}\")
print(f\"  算法: {data['algorithm']}\")
print(f\"  耗时: {data['processing_time_ms']}ms\")
print(f\"  判断: {'✅ 真实' if data['is_authentic'] else '❌ 伪造'}\")
"

echo ""
echo "=========================================="
echo "✅ 测试完成!"
echo "=========================================="
echo ""
echo "💡 提示:"
echo "  - 打开浏览器访问 http://localhost:3000"
echo "  - 在'验证算法'下拉框中选择'GNN'"
echo "  - 上传签名图像进行测试"
echo ""
