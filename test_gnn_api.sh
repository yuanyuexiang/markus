#!/bin/bash
echo "🧠 测试GNN API集成"
echo "="

# 使用现有的清洁后签名
TEMPLATE="backend/uploaded_samples/debug/template_cleaned_20251029_124648.png"
QUERY="backend/uploaded_samples/debug/query_cleaned_20251029_124648.png"

echo "📁 测试图像: template_cleaned & query_cleaned"
echo ""

echo "🧠 发送GNN验证请求..."
curl -X POST http://localhost:8000/api/verify \
  -F "template_image=@$TEMPLATE" \
  -F "query_image=@$QUERY" \
  -F "verification_type=signature" \
  -F "algorithm=gnn" \
  2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('\n结果:')
print(f'  算法: {data.get(\"algorithm\")}')
print(f'  相似度: {data.get(\"similarity\")}')
print(f'  距离: {data.get(\"euclidean_distance\")}')
print(f'  匹配: {data.get(\"is_authentic\")}')
if data.get('gnn_keypoints_template'):
    print(f'  GNN关键点-模板: {data.get(\"gnn_keypoints_template\")}')
    print(f'  GNN关键点-查询: {data.get(\"gnn_keypoints_query\")}')
    print(f'  GNN距离: {data.get(\"gnn_distance\")}')
else:
    print('  ⚠️ 未使用GNN (可能回退到其他算法)')
"

echo ""
echo "✅ 测试完成"
