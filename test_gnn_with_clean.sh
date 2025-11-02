#!/bin/bash

echo "🧪 测试 GNN 自动清洁功能"
echo "=" * 60

TEMPLATE="backend/uploaded_samples/signature_template_20251029_185914.png"
QUERY="backend/uploaded_samples/signature_query_20251029_185914.png"

echo "📁 测试图片: 您刚才上传的两张签名"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧠 GNN 算法 (现在会自动清洁)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

curl -X POST http://localhost:8000/api/verify \
  -F "template_image=@$TEMPLATE" \
  -F "query_image=@$QUERY" \
  -F "verification_type=signature" \
  -F "algorithm=gnn" \
  2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'\n结果:')
print(f'  算法: {data.get(\"algorithm\")}')
print(f'  相似度: {data.get(\"similarity\"):.4f} ({data.get(\"similarity\")*100:.2f}%)')
print(f'  GNN距离: {data.get(\"gnn_distance\"):.4f}')
print(f'  关键点-模板: {data.get(\"gnn_keypoints_template\")}')
print(f'  关键点-查询: {data.get(\"gnn_keypoints_query\")}')
print(f'  阈值: {data.get(\"threshold\"):.4f}')
print(f'  匹配结果: {\"✅ 相同\" if data.get(\"is_authentic\") else \"❌ 不同\"}')
print(f'\n📊 对比之前的结果:')
print(f'  之前(无清洁): 距离 1.3555, 置信度 21.48% → ❌ 不同')
print(f'  现在(自动清洁): 距离 {data.get(\"gnn_distance\"):.4f}, 置信度 {data.get(\"similarity\")*100:.2f}% → {\"✅ 相同\" if data.get(\"is_authentic\") else \"❌ 不同\"}')
"

echo ""
echo "✅ 测试完成!"
echo ""
echo "💡 提示: 清洁后的图片保存在 backend/uploaded_samples/debug/ 目录"

