#!/bin/bash

echo "🧪 SigNet/CLIP 快速冒烟测试"
echo "================================"
echo ""

echo "📝 测试1: 同一张签名(完全相同文件)"
echo "期望: 欧氏距离接近0，相似度接近100%，判定通过"
echo ""

curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@test_images/signature_template.png" \
  -F "query_image=@test_images/signature_template.png" \
  -F "verification_type=signature" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"算法: {data.get('algorithm')}\")
print(f\"相似度: {data.get('final_score', 0)*100:.2f}%\")
print(f\"欧氏距离: {data.get('euclidean_distance')}\")
print(f\"阈值: {data.get('threshold')}\")
print(f\"判定结果: {'通过' if data.get('is_authentic') else '拒绝'}\")
print(f\"置信度: {str(data.get('confidence','')).upper()}\")
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📝 测试2: 模板 vs real(同一人不同写法/质量差异)"
echo "(该测试只展示结果，不做强断言)"
echo ""

curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@test_images/signature_template.png" \
  -F "query_image=@test_images/signature_real.png" \
  -F "verification_type=signature" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"算法: {data.get('algorithm')}\")
print(f\"相似度: {data.get('final_score', 0)*100:.2f}%\")
print(f\"欧氏距离: {data.get('euclidean_distance')}\")
print(f\"阈值: {data.get('threshold')}\")
print(f\"判定结果: {'通过' if data.get('is_authentic') else '拒绝'}\")
print(f\"置信度: {str(data.get('confidence','')).upper()}\")
" 
