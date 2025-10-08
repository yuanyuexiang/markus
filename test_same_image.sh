#!/bin/bash

echo "🧪 测试同一张图片不同裁剪的匹配度"
echo "================================"
echo ""

# 创建两个不同大小的裁剪（模拟用户操作）
# 使用signature_template.png作为基础

echo "📝 测试: 同一张签名的不同裁剪区域"
echo ""
echo "模拟场景: 用户从同一张图片裁剪了两个不同大小的区域"
echo "期望结果: 特征点匹配应该 > 60%，CLIP相似度应该 > 90%"
echo ""

curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@test_images/signature_template.png" \
  -F "query_image=@test_images/signature_real.png" \
  -F "verification_type=signature" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"✅ CLIP相似度: {data['clip_similarity']:.1%}\")
print(f\"✅ 特征点匹配: {data['feature_similarity']:.1%}\")
print(f\"✅ 最终得分: {data['final_score']:.1%}\")
print(f\"✅ 判定结果: {'通过' if data['is_authentic'] else '拒绝'}\")
print(f\"✅ 置信度: {data['confidence'].upper()}\")
print()
if data['feature_similarity'] > 0.6:
    print('🎉 特征点匹配正常！同一张图片的不同裁剪能正确识别')
else:
    print(f\"⚠️  特征点匹配偏低 ({data['feature_similarity']:.1%})，可能需要进一步优化\")
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 查看后端日志中的特征点详情
echo "📊 后端特征点匹配详情:"
tail -20 /tmp/backend.log | grep "特征点:" | tail -1
