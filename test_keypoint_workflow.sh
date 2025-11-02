#!/bin/bash

echo "=========================================="
echo "🎯 签名关键点标注完整工作流程测试"
echo "=========================================="

# 检查是否有测试图像
SAMPLE_DIR="backend/uploaded_samples/debug"
if [ ! -d "$SAMPLE_DIR" ]; then
    echo "❌ 找不到测试图像目录: $SAMPLE_DIR"
    exit 1
fi

# 查找第一个清洁后的图像
SAMPLE_IMAGE=$(ls $SAMPLE_DIR/template_cleaned_*.png 2>/dev/null | head -n 1)

if [ -z "$SAMPLE_IMAGE" ]; then
    echo "❌ 找不到测试图像"
    exit 1
fi

echo ""
echo "📁 使用测试图像: $SAMPLE_IMAGE"
echo ""

# 步骤1: 自动标注
echo "步骤1️⃣: 创建自动标注脚本"
cat > auto_annotate_test.py << 'EOF'
import sys
sys.path.insert(0, '.')
from keypoint_annotator import SignatureKeypointAnnotator

# 自动标注
annotator = SignatureKeypointAnnotator(sys.argv[1])
print("\n🤖 执行自动检测...")
annotator.auto_detect_keypoints()

# 保存结果
output = annotator.save_annotations('test_keypoints_auto.json')
print(f"\n✅ 自动标注完成!")
print(f"   生成文件: {output}")
print(f"   可视化: test_keypoints_auto.png")

# 统计
stats = annotator.get_statistics()
print("\n📊 检测结果:")
for kp_type, count in stats.items():
    label = annotator.KEYPOINT_TYPES[kp_type]['label']
    print(f"   {label}: {count}")
print(f"   总计: {sum(stats.values())} 个关键点")
EOF

python3 auto_annotate_test.py "$SAMPLE_IMAGE"

if [ $? -ne 0 ]; then
    echo "❌ 自动标注失败"
    exit 1
fi

echo ""
echo "=========================================="

# 步骤2: 分析标注数据
if [ -f "test_keypoints_auto.json" ]; then
    echo ""
    echo "步骤2️⃣: 分析标注数据"
    python3 analyze_keypoints.py test_keypoints_auto.json
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 分析完成,生成文件:"
        [ -f "keypoint_distribution.png" ] && echo "   - keypoint_distribution.png (分布图)"
        [ -f "signature1_graph.npz" ] && echo "   - signature1_graph.npz (GNN训练数据)"
    fi
fi

echo ""
echo "=========================================="
echo "步骤3️⃣: 手动标注说明"
echo "=========================================="
echo ""
echo "要进行手动标注,请运行:"
echo "  python3 keypoint_annotator.py $SAMPLE_IMAGE"
echo ""
echo "操作指南:"
echo "  1/2/3/4  - 切换关键点类型"
echo "  左键     - 添加关键点"
echo "  右键     - 删除关键点"
echo "  A        - 自动检测"
echo "  S        - 保存结果"
echo "  Q/ESC    - 退出"
echo ""
echo "=========================================="
echo "✅ 工作流程测试完成!"
echo "=========================================="
echo ""
echo "📚 相关文档:"
echo "  - KEYPOINT_ANNOTATION_GUIDE.md (详细标注指南)"
echo "  - keypoint_annotator.py (标注工具)"
echo "  - analyze_keypoints.py (分析工具)"
echo ""
