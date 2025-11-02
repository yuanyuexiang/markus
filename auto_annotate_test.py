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
