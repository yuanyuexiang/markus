#!/usr/bin/env python3
"""测试SigNet对同一图片的一致性"""
import sys
sys.path.insert(0, '/Users/yuanyuexiang/Desktop/workspace/markus/backend')

from signet_model import SigNetModel
from preprocess.normalize import preprocess_signature
import numpy as np
from PIL import Image

# 加载模型
model = SigNetModel('models/signet.pkl')

# 加载同一张图片
img_path = 'uploaded_samples/signature_query_20251007_182850.png'
img = np.array(Image.open(img_path).convert('L'))

# 预处理
img_processed = preprocess_signature(img, canvas_size=(952, 1360))

# 多次推理
print("\n🔬 测试同一图片的推理一致性:")
print("="*60)

features = []
for i in range(5):
    feat = model.get_feature_vector(img_processed)
    features.append(feat)
    print(f"第{i+1}次推理 - 特征向量前5维: {feat[:5]}")

# 计算两两之间的欧氏距离
print("\n📏 两两之间的欧氏距离:")
print("="*60)
for i in range(len(features)):
    for j in range(i+1, len(features)):
        dist = np.linalg.norm(features[i] - features[j])
        print(f"推理{i+1} vs 推理{j+1}: {dist:.10f}")

# 计算与第一次的最大差异
max_diff = 0
for i in range(1, len(features)):
    diff = np.abs(features[0] - features[i]).max()
    max_diff = max(max_diff, diff)
    
print(f"\n📊 最大元素差异: {max_diff:.2e}")

if max_diff < 1e-6:
    print("✅ 结论: 推理完全一致!")
elif max_diff < 1e-3:
    print("⚠️  结论: 推理存在微小差异(可能是浮点误差)")
else:
    print("❌ 结论: 推理存在明显差异(可能是BatchNorm问题)")
