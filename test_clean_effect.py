#!/usr/bin/env python3
"""测试签名清洁效果"""
import cv2
import numpy as np
import sys
sys.path.insert(0, 'backend')

from preprocess.auto_crop import clean_signature_with_morph

# 读取原图
original_path = 'backend/uploaded_samples/signature_template_20251029_124437.png'
original = cv2.imread(original_path, 0)

if original is None:
    print(f"无法读取图片: {original_path}")
    sys.exit(1)

print(f"原图尺寸: {original.shape}")
print(f"原图前景比例: {(original < 128).sum() / original.size * 100:.2f}%")

# 清洁
cleaned_binary = clean_signature_with_morph(original, mode='conservative')
print(f"\n清洁后（二值）:")
print(f"  尺寸: {cleaned_binary.shape}")
print(f"  前景(255)比例: {(cleaned_binary == 255).sum() / cleaned_binary.size * 100:.2f}%")
print(f"  背景(0)比例: {(cleaned_binary == 0).sum() / cleaned_binary.size * 100:.2f}%")

# 反转（让签名变黑，背景变白）
cleaned_display = cv2.bitwise_not(cleaned_binary)
print(f"\n反转后（用于展示）:")
print(f"  前景(0)比例: {(cleaned_display == 0).sum() / cleaned_display.size * 100:.2f}%")
print(f"  背景(255)比例: {(cleaned_display == 255).sum() / cleaned_display.size * 100:.2f}%")

# 保存对比
cv2.imwrite('test_original.png', original)
cv2.imwrite('test_cleaned_binary.png', cleaned_binary)
cv2.imwrite('test_cleaned_display.png', cleaned_display)

print(f"\n✅ 已保存:")
print(f"  test_original.png - 原图")
print(f"  test_cleaned_binary.png - 清洁后二值图（前景255）")
print(f"  test_cleaned_display.png - 反转后展示图（前景0）")

# 对比分析
print(f"\n📊 对比分析:")
print(f"  原图包含 {len(np.unique(original))} 种灰度值")
print(f"  清洁后只有 {len(np.unique(cleaned_display))} 种值（纯黑白）")

# 检查是否真的清洁了
if (cleaned_display == 0).sum() > (original < 128).sum() * 0.5:
    print(f"  ⚠️  警告: 清洁后前景反而增加了，可能清洁效果不佳")
else:
    print(f"  ✅ 清洁效果正常，前景减少")
