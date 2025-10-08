#!/usr/bin/env python3
"""
测试二值化预处理的效果
生成对比图片：原图 vs 二值化
"""
import cv2
import numpy as np
from PIL import Image
import sys

def preprocess_for_feature_matching(img_cv: np.ndarray) -> np.ndarray:
    """
    图像预处理用于特征点匹配
    1. 自适应二值化：去除背景噪声
    2. 形态学处理：连接断点，去除小噪点
    """
    # 自适应阈值二值化
    binary = cv2.adaptiveThreshold(
        img_cv, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        blockSize=11,
        C=2
    )
    
    # 形态学闭运算：连接断开的笔画
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 去除小噪点
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary

def visualize_preprocessing(image_path: str):
    """可视化预处理效果"""
    # 读取图片
    img = Image.open(image_path).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    # 二值化处理
    binary = preprocess_for_feature_matching(img_cv)
    
    # ORB特征点检测
    orb = cv2.ORB_create(nfeatures=1000)
    
    # 原图特征点
    kp_gray, des_gray = orb.detectAndCompute(img_cv, None)
    img_gray_kp = cv2.drawKeypoints(cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR), 
                                     kp_gray, None, color=(0,255,0))
    
    # 二值化图特征点
    kp_binary, des_binary = orb.detectAndCompute(binary, None)
    img_binary_kp = cv2.drawKeypoints(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), 
                                       kp_binary, None, color=(0,255,0))
    
    # 拼接对比
    comparison = np.hstack([
        cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
        img_gray_kp,
        img_binary_kp
    ])
    
    # 添加文字标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Original', (10, 30), font, 0.7, (0,0,255), 2)
    cv2.putText(comparison, 'Binary', (img_cv.shape[1]+10, 30), font, 0.7, (0,0,255), 2)
    cv2.putText(comparison, f'Features: {len(kp_gray)}', (img_cv.shape[1]*2+10, 30), font, 0.7, (0,255,0), 2)
    cv2.putText(comparison, f'Features: {len(kp_binary)}', (img_cv.shape[1]*3+10, 30), font, 0.7, (0,255,0), 2)
    
    # 保存结果
    output_path = image_path.replace('.png', '_binary_comparison.png')
    cv2.imwrite(output_path, comparison)
    
    print(f"✅ 已生成对比图: {output_path}")
    print(f"   原图特征点: {len(kp_gray)}")
    print(f"   二值化特征点: {len(kp_binary)}")
    print(f"   特征点变化: {len(kp_binary) - len(kp_gray):+d}")
    
    return len(kp_gray), len(kp_binary)

if __name__ == "__main__":
    print("🔍 二值化预处理效果测试")
    print("=" * 60)
    print()
    
    test_images = [
        "test_images/signature_template.png",
        "test_images/signature_fake.png",
        "test_images/seal_template.png",
        "test_images/seal_fake.png"
    ]
    
    for img_path in test_images:
        print(f"📝 处理: {img_path}")
        try:
            gray_kp, binary_kp = visualize_preprocessing(img_path)
            print()
        except Exception as e:
            print(f"   ❌ 错误: {e}")
            print()
    
    print("🎉 测试完成！")
    print("查看生成的 *_binary_comparison.png 文件来对比效果")
