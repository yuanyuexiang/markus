#!/usr/bin/env python3
"""
无GUI版本的关键点自动检测演示
"""

import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path


def skeletonize(binary):
    """骨架提取 - 使用形态学细化"""
    # 确保是二值图 (0和255)
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    
    # 使用OpenCV的形态学细化
    size = np.size(binary)
    skeleton = np.zeros(binary.shape, np.uint8)
    
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    
    temp = binary.copy()
    
    while not done:
        eroded = cv2.erode(temp, element)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        subset = cv2.subtract(eroded, opened)
        skeleton = cv2.bitwise_or(skeleton, subset)
        temp = eroded.copy()
        
        zeros = size - cv2.countNonZero(temp)
        if zeros == size:
            done = True
    
    return skeleton


def detect_endpoints(skeleton):
    """检测端点"""
    endpoints = []
    h, w = skeleton.shape
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skeleton[y, x] == 0:
                continue
            
            neighbors = skeleton[y-1:y+2, x-1:x+2].copy()
            neighbors[1, 1] = 0
            neighbor_count = np.count_nonzero(neighbors)
            
            if neighbor_count == 1:
                endpoints.append({'x': int(x), 'y': int(y), 'type': 'endpoint'})
    
    return endpoints


def detect_junctions(skeleton):
    """检测交叉点"""
    junctions = []
    h, w = skeleton.shape
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skeleton[y, x] == 0:
                continue
            
            neighbors = skeleton[y-1:y+2, x-1:x+2].copy()
            neighbors[1, 1] = 0
            neighbor_count = np.count_nonzero(neighbors)
            
            if neighbor_count >= 3:
                junctions.append({'x': int(x), 'y': int(y), 'type': 'junction'})
    
    return junctions


def detect_corners(skeleton):
    """检测转折点"""
    skeleton_float = np.float32(skeleton)
    harris = cv2.cornerHarris(skeleton_float, blockSize=2, ksize=3, k=0.04)
    
    threshold = 0.01 * harris.max() if harris.max() > 0 else 0
    corners_pos = np.argwhere(harris > threshold)
    
    corners = [{'x': int(x), 'y': int(y), 'type': 'corner'} 
               for y, x in corners_pos]
    
    return corners


def auto_detect_keypoints(image_path):
    """自动检测关键点"""
    print(f"\n📂 加载图像: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    
    # 转灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    print(f"   图像尺寸: {gray.shape[1]} x {gray.shape[0]}")
    
    # 二值化 - 检测是黑底白字还是白底黑字
    mean_val = np.mean(gray)
    if mean_val > 127:
        # 白底黑字,直接使用(前景=0,背景=255)
        # 需要反转为前景=255,背景=0
        binary = cv2.bitwise_not(gray)
        print("   检测到白底黑字,已反转")
    else:
        # 黑底白字
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        print("   检测到黑底白字")
    
    # 确保是二值图
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    
    print("🔍 提取骨架...")
    skeleton = skeletonize(binary)
    
    print("🔍 检测端点...")
    endpoints = detect_endpoints(skeleton)
    
    print("🔍 检测交叉点...")
    junctions = detect_junctions(skeleton)
    
    print("🔍 检测转折点...")
    corners = detect_corners(skeleton)
    
    # 合并所有关键点
    all_keypoints = endpoints + junctions + corners
    
    # 统计
    stats = {
        'endpoint': len(endpoints),
        'junction': len(junctions),
        'corner': len(corners),
        'bifurcation': 0
    }
    
    print(f"\n✅ 检测完成,共 {len(all_keypoints)} 个关键点:")
    print(f"   端点: {stats['endpoint']}")
    print(f"   交叉点: {stats['junction']}")
    print(f"   转折点: {stats['corner']}")
    
    # 保存结果
    data = {
        'image_path': str(image_path),
        'image_size': {
            'width': int(gray.shape[1]),
            'height': int(gray.shape[0])
        },
        'timestamp': datetime.now().isoformat(),
        'keypoints': all_keypoints,
        'statistics': stats
    }
    
    # 可视化
    vis_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    colors = {
        'endpoint': (0, 0, 255),    # 红色
        'junction': (0, 255, 0),    # 绿色
        'corner': (255, 0, 0)       # 蓝色
    }
    
    for kp in all_keypoints:
        color = colors.get(kp['type'], (255, 255, 255))
        cv2.circle(vis_image, (kp['x'], kp['y']), 5, color, -1)
        cv2.circle(vis_image, (kp['x'], kp['y']), 7, color, 2)
    
    return data, vis_image, skeleton


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python3 auto_detect_keypoints.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("="*60)
    print("🤖 签名关键点自动检测")
    print("="*60)
    
    # 检测
    data, vis_image, skeleton = auto_detect_keypoints(image_path)
    
    # 保存JSON
    base_name = Path(image_path).stem
    json_path = f"keypoints_{base_name}_auto.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 JSON数据已保存: {json_path}")
    
    # 保存可视化
    vis_path = f"keypoints_{base_name}_auto_vis.png"
    cv2.imwrite(vis_path, vis_image)
    print(f"🖼️  可视化已保存: {vis_path}")
    
    # 保存骨架
    skeleton_path = f"keypoints_{base_name}_skeleton.png"
    cv2.imwrite(skeleton_path, skeleton)
    print(f"🦴 骨架图已保存: {skeleton_path}")
    
    print("\n" + "="*60)
    print("✅ 检测完成!")
    print("="*60)
