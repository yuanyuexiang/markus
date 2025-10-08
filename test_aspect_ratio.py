#!/usr/bin/env python3
"""
测试不同长宽比裁剪的特征匹配效果
"""
import cv2
import numpy as np
from PIL import Image
import requests
import base64
import io

def crop_image_with_ratio(img_path, ratio_w, ratio_h):
    """按照指定比例裁剪图片中心区域"""
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # 计算目标长宽比
    target_aspect = ratio_w / ratio_h
    current_aspect = w / h
    
    if current_aspect > target_aspect:
        # 图片太宽，裁剪宽度
        new_w = int(h * target_aspect)
        new_h = h
        x = (w - new_w) // 2
        y = 0
    else:
        # 图片太高，裁剪高度
        new_w = w
        new_h = int(w / target_aspect)
        x = 0
        y = (h - new_h) // 2
    
    cropped = img[y:y+new_h, x:x+new_w]
    
    # 转换为PIL Image并保存
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cropped_rgb)
    
    # 保存到字节流
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return buffer, pil_img.size

def test_different_ratios():
    """测试同一张签名，不同比例裁剪的匹配效果"""
    base_img = "test_images/signature_template.png"
    
    print("🧪 测试不同长宽比裁剪的匹配效果")
    print("=" * 60)
    
    # 测试配置：基准 vs 不同比例
    test_cases = [
        ("1:1正方形 vs 1:1正方形", (1, 1), (1, 1)),
        ("1:1正方形 vs 2:1横向", (1, 1), (2, 1)),
        ("1:1正方形 vs 1:2纵向", (1, 1), (1, 2)),
        ("2:1横向 vs 3:1超宽", (2, 1), (3, 1)),
        ("1:2纵向 vs 1:3超高", (1, 2), (1, 3)),
    ]
    
    for test_name, ratio1, ratio2 in test_cases:
        print(f"\n📝 {test_name}")
        
        # 裁剪两张图片
        img1_buffer, size1 = crop_image_with_ratio(base_img, *ratio1)
        img2_buffer, size2 = crop_image_with_ratio(base_img, *ratio2)
        
        print(f"   图片1尺寸: {size1[0]}x{size1[1]} (比例{ratio1[0]}:{ratio1[1]})")
        print(f"   图片2尺寸: {size2[0]}x{size2[1]} (比例{ratio2[0]}:{ratio2[1]})")
        
        # 调用API验证
        files = {
            'template_image': ('template.png', img1_buffer, 'image/png'),
            'query_image': ('query.png', img2_buffer, 'image/png'),
        }
        data = {'verification_type': 'signature'}
        
        try:
            response = requests.post('http://localhost:8000/api/verify', files=files, data=data)
            result = response.json()
            
            if result['success']:
                print(f"   ✅ CLIP相似度: {result['clip_similarity']*100:.1f}%")
                print(f"   ✅ 特征匹配: {result['feature_similarity']*100:.1f}%")
                print(f"   ✅ 最终得分: {result['final_score']*100:.1f}%")
                print(f"   ✅ 判定: {result['recommendation']}")
            else:
                print(f"   ❌ 失败: {result.get('error', '未知错误')}")
        except Exception as e:
            print(f"   ❌ 请求失败: {e}")

if __name__ == "__main__":
    test_different_ratios()
