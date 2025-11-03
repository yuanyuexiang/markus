#!/usr/bin/env python3
"""
分析测试图片的笔画特征差异
"""
import sys
sys.path.append('backend')

from stroke_analyzer import SignatureStrokeAnalyzer
import cv2
from pathlib import Path

def analyze_pair(file1, file2, label):
    """分析一对图片"""
    print(f"\n{'='*70}")
    print(f"📝 {label}")
    print(f"   图片1: {file1.name}")
    print(f"   图片2: {file2.name}")
    print(f"{'='*70}")
    
    analyzer = SignatureStrokeAnalyzer()
    
    img1 = cv2.imread(str(file1))
    img2 = cv2.imread(str(file2))
    
    f1 = analyzer.extract_features(img1)
    f2 = analyzer.extract_features(img2)
    
    diff = analyzer.calculate_difference(f1, f2)
    should_reject, reason = analyzer.should_fast_reject(f1, f2)
    
    print(f"\n特征对比:")
    print(f"{'特征':<20} {'图片1':>15} {'图片2':>15} {'差异':>10}")
    print(f"{'-'*70}")
    print(f"{'笔画数':<20} {f1['stroke_count']:>15} {f2['stroke_count']:>15} {diff['stroke_count_diff']*100:>9.1f}%")
    print(f"{'密度':<20} {f1['density']*100:>14.2f}% {f2['density']*100:>14.2f}% {diff['density_diff']*100:>9.1f}%")
    print(f"{'宽高比':<20} {f1['aspect_ratio']:>15.2f} {f2['aspect_ratio']:>15.2f} {diff['aspect_ratio_diff']*100:>9.1f}%")
    print(f"{'边界框面积':<20} {f1['bbox_area']:>15.0f} {f2['bbox_area']:>15.0f} {diff['bbox_area_diff']*100:>9.1f}%")
    print(f"{'综合评分':<20} {'':>15} {'':>15} {diff['combined_score']:>10.2f}")
    
    print(f"\n是否拒绝: {'🔴 是' if should_reject else '🟢 否'}")
    if should_reject:
        print(f"拒绝原因: {reason}")
    else:
        print(f"通过原因: 所有特征差异都在阈值范围内")

def main():
    test_dir = Path("test_images")
    
    print("🔍 笔画特征差异分析")
    
    # 分析所有测试对
    analyze_pair(
        test_dir / "signature_template.png",
        test_dir / "signature_real.png",
        "真实签名匹配 (期望:通过)"
    )
    
    analyze_pair(
        test_dir / "signature_template.png",
        test_dir / "signature_fake.png",
        "伪造签名识别 (期望:拒绝)"
    )
    
    analyze_pair(
        test_dir / "signature_template.png",
        test_dir / "seal_template.png",
        "签名vs图章 (期望:拒绝)"
    )
    
    analyze_pair(
        test_dir / "seal_template.png",
        test_dir / "seal_real.png",
        "真实图章匹配 (期望:通过)"
    )
    
    analyze_pair(
        test_dir / "seal_template.png",
        test_dir / "seal_fake.png",
        "伪造图章识别 (期望:拒绝)"
    )
    
    print(f"\n{'='*70}")
    print("\n💡 阈值参考:")
    print("   笔画数量差异: > 45%")
    print("   宽高比差异: > 50%")
    print("   密度差异: > 50%")
    print("   边界框面积差异: > 60%")
    print("   综合评分: > 1.2")

if __name__ == '__main__':
    main()
