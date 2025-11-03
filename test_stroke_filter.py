#!/usr/bin/env python3
"""
测试笔画筛选功能

测试场景:
1. 完全不同的签名 (应该被快速拒绝)
2. 相似的签名 (应该通过筛选,进入深度学习)
3. 同一人不同时间的签名 (应该通过筛选)
"""

import requests
import sys
from pathlib import Path

API_URL = "http://localhost:8000/api/verify"

def test_stroke_filter():
    """测试笔画筛选功能"""
    
    # 查找测试图片
    test_dir = Path(__file__).parent / "uploaded_samples"
    
    if not test_dir.exists():
        print("❌ 测试目录不存在:", test_dir)
        print("💡 请先运行一次签名验证生成测试样本")
        return
    
    # 获取最新的两张图片
    images = sorted(test_dir.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if len(images) < 2:
        print("❌ 测试图片不足,至少需要2张")
        print("💡 请先运行一次签名验证生成测试样本")
        return
    
    print("🔍 测试笔画筛选功能\n")
    print("=" * 60)
    
    # 测试1: 使用最近上传的两张图片
    print(f"\n📝 测试1: 对比最近的两张图片")
    print(f"   模板: {images[0].name}")
    print(f"   查询: {images[1].name}")
    
    with open(images[0], 'rb') as f1, open(images[1], 'rb') as f2:
        files = {
            'template_image': ('template.png', f1, 'image/png'),
            'query_image': ('query.png', f2, 'image/png')
        }
        data = {
            'algorithm': 'signet',
            'verification_type': 'signature'
        }
        
        response = requests.post(API_URL, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ 验证成功")
            print(f"   算法: {result.get('algorithm_used', 'N/A')}")
            print(f"   处理时间: {result.get('processing_time_ms', 0):.2f}ms")
            
            if result.get('fast_reject'):
                print(f"\n⚡ 快速拒绝!")
                print(f"   原因: {result.get('reject_reason')}")
                print(f"\n📊 笔画特征对比:")
                
                template_f = result['stroke_features']['template']
                query_f = result['stroke_features']['query']
                diffs = result['stroke_features']['differences']
                
                print(f"   {'特征':<15} {'模板':>10} {'查询':>10} {'差异':>10}")
                print(f"   {'-'*50}")
                print(f"   {'笔画数':<15} {template_f['stroke_count']:>10} {query_f['stroke_count']:>10} {diffs['stroke_count_diff']*100:>9.1f}%")
                print(f"   {'密度':<15} {template_f['density']*100:>9.2f}% {query_f['density']*100:>9.2f}% {diffs['density_diff']*100:>9.1f}%")
                print(f"   {'宽高比':<15} {template_f['aspect_ratio']:>10.2f} {query_f['aspect_ratio']:>10.2f} {diffs['aspect_ratio_diff']*100:>9.1f}%")
                print(f"   {'边界框面积':<15} {template_f['bbox_area']:>10.0f} {query_f['bbox_area']:>10.0f} {diffs['bbox_area_diff']*100:>9.1f}%")
                print(f"   {'综合评分':<15} {'':>10} {'':>10} {diffs['combined_score']:>10.2f}")
                
                print(f"\n💡 快速拒绝节省了深度学习计算时间")
                
            else:
                print(f"\n✅ 笔画特征检查通过,使用深度学习验证")
                print(f"   相似度: {result.get('final_score', 0) * 100:.1f}%")
                print(f"   置信度: {result.get('confidence', 'N/A')}")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(response.text)
    
    # 测试2: 如果有更多图片,测试不同组合
    if len(images) >= 4:
        print(f"\n" + "=" * 60)
        print(f"\n📝 测试2: 对比较早的两张图片")
        print(f"   模板: {images[2].name}")
        print(f"   查询: {images[3].name}")
        
        with open(images[2], 'rb') as f1, open(images[3], 'rb') as f2:
            files = {
                'template_image': ('template.png', f1, 'image/png'),
                'query_image': ('query.png', f2, 'image/png')
            }
            data = {
                'algorithm': 'gnn',
                'verification_type': 'signature'
            }
            
            response = requests.post(API_URL, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"\n✅ 验证成功")
                print(f"   算法: {result.get('algorithm_used', 'N/A')}")
                print(f"   处理时间: {result.get('processing_time_ms', 0):.2f}ms")
                
                if result.get('fast_reject'):
                    print(f"\n⚡ 快速拒绝! (节省了GNN计算)")
                    print(f"   原因: {result.get('reject_reason')}")
                else:
                    print(f"\n✅ 进入GNN深度验证")
                    print(f"   相似度: {result.get('final_score', 0) * 100:.1f}%")
    
    print("\n" + "=" * 60)
    print("\n✨ 测试完成!")
    print("\n💡 使用建议:")
    print("   1. 笔画筛选在验证前快速拒绝明显不同的签名")
    print("   2. 处理时间从数百毫秒降低到几毫秒")
    print("   3. 节省GPU/CPU资源,提升系统吞吐量")
    print("   4. 阈值保守设置,避免误杀真实签名")

if __name__ == '__main__':
    try:
        test_stroke_filter()
    except KeyboardInterrupt:
        print("\n\n⚠️  测试中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
