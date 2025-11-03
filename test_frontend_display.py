#!/usr/bin/env python3
"""
快速测试前端显示 - 模拟快速拒绝场景
"""
import requests
from pathlib import Path

API_URL = "http://localhost:8000/api/verify"

def test_fast_reject_display():
    """测试快速拒绝在前端的显示"""
    test_dir = Path("test_images")
    
    # 测试: 签名 vs 图章 (应该快速拒绝)
    print("🧪 测试快速拒绝前端显示...")
    print(f"   场景: 签名 vs 图章 (完全不同类型)")
    
    with open(test_dir / "signature_template.png", 'rb') as f1:
        with open(test_dir / "seal_template.png", 'rb') as f2:
            files = {
                'template_image': ('template.png', f1, 'image/png'),
                'query_image': ('query.png', f2, 'image/png')
            }
            data = {
                'algorithm': 'signet',
                'verification_type': 'signature'
            }
            
            print(f"\n📤 发送请求...")
            response = requests.post(API_URL, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"\n✅ 响应成功!")
                print(f"\n📊 返回数据结构:")
                print(f"   success: {result.get('success')}")
                print(f"   fast_reject: {result.get('fast_reject')}")
                print(f"   match: {result.get('match')}")
                print(f"   final_score: {result.get('final_score')}")
                print(f"   confidence: {result.get('confidence')}")
                print(f"   algorithm: {result.get('algorithm')}")
                print(f"   type: {result.get('type')}")
                print(f"   reject_reason: {result.get('reject_reason')}")
                print(f"   processing_time_ms: {result.get('processing_time_ms')}")
                
                if result.get('stroke_features'):
                    print(f"\n📐 笔画特征数据:")
                    print(f"   ✓ template: {list(result['stroke_features']['template'].keys())}")
                    print(f"   ✓ query: {list(result['stroke_features']['query'].keys())}")
                    print(f"   ✓ differences: {list(result['stroke_features']['differences'].keys())}")
                
                print(f"\n🎯 前端显示测试:")
                if result.get('success') and result.get('fast_reject'):
                    print(f"   ✅ 数据完整,前端应正常显示快速拒绝结果")
                    print(f"\n💡 请在浏览器中测试:")
                    print(f"   1. 打开 http://localhost:8000")
                    print(f"   2. 上传同样的图片 (signature_template.png vs seal_template.png)")
                    print(f"   3. 应该看到:")
                    print(f"      - 大标题显示 '0.0%'")
                    print(f"      - 算法显示 '⚡ 笔画筛选器'")
                    print(f"      - 漂亮的特征对比表格")
                    print(f"      - 拒绝原因说明")
                    print(f"      - 处理时间 ~{result.get('processing_time_ms', 0):.0f}ms")
                else:
                    print(f"   ⚠️  数据可能不完整,前端可能显示错误")
                    
            else:
                print(f"❌ 请求失败: {response.status_code}")
                print(response.text)

if __name__ == '__main__':
    test_fast_reject_display()
