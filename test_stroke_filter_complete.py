#!/usr/bin/env python3
"""
笔画特征筛选专项测试

测试三种场景:
1. 真实签名 vs 真实签名 (同一人) - 应通过筛选
2. 模板签名 vs 伪造签名 (不同人) - 应快速拒绝
3. 签名 vs 图章 - 应快速拒绝
"""

import requests
import sys
from pathlib import Path
import time

API_URL = "http://localhost:8000/api/verify"

def test_case(name, template_file, query_file, expected_reject=False):
    """测试单个案例"""
    print(f"\n{'='*70}")
    print(f"📝 测试: {name}")
    print(f"   模板: {template_file.name}")
    print(f"   查询: {query_file.name}")
    print(f"   期望: {'快速拒绝' if expected_reject else '通过筛选'}")
    print(f"{'='*70}")
    
    with open(template_file, 'rb') as f1, open(query_file, 'rb') as f2:
        files = {
            'template_image': ('template.png', f1, 'image/png'),
            'query_image': ('query.png', f2, 'image/png')
        }
        data = {
            'algorithm': 'signet',
            'verification_type': 'signature'
        }
        
        start = time.time()
        response = requests.post(API_URL, files=files, data=data)
        elapsed = (time.time() - start) * 1000
        
        if response.status_code != 200:
            print(f"❌ 请求失败: {response.status_code}")
            print(response.text)
            return False
        
        result = response.json()
        is_fast_reject = result.get('fast_reject', False)
        
        print(f"\n⏱️  总处理时间: {elapsed:.2f}ms")
        print(f"   后端处理: {result.get('processing_time_ms', 0):.2f}ms")
        
        if is_fast_reject:
            print(f"\n⚡ 快速拒绝!")
            print(f"   原因: {result.get('reject_reason')}")
            print(f"   算法: {result.get('algorithm_used')}")
            
            if 'stroke_features' in result:
                template_f = result['stroke_features']['template']
                query_f = result['stroke_features']['query']
                diffs = result['stroke_features']['differences']
                
                print(f"\n📊 笔画特征对比:")
                print(f"   {'特征':<18} {'模板':>12} {'查询':>12} {'差异':>12}")
                print(f"   {'-'*60}")
                
                features = [
                    ('笔画数', 'stroke_count', '', ''),
                    ('密度', 'density', '%', 100),
                    ('宽高比', 'aspect_ratio', '', 1),
                    ('边界框面积', 'bbox_area', 'px', 1),
                ]
                
                for fname, fkey, unit, multiplier in features:
                    t_val = template_f[fkey]
                    q_val = query_f[fkey]
                    diff = diffs[f'{fkey}_diff']
                    
                    if multiplier and multiplier != 1:
                        t_val *= multiplier
                        q_val *= multiplier
                    
                    t_str = f"{t_val:.2f}{unit}" if multiplier else f"{int(t_val)}{unit}"
                    q_str = f"{q_val:.2f}{unit}" if multiplier else f"{int(q_val)}{unit}"
                    diff_str = f"{diff*100:.1f}%"
                    
                    # 高亮差异大的特征
                    highlight = "🔴" if diff > 0.5 else "🟢" if diff > 0.3 else "⚪"
                    print(f"   {highlight} {fname:<15} {t_str:>12} {q_str:>12} {diff_str:>12}")
                
                if 'combined_score' in diffs:
                    print(f"\n   综合评分: {diffs['combined_score']:.2f}")
                    print(f"   (评分 > 1.2 触发拒绝)")
        else:
            print(f"\n✅ 通过笔画筛选,进入深度学习验证")
            print(f"   算法: {result.get('algorithm_used')}")
            print(f"   相似度: {result.get('final_score', 0) * 100:.1f}%")
            print(f"   置信度: {result.get('confidence', 'N/A')}")
        
        # 验证是否符合预期
        success = is_fast_reject == expected_reject
        
        if success:
            print(f"\n✅ 测试通过!")
        else:
            print(f"\n❌ 测试失败!")
            print(f"   期望: {'快速拒绝' if expected_reject else '通过筛选'}")
            print(f"   实际: {'快速拒绝' if is_fast_reject else '通过筛选'}")
        
        return success

def main():
    """主测试流程"""
    test_dir = Path("test_images")
    
    if not test_dir.exists():
        print("❌ test_images 目录不存在")
        print("💡 请先运行: python3 generate_test_images.py")
        return 1
    
    print("🔍 笔画特征筛选 - 专项测试")
    print("测试目标: 验证快速拒绝功能是否正常工作\n")
    
    results = []
    
    # 测试1: 真实签名 vs 真实签名 (应通过筛选)
    test1 = test_case(
        "真实签名匹配",
        test_dir / "signature_template.png",
        test_dir / "signature_real.png",
        expected_reject=False
    )
    results.append(("真实签名匹配", test1))
    
    # 测试2: 模板签名 vs 伪造签名 (应快速拒绝)
    test2 = test_case(
        "伪造签名识别",
        test_dir / "signature_template.png",
        test_dir / "signature_fake.png",
        expected_reject=True
    )
    results.append(("伪造签名识别", test2))
    
    # 测试3: 签名 vs 图章 (应快速拒绝)
    test3 = test_case(
        "签名vs图章 (类型混淆)",
        test_dir / "signature_template.png",
        test_dir / "seal_template.png",
        expected_reject=True
    )
    results.append(("签名vs图章", test3))
    
    # 测试4: 图章 vs 图章 (应通过筛选)
    test4 = test_case(
        "真实图章匹配",
        test_dir / "seal_template.png",
        test_dir / "seal_real.png",
        expected_reject=False
    )
    results.append(("真实图章匹配", test4))
    
    # 测试5: 图章 vs 伪造图章 (应快速拒绝)
    test5 = test_case(
        "伪造图章识别",
        test_dir / "seal_template.png",
        test_dir / "seal_fake.png",
        expected_reject=True
    )
    results.append(("伪造图章识别", test5))
    
    # 汇总结果
    print(f"\n{'='*70}")
    print("📊 测试汇总")
    print(f"{'='*70}\n")
    
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {status}  {name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\n通过率: {passed}/{total} ({passed/total*100:.0f}%)\n")
    
    if passed == total:
        print("🎉 所有测试通过!")
        print("\n💡 笔画筛选功能工作正常:")
        print("   ✓ 能正确识别明显不同的签名/图章")
        print("   ✓ 不会误杀相似的真实签名/图章")
        print("   ✓ 显著降低深度学习计算负担")
        return 0
    else:
        print("⚠️  部分测试失败,请检查阈值设置")
        print("\n💡 调整建议:")
        print("   - 如果误杀真实签名,提高阈值(stroke_analyzer.py)")
        print("   - 如果放过明显伪造,降低阈值")
        return 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  测试中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
