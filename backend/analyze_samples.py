#!/usr/bin/env python3
"""
分析已保存的真实样本
"""
import os
import glob
from PIL import Image
import sys

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

import clip
import torch
import torch.nn.functional as F

def analyze_samples():
    """分析所有保存的样本对"""
    
    # 加载CLIP模型
    device = "cpu"
    print("📦 加载CLIP模型...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("✅ 模型加载完成\n")
    
    samples_dir = "uploaded_samples"
    
    if not os.path.exists(samples_dir):
        print(f"❌ 样本目录不存在: {samples_dir}")
        return
    
    # 获取所有模板文件
    template_files = sorted(glob.glob(f"{samples_dir}/*_template_*.png"))
    
    if not template_files:
        print(f"⚠️ 未找到任何样本，请先使用系统上传图片")
        return
    
    pairs = []
    for template_file in template_files:
        # 提取时间戳：文件名格式 {type}_{role}_{timestamp}.png
        basename = os.path.basename(template_file)  # 例如: signature_template_20251007_032012.png
        parts = basename.replace('.png', '').split('_')
        
        # parts = ['signature', 'template', '20251007', '032012']
        type_name = parts[0]  # signature 或 seal
        timestamp = '_'.join(parts[2:])  # 20251007_032012
        
        # 构建对应的query文件名
        query_file = os.path.join(samples_dir, f"{type_name}_query_{timestamp}.png")
        
        if os.path.exists(query_file):
            pairs.append((template_file, query_file, type_name, timestamp))
    
    if not pairs:
        print(f"⚠️ 未找到配对样本")
        return
    
    print(f"🔍 找到 {len(pairs)} 对样本")
    print("=" * 80)
    print(f"{'类型':<12} {'时间戳':<18} {'相似度':<10} {'判断':<10} {'建议'}")
    print("=" * 80)
    
    signature_results = []
    seal_results = []
    
    for template_path, query_path, type_name, timestamp in pairs:
        # 加载图片
        img1 = Image.open(template_path).convert('RGB')
        img2 = Image.open(query_path).convert('RGB')
        
        # 预处理
        img1_input = preprocess(img1).unsqueeze(0).to(device)
        img2_input = preprocess(img2).unsqueeze(0).to(device)
        
        # 提取特征
        with torch.no_grad():
            f1 = model.encode_image(img1_input)
            f2 = model.encode_image(img2_input)
            f1 = f1 / f1.norm(dim=-1, keepdim=True)
            f2 = f2 / f2.norm(dim=-1, keepdim=True)
            similarity = float(F.cosine_similarity(f1, f2))
        
        # 判断
        if type_name == 'signature':
            threshold = 0.85
            signature_results.append(similarity)
        else:
            threshold = 0.88
            seal_results.append(similarity)
        
        is_pass = similarity >= threshold
        status = "✅ 通过" if is_pass else "❌ 拒绝"
        
        if similarity > threshold + 0.05:
            confidence = "高置信度"
        elif similarity < threshold - 0.10:
            confidence = "低置信度"
        else:
            confidence = "中等置信度"
        
        type_display = "签名" if type_name == 'signature' else "印章"
        
        print(f"{type_display:<10} {timestamp:<18} {similarity:>6.1%}    {status:<10} {confidence}")
        
        # 保存结果
        result = {
            'type': type_name,
            'timestamp': timestamp,
            'similarity': similarity,
            'threshold': threshold,
            'is_pass': is_pass
        }
    
    print("=" * 80)
    print()
    
    # 统计分析
    if signature_results:
        print("📊 签名样本统计:")
        print(f"   样本数量: {len(signature_results)}")
        print(f"   平均相似度: {sum(signature_results)/len(signature_results):.1%}")
        print(f"   最高相似度: {max(signature_results):.1%}")
        print(f"   最低相似度: {min(signature_results):.1%}")
        print(f"   通过率(>85%): {sum(1 for s in signature_results if s >= 0.85)/len(signature_results):.1%}")
        print()
    
    if seal_results:
        print("📊 印章样本统计:")
        print(f"   样本数量: {len(seal_results)}")
        print(f"   平均相似度: {sum(seal_results)/len(seal_results):.1%}")
        print(f"   最高相似度: {max(seal_results):.1%}")
        print(f"   最低相似度: {min(seal_results):.1%}")
        print(f"   通过率(>88%): {sum(1 for s in seal_results if s >= 0.88)/len(seal_results):.1%}")
        print()
    
    # 找出问题样本
    print("🔍 潜在问题样本:")
    
    sig_idx = 0
    seal_idx = 0
    for template_path, query_path, type_name, timestamp in pairs:
        if type_name == 'signature':
            if sig_idx < len(signature_results):
                similarity = signature_results[sig_idx]
                sig_idx += 1
            else:
                continue
            threshold = 0.85
        else:
            if seal_idx < len(seal_results):
                similarity = seal_results[seal_idx]
                seal_idx += 1
            else:
                continue
            threshold = 0.88
        
        # 接近阈值的样本（需要人工确认）
        if abs(similarity - threshold) < 0.05:
            print(f"   ⚠️ {type_name} {timestamp}: {similarity:.1%} (接近阈值 {threshold:.0%})")
    
    print()
    print("💡 建议:")
    print("   1. 查看 uploaded_samples/ 目录中的图片")
    print("   2. 手动标注哪些是同一个人/印章")
    print("   3. 根据真实标注调整阈值")
    print("   4. 发现CLIP误判的case")

if __name__ == "__main__":
    analyze_samples()
