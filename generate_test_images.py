"""
生成测试用的签名和图章图片
"""
from PIL import Image, ImageDraw, ImageFont
import os
import random

def create_signature_sample(text, filename, add_variation=False):
    """创建签名样本"""
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # 模拟手写签名（斜体效果）
    try:
        # 尝试使用系统字体
        font = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", 60)
    except:
        font = ImageFont.load_default()
    
    # 位置微调（模拟真实手写的轻微差异）
    x_offset = random.randint(-5, 5) if add_variation else 0
    y_offset = random.randint(-3, 3) if add_variation else 0
    
    # 绘制签名文字
    draw.text((50 + x_offset, 70 + y_offset), text, fill='black', font=font)
    
    # 添加轻微噪点（模拟纸张质感）
    if add_variation:
        import numpy as np
        img_array = np.array(img)
        noise = np.random.randint(-5, 5, img_array.shape, dtype=np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
    
    img.save(filename)
    print(f"✅ 已生成签名样本: {filename}")

def create_seal_sample(company_name, filename, add_variation=False, is_fake=False):
    """创建图章样本"""
    img = Image.new('RGB', (300, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # 位置微调
    offset = random.randint(-3, 3) if add_variation else 0
    
    if is_fake:
        # 伪造图章：使用方形边框和不同的形状
        draw.rectangle([50, 50, 250, 250], outline='red', width=8)
        # 绘制简单的圆形而不是五角星
        draw.ellipse([120, 100, 180, 160], fill='red')
    else:
        # 真实图章：圆形边框和五角星
        draw.ellipse([50+offset, 50+offset, 250+offset, 250+offset], outline='red', width=8)
        
        # 绘制五角星（简化）
        star_offset = offset
        draw.polygon([(150+star_offset, 80), (165+star_offset, 120), (205+star_offset, 120), 
                      (175+star_offset, 145), (190+star_offset, 185), (150+star_offset, 160), 
                      (110+star_offset, 185), (125+star_offset, 145), 
                      (95+star_offset, 120), (135+star_offset, 120)], fill='red')
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 24)
    except:
        font = ImageFont.load_default()
    
    # 绘制公司名称（环形文字简化为直线）
    draw.text((80+offset, 200), company_name, fill='red', font=font)
    
    # 添加轻微噪点
    if add_variation:
        import numpy as np
        img_array = np.array(img)
        noise = np.random.randint(-3, 3, img_array.shape, dtype=np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
    
    img.save(filename)
    print(f"✅ 已生成图章样本: {filename}")

if __name__ == "__main__":
    # 创建测试图片目录
    os.makedirs("test_images", exist_ok=True)
    
    print("🎨 正在生成测试图片...")
    
    # 生成签名样本
    create_signature_sample("张三", "test_images/signature_template.png")
    create_signature_sample("张三", "test_images/signature_real.png", add_variation=True)  # 有轻微差异
    create_signature_sample("李四", "test_images/signature_fake.png")  # 完全不同
    
    # 生成图章样本
    create_seal_sample("某某公司", "test_images/seal_template.png")
    create_seal_sample("某某公司", "test_images/seal_real.png", add_variation=True)  # 有轻微差异
    create_seal_sample("其他公司", "test_images/seal_fake.png", is_fake=True)  # 完全不同的图章
    
    print("\n🎉 测试图片生成完成！")
    print("\n📁 测试图片位置: ./test_images/")
    print("   - signature_template.png (模板签名)")
    print("   - signature_real.png (真实签名 - 应该匹配)")
    print("   - signature_fake.png (伪造签名 - 应该不匹配)")
    print("   - seal_template.png (模板图章)")
    print("   - seal_real.png (真实图章 - 应该匹配)")
    print("   - seal_fake.png (伪造图章 - 应该不匹配)")
