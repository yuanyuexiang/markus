# 真实样本收集系统

## 功能说明

系统现在会**自动保存**您每次上传并裁剪的图片，这些真实数据将用于：
1. ✅ 分析CLIP算法的实际表现
2. ✅ 优化阈值设置
3. ✅ 发现算法缺陷
4. ✅ 训练更好的模型（未来）

## 保存位置

```bash
/Users/yuanyuexiang/Desktop/workspace/markus/backend/uploaded_samples/
```

## 文件命名规则

```
{类型}_{角色}_{时间戳}.png

类型: signature (签名) 或 seal (印章)
角色: template (模板) 或 query (待验证)
时间戳: YYYYmmdd_HHMMSS

示例:
- signature_template_20251007_032012.png  # 签名模板
- signature_query_20251007_032012.png     # 待验证签名
- seal_template_20251007_032013.png       # 印章模板
- seal_query_20251007_032013.png          # 待验证印章
```

## 使用方法

### 1. 正常使用系统
- 打开 http://localhost:3000
- 上传并裁剪图片
- 点击验证
- **图片会自动保存！**

### 2. 查看保存的样本

```bash
cd /Users/yuanyuexiang/Desktop/workspace/markus/backend/uploaded_samples
ls -lh  # 查看所有样本

# 按时间排序
ls -lt

# 只看签名
ls -lh signature_*

# 只看印章
ls -lh seal_*
```

### 3. 分析CLIP相似度

创建分析脚本来批量测试：

```bash
cd /Users/yuanyuexiang/Desktop/workspace/markus/backend
```

创建 `analyze_samples.py`:

```python
import os
import glob
from PIL import Image
import clip
import torch

# 加载CLIP模型
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

samples_dir = "uploaded_samples"

# 获取所有配对
pairs = []
files = sorted(glob.glob(f"{samples_dir}/*_template_*.png"))

for template_file in files:
    timestamp = template_file.split('_')[-1].replace('.png', '')
    type_name = template_file.split('_')[0].split('/')[-1]
    query_file = f"{samples_dir}/{type_name}_query_{timestamp}.png"
    
    if os.path.exists(query_file):
        pairs.append((template_file, query_file, type_name))

print(f"找到 {len(pairs)} 对样本")
print("=" * 60)

for template_path, query_path, type_name in pairs:
    # 加载图片
    img1 = preprocess(Image.open(template_path)).unsqueeze(0).to(device)
    img2 = preprocess(Image.open(query_path)).unsqueeze(0).to(device)
    
    # 提取特征
    with torch.no_grad():
        f1 = model.encode_image(img1)
        f2 = model.encode_image(img2)
        f1 = f1 / f1.norm(dim=-1, keepdim=True)
        f2 = f2 / f2.norm(dim=-1, keepdim=True)
        similarity = float(torch.nn.functional.cosine_similarity(f1, f2))
    
    timestamp = template_path.split('_')[-1].replace('.png', '')
    print(f"{type_name:10} {timestamp} | 相似度: {similarity:.1%}")

print("=" * 60)
```

运行分析：
```bash
python3 analyze_samples.py
```

### 4. 找出问题样本

如果发现CLIP相似度异常的样本：

```bash
# 打开查看
open uploaded_samples/signature_template_20251007_032012.png
open uploaded_samples/signature_query_20251007_032012.png
```

### 5. 导出数据集

```bash
# 复制到专门的数据集目录
mkdir -p ~/Desktop/signature_dataset
cp uploaded_samples/signature_* ~/Desktop/signature_dataset/

mkdir -p ~/Desktop/seal_dataset
cp uploaded_samples/seal_* ~/Desktop/seal_dataset/
```

## 数据分析

### 统计信息

```bash
# 签名样本数量
ls uploaded_samples/signature_template_* | wc -l

# 印章样本数量
ls uploaded_samples/seal_template_* | wc -l

# 总样本数
ls uploaded_samples/*.png | wc -l
```

### 清理旧样本

```bash
# 删除今天之前的样本
find uploaded_samples -name "*.png" -mtime +1 -delete

# 删除所有样本（慎用！）
rm uploaded_samples/*.png
```

## 优化建议

### 1. 收集多样化样本
- ✅ 不同的人签名
- ✅ 同一人不同时间的签名
- ✅ 不同风格的签名（正楷、草书、英文）
- ✅ 不同公司的印章
- ✅ 同一印章不同盖印质量

### 2. 标注样本
建议创建 `samples.csv` 记录：
```csv
template,query,type,is_same_person,clip_similarity,notes
signature_template_20251007_032012.png,signature_query_20251007_032012.png,signature,yes,0.953,同一人不同裁剪
signature_template_20251007_032013.png,signature_query_20251007_032013.png,signature,no,0.751,完全不同的人
```

### 3. 分析误判案例
- CLIP相似度>85%但实际不同 → 误报
- CLIP相似度<85%但实际相同 → 漏报

找出这些case可以帮助调优阈值。

## 隐私声明

⚠️ **注意**：
- 这些图片包含真实签名/印章信息
- 请妥善保管，不要公开分享
- 定期清理不需要的样本
- 敏感数据请及时删除

## 下一步计划

### 短期（基于CLIP）
1. 收集100+真实样本
2. 分析CLIP相似度分布
3. 通过ROC曲线优化阈值
4. 发现CLIP的局限性

### 长期（训练专用模型）
1. 使用收集的样本fine-tune CLIP
2. 或训练孪生网络（Siamese Network）
3. 专门针对签名/印章验证优化
4. 达到99%+准确率

## 当前状态

✅ 系统已启动，每次验证都会自动保存样本
✅ 保存路径：`backend/uploaded_samples/`
✅ 可以直接使用，无需额外配置

**现在就去测试，每次裁剪的图片都会被保存下来！** 🎉
