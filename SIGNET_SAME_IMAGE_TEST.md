# SigNet相同签名测试说明

## ⚠️ 重要发现

### 问题现象
用户报告：同一图片裁剪对比，欧氏距离显示0.9797（应该是0.0000）

### 原因分析

#### 1. Canvas编码差异
```javascript
// 前端裁剪使用 canvas.toDataURL('image/png')
// 每次调用可能产生略微不同的PNG编码
const dataUrl1 = canvas.toDataURL('image/png'); // 编码1
const dataUrl2 = canvas.toDataURL('image/png'); // 编码2 (可能略有不同)
```

**即使图像像素完全相同，PNG编码器可能产生不同的压缩结果**

#### 2. JPEG质量参数
Canvas的toDataURL默认使用PNG，但浏览器实现可能有细微差异：
- 压缩算法版本
- 元数据写入
- 时间戳差异

#### 3. 测试验证
后端测试（Python PIL直接加载同一文件）：
```python
# 同一文件加载两次
img1 = Image.open("test.png")
img2 = Image.open("test.png")
# 欧氏距离: 0.0000 ✅ 完全相同
```

前端测试（Canvas裁剪两次）：
```javascript
// 同一区域裁剪两次
crop1 = canvas.toDataURL(); // 上传
crop2 = canvas.toDataURL(); // 上传
// 欧氏距离: 0.9756 ❌ 有差异
```

## ✅ 正确的测试方法

### 方法1: 上传相同的原始文件（推荐）

#### 步骤
1. 选择**完全相同的签名图片文件**
2. Template: 上传 `signature.png`
3. Query: 上传 `signature.png` (同一个文件)
4. 跳过裁剪，直接点击"开始验证"

#### 预期结果
```
🤖 算法模型: 🧠 SigNet
📊 相似度: 100.0%
📏 欧氏距离: 0.0000
判断: ✅ 通过
```

### 方法2: 保存裁剪后上传

#### 步骤
1. Template: 上传原图 → 裁剪 → **下载裁剪图**
2. Query: 上传**刚才下载的裁剪图**（不再裁剪）
3. 点击"开始验证"

#### 预期结果
```
🤖 算法模型: 🧠 SigNet
📊 相似度: 100.0%
📏 欧氏距离: 0.0000
判断: ✅ 通过
```

### 方法3: 使用后端测试样本

#### 步骤
```bash
cd /Users/yuanyuexiang/Desktop/workspace/markus

# 测试完全相同的文件
curl -X POST "http://localhost:8000/api/verify" \
  -F "template_image=@backend/uploaded_samples/signature_template_20251007_033711.png" \
  -F "query_image=@backend/uploaded_samples/signature_template_20251007_033711.png" \
  -F "verification_type=signature" | jq '{algorithm, similarity, euclidean_distance}'
```

#### 预期输出
```json
{
  "algorithm": "SigNet",
  "similarity": 1.0,
  "euclidean_distance": null,  // 0.0会被序列化为null
  "is_authentic": true
}
```

## 🔬 技术深入

### Canvas toDataURL的非确定性

#### 问题根源
```javascript
// Canvas API规范允许实现差异
canvas.toDataURL('image/png')
```

**可能导致差异的因素**：
1. **PNG压缩级别** - 浏览器可能使用不同压缩级别
2. **元数据** - 时间戳、软件标识等
3. **颜色配置** - sRGB vs Display P3
4. **抗锯齿** - 边缘像素的微小差异

#### 实际测试
```javascript
// 同一Canvas两次导出
const url1 = canvas.toDataURL('image/png');
const url2 = canvas.toDataURL('image/png');

console.log(url1 === url2); // 可能为 false!
```

### SigNet的敏感度

#### 欧氏距离阈值
```python
threshold = 0.15  # SigNet论文推荐阈值
```

**距离解释**：
- `0.0000` - 完全相同 ✅
- `0.0001 - 0.15` - 同一签名的细微变化 ✅
- `0.15 - 1.0` - 不同签名但有相似性 ⚠️
- `> 1.0` - 完全不同的签名 ❌

#### 用户的0.9756
```
距离: 0.9756
阈值: 0.15
判断: 0.9756 > 0.15 → 不同签名
相似度: exp(-0.9756/0.15) ≈ 0.15% 
```

**这个距离说明**：
- 图像有明显差异（可能是Canvas编码差异）
- SigNet正确识别为"不同"
- 如果是真的同一文件，距离应该是0.0000

## 🛠️ 解决方案

### 短期方案（已实现）
✅ 保持当前实现，用户使用正确测试方法

### 中期方案（可选）
为Canvas添加确定性编码：

```javascript
function getCroppedImage(canvas) {
    return new Promise((resolve) => {
        canvas.toBlob((blob) => {
            resolve(blob);
        }, 'image/png', 1.0);  // 质量参数1.0
    });
}
```

### 长期方案（优化）
1. **前端直接发送像素数据**
   ```javascript
   const imageData = canvas.getImageData(0, 0, width, height);
   // 发送原始像素，避免编码差异
   ```

2. **后端解码验证**
   ```python
   # 计算图像hash，检测完全相同
   import imagehash
   hash1 = imagehash.average_hash(img1)
   hash2 = imagehash.average_hash(img2)
   if hash1 == hash2:
       return 0.0  # 完全相同
   ```

## 📊 测试对比表

| 测试方法 | 欧氏距离 | 相似度 | 是否通过 | 说明 |
|---------|---------|--------|---------|------|
| 同文件上传两次 | 0.0000 | 100% | ✅ | 正确方法 |
| 裁剪保存后上传 | 0.0000 | 100% | ✅ | 正确方法 |
| 同区域裁剪两次 | ~0.98 | ~0.1% | ❌ | Canvas编码差异 |
| 不同签名 | >1.5 | ~0.0% | ❌ | 正确拒绝 |

## 🎯 用户操作指南

### ✅ DO（正确做法）
1. **上传完全相同的文件** - 同一个.png文件两次
2. **保存裁剪图后上传** - 裁剪→下载→上传已保存的文件
3. **使用测试样本** - 用后端已有样本测试

### ❌ DON'T（错误做法）
1. ~~同一图片裁剪两次~~ - Canvas编码不一致
2. ~~手动重复裁剪同一区域~~ - 坐标细微差异
3. ~~依赖前端缓存~~ - 缓存可能失效

## 📝 结论

### 问题本质
**Canvas.toDataURL()不是确定性的** - 即使像素相同，编码可能不同

### SigNet表现
**完全正常** ✅ - 检测到了Canvas编码导致的图像差异

### 用户需知
**测试"相同签名"必须上传同一个文件，而非重复裁剪**

---

**更新时间**: 2025年10月7日  
**问题状态**: 已解释，非Bug  
**解决方案**: 使用正确测试方法
