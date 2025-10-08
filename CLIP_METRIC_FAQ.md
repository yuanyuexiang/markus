# ❓ 图章CLIP比的是欧氏距离还是余弦相似度？

## ✅ 答案：余弦相似度 (Cosine Similarity)

### 代码证明

**位置**: `backend/main.py` 第225-241行

```python
def compute_clip_similarity(template_img, query_img):
    """使用CLIP计算图像相似度"""
    # CLIP预处理和特征提取
    template_input = clip_preprocess(template_img).unsqueeze(0).to(device)
    query_input = clip_preprocess(query_img).unsqueeze(0).to(device)
    
    # 提取特征
    with torch.no_grad():
        template_features = clip_model.encode_image(template_input)
        query_features = clip_model.encode_image(query_input)
        
        # L2归一化
        template_features = template_features / template_features.norm(dim=-1, keepdim=True)
        query_features = query_features / query_features.norm(dim=-1, keepdim=True)
    
    # 计算余弦相似度 ✅
    similarity = float(F.cosine_similarity(template_features, query_features))
    return similarity
```

**关键行**: 第241行使用 `F.cosine_similarity()` 计算**余弦相似度**。

### 实际验证结果

根据你提供的验证结果：

```
验证结果
印章图章 · 置信度: HIGH
99.6%

🤖 算法模型: CLIP
📊 相似度: 99.6%
📏 欧氏距离: N/A  ✅ 正确！因为CLIP不使用欧氏距离
```

**欧氏距离显示 "N/A" 是正确的行为**，因为：
1. CLIP 只返回余弦相似度
2. API 中 `euclidean_distance` 为 `null`
3. 前端检测到 `null` 后显示 "N/A"

### API 响应对比

#### 签名验证 (SigNet - 欧氏距离)
```json
{
  "algorithm": "SigNet[classical]",
  "similarity": 1.0,
  "euclidean_distance": 0.0,      // ✅ 有值
  "ssim": 0.9266,
  "signet_pipeline": "classical",
  "confidence": "high"
}
```

#### 印章验证 (CLIP - 余弦相似度)
```json
{
  "algorithm": "CLIP",
  "similarity": 0.9959,            // 99.59% (即你看到的99.6%)
  "euclidean_distance": null,      // ✅ null → 前端显示N/A
  "ssim": null,
  "signet_pipeline": null,
  "confidence": "high"
}
```

## 为什么这样设计？

### SigNet 使用欧氏距离
- **原因**: 签名验证需要精确匹配笔迹细节
- **特点**: 
  - 对像素级差异敏感
  - 距离越小越相似
  - 0 表示完全相同

### CLIP 使用余弦相似度
- **原因**: 印章验证关注整体视觉语义
- **特点**:
  - 对图像整体理解好
  - 抗缩放和轻微变换
  - 1 表示完全相同

## 数学公式

### 余弦相似度
$$
\text{cosine\_similarity} = \frac{A \cdot B}{||A|| \times ||B||} = \cos(\theta)
$$

其中：
- $A, B$ 是两个特征向量
- $||A||, ||B||$ 是向量的L2范数
- $\theta$ 是向量间的夹角

**值域**: $[-1, 1]$
- $1$ = 完全相同（夹角0°）
- $0$ = 无关（夹角90°）
- $-1$ = 完全相反（夹角180°）

### 欧氏距离
$$
d = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}
$$

**值域**: $[0, +\infty)$
- $0$ = 完全相同
- 越大越不同

## 总结

| 项目 | SigNet (签名) | CLIP (印章) |
|------|--------------|-------------|
| **度量方式** | 欧氏距离 | 余弦相似度 ✅ |
| **欧氏距离字段** | 有值 (如 0.0) | null (显示N/A) ✅ |
| **相似度范围** | 0-1 | 0-1 |
| **适用场景** | 笔迹细节匹配 | 整体视觉相似 |
| **抗变换能力** | 低 | 高 |

## 前端显示逻辑

**代码**: `frontend/app.js` (推测)

```javascript
if (euclideanDist !== null && euclideanDist !== undefined) {
    element.textContent = euclideanDist.toFixed(4);
} else {
    element.textContent = 'N/A';  // ✅ CLIP时会走这里
}
```

**结论**: 
- 印章验证使用 **CLIP + 余弦相似度**
- 欧氏距离显示 **N/A 是正确的**
- 99.6% 相似度来自余弦相似度计算

---
**日期**: 2025-10-08  
**状态**: ✅ 已验证  
**详细文档**: `SIMILARITY_METRICS.md`
