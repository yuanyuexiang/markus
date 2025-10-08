# 相似度度量方式说明

## 算法对比

### 📐 SigNet (签名验证)
- **度量方式**: 欧氏距离 (Euclidean Distance)
- **特征**: 2048维向量
- **计算公式**: 
  ```python
  distance = np.linalg.norm(feat1 - feat2)
  similarity = np.exp(-distance / 0.15)
  ```
- **输出指标**:
  - ✅ 相似度 (0-1)
  - ✅ 欧氏距离 (数值)
  - ✅ SSIM 结构相似度
  - ✅ 预处理管线类型

### 🎨 CLIP (印章验证)
- **度量方式**: 余弦相似度 (Cosine Similarity)
- **特征**: L2归一化的嵌入向量
- **计算公式**:
  ```python
  # L2归一化
  feat1 = feat1 / feat1.norm()
  feat2 = feat2 / feat2.norm()
  # 余弦相似度
  similarity = F.cosine_similarity(feat1, feat2)
  ```
- **输出指标**:
  - ✅ 相似度 (0-1)
  - ❌ 欧氏距离 (不适用，显示N/A)

## 为什么不同？

### SigNet - 欧氏距离
- **优势**: 对笔迹细节敏感，能捕捉细微差异
- **适用**: 签名验证（同一个人手写签名的微小变化）
- **特点**: 距离越小越相似，0为完全相同

### CLIP - 余弦相似度  
- **优势**: 对图像整体语义理解好，抗几何变换
- **适用**: 印章图章验证（更关注整体视觉相似性）
- **特点**: 值域[-1, 1]，1为完全相同，0为无关，-1为完全相反

## API 响应差异

### 签名验证响应
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

### 印章验证响应
```json
{
  "algorithm": "CLIP",
  "similarity": 0.996,
  "euclidean_distance": null,     // ❌ null (前端显示N/A)
  "ssim": null,
  "signet_pipeline": null,
  "confidence": "high"
}
```

## 前端显示逻辑

当前前端代码：
```javascript
// 欧氏距离显示
if (euclideanDist !== null && euclideanDist !== undefined) {
    document.getElementById('euclidean-distance').textContent = euclideanDist.toFixed(4);
} else {
    document.getElementById('euclidean-distance').textContent = 'N/A';
}
```

**结论**: 印章使用CLIP时，欧氏距离显示"N/A"是**正常且正确**的行为，因为CLIP不使用欧氏距离。

## 相似度值对比

| 验证类型 | 算法 | 度量方式 | 典型阈值 | 相似度范围 |
|---------|------|---------|---------|-----------|
| 签名 | SigNet | 欧氏距离 | 0.75 | 0-1 (距离→相似度转换) |
| 印章 | CLIP | 余弦相似度 | 0.88 | 0-1 (直接输出) |

## 数学原理

### 欧氏距离 (L2 Distance)
$$d = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$
- 衡量向量空间中的**绝对距离**
- 对特征值的**绝对大小**敏感

### 余弦相似度 (Cosine Similarity)
$$\text{similarity} = \frac{A \cdot B}{||A|| \times ||B||} = \cos(\theta)$$
- 衡量向量的**方向相似性**
- 对特征值的**缩放不敏感**
- L2归一化后，余弦相似度 = 1 - 欧氏距离²/2

## 推荐实践

### 何时查看欧氏距离
- ✅ 签名验证时（SigNet）
- ✅ 需要精确匹配笔迹细节时
- ✅ 调试预处理效果时

### 何时忽略欧氏距离
- ❌ 印章验证时（CLIP，显示N/A）
- ❌ 使用余弦相似度的场景
- ❌ 关注整体视觉相似性时

---
**版本**: v2.1  
**更新时间**: 2025-10-08  
**状态**: ✅ 正常工作
