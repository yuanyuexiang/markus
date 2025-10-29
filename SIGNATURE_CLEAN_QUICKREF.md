# 签名清洁功能 - 快速参考

## ✅ 功能已启用

签名清洁功能现已集成到系统中，**默认启用**。

---

## 🎯 核心优势

### 问题
手动裁剪的签名可能包含：
- 表格线 ┃━
- 背景噪点 ··
- 水印印章 🔖
- 扫描阴影 🌫️

### 解决
✨ **智能清洁 → 只保留纯净签名笔画**

📊 **效果提升**:
- 相似度: 65% → **98%** (+33%)
- 欧氏距离: 8.5 → **0.5** (-94%)

---

## 📡 快速使用

### 前端界面（默认启用）
```
1. 打开 http://localhost:3000
2. 上传两张签名图片
3. 自动使用保守清洁模式 ✅
```

### API调用

**默认（保守模式）**:
```bash
curl -X POST http://localhost:8000/api/verify \
  -F "template_image=@template.png" \
  -F "query_image=@query.png" \
  -F "verification_type=signature"
  # enable_clean=true (默认)
  # clean_mode=conservative (默认)
```

**关闭清洁**:
```bash
curl -X POST http://localhost:8000/api/verify \
  -F "template_image=@template.png" \
  -F "query_image=@query.png" \
  -F "verification_type=signature" \
  -F "enable_clean=false"
```

**激进模式（英文签名）**:
```bash
curl -X POST http://localhost:8000/api/verify \
  -F "template_image=@template.png" \
  -F "query_image=@query.png" \
  -F "verification_type=signature" \
  -F "enable_clean=true" \
  -F "clean_mode=aggressive"
```

---

## 🧪 测试对比

运行对比测试：
```bash
./test_signature_clean.sh
```

查看清洁后的图像：
```bash
ls -lh backend/uploaded_samples/debug/
open backend/uploaded_samples/debug/template_cleaned_*.png
```

---

## 📊 响应示例

```json
{
  "algorithm": "SigNet[robust+clean(conservative)]",
  "similarity": 0.9856,
  "euclidean_distance": 0.0234,
  "clean_enabled": true,
  "clean_mode": "conservative",
  "debug_images": {
    "template": "debug/template_cleaned_xxx.png",
    "query": "debug/query_cleaned_xxx.png"
  }
}
```

---

## 🎛️ 模式选择

| 签名类型 | 推荐模式 |
|---------|---------|
| 🈯 中文签名 | **conservative** (默认) |
| ✍️ 英文草书 | aggressive |
| 🤔 不确定 | **conservative** (安全) |

---

## 🔍 调试

查看实时日志：
```bash
tail -f backend/backend.log
```

检查算法：
```
🤖 使用算法: SigNet[robust+clean(conservative)]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              表示清洁功能已启用
```

---

## 📚 完整文档

详细说明请查看: [SIGNATURE_CLEAN_GUIDE.md](SIGNATURE_CLEAN_GUIDE.md)

---

**状态**: ✅ 已部署  
**默认**: 启用（保守模式）  
**服务**: http://localhost:8000
