# 签名清洁图片展示功能 - 部署完成

## ✅ 已完成的功能

### 🎨 前端展示
- ✅ 在验证结果区域添加了"签名清洁效果对比"板块
- ✅ 左右并排展示模板和查询图片的清洁结果
- ✅ 显示当前使用的清洁模式（conservative/aggressive）
- ✅ 智能显示：只在签名验证且启用清洁时显示
- ✅ 响应式布局，图片自适应大小

### 🔧 后端支持
- ✅ 添加静态文件服务：`/uploaded_samples` 路径
- ✅ 自动保存清洁后图片到 `debug/` 目录
- ✅ API 响应包含 `debug_images` 字段
- ✅ CORS 配置支持跨域访问图片

### 📊 展示效果

#### 验证结果页面新增区域：

```
┌─────────────────────────────────────────────┐
│         验证结果                             │
│  相似度: 98.5%  置信度: HIGH                 │
├─────────────────────────────────────────────┤
│  🤖 算法模型    📊 相似度    📏 欧氏距离     │
│  🧠 SigNet     98.5%        0.0234          │
├─────────────────────────────────────────────┤
│  🧹 签名清洁效果对比                         │
│  ┌───────────────┐  ┌───────────────┐      │
│  │ 📋 模板清洁    │  │ 🔎 查询清洁    │      │
│  │    结果       │  │    结果       │      │
│  │  ┌─────────┐ │  │  ┌─────────┐ │      │
│  │  │         │ │  │  │         │ │      │
│  │  │ [图片]  │ │  │  │ [图片]  │ │      │
│  │  │         │ │  │  │         │ │      │
│  │  └─────────┘ │  │  └─────────┘ │      │
│  └───────────────┘  └───────────────┘      │
│  ℹ️ 清洁模式: conservative                  │
│  清洁功能已自动去除背景杂质，保留纯净签名笔画 │
└─────────────────────────────────────────────┘
```

---

## 🎯 使用方法

### 方式1: 通过前端界面（推荐）

1. **打开浏览器**
   ```
   http://localhost:3000
   ```

2. **上传签名图片**
   - 选择"手写签名"模式
   - 上传模板签名
   - 上传查询签名
   - （可选）使用裁剪工具精确裁剪

3. **开始验证**
   - 点击"开始验证"按钮
   - 等待1-2秒处理

4. **查看结果**
   - 向下滚动到"验证结果"区域
   - 查看相似度、欧氏距离等指标
   - **查看"签名清洁效果对比"**
     - 左图：模板清洁后的结果
     - 右图：查询清洁后的结果
     - 对比原图，看杂质是否被移除

### 方式2: 通过 API 调用

```bash
curl -X POST http://localhost:8000/api/verify \
  -F "template_image=@template.png" \
  -F "query_image=@query.png" \
  -F "verification_type=signature" \
  -F "enable_clean=true" \
  -F "clean_mode=conservative"
```

**响应示例**:
```json
{
  "success": true,
  "similarity": 0.9856,
  "euclidean_distance": 0.0234,
  "clean_enabled": true,
  "clean_mode": "conservative",
  "debug_images": {
    "template": "debug/template_cleaned_20251029_110231.png",
    "query": "debug/query_cleaned_20251029_110231.png"
  }
}
```

**访问清洁后的图片**:
```
http://localhost:8000/uploaded_samples/debug/template_cleaned_20251029_110231.png
http://localhost:8000/uploaded_samples/debug/query_cleaned_20251029_110231.png
```

---

## 📂 文件位置

### 前端代码
- **HTML**: `frontend/index.html`
  - 第 568-592 行：清洁图片展示区域

- **JavaScript**: `frontend/app.js`
  - `displayResult()` 函数：处理清洁图片显示逻辑

### 后端代码
- **API**: `backend/main.py`
  - 第 32 行：静态文件服务挂载
  - `compute_signet_similarity()` 函数：保存清洁后图片

### 存储位置
- **清洁后图片**: `backend/uploaded_samples/debug/`
  - 格式: `{template|query}_cleaned_YYYYMMDD_HHMMSS.png`
  - 自动生成，验证后保存

---

## 🔍 对比要点

### 清洁效果检查清单

查看清洁后的图片，确认：

- [ ] **表格线被移除** - 原图中的横线、竖线应消失
- [ ] **背景噪点被清除** - 灰色点、污渍应消失
- [ ] **签名笔画完整** - 所有签名笔画应保留
- [ ] **笔画无断裂** - 笔画不应因清洁而断开
- [ ] **图像纯净** - 只有黑色笔画 + 白色背景
- [ ] **中文笔画保护** - "点"、"撇"、"捺"等细小笔画保留

### 对比示例

**原始图片**:
```
╔═══════════════╗
║ ┌───┬───┬───┐║
║ │   │张三│   │║ ← 有表格线
║ ├───┼───┼───┤║
║ │   │    │   │║
║ └───┴───┴───┘║
║ ··· 噪点 ···  ║ ← 有背景噪点
╚═══════════════╝
```

**清洁后图片**:
```
╔═══════════════╗
║               ║
║      张三     ║ ← 只有签名
║               ║
║               ║
║               ║ ← 纯白背景
╚═══════════════╝
```

---

## 🐛 故障排查

### 问题1: 看不到清洁图片

**症状**: 验证完成后，没有显示"签名清洁效果对比"区域

**排查步骤**:

1. **检查是否为签名验证**
   ```
   验证类型必须是"手写签名"，图章不会显示清洁图片
   ```

2. **检查清洁是否启用**
   ```javascript
   // 在浏览器控制台查看
   console.log(result.clean_enabled);  // 应该是 true
   console.log(result.debug_images);   // 应该有值
   ```

3. **检查后端日志**
   ```bash
   tail -f backend/backend.log
   # 查找类似输出：
   # ⚠️ 保存调试图像失败: xxx  (如果有错误)
   ```

4. **检查 debug 目录**
   ```bash
   ls -lh backend/uploaded_samples/debug/
   # 应该能看到最新生成的 png 文件
   ```

### 问题2: 图片显示 404

**症状**: 清洁图片区域显示，但图片加载失败

**排查步骤**:

1. **检查静态文件服务**
   ```bash
   curl http://localhost:8000/uploaded_samples/debug/
   # 应该返回目录列表或文件
   ```

2. **检查文件路径**
   ```javascript
   // 在浏览器控制台查看
   const img = document.getElementById('templateCleanedImage');
   console.log(img.src);
   // 应该是: http://localhost:8000/uploaded_samples/debug/xxx.png
   ```

3. **手动访问图片**
   ```
   直接在浏览器打开图片URL，看是否能访问
   ```

### 问题3: 图片是空白的

**症状**: 图片能加载，但显示全白或全黑

**原因**: 清洁算法过度处理，删除了所有内容

**解决方案**:

1. **降低清洁强度**
   ```python
   # 在 backend/preprocess/auto_crop.py 中调整
   if area < 5:  # 改为 3，保留更多小笔画
   ```

2. **关闭清洁功能测试**
   ```bash
   curl -X POST http://localhost:8000/api/verify \
     -F "enable_clean=false"  # 对比效果
   ```

---

## 📊 技术实现

### 前端关键代码

```javascript
// frontend/app.js - displayResult() 函数
if (result.type === 'signature' && result.debug_images && result.clean_enabled) {
    const backendUrl = 'http://localhost:8000';
    const templatePath = result.debug_images.template;
    const queryPath = result.debug_images.query;
    
    document.getElementById('templateCleanedImage').src = 
        `${backendUrl}/uploaded_samples/${templatePath}`;
    document.getElementById('queryCleanedImage').src = 
        `${backendUrl}/uploaded_samples/${queryPath}`;
    
    cleanedComparison.style.display = 'block';
}
```

### 后端关键代码

```python
# backend/main.py - 静态文件服务
app.mount("/uploaded_samples", 
          StaticFiles(directory="uploaded_samples"), 
          name="uploaded_samples")

# backend/main.py - compute_signet_similarity()
debug_dir = 'uploaded_samples/debug'
os.makedirs(debug_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
cv2.imwrite(f'{debug_dir}/template_cleaned_{timestamp}.png', 
            template_processed)
result['debug_images'] = {
    'template': f'debug/template_cleaned_{timestamp}.png',
    'query': f'debug/query_cleaned_{timestamp}.png'
}
```

---

## 🎯 下一步优化建议

1. **添加图片对比滑块**
   - 左右拖动查看原图/清洁图对比

2. **添加放大查看功能**
   - 点击图片可以放大查看细节

3. **添加下载按钮**
   - 一键下载清洁后的图片

4. **添加历史记录**
   - 保存每次验证的清洁图片

5. **添加清洁参数调节**
   - UI 界面调整清洁强度

---

## ✅ 测试检查清单

部署后的测试：

- [x] 服务启动成功
- [x] 前端可以访问
- [x] 上传签名图片成功
- [x] 验证功能正常
- [ ] 清洁图片区域显示 ← **需要你测试**
- [ ] 图片能正常加载 ← **需要你测试**
- [ ] 清洁效果符合预期 ← **需要你测试**
- [ ] 表格线被移除 ← **需要你测试**
- [ ] 签名笔画完整 ← **需要你测试**

---

## 📞 支持

遇到问题请查看：
- 📚 完整文档: `SIGNATURE_CLEAN_GUIDE.md`
- 🎯 快速参考: `SIGNATURE_CLEAN_QUICKREF.md`
- 🧪 测试指南: `test_clean_display.sh`

---

**部署时间**: 2025-10-29  
**状态**: ✅ 已部署并运行  
**访问地址**: http://localhost:3000
