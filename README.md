# 签名图章验证系统

基于 SigNet + CLIP 的签名与图章智能验证系统。

## 快速开始

### 1. 启动服务

```bash
./start_services.sh
```

服务地址：
- 🎨 前端界面: http://localhost:3000
- 🔧 后端API: http://localhost:8000
- 📚 API文档: http://localhost:8000/docs

### 2. 停止服务

```bash
./stop_services.sh
```

### 3. 查看日志

```bash
# 后端日志
tail -f backend/backend.log

# 前端日志
tail -f frontend/frontend.log
```

## 功能特性

### ✅ 签名验证 (SigNet)
- 使用专业深度学习模型 SigNet
- 支持手工裁剪的小图片
- 双路径预处理自动选择最优结果
- SSIM 结构相似度辅助判断
- 欧氏距离 + 相似度百分比展示

### ✅ 图章验证 (CLIP)
- 基于 OpenAI CLIP 视觉-语言模型
- 余弦相似度计算
- 高精度印章匹配

### ✅ 智能评估
- 自动置信度评估 (HIGH/MEDIUM/LOW)
- 智能推荐建议
- 实时处理时间统计

## 技术栈

### 后端
- FastAPI (Python Web框架)
- TensorFlow 2.x (SigNet模型)
- PyTorch + CLIP (印章验证)
- OpenCV (图像预处理)

### 前端
- 原生 JavaScript + HTML5 Canvas
- 拖拽上传 + 手绘签名
- 响应式设计

## 项目结构

```
markus/
├── backend/                 # 后端服务
│   ├── main.py             # FastAPI主程序
│   ├── signet_model.py     # SigNet模型定义
│   ├── preprocess/         # 预处理模块
│   │   ├── normalize.py    # 传统预处理
│   │   └── auto_crop.py    # 鲁棒自动裁剪
│   ├── models/             # 模型文件
│   │   └── signet.pkl      # SigNet权重
│   └── venv/               # Python虚拟环境
├── frontend/               # 前端界面
│   ├── index.html          # 主页面
│   └── app.js              # 前端逻辑
├── start_services.sh       # 启动脚本
├── stop_services.sh        # 停止脚本
├── BUG_FIX_REPORT.md      # Bug修复报告
└── SERVICE_STATUS.md       # 服务状态文档
```

## 最近更新 (v2.1)

### 🐛 Bug修复
修复了 `preprocess/normalize.py` 中二值化逻辑错误导致的手工裁剪签名相似度为0%的问题。

**修复前**:
```python
r, c = np.where(binarized_image == 0)  # ❌ 查找背景
```

**修复后**:
```python
r, c = np.where(binarized_image)       # ✅ 查找前景
```

**效果对比**:
- 同一签名相似度: 0% → **100%** ✅
- 欧氏距离: 19.55 → **0.0** ✅
- 置信度: LOW → **HIGH** ✅

### ✨ 新增功能
1. 双路径预处理策略（自动选择最优）
2. SSIM 结构相似度辅助
3. 增强的API响应字段（`ssim`, `signet_pipeline`）
4. 完整的测试脚本和文档

详见: [BUG_FIX_REPORT.md](BUG_FIX_REPORT.md)

## API使用示例

### 验证签名

```bash
curl -X POST http://localhost:8000/api/verify \
  -F "template_image=@template.png" \
  -F "query_image=@query.png" \
  -F "verification_type=signature"
```

### 响应示例

```json
{
  "success": true,
  "type": "signature",
  "algorithm": "SigNet[classical]",
  "similarity": 1.0,
  "euclidean_distance": 0.0,
  "ssim": 0.9266,
  "signet_pipeline": "classical",
  "final_score": 1.0,
  "confidence": "high",
  "is_authentic": true,
  "threshold": 0.75,
  "recommendation": "高置信度通过 - 签名高度相似，可自动接受",
  "processing_time_ms": 8855.38
}
```

## 测试脚本

```bash
# 测试修复效果
./test_fix.sh

# 测试SigNet vs CLIP
./test_signet_vs_clip.sh

# 测试同一签名
./test_same_image.sh

# 测试不同签名
./test_different_signatures.sh
```

## 环境要求

- Python 3.8+
- Node.js (可选，仅用于某些开发工具)
- macOS / Linux (推荐)

## 安装依赖

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 开发建议

### 调整阈值
编辑 `backend/main.py`:
```python
threshold = 0.75  # SigNet阈值，可根据实际数据调整
```

### 添加新的预处理方法
在 `backend/preprocess/` 目录下创建新模块。

### 自定义置信度规则
修改 `backend/main.py` 中的置信度评估逻辑。

## 常见问题

### Q: 相似度总是很低？
A: 确保上传的是灰度图或清晰的签名图片，避免过多背景噪声。

### Q: 服务启动失败？
A: 检查端口8000和3000是否被占用，查看日志文件排查错误。

### Q: CLIP模型加载慢？
A: 首次加载需要下载模型，请耐心等待或使用缓存的模型文件。

## 许可证

MIT License

## 贡献者

欢迎提交 Issue 和 Pull Request！

---
**版本**: v2.1  
**更新日期**: 2025-10-08  
**状态**: ✅ 生产就绪
