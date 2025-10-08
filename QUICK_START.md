# 🚀 5分钟快速上手指南

## 📋 前置条件

- ✅ Mac 电脑 (已测试)
- ✅ Python 3.12 (已安装)
- ✅ 网络连接 (已完成依赖安装)

## 🎯 立即开始

### 第一步: 启动系统 (10秒)

**选项A: 一键启动**
```bash
cd /Users/yuanyuexiang/Desktop/workspace/markus
./start.sh
```

**选项B: 手动启动** (如果一键启动失败)
```bash
# 终端1 - 后端
cd /Users/yuanyuexiang/Desktop/workspace/markus/backend
venv/bin/python main.py

# 终端2 - 前端
cd /Users/yuanyuexiang/Desktop/workspace/markus/frontend  
python3 -m http.server 3000
```

### 第二步: 打开系统 (5秒)

浏览器访问: **http://localhost:3000**

### 第三步: 开始验证 (1分钟)

#### 🧪 使用测试图片快速验证

**验证签名**:
1. 点击 "📝 手写签名验证"
2. 上传模板图片: `test_images/signature_template.png`
3. 上传待验证图片: `test_images/signature_real.png`
4. 使用 "⬜ 矩形" 工具框选签名区域
5. 点击 "✂️ 裁剪" (两张图都要裁剪)
6. 点击 "🚀 开始验证"
7. 查看结果 - 应该显示 ✅ 高相似度

**验证图章**:
1. 点击 "🔴 印章图章验证"  
2. 上传模板图片: `test_images/seal_template.png`
3. 上传待验证图片: `test_images/seal_real.png`
4. 使用 "⭕ 圆形" 工具圈选图章
5. 点击 "✂️ 裁剪" (两张图都要裁剪)
6. 点击 "🚀 开始验证"
7. 查看结果 - 应该显示 ✅ 高相似度

#### 📸 使用自己的图片

1. 准备两张图片:
   - 模板图片 (已知真实的签名/图章)
   - 待验证图片 (需要验证真伪的签名/图章)

2. 拖拽上传到对应位置

3. 裁剪技巧:
   - **签名**: 矩形框选,包含完整签名,留一点边距
   - **图章**: 圆形圈选,完整包含印章

4. 裁剪后点击"开始验证"

5. 结果判读:
   - 相似度 > 78% (签名) / 82% (图章) = 可能真实
   - 置信度 High = 可信度高
   - 置信度 Medium = 建议人工复审
   - 置信度 Low = 必须专家复审

## 📊 测试结果示例

### ✅ 应该通过的测试
```
模板: signature_template.png
验证: signature_real.png
预期结果: 
  - 相似度: 90-95%
  - 置信度: High
  - 判断: ✅ 可能为真实
```

### ❌ 应该拒绝的测试
```
模板: signature_template.png  
验证: signature_fake.png
预期结果:
  - 相似度: 20-40%
  - 置信度: Low
  - 判断: ❌ 可能为伪造
```

## 🛑 停止系统

```bash
cd /Users/yuanyuexiang/Desktop/workspace/markus
./stop.sh
```

或者直接关闭终端窗口即可。

## ⚡ 常见问题

**Q: 验证速度慢怎么办?**
A: Mac M1/M2 约200-300ms, Intel Mac 约400-500ms是正常的。如果超过1秒，请检查CPU占用。

**Q: 准确率不高怎么办?**
A: 
1. 确保图片清晰、光照均匀
2. 裁剪时包含完整签名/图章
3. 零样本学习准确率85-92%已是极限
4. 低置信度时请人工复审

**Q: 裁剪不准怎么办?**
A: 点击 "🔄 重置" 重新选择区域

**Q: 无法连接后端?**
A: 检查 http://localhost:8000 是否可访问，确认后端已启动

## 📚 更多信息

- 完整文档: `README.md`
- 项目状态: `PROJECT_STATUS.md`
- 技术方案: `签名图章验证系统-轻量化方案.md`

## 🎉 现在开始验证吧！

**你已经准备好了！访问 http://localhost:3000 开始你的第一次验证！**

---

💡 **小提示**: 
- 第一次验证时CLIP模型需要加载,可能需要等待5-10秒
- 后续验证会更快
- 建议使用Chrome或Safari浏览器以获得最佳体验
