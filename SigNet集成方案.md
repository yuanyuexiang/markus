# SigNet签名验证系统集成方案

## 📋 方案概述

**目标**: 将专业的SigNet签名验证模型集成到现有系统,替代CLIP解决93.9%误判问题

**架构**:
- 签名验证 → **SigNet** (专业模型)
- 印章验证 → **CLIP** (保持不变)

## 🔍 选择的模型: SigNet (sigver_wiwd)

**仓库**: https://github.com/luizgh/sigver_wiwd

**优势**:
1. ✅ **专门为离线签名验证设计**
2. ✅ **预训练模型开箱即用** (在GPDS数据集上训练)
3. ✅ **提供TensorFlow和Theano两种实现**
4. ✅ **准确率高** - 在多个数据集上验证
5. ✅ **支持多种尺寸** (SigNet-SPP模型)
6. ✅ **MIT开源协议**

**论文**:
- [1] Learning Features for Offline Handwritten Signature Verification using Deep Convolutional Neural Networks
- [2] Fixed-sized representation learning from Offline Handwritten Signatures

## 🛠️ 实施步骤

### 第1步: 安装依赖

```bash
cd /Users/yuanyuexiang/Desktop/workspace/markus/backend

# 安装SigNet依赖
venv/bin/pip install tensorflow==2.8.0  # 或使用最新兼容版本
venv/bin/pip install scipy
venv/bin/pip install six
venv/bin/pip install pillow
```

### 第2步: 下载预训练模型

```bash
# 创建models目录
mkdir -p models

# 下载SigNet模型
# 方式1: 手动从Google Drive下载
# https://drive.google.com/file/d/1KffsnZu8-33wXklsodofw-a-KX6tAsVN/view?usp=share_link

# 或使用gdown自动下载
venv/bin/pip install gdown
venv/bin/python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1KffsnZu8-33wXklsodofw-a-KX6tAsVN', 'models/signet.pkl', quiet=False)"
```

### 第3步: 复制SigNet核心代码

从仓库复制以下文件到backend目录:
- `tf_signet.py` - TensorFlow模型定义
- `tf_cnn_model.py` - CNN模型封装
- `lasagne_to_tf.py` - 权重转换工具
- `preprocess/normalize.py` - 签名预处理

### 第4步: 修改backend/main.py

```python
# 在文件开头添加SigNet导入
import tensorflow as tf
from tf_signet import build_architecture
from tf_cnn_model import TF_CNNModel
from preprocess.normalize import preprocess_signature
import tf_signet

# 全局变量存储模型
signet_model = None
signet_session = None

def load_signet_model():
    """加载SigNet模型(只在启动时加载一次)"""
    global signet_model, signet_session
    
    if signet_model is None:
        print("📦 加载SigNet签名验证模型...")
        model_weight_path = 'models/signet.pkl'
        signet_model = TF_CNNModel(tf_signet, model_weight_path)
        
        signet_session = tf.Session()
        signet_session.run(tf.global_variables_initializer())
        print("✅ SigNet模型加载完成")
    
    return signet_model, signet_session

def compute_signature_similarity_signet(template_img, query_img):
    """使用SigNet计算签名相似度"""
    model, sess = load_signet_model()
    
    # SigNet要求的画布大小
    canvas_size = (952, 1360)
    
    # 转换PIL Image为numpy array
    template_np = np.array(template_img.convert('L'))
    query_np = np.array(query_img.convert('L'))
    
    # 预处理签名
    template_processed = preprocess_signature(template_np, canvas_size)
    query_processed = preprocess_signature(query_np, canvas_size)
    
    # 提取特征向量
    template_features = model.get_feature_vector(sess, template_processed)
    query_features = model.get_feature_vector(sess, query_processed)
    
    # 计算欧氏距离
    euclidean_dist = np.linalg.norm(template_features - query_features)
    
    # 转换为相似度分数 (距离越小,相似度越高)
    # SigNet论文中阈值约为0.145
    # 我们将距离转换为0-100的相似度分数
    similarity = max(0, 100 - euclidean_dist * 50)  # 可调整系数
    
    return similarity / 100.0  # 返回0-1之间的值

# 修改verify_images函数
@app.post("/api/verify")
async def verify_images(
    template: UploadFile = File(...),
    query: UploadFile = File(...),
    type: str = Form(...)
):
    try:
        template_img = Image.open(template.file).convert("RGB")
        query_img = Image.open(query.file).convert("RGB")
        
        # 自动保存(保持不变)
        save_dir = "uploaded_samples"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        template_path = f"{save_dir}/{type}_template_{timestamp}.png"
        query_path = f"{save_dir}/{type}_query_{timestamp}.png"
        template_img.save(template_path)
        query_img.save(query_path)
        
        # 根据类型选择算法
        if type == "signature":
            # 签名用SigNet
            similarity = compute_signature_similarity_signet(template_img, query_img)
            threshold = 0.80  # SigNet的阈值需要根据实际数据调整
        else:
            # 印章继续用CLIP
            similarity = compute_clip_similarity(template_img, query_img)
            threshold = 0.88
        
        is_match = similarity >= threshold
        
        return {
            "similarity": float(similarity),
            "threshold": float(threshold),
            "match": bool(is_match),
            "algorithm": "SigNet" if type == "signature" else "CLIP"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 第5步: 更新前端显示

修改`frontend/index.html`和`frontend/app.js`:

```javascript
// 在显示结果时添加算法信息
result.innerHTML = `
    <div class="result-item">
        <span>算法:</span>
        <span>${data.algorithm}</span>
    </div>
    <div class="result-item">
        <span>CLIP相似度:</span>
        <span>${(data.similarity * 100).toFixed(1)}%</span>
    </div>
    <div class="result-item">
        <span>阈值:</span>
        <span>${(data.threshold * 100).toFixed(1)}%</span>
    </div>
    <div class="result-item ${data.match ? 'match' : 'no-match'}">
        <span>结果:</span>
        <span>${data.match ? '✅ 通过' : '❌ 拒绝'}</span>
    </div>
`;
```

## 📊 阈值调整策略

**初始阈值**:
- 签名(SigNet): 0.80 (需根据实际数据调整)
- 印章(CLIP): 0.88

**调整方法**:
1. 收集100+真实样本对
2. 运行analyze_samples.py分析
3. 根据ROC曲线确定最佳阈值
4. 目标: FAR(误接受率) < 5%, FRR(误拒绝率) < 10%

## 🔬 测试验证

### 测试1: 验证SigNet安装

```bash
cd backend
venv/bin/python << EOF
from scipy.misc import imread
from preprocess.normalize import preprocess_signature
import tf_signet
from tf_cnn_model import TF_CNNModel
import tensorflow as tf

# 加载模型
model = TF_CNNModel(tf_signet, 'models/signet.pkl')
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("✅ SigNet模型加载成功")
EOF
```

### 测试2: 对比CLIP vs SigNet

```python
# 使用之前93.9%误判的样本测试
template = "uploaded_samples/signature_template_20251007_033711.png"
query = "uploaded_samples/signature_query_20251007_033711.png"

# CLIP结果: 93.9% (误判)
# SigNet结果: ? (期望 < 80%)
```

## 📈 预期效果

**CLIP的问题**:
- 两个不同签名: 93.9% 相似度 ❌
- 无法区分笔画细节

**SigNet的优势**:
- 专门学习签名特征
- 对笔画、压力、动态特征敏感
- 期望不同签名 < 70% 相似度 ✅

## 🚀 部署流程

```bash
# 1. 安装依赖
cd /Users/yuanyuexiang/Desktop/workspace/markus/backend
venv/bin/pip install tensorflow scipy six pillow gdown

# 2. 下载模型
mkdir -p models
venv/bin/python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1KffsnZu8-33wXklsodofw-a-KX6tAsVN', 'models/signet.pkl', quiet=False)"

# 3. 复制SigNet代码
# 从 https://github.com/luizgh/sigver_wiwd 下载:
# - tf_signet.py
# - tf_cnn_model.py  
# - lasagne_to_tf.py
# - preprocess/normalize.py

# 4. 更新backend/main.py (按上述代码修改)

# 5. 重启服务
cd /Users/yuanyuexiang/Desktop/workspace/markus
./restart_services_v2.sh

# 6. 测试
# 上传之前93.9%误判的样本,查看SigNet结果
```

## 🔄 备选方案

如果SigNet集成遇到困难,可以考虑:

1. **简化方案**: 使用预计算的SigNet特征 (从论文提供的数据集)
2. **PyTorch版本**: 使用 https://github.com/luizgh/sigver (训练代码)
3. **轻量级方案**: 传统CV特征(Hu矩 + ORB + 笔画方向)

## 📝 注意事项

1. **TensorFlow版本**: 使用TF 2.x,需要启用兼容模式
2. **预处理一致性**: 确保图像预处理与训练时一致
3. **模型加载**: 只在启动时加载一次,避免重复加载
4. **阈值标定**: 使用真实数据确定最佳阈值
5. **印章继续用CLIP**: 印章视觉差异明显,CLIP够用

## 🎯 成功标准

- ✅ 不同签名相似度 < 70%
- ✅ 相同签名相似度 > 85%
- ✅ 93.9%误判问题解决
- ✅ 系统稳定运行

---

**下一步**: 开始安装依赖并下载模型!
