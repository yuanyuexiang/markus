"""
签名图章验证系统 - 后端API
签名: SigNet专业模型
印章: CLIP图像相似度
"""
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import clip
import torch
from PIL import Image
import io
import cv2
import numpy as np
from typing import Literal
import torch.nn.functional as F
import time
import os
from datetime import datetime

app = FastAPI(title="签名图章验证系统")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GNN验证器延迟加载
gnn_verifier = None

def load_gnn_verifier():
    """延迟加载GNN验证器"""
    global gnn_verifier
    if gnn_verifier is not None:
        return gnn_verifier
    
    try:
        import sys
        import os
        # 添加backend目录到Python路径
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
        
        from gnn_verifier import get_gnn_verifier
        gnn_verifier = get_gnn_verifier()
        print("✅ GNN验证器加载完成")
        return gnn_verifier
    except Exception as e:
        import traceback
        print(f"⚠️ GNN验证器加载失败: {e}")
        print(traceback.format_exc())
        return None

# 挂载静态文件服务，用于访问上传的样本和调试图片
app.mount("/uploaded_samples", StaticFiles(directory="uploaded_samples"), name="uploaded_samples")

# 全局加载CLIP模型(印章使用)
print("🔄 正在加载CLIP模型...")
device = "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
print("✅ CLIP模型加载完成")

# SigNet延迟加载(避免TensorFlow启动阻塞)
signet_model = None
_signet_imports = {}

def load_signet_model():
    """延迟加载SigNet模型(首次调用时加载)"""
    global signet_model, _signet_imports
    if signet_model is not None:
        return signet_model
    
    try:
        from signet_model import SigNetModel
        from preprocess.normalize import preprocess_signature
        _signet_imports['SigNetModel'] = SigNetModel
        _signet_imports['preprocess_signature'] = preprocess_signature
        
        signet_model = SigNetModel('models/signet.pkl')
        print("✅ SigNet签名验证模型加载完成")
        return signet_model
    except Exception as e:
        print(f"⚠️ SigNet模型加载失败: {e}")
        return None

def preprocess_for_feature_matching(img_cv: np.ndarray) -> np.ndarray:
    """
    图像预处理用于特征点匹配
    1. 自适应二值化：去除背景噪声
    2. 形态学处理：连接断点，去除小噪点
    """
    # 自适应阈值二值化（对不均匀光照更鲁棒）
    binary = cv2.adaptiveThreshold(
        img_cv, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        blockSize=11,  # 邻域大小
        C=2  # 常数偏移
    )
    
    # 形态学闭运算：连接断开的笔画
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 去除小噪点
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary

def normalize_and_resize(img: np.ndarray, target_size: int = None) -> np.ndarray:
    """
    智能标准化：保持长宽比，自适应选择目标尺寸
    - 小图片（<300px）：放大到至少300px
    - 大图片（>800px）：缩小到800px以内
    - 中等图片：保持原样
    """
    h, w = img.shape[:2]
    max_dim = max(h, w)
    
    # 根据原图尺寸自适应目标尺寸
    if target_size is None:
        if max_dim < 300:
            target_size = 300  # 小图放大
        elif max_dim > 800:
            target_size = 800  # 大图缩小
        else:
            target_size = max_dim  # 中等图保持
    
    # 计算缩放比例
    scale = target_size / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 等比例缩放
    if scale != 1.0:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    else:
        resized = img.copy()
    
    # 创建正方形画布（白色背景）
    canvas = np.ones((target_size, target_size), dtype=np.uint8) * 255
    
    # 将缩放后的图片居中放置
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas, scale

def compute_feature_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """
    特征点匹配已禁用 - 只使用CLIP
    """
    return 0.0  # 不再使用特征点匹配

def compute_ssim_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算结构相似度（SSIM）作为备选方案"""
    try:
        from skimage.metrics import structural_similarity as ssim
        score = ssim(img1, img2)
        return max(0.0, min(1.0, score))
    except:
        # 如果ssim不可用，使用简单的MSE
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        similarity = 1.0 / (1.0 + mse / 1000.0)
        return float(similarity)

def compute_signet_similarity(template_img, query_img, enable_clean=True, clean_mode='conservative'):
    """使用SigNet计算签名相似度，支持签名清洁功能
    
    Args:
        template_img: 模板图像
        query_img: 查询图像
        enable_clean: 是否启用签名清洁
        clean_mode: 清洁模式 'conservative'(中文) 或 'aggressive'(英文)
    
    Returns:
        包含相似度、距离、SSIM、处理流程信息的字典
    """
    model = load_signet_model()
    if model is None:
        return None
    
    try:
        preprocess_signature = _signet_imports['preprocess_signature']
        # 延迟导入增强预处理
        try:
            from preprocess.auto_crop import robust_preprocess, robust_preprocess_with_clean, clean_signature_with_morph
        except Exception as _e:
            robust_preprocess = None
            robust_preprocess_with_clean = None
            clean_signature_with_morph = None
        
        # 转换PIL Image到numpy array (灰度图)
        template_np = np.array(template_img.convert('L'))
        query_np = np.array(query_img.convert('L'))
        
        # 保存高分辨率的清洁图片（用于前端展示）
        template_cleaned_display = None
        query_cleaned_display = None
        
        if enable_clean and clean_signature_with_morph is not None:
            # 对原始图片进行清洁，保持原始尺寸
            template_cleaned_display = clean_signature_with_morph(template_np, mode=clean_mode)
            query_cleaned_display = clean_signature_with_morph(query_np, mode=clean_mode)
            # 反转（清洁后是前景255，需要转成背景255）
            template_cleaned_display = cv2.bitwise_not(template_cleaned_display)
            query_cleaned_display = cv2.bitwise_not(query_cleaned_display)
        
        # 传统预处理路径（回退）
        fallback_template = preprocess_signature(template_np, canvas_size=(952, 1360))
        fallback_query = preprocess_signature(query_np, canvas_size=(952, 1360))

        # 选择增强预处理路径
        if enable_clean and robust_preprocess_with_clean is not None:
            # 使用带清洁的预处理
            t_auto = robust_preprocess_with_clean(template_np, clean_mode=clean_mode)
            q_auto = robust_preprocess_with_clean(query_np, clean_mode=clean_mode)
            pipeline_name = f'robust+clean({clean_mode})'
        elif robust_preprocess is not None:
            # 使用无清洁的鲁棒预处理
            t_auto = robust_preprocess(template_np)
            q_auto = robust_preprocess(query_np)
            pipeline_name = 'robust'
        else:
            t_auto = q_auto = None
            pipeline_name = 'classical'

        auto_valid = (
            t_auto is not None and q_auto is not None and
            t_auto.shape == (150, 220) and q_auto.shape == (150, 220)
        )

        # 计算两条路径的距离
        dist_fallback = model.compute_similarity(fallback_template, fallback_query)
        dist_auto = None
        if auto_valid:
            dist_auto = model.compute_similarity(t_auto, q_auto)

        # 选择更优路径
        if dist_auto is not None and dist_auto <= dist_fallback:
            euclidean_dist = dist_auto
            pipeline = pipeline_name
            ssim_inputs = (t_auto, q_auto)
        else:
            euclidean_dist = dist_fallback
            pipeline = 'classical'
            # SSIM 优先使用自动裁剪结果
            if auto_valid:
                ssim_inputs = (t_auto, q_auto)
            else:
                ssim_inputs = (fallback_template, fallback_query)
        
        # 转换为相似度分数(0-1)
        threshold_dist = 0.15  # SigNet论文阈值
        similarity = np.exp(-euclidean_dist / threshold_dist)

        # 结构相似度辅助
        try:
            ssim_score = compute_ssim_similarity(ssim_inputs[0], ssim_inputs[1])
        except Exception:
            ssim_score = None
        
        result = {
            'similarity': float(similarity),
            'distance': float(euclidean_dist),
            'ssim': float(ssim_score) if ssim_score is not None else None,
            'pipeline': pipeline,
            'clean_enabled': enable_clean,
            'clean_mode': clean_mode if enable_clean else None
        }
        
        # 保存高分辨率清洁图片用于前端展示
        if enable_clean and template_cleaned_display is not None and query_cleaned_display is not None:
            try:
                debug_dir = 'uploaded_samples/debug'
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # 保存原始尺寸的清洁图片
                template_path = f'{debug_dir}/template_cleaned_{timestamp}.png'
                query_path = f'{debug_dir}/query_cleaned_{timestamp}.png'
                
                cv2.imwrite(template_path, template_cleaned_display)
                cv2.imwrite(query_path, query_cleaned_display)
                
                result['debug_images'] = {
                    'template': f'debug/template_cleaned_{timestamp}.png',
                    'query': f'debug/query_cleaned_{timestamp}.png'
                }
                
                print(f"✅ 已保存清洁图片: {template_path} ({template_cleaned_display.shape})")
                print(f"✅ 已保存清洁图片: {query_path} ({query_cleaned_display.shape})")
                
            except Exception as e:
                print(f"⚠️ 保存调试图像失败: {e}")
                import traceback
                traceback.print_exc()
        
        return result
    except Exception as e:
        print(f"⚠️ SigNet处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

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
    
    # 计算余弦相似度
    similarity = float(F.cosine_similarity(template_features, query_features))
    return similarity

@app.post("/api/verify")
async def verify_signature(
    template_image: UploadFile = File(...),
    query_image: UploadFile = File(...),
    verification_type: str = Form(default="signature"),
    algorithm: str = Form(default="signet"),  # 新增: signet, gnn, clip
    enable_clean: bool = Form(default=True),
    clean_mode: str = Form(default="conservative")
):
    """
    验证签名或图章的相似度
    
    算法选项:
    - signet: SigNet专业模型(默认,适合签名)
    - gnn: 图神经网络(基于关键点结构)
    - clip: CLIP视觉模型(适合印章)
    
    参数:
    - algorithm: 验证算法 ('signet', 'gnn', 'clip')
    - enable_clean: 是否启用签名清洁（去除杂质）
    - clean_mode: 清洁模式 'conservative'(中文签名) 或 'aggressive'(英文签名)
    """
    start_time = time.time()

    try:
        # 读取图片并直接转为灰度L
        template_img = Image.open(io.BytesIO(await template_image.read())).convert('L')
        query_img = Image.open(io.BytesIO(await query_image.read())).convert('L')

        # 🔥 保存用户上传的真实裁剪图片
        save_dir = "uploaded_samples"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        template_path = os.path.join(save_dir, f"{verification_type}_template_{timestamp}.png")
        query_path = os.path.join(save_dir, f"{verification_type}_query_{timestamp}.png")

        template_img.save(template_path)
        query_img.save(query_path)
        print(f"💾 已保存样本: {template_path}, {query_path}")
        
        # 根据算法选择计算相似度
        algorithm_used = ""
        euclidean_distance = None
        result = None
        gnn_info = {}
        
        # 强制算法选择逻辑
        if algorithm == "gnn":
            # 使用GNN验证
            print("🧠 使用GNN算法...")
            gnn = load_gnn_verifier()
            if gnn is not None and gnn.model is not None:
                # 转为numpy数组
                template_np = np.array(template_img)
                query_np = np.array(query_img)
                
                # GNN自动使用签名清洁功能 (保守模式适合中文签名)
                try:
                    from preprocess.auto_crop import clean_signature_with_morph
                    # 清洁图片去除噪声
                    template_cleaned = clean_signature_with_morph(template_np, mode='conservative')
                    query_cleaned = clean_signature_with_morph(query_np, mode='conservative')
                    # 反转 (清洁后是前景255,需要转成背景255)
                    template_cleaned = cv2.bitwise_not(template_cleaned)
                    query_cleaned = cv2.bitwise_not(query_cleaned)
                    
                    # 保存清洁后的图片用于调试
                    debug_dir = os.path.join(save_dir, "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    template_clean_path = os.path.join(debug_dir, f"template_cleaned_{timestamp}.png")
                    query_clean_path = os.path.join(debug_dir, f"query_cleaned_{timestamp}.png")
                    cv2.imwrite(template_clean_path, template_cleaned)
                    cv2.imwrite(query_clean_path, query_cleaned)
                    print(f"✅ GNN已保存清洁图片: {template_clean_path}, {query_clean_path}")
                    
                    # 使用清洁后的图片进行GNN验证
                    gnn_result = gnn.verify(template_cleaned, query_cleaned)
                except Exception as clean_error:
                    print(f"⚠️ 签名清洁失败,使用原始图片: {clean_error}")
                    gnn_result = gnn.verify(template_np, query_np)
                
                if 'error' not in gnn_result:
                    # GNN成功
                    similarity = gnn_result['confidence']
                    euclidean_distance = gnn_result['distance']
                    gnn_info = {
                        'keypoints_template': gnn_result['keypoints_template'],
                        'keypoints_query': gnn_result['keypoints_query'],
                        'gnn_distance': gnn_result['distance'],
                        'gnn_threshold': gnn_result['threshold']
                    }
                    algorithm_used = "GNN"
                    threshold = 0.5  # GNN使用confidence阈值
                else:
                    # GNN失败,回退到SigNet
                    print(f"⚠️ GNN失败: {gnn_result['error']}, 回退到SigNet")
                    algorithm = "signet"
            else:
                # GNN未加载,回退到SigNet
                print("⚠️ GNN模型未加载, 回退到SigNet")
                algorithm = "signet"
        
        if algorithm == "signet":
            # 使用SigNet验证（支持清洁功能）
            print("🔬 使用SigNet算法...")
            result = compute_signet_similarity(
                template_img, 
                query_img, 
                enable_clean=enable_clean,
                clean_mode=clean_mode
            )
            if result is not None:
                similarity = result['similarity']
                euclidean_distance = result['distance']
                signature_ssim = result.get('ssim')
                pipeline = result.get('pipeline', 'SigNet')
                clean_info = f"+clean({clean_mode})" if result.get('clean_enabled') else ""
                algorithm_used = f"SigNet[{pipeline}{clean_info}]"
                threshold = 0.92  # SigNet阈值
                if signature_ssim is not None:
                    # SSIM为人工裁剪/轻微变形提供兜底
                    similarity = max(similarity, min(0.98, signature_ssim * 0.95))
            else:
                # SigNet失败,回退到CLIP
                similarity = compute_clip_similarity(template_img, query_img)
                algorithm_used = "CLIP(fallback)"
                threshold = 0.85
        
        elif algorithm == "clip":
            # 使用CLIP验证
            print("🎨 使用CLIP算法...")
            similarity = compute_clip_similarity(template_img, query_img)
            algorithm_used = "CLIP"
            threshold = 0.88 if verification_type == "seal" else 0.85
        
        # 如果还没有设置algorithm_used(说明上面的逻辑没有执行),使用默认
        if not algorithm_used:
            if verification_type == "signature":
                result = compute_signet_similarity(
                    template_img, 
                    query_img, 
                    enable_clean=enable_clean,
                    clean_mode=clean_mode
                )
                if result is not None:
                    similarity = result['similarity']
                    euclidean_distance = result['distance']
                    algorithm_used = "SigNet"
                    threshold = 0.92
                else:
                    similarity = compute_clip_similarity(template_img, query_img)
                    algorithm_used = "CLIP(fallback)"
                    threshold = 0.85
            else:
                # 印章用CLIP
                similarity = compute_clip_similarity(template_img, query_img)
                algorithm_used = "CLIP"
                threshold = 0.88
        
        # 打印调试信息
        print(f"\n{'='*60}")
        print(f"🔍 验证类型: {verification_type}")
        print(f"🤖 使用算法: {algorithm_used}")
        print(f"🎯 相似度: {similarity:.4f}")
        if euclidean_distance is not None:
            print(f"📏 欧氏距离: {euclidean_distance:.4f}")
        if verification_type == "signature" and result is not None and result.get('ssim') is not None:
            print(f"🧮 SSIM: {result['ssim']:.4f}")
        if gnn_info:
            print(f"🧠 GNN关键点: template={gnn_info['keypoints_template']}, query={gnn_info['keypoints_query']}")
        print(f"📊 阈值: {threshold:.4f}")
        print(f"{'='*60}\n")
        
        # 使用计算出的相似度
        final_score = similarity
        
        # 置信度评估（只基于CLIP）
        if final_score > threshold + 0.05:
            confidence = 'high'
        elif final_score < threshold - 0.10:
            confidence = 'low'
        else:
            confidence = 'medium'
        
        # 生成建议
        type_name = "签名" if verification_type == 'signature' else "图章"
        if confidence == 'high':
            if final_score > threshold:
                recommendation = f"高置信度通过 - {type_name}高度相似，可自动接受"
            else:
                recommendation = f"高置信度拒绝 - {type_name}差异明显，可自动拒绝"
        elif confidence == 'medium':
            recommendation = f"中等置信度 - {type_name}相似度{final_score:.1%}，建议人工复审"
        else:
            recommendation = f"低置信度 - {type_name}特征不明确，强烈建议专家复审"
        
        processing_time = time.time() - start_time
        
        response_data = {
            'success': True,
            'type': verification_type,
            'algorithm': algorithm_used,
            'similarity': round(similarity, 4),
            'euclidean_distance': round(euclidean_distance, 4) if euclidean_distance is not None else None,
            'ssim': round(result['ssim'], 4) if verification_type == "signature" and result is not None and result.get('ssim') is not None else None,
            'signet_pipeline': result.get('pipeline') if verification_type == "signature" and result is not None else None,
            'final_score': round(final_score, 4),
            'confidence': confidence,
            'is_authentic': final_score > threshold and confidence != 'low',
            'threshold': threshold,
            'recommendation': recommendation,
            'processing_time_ms': round(processing_time * 1000, 2),
            'clean_enabled': enable_clean if verification_type == "signature" else None,
            'clean_mode': clean_mode if verification_type == "signature" and enable_clean else None,
            # GNN特有信息
            'gnn_keypoints_template': gnn_info.get('keypoints_template') if gnn_info else None,
            'gnn_keypoints_query': gnn_info.get('keypoints_query') if gnn_info else None,
            'gnn_distance': gnn_info.get('gnn_distance') if gnn_info else None
        }
        
        # 添加调试图像路径（如果有）
        if verification_type == "signature" and result is not None and 'debug_images' in result:
            response_data['debug_images'] = result['debug_images']
        
        return response_data
        
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        )

# 挂载前端静态文件 (用于单容器部署)
# 注意: 静态文件路由必须放在最后,避免覆盖API路由
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
    print(f"✅ 前端静态文件已挂载: {frontend_path}")
else:
    print(f"⚠️ 前端目录不存在: {frontend_path}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 启动服务器: http://localhost:8000")
    print("📖 API文档: http://localhost:8000/docs")
    print("🎨 前端界面: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
