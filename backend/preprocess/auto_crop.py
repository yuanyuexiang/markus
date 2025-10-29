"""自动裁剪与标准化预处理

目标: 提高对手工裁剪差异的鲁棒性，尽量对齐签名主笔迹区域，减少空白比例差异带来的嵌入偏移。

核心步骤:
1. 灰度 & 去噪 (可选的轻量模糊)
2. 自适应/OTSU 二值化
3. 形态学处理 (开/闭运算) 连接断裂、移除椒盐噪点
4. 寻找笔迹前景连通域，按面积过滤，合并有效区域外接矩形
5. 对裁剪结果按长边缩放到 target_long_side，并在固定画布(canvas_size)上做“等比例 + 居中 + 留边”放置
6. 输出与原 normalize + crop + resize 流程尺寸一致的 (150,220) 输入大小

设计原则:
- 不改变最终网络输入尺寸
- 失败时回退原始图像，让上层逻辑可继续
- 保持纯 numpy / cv2，避免引入重库
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, Optional

# 最终网络输入尺寸 (H,W)
FINAL_SIZE = (150, 220)
CANVAS_SIZE = (952, 1360)  # 与原 normalize 一致 (H,W)

def _foreground_mask(img: np.ndarray) -> np.ndarray:
    """生成签名前景掩膜 (前景=255, 背景=0)。
    使用 OTSU + 自适应阈值兜底，并叠加形态学操作增强笔画连贯性。
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    try:
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    except Exception:
        mask = None
    if mask is None or mask.mean() in (0, 255):
        mask = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25, 5
        )

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)
    return mask

def clean_signature_conservative(binary: np.ndarray) -> np.ndarray:
    """保守清洁模式 - 适合中文签名
    
    只删除明显的杂质，保护所有可能是签名笔画的部分。
    
    策略：
    1. 只删除极小噪点 (< 5像素)
    2. 保护中心区域的所有笔画
    3. 允许大长宽比 (长横、长竖)
    4. 基于密度过滤孤立杂质
    
    Args:
        binary: 二值图像，前景=255，背景=0
        
    Returns:
        清洁后的二值图像
    """
    if binary.size == 0:
        return binary
        
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    if num_labels <= 1:  # 只有背景
        return binary
    
    h, w = binary.shape
    
    # 第1步：计算签名主体中心（用于密度过滤）
    valid_centers = []
    valid_areas = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 5:  # 只考虑非极小噪点
            valid_centers.append(centroids[i])
            valid_areas.append(area)
    
    if not valid_centers:
        return binary
    
    # 加权质心（大笔画权重更高）
    valid_centers = np.array(valid_centers)
    valid_areas = np.array(valid_areas)
    weights = valid_areas / valid_areas.sum()
    main_center = np.average(valid_centers, axis=0, weights=weights)
    
    # 计算密集区域半径（包含85%的笔画）
    distances = np.linalg.norm(valid_centers - main_center, axis=1)
    if len(distances) > 0:
        radius = np.percentile(distances, 85)
    else:
        radius = max(h, w)
    
    # 第2步：智能过滤
    mask = np.zeros_like(binary)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        
        # 距离主体中心的距离
        distance_to_center = np.linalg.norm(centroids[i] - main_center)
        
        # 过滤规则（保守策略）
        
        # a) 删除极小噪点
        if area < 5:
            continue
        
        # b) 删除极大背景块
        if area > h * w * 0.7:
            continue
        
        # c) 长宽比过滤（允许到30，保护长横竖）
        aspect_ratio = max(width, height) / (min(width, height) + 1e-5)
        if aspect_ratio > 30:
            # 但如果在中心区域，可能是签名的一部分
            center_x = x + width / 2
            center_y = y + height / 2
            if not (0.2 * w < center_x < 0.8 * w and 0.2 * h < center_y < 0.8 * h):
                continue
        
        # d) 密度过滤：远离主体且很小的是杂质
        if distance_to_center > radius * 1.8 and area < 30:
            continue
        
        # e) 边缘小块过滤
        margin = 3
        at_edge = (x < margin or y < margin or 
                   x + width > w - margin or y + height > h - margin)
        if at_edge and area < 30:
            continue
        
        # 保留这个连通域
        mask[labels == i] = 255
    
    return mask

def clean_signature_with_morph(gray: np.ndarray, mode='conservative') -> np.ndarray:
    """完整的签名清洁流程
    
    Args:
        gray: 灰度图像（背景亮，签名暗）
        mode: 'conservative' (中文) 或 'aggressive' (英文)
        
    Returns:
        清洁后的二值图像（前景255，背景0）
    """
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    
    # 策略：使用保守的固定阈值，只保留明显的签名笔画
    # 设置阈值为120，只保留较暗的部分（真正的笔画）
    # 这样可以避免把灰色过渡区也当作前景
    threshold = 120
    
    # 简单阈值二值化
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    if mode == 'conservative':
        # 温和去噪（去除小噪点）
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_tiny, iterations=1)
        
        # 保守清洁（去除离散杂质）
        cleaned = clean_signature_conservative(binary)
        
    else:  # aggressive
        # 更强的形态学处理
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # 清洁
        cleaned = clean_signature_conservative(binary)
    
    return cleaned

def _largest_foreground_bbox(mask: np.ndarray, min_area_ratio: float = 0.0002) -> Optional[Tuple[int,int,int,int]]:
    """寻找前景连通域外接矩形 (合并所有满足面积要求的区域).
    返回 (y_min, y_max, x_min, x_max) 或 None
    """
    # 掩膜本身前景为白色(255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h, w = mask.shape
    min_area = h * w * min_area_ratio
    ys = []
    xs = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < min_area:
            continue
        ys.extend([y, y+ch])
        xs.extend([x, x+cw])
    if not ys or not xs:
        return None
    y_min, y_max = min(ys), max(ys)
    x_min, x_max = min(xs), max(xs)
    return (y_min, y_max, x_min, x_max)

def auto_align_signature(gray: np.ndarray, target_canvas=CANVAS_SIZE, final_size=FINAL_SIZE,
                         margin_ratio: float = 0.15, max_scale: float = 8.0) -> Optional[np.ndarray]:
    """自动裁剪 + 居中标准化.

    margin_ratio: 额外留白边界比例，减少裁剪过紧导致的笔迹差异放大。
    返回 final_size 尺寸 (H,W) 的 uint8 图像，失败返回 None。
    """
    try:
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        mask = _foreground_mask(gray)
        bbox = _largest_foreground_bbox(mask)
        if bbox is None:
            return None
        y0,y1,x0,x1 = bbox
        # 加 margin
        h, w = gray.shape
        margin_y = int((y1 - y0) * margin_ratio)
        margin_x = int((x1 - x0) * margin_ratio)
        y0 = max(0, y0 - margin_y)
        y1 = min(h, y1 + margin_y)
        x0 = max(0, x0 - margin_x)
        x1 = min(w, x1 + margin_x)
        cropped = gray[y0:y1, x0:x1]
        if cropped.size == 0:
            return None
        # 等比例缩放放入 target_canvas
        canvas_h, canvas_w = target_canvas
        ch, cw = cropped.shape
        scale = min(canvas_h / ch, canvas_w / cw)
        if scale > max_scale:
            scale = max_scale
        new_h = max(1, int(ch * scale))
        new_w = max(1, int(cw * scale))
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255
        off_y = (canvas_h - new_h) // 2
        off_x = (canvas_w - new_w) // 2
        canvas[off_y:off_y+new_h, off_x:off_x+new_w] = resized
        # 最终缩放到 final_size
        final = cv2.resize(canvas, (final_size[1], final_size[0]), interpolation=cv2.INTER_AREA)
        return final
    except Exception:
        return None

def robust_preprocess(gray: np.ndarray) -> np.ndarray:
    """高鲁棒性预处理入口.
    1. 自动对齐(成功) -> 输出
    2. 失败 -> 原样等比例放入画布再缩放
    """
    out = auto_align_signature(gray)
    if out is not None:
        return out
    # 简单回退: 放入白画布
    h,w = gray.shape
    canvas = np.ones(CANVAS_SIZE, dtype=np.uint8)*255
    scale = min(CANVAS_SIZE[0]/h, CANVAS_SIZE[1]/w)
    new_h = max(1, int(h*scale))
    new_w = max(1, int(w*scale))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
    oy = (CANVAS_SIZE[0]-new_h)//2
    ox = (CANVAS_SIZE[1]-new_w)//2
    canvas[oy:oy+new_h, ox:ox+new_w] = resized
    final = cv2.resize(canvas, (FINAL_SIZE[1], FINAL_SIZE[0]), interpolation=cv2.INTER_AREA)
    return final

def robust_preprocess_with_clean(gray: np.ndarray, clean_mode='conservative') -> np.ndarray:
    """高鲁棒性预处理 + 签名清洁
    
    流程：
    1. 清洁签名（去除杂质）
    2. 自动裁剪对齐
    3. 标准化到目标尺寸
    
    Args:
        gray: 输入灰度图
        clean_mode: 'conservative'(中文) 或 'aggressive'(英文)
        
    Returns:
        处理后的标准化图像
    """
    try:
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        
        # 1. 清洁签名
        cleaned_binary = clean_signature_with_morph(gray, mode=clean_mode)
        
        # 2. 转回灰度（反转，让签名是黑色）
        cleaned_gray = cv2.bitwise_not(cleaned_binary)
        
        # 3. 自动对齐
        out = auto_align_signature(cleaned_gray)
        if out is not None:
            return out
            
        # 4. 失败回退
        h, w = cleaned_gray.shape
        canvas = np.ones(CANVAS_SIZE, dtype=np.uint8) * 255
        scale = min(CANVAS_SIZE[0]/h, CANVAS_SIZE[1]/w)
        new_h = max(1, int(h*scale))
        new_w = max(1, int(w*scale))
        resized = cv2.resize(cleaned_gray, (new_w, new_h), 
                            interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
        oy = (CANVAS_SIZE[0]-new_h)//2
        ox = (CANVAS_SIZE[1]-new_w)//2
        canvas[oy:oy+new_h, ox:ox+new_w] = resized
        final = cv2.resize(canvas, (FINAL_SIZE[1], FINAL_SIZE[0]), interpolation=cv2.INTER_AREA)
        return final
        
    except Exception:
        # 完全失败，回退到无清洁版本
        return robust_preprocess(gray)

__all__ = [
    'robust_preprocess',
    'robust_preprocess_with_clean',
    'auto_align_signature',
    'clean_signature_with_morph',
    'clean_signature_conservative'
]
