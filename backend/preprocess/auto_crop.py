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

__all__ = [
    'robust_preprocess',
    'auto_align_signature'
]
