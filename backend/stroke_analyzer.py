"""
签名笔画特征提取和快速过滤模块

通过提取签名的基本笔画特征(笔画数量、密度、宽高比等)进行快速过滤,
识别明显不同的签名,避免不必要的深度学习计算。
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class SignatureStrokeAnalyzer:
    """签名笔画特征分析器"""
    
    def __init__(self):
        """初始化分析器"""
        pass
    
    def extract_features(self, image: np.ndarray) -> Dict:
        """
        提取签名的笔画特征
        
        Args:
            image: 输入图像 (numpy array, BGR 格式)
            
        Returns:
            特征字典,包含:
            - stroke_count: 笔画数量(连通域数量)
            - total_pixels: 签名像素总数
            - density: 签名密度(签名像素占比)
            - aspect_ratio: 宽高比
            - bbox_area: 外接矩形面积
            - avg_stroke_length: 平均笔画长度
            - stroke_thickness: 平均笔画粗细
        """
        # 转为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 二值化 (假设背景是白色,签名是黑色)
        # 如果是白底黑字,直接阈值化;如果是黑底白字,需要反转
        mean_val = np.mean(gray)
        if mean_val > 127:  # 白底黑字
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:  # 黑底白字
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 对小尺寸/低墨迹裁剪：MORPH_OPEN(3x3) 很容易把细笔画直接抹掉，导致 stroke_count=0
        h, w = binary.shape[:2]
        min_dim = min(h, w)
        kernel_size = 3 if min_dim >= 80 else 2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        total_pixels_raw = int(np.sum(binary > 0))
        if total_pixels_raw < 200:
            # 低墨迹时优先做闭运算连接断点，避免开运算误删
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        else:
            # 正常情况下用开运算去噪
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 1. 计算签名像素总数
        total_pixels = int(np.sum(binary > 0))
        
        # 2. 计算密度
        total_area = binary.shape[0] * binary.shape[1]
        density = total_pixels / total_area if total_area > 0 else 0
        
        # 3. 找到签名的外接矩形
        coords = cv2.findNonZero(binary)
        # 墨迹太少/几乎空白：笔画特征不可靠，直接标记 invalid，交给深度模型
        if coords is None or len(coords) < 10 or total_pixels < 50:
            # 图像太空或没有签名
            return {
                'stroke_count': 0,
                'total_pixels': int(total_pixels),
                'density': float(density),
                'aspect_ratio': 0,
                'bbox_area': 0,
                'avg_stroke_length': 0,
                'stroke_thickness': 0,
                'valid': False
            }
        
        x, y, w, h = cv2.boundingRect(coords)
        bbox_area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        # 4. 检测连通域(笔画数量)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        # 减去背景(标签0)
        stroke_count = num_labels - 1
        
        # 过滤掉太小的连通域(噪点)
        # 使用与图像面积相关的阈值，避免小图/细笔画全被当作噪点
        min_area = max(3, int(total_area * 0.0002))
        valid_strokes = 0
        total_stroke_area = 0
        
        for i in range(1, num_labels):  # 跳过背景
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                valid_strokes += 1
                total_stroke_area += area
        
        stroke_count = valid_strokes

        # 连通域全被过滤掉：说明笔画非常细碎/图像过小，特征不可靠，避免触发快速拒绝
        # （否则会出现 stroke_count=0 vs 2 这类 100% 差异误杀同一人的签名）
        if stroke_count == 0:
            return {
                'stroke_count': 0,
                'total_pixels': int(total_pixels),
                'density': float(density),
                'aspect_ratio': float(aspect_ratio),
                'bbox_area': int(bbox_area),
                'avg_stroke_length': 0,
                'stroke_thickness': 0,
                'valid': False
            }
        
        # 5. 估算平均笔画长度
        # 使用骨架化来估算笔画长度
        try:
            skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except:
            # 如果ximgproc不可用,使用形态学操作近似
            kernel = np.ones((3,3), np.uint8)
            skeleton = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel, iterations=1)
        
        skeleton_pixels = np.sum(skeleton > 0)
        avg_stroke_length = skeleton_pixels / stroke_count if stroke_count > 0 else 0
        
        # 6. 估算平均笔画粗细
        stroke_thickness = total_pixels / skeleton_pixels if skeleton_pixels > 0 else 1
        
        features = {
            'stroke_count': int(stroke_count),
            'total_pixels': int(total_pixels),
            'density': float(density),
            'aspect_ratio': float(aspect_ratio),
            'bbox_area': int(bbox_area),
            'avg_stroke_length': float(avg_stroke_length),
            'stroke_thickness': float(stroke_thickness),
            'valid': True
        }
        
        return features
    
    def calculate_difference(self, features1: Dict, features2: Dict) -> Dict:
        """
        计算两组特征的差异
        
        Args:
            features1: 第一张图片的特征
            features2: 第二张图片的特征
            
        Returns:
            差异字典,包含各项特征的差异百分比
        """
        if not features1['valid'] or not features2['valid']:
            return {
                'valid': False,
                'reason': '其中一张图片无法提取有效特征'
            }
        
        def safe_diff_ratio(val1, val2):
            """安全计算差异比例"""
            if val1 == 0 and val2 == 0:
                return 0.0
            if val1 == 0 or val2 == 0:
                return 1.0  # 100% 差异
            return abs(val1 - val2) / max(val1, val2)
        
        diff = {
            'stroke_count_diff': safe_diff_ratio(features1['stroke_count'], features2['stroke_count']),
            'density_diff': safe_diff_ratio(features1['density'], features2['density']),
            'aspect_ratio_diff': safe_diff_ratio(features1['aspect_ratio'], features2['aspect_ratio']),
            'bbox_area_diff': safe_diff_ratio(features1['bbox_area'], features2['bbox_area']),
            'stroke_length_diff': safe_diff_ratio(features1['avg_stroke_length'], features2['avg_stroke_length']),
            'stroke_thickness_diff': safe_diff_ratio(features1['stroke_thickness'], features2['stroke_thickness']),
            'valid': True
        }
        
        # 计算综合评分
        combined_score = (
            diff['stroke_count_diff'] +
            diff['aspect_ratio_diff'] +
            diff['density_diff']
        )
        diff['combined_score'] = combined_score
        
        return diff
    
    def should_fast_reject(self, features1: Dict, features2: Dict, 
                          thresholds: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        判断是否应该快速拒绝(不同人签名)
        
        Args:
            features1: 第一张图片的特征
            features2: 第二张图片的特征
            thresholds: 阈值字典 (可选,使用默认值)
            
        Returns:
            (should_reject, reason) 元组
            - should_reject: True 表示应该拒绝
            - reason: 拒绝原因
        """
        # 默认阈值 (适中设置,平衡误杀和漏检)
        default_thresholds = {
            'stroke_count_diff_max': 0.45,     # 笔画数量差异 > 45%
            'aspect_ratio_diff_max': 0.50,    # 宽高比差异 > 50%
            'density_diff_max': 0.50,         # 密度差异 > 50%
            'bbox_area_diff_max': 0.60,       # 面积差异 > 60%
            'combined_score_max': 1.2,        # 综合得分 > 1.2
        }
        
        if thresholds:
            default_thresholds.update(thresholds)
        thresholds = default_thresholds
        
        # 计算差异
        diff = self.calculate_difference(features1, features2)
        
        if not diff['valid']:
            return False, ""  # 无法判断,不拒绝
        
        # 规则 1: 笔画数量差异过大
        if diff['stroke_count_diff'] > thresholds['stroke_count_diff_max']:
            return True, f"笔画数量差异过大 ({diff['stroke_count_diff']:.1%}), 疑似不同人签名"
        
        # 规则 2: 宽高比差异过大
        if diff['aspect_ratio_diff'] > thresholds['aspect_ratio_diff_max']:
            return True, f"签名宽高比差异过大 ({diff['aspect_ratio_diff']:.1%}), 疑似不同人签名"
        
        # 规则 3: 密度差异过大
        if diff['density_diff'] > thresholds['density_diff_max']:
            return True, f"签名密度差异过大 ({diff['density_diff']:.1%}), 疑似不同人签名"
        
        # 规则 4: 外接矩形面积差异过大
        if diff['bbox_area_diff'] > thresholds['bbox_area_diff_max']:
            return True, f"签名大小差异过大 ({diff['bbox_area_diff']:.1%}), 疑似不同人签名"
        
        # 规则 5: 综合评分 (多个指标同时异常)
        combined_score = (
            diff['stroke_count_diff'] +
            diff['aspect_ratio_diff'] +
            diff['density_diff']
        )
        
        if combined_score > thresholds['combined_score_max']:
            return True, f"综合特征差异过大 (得分: {combined_score:.2f}), 疑似不同人签名"
        
        # 通过所有检查
        return False, ""


def quick_signature_check(image1: np.ndarray, image2: np.ndarray, thresholds: Optional[Dict] = None) -> Dict:
    """
    快速签名预检查 (便捷函数)
    
    Args:
        image1: 第一张签名图片
        image2: 第二张签名图片
        
    Returns:
        检查结果字典:
        {
            'should_reject': bool,  # 是否应该拒绝
            'reason': str,          # 拒绝原因
            'features1': dict,      # 图片1的特征
            'features2': dict,      # 图片2的特征
            'differences': dict     # 特征差异
        }
    """
    analyzer = SignatureStrokeAnalyzer()
    
    # 提取特征
    features1 = analyzer.extract_features(image1)
    features2 = analyzer.extract_features(image2)
    
    # 计算差异
    differences = analyzer.calculate_difference(features1, features2)
    
    # 判断是否拒绝（允许外部传入更保守/更激进的阈值）
    should_reject, reason = analyzer.should_fast_reject(features1, features2, thresholds=thresholds)
    
    return {
        'should_reject': should_reject,
        'reason': reason,
        'template_features': features1,
        'query_features': features2,
        'differences': differences
    }
