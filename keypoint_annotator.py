#!/usr/bin/env python3
"""
签名关键点标注工具
支持手动和自动关键点检测
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime


class SignatureKeypointAnnotator:
    """签名关键点标注器"""
    
    # 关键点类型定义
    KEYPOINT_TYPES = {
        'endpoint': {'color': (0, 0, 255), 'label': '端点', 'key': '1'},      # 红色
        'junction': {'color': (0, 255, 0), 'label': '交叉点', 'key': '2'},    # 绿色
        'corner': {'color': (255, 0, 0), 'label': '转折点', 'key': '3'},      # 蓝色
        'bifurcation': {'color': (255, 255, 0), 'label': '分叉点', 'key': '4'} # 青色
    }
    
    def __init__(self, image_path):
        """初始化标注器"""
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 如果是二值图,转为RGB方便显示彩色标记
        if len(self.original_image.shape) == 2:
            self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        else:
            self.display_image = self.original_image.copy()
        
        self.keypoints = []  # 存储: [{'x': x, 'y': y, 'type': 'endpoint'}, ...]
        self.current_type = 'endpoint'
        self.window_name = '签名关键点标注工具'
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 添加关键点
            self.add_keypoint(x, y, self.current_type)
            self.redraw()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 删除最近的关键点
            self.remove_nearest_keypoint(x, y)
            self.redraw()
    
    def add_keypoint(self, x, y, kp_type):
        """添加关键点"""
        self.keypoints.append({
            'x': x,
            'y': y,
            'type': kp_type
        })
        print(f"✅ 添加{self.KEYPOINT_TYPES[kp_type]['label']}: ({x}, {y})")
    
    def remove_nearest_keypoint(self, x, y, threshold=20):
        """删除最近的关键点"""
        if not self.keypoints:
            return
        
        # 找到最近的关键点
        distances = [np.sqrt((kp['x']-x)**2 + (kp['y']-y)**2) for kp in self.keypoints]
        min_idx = np.argmin(distances)
        
        if distances[min_idx] < threshold:
            removed = self.keypoints.pop(min_idx)
            print(f"🗑️  删除{self.KEYPOINT_TYPES[removed['type']]['label']}: ({removed['x']}, {removed['y']})")
    
    def redraw(self):
        """重绘图像和关键点"""
        # 重置显示图像
        if len(self.original_image.shape) == 2:
            self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        else:
            self.display_image = self.original_image.copy()
        
        # 绘制所有关键点
        for kp in self.keypoints:
            color = self.KEYPOINT_TYPES[kp['type']]['color']
            cv2.circle(self.display_image, (kp['x'], kp['y']), 5, color, -1)
            cv2.circle(self.display_image, (kp['x'], kp['y']), 7, color, 2)
        
        # 显示提示信息
        self.draw_instructions()
        cv2.imshow(self.window_name, self.display_image)
    
    def draw_instructions(self):
        """绘制操作说明"""
        instructions = [
            f"当前模式: {self.KEYPOINT_TYPES[self.current_type]['label']}",
            "1-端点 | 2-交叉点 | 3-转折点 | 4-分叉点",
            "左键-添加 | 右键-删除 | S-保存 | A-自动检测 | Q-退出"
        ]
        
        y_offset = 30
        for i, text in enumerate(instructions):
            cv2.putText(self.display_image, text, (10, y_offset + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def auto_detect_keypoints(self):
        """自动检测关键点"""
        print("\n🤖 开始自动检测关键点...")
        
        # 转为灰度图
        if len(self.original_image.shape) == 3:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.original_image.copy()
        
        # 二值化
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # 骨架提取
        skeleton = self.skeletonize(binary)
        
        # 检测不同类型的关键点
        endpoints = self.detect_endpoints(skeleton)
        junctions = self.detect_junctions(skeleton)
        corners = self.detect_corners(skeleton)
        
        # 添加到关键点列表
        auto_keypoints = []
        for x, y in endpoints:
            auto_keypoints.append({'x': int(x), 'y': int(y), 'type': 'endpoint'})
        for x, y in junctions:
            auto_keypoints.append({'x': int(x), 'y': int(y), 'type': 'junction'})
        for x, y in corners:
            auto_keypoints.append({'x': int(x), 'y': int(y), 'type': 'corner'})
        
        print(f"✅ 自动检测到 {len(auto_keypoints)} 个关键点:")
        print(f"   - 端点: {len(endpoints)}")
        print(f"   - 交叉点: {len(junctions)}")
        print(f"   - 转折点: {len(corners)}")
        
        # 合并到现有关键点(避免重复)
        for kp in auto_keypoints:
            # 检查是否已存在相近的关键点
            is_duplicate = False
            for existing_kp in self.keypoints:
                dist = np.sqrt((existing_kp['x']-kp['x'])**2 + (existing_kp['y']-kp['y'])**2)
                if dist < 10:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                self.keypoints.append(kp)
        
        self.redraw()
    
    def skeletonize(self, binary):
        """骨架提取 (Zhang-Suen算法)"""
        skeleton = binary.copy()
        skeleton[skeleton > 0] = 1
        
        # 简化版骨架提取
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        while not done:
            eroded = cv2.erode(skeleton, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(skeleton, temp)
            skeleton = eroded.copy()
            
            done = cv2.countNonZero(temp) == 0
        
        return skeleton * 255
    
    def detect_endpoints(self, skeleton):
        """检测端点 (邻居数=1)"""
        endpoints = []
        h, w = skeleton.shape
        
        for y in range(1, h-1):
            for x in range(1, w-1):
                if skeleton[y, x] == 0:
                    continue
                
                # 计算8邻域中的前景像素数
                neighbors = skeleton[y-1:y+2, x-1:x+2].copy()
                neighbors[1, 1] = 0  # 排除中心点
                neighbor_count = np.count_nonzero(neighbors)
                
                if neighbor_count == 1:
                    endpoints.append((x, y))
        
        return endpoints
    
    def detect_junctions(self, skeleton):
        """检测交叉点 (邻居数>=3)"""
        junctions = []
        h, w = skeleton.shape
        
        for y in range(1, h-1):
            for x in range(1, w-1):
                if skeleton[y, x] == 0:
                    continue
                
                neighbors = skeleton[y-1:y+2, x-1:x+2].copy()
                neighbors[1, 1] = 0
                neighbor_count = np.count_nonzero(neighbors)
                
                if neighbor_count >= 3:
                    junctions.append((x, y))
        
        return junctions
    
    def detect_corners(self, skeleton):
        """检测转折点 (Harris角点)"""
        # 使用Harris角点检测
        skeleton_float = np.float32(skeleton)
        harris = cv2.cornerHarris(skeleton_float, blockSize=2, ksize=3, k=0.04)
        
        # 阈值筛选
        threshold = 0.01 * harris.max()
        corners = np.argwhere(harris > threshold)
        
        # 转换为(x, y)格式
        return [(int(x), int(y)) for y, x in corners]
    
    def save_annotations(self, output_path=None):
        """保存标注结果"""
        if output_path is None:
            base_name = Path(self.image_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"keypoints_{base_name}_{timestamp}.json"
        
        # 准备保存数据
        data = {
            'image_path': self.image_path,
            'image_size': {
                'width': self.original_image.shape[1],
                'height': self.original_image.shape[0]
            },
            'timestamp': datetime.now().isoformat(),
            'keypoints': self.keypoints,
            'statistics': self.get_statistics()
        }
        
        # 保存JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 标注结果已保存到: {output_path}")
        
        # 同时保存可视化图像
        vis_path = output_path.replace('.json', '.png')
        cv2.imwrite(vis_path, self.display_image)
        print(f"🖼️  可视化图像已保存到: {vis_path}")
        
        return output_path
    
    def get_statistics(self):
        """获取标注统计信息"""
        stats = {kp_type: 0 for kp_type in self.KEYPOINT_TYPES.keys()}
        for kp in self.keypoints:
            stats[kp['type']] += 1
        return stats
    
    def run(self):
        """运行标注工具"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.redraw()
        print("\n" + "="*60)
        print("🎯 签名关键点标注工具")
        print("="*60)
        print("操作说明:")
        print("  1/2/3/4 - 切换关键点类型")
        print("  左键    - 添加关键点")
        print("  右键    - 删除最近的关键点")
        print("  A       - 自动检测关键点")
        print("  S       - 保存标注结果")
        print("  Q/ESC   - 退出")
        print("="*60 + "\n")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # 切换关键点类型
            if key == ord('1'):
                self.current_type = 'endpoint'
                print(f"🔄 切换到: {self.KEYPOINT_TYPES['endpoint']['label']}")
                self.redraw()
            elif key == ord('2'):
                self.current_type = 'junction'
                print(f"🔄 切换到: {self.KEYPOINT_TYPES['junction']['label']}")
                self.redraw()
            elif key == ord('3'):
                self.current_type = 'corner'
                print(f"🔄 切换到: {self.KEYPOINT_TYPES['corner']['label']}")
                self.redraw()
            elif key == ord('4'):
                self.current_type = 'bifurcation'
                print(f"🔄 切换到: {self.KEYPOINT_TYPES['bifurcation']['label']}")
                self.redraw()
            
            # 自动检测
            elif key == ord('a') or key == ord('A'):
                self.auto_detect_keypoints()
            
            # 保存
            elif key == ord('s') or key == ord('S'):
                self.save_annotations()
            
            # 退出
            elif key == ord('q') or key == ord('Q') or key == 27:  # ESC
                print("\n👋 退出标注工具")
                break
        
        cv2.destroyAllWindows()
        
        # 返回统计信息
        stats = self.get_statistics()
        print("\n📊 标注统计:")
        for kp_type, count in stats.items():
            print(f"   {self.KEYPOINT_TYPES[kp_type]['label']}: {count}")
        print(f"   总计: {len(self.keypoints)} 个关键点\n")
        
        return self.keypoints


def batch_annotate(image_dir, output_dir='annotations'):
    """批量标注工具"""
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(Path(image_dir).glob(ext))
    
    print(f"\n📁 找到 {len(image_files)} 个图像文件")
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"处理 [{i}/{len(image_files)}]: {img_path.name}")
        print('='*60)
        
        annotator = SignatureKeypointAnnotator(str(img_path))
        annotator.run()
        
        # 保存到输出目录
        output_path = os.path.join(output_dir, f"{img_path.stem}_keypoints.json")
        annotator.save_annotations(output_path)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  单个图像标注: python keypoint_annotator.py <image_path>")
        print("  批量标注:     python keypoint_annotator.py <image_dir> --batch")
        print("\n示例:")
        print("  python keypoint_annotator.py backend/uploaded_samples/debug/template_cleaned_20251029_124648.png")
        print("  python keypoint_annotator.py backend/uploaded_samples/debug --batch")
        sys.exit(1)
    
    if '--batch' in sys.argv:
        batch_annotate(sys.argv[1])
    else:
        annotator = SignatureKeypointAnnotator(sys.argv[1])
        annotator.run()
