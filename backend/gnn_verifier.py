"""
GNN签名验证模块
用于加载训练好的GNN模型并进行签名验证
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import cv2
import os


class SignatureGNN(nn.Module):
    """签名验证图神经网络"""
    
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=128):
        super(SignatureGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        
        return x


class GNNSignatureVerifier:
    """GNN签名验证器"""
    
    def __init__(self, model_path='signature_gnn_model.pth'):
        """初始化验证器并加载模型"""
        self.model = SignatureGNN(input_dim=6, hidden_dim=64, output_dim=128)

        # 兼容不同工作目录：默认模型路径按 backend 目录解析
        if not os.path.isabs(model_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, model_path)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
            print(f"✓ GNN模型已加载: {model_path}")
        else:
            print(f"⚠️ GNN模型文件不存在: {model_path}")
            self.model = None
        
        self.threshold = 0.5  # 距离阈值
    
    def extract_keypoints(self, image):
        """从图像提取关键点"""
        # 确保是灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 二值化
        mean_val = np.mean(gray)
        if mean_val > 127:
            # 白底黑字,反转
            binary = cv2.bitwise_not(gray)
        else:
            binary = gray.copy()
        
        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
        
        # 骨架提取
        skeleton = self._skeletonize(binary)
        
        # 检测关键点
        endpoints = self._detect_endpoints(skeleton)
        junctions = self._detect_junctions(skeleton)
        corners = self._detect_corners(skeleton)
        
        # 合并关键点
        keypoints = []
        for x, y in endpoints:
            keypoints.append({'x': int(x), 'y': int(y), 'type': 'endpoint'})
        for x, y in junctions:
            keypoints.append({'x': int(x), 'y': int(y), 'type': 'junction'})
        for x, y in corners:
            keypoints.append({'x': int(x), 'y': int(y), 'type': 'corner'})
        
        return keypoints
    
    def _skeletonize(self, binary):
        """骨架提取"""
        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
        
        size = np.size(binary)
        skeleton = np.zeros(binary.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        temp = binary.copy()
        
        while not done:
            eroded = cv2.erode(temp, element)
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
            subset = cv2.subtract(eroded, opened)
            skeleton = cv2.bitwise_or(skeleton, subset)
            temp = eroded.copy()
            
            zeros = size - cv2.countNonZero(temp)
            if zeros == size:
                done = True
        
        return skeleton
    
    def _detect_endpoints(self, skeleton):
        """检测端点"""
        endpoints = []
        h, w = skeleton.shape
        
        for y in range(1, h-1):
            for x in range(1, w-1):
                if skeleton[y, x] == 0:
                    continue
                
                neighbors = skeleton[y-1:y+2, x-1:x+2].copy()
                neighbors[1, 1] = 0
                neighbor_count = np.count_nonzero(neighbors)
                
                if neighbor_count == 1:
                    endpoints.append((x, y))
        
        return endpoints
    
    def _detect_junctions(self, skeleton):
        """检测交叉点"""
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
    
    def _detect_corners(self, skeleton):
        """检测转折点"""
        skeleton_float = np.float32(skeleton)
        harris = cv2.cornerHarris(skeleton_float, blockSize=2, ksize=3, k=0.04)
        
        threshold = 0.01 * harris.max() if harris.max() > 0 else 0
        corners_pos = np.argwhere(harris > threshold)
        
        corners = [(int(x), int(y)) for y, x in corners_pos]
        return corners
    
    def create_graph(self, keypoints, width, height):
        """从关键点创建图"""
        if len(keypoints) == 0:
            # 空图
            x = torch.zeros((1, 6), dtype=torch.float)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            return Data(x=x, edge_index=edge_index)
        
        # 提取节点特征
        type_to_idx = {'endpoint': 0, 'junction': 1, 'corner': 2, 'bifurcation': 3}
        
        features = []
        for kp in keypoints:
            x_norm = kp['x'] / width
            y_norm = kp['y'] / height
            
            type_onehot = [0, 0, 0, 0]
            type_idx = type_to_idx.get(kp['type'], 0)
            type_onehot[type_idx] = 1
            
            feat = [x_norm, y_norm] + type_onehot
            features.append(feat)
        
        x = torch.tensor(features, dtype=torch.float)
        
        # 计算边
        edges = []
        n = len(keypoints)
        max_distance = 50
        
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = keypoints[i]['x'], keypoints[i]['y']
                x2, y2 = keypoints[j]['x'], keypoints[j]['y']
                
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                
                if dist <= max_distance:
                    edges.append([i, j])
                    edges.append([j, i])
        
        if not edges:
            edges = [[i, i] for i in range(n)]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)
    
    def verify(self, template_image, query_image):
        """
        验证两个签名是否匹配
        
        Args:
            template_image: 模板签名图像(numpy array)
            query_image: 查询签名图像(numpy array)
        
        Returns:
            dict: {
                'match': bool,
                'distance': float,
                'confidence': float,
                'keypoints_template': int,
                'keypoints_query': int
            }
        """
        if self.model is None:
            return {
                'match': False,
                'distance': 1.0,
                'confidence': 0.0,
                'error': 'GNN模型未加载'
            }
        
        try:
            # 提取关键点
            template_kp = self.extract_keypoints(template_image)
            query_kp = self.extract_keypoints(query_image)
            
            # 创建图
            template_h, template_w = template_image.shape[:2]
            query_h, query_w = query_image.shape[:2]
            
            graph1 = self.create_graph(template_kp, template_w, template_h)
            graph2 = self.create_graph(query_kp, query_w, query_h)
            
            # 前向传播
            with torch.no_grad():
                batch1 = torch.zeros(graph1.x.size(0), dtype=torch.long)
                batch2 = torch.zeros(graph2.x.size(0), dtype=torch.long)
                
                emb1 = self.model(graph1.x, graph1.edge_index, batch1)
                emb2 = self.model(graph2.x, graph2.edge_index, batch2)
                
                # 计算欧氏距离
                distance = F.pairwise_distance(emb1, emb2).item()
            
            # 判断是否匹配 (训练时使用margin=1.0, 所以阈值应该在0-1之间)
            match = distance < self.threshold
            
            # 计算置信度 (距离范围: 相同签名<0.5, 不同签名>1.0)
            # 使用sigmoid函数将距离映射到[0,1]区间
            # distance越小,confidence越高
            if distance < self.threshold:
                # 匹配的情况: 将[0, threshold]映射到[1.0, 0.5]
                confidence = 1.0 - (distance / self.threshold) * 0.5
            else:
                # 不匹配的情况: 将[threshold, 2.0]映射到[0.5, 0.0]
                max_distance = 2.0  # 假设最大距离为2.0
                confidence = max(0.0, 0.5 * (1.0 - (distance - self.threshold) / (max_distance - self.threshold)))
            
            return {
                'match': match,
                'distance': float(distance),
                'confidence': float(confidence),
                'threshold': float(self.threshold),
                'keypoints_template': len(template_kp),
                'keypoints_query': len(query_kp)
            }
        
        except Exception as e:
            return {
                'match': False,
                'distance': 1.0,
                'confidence': 0.0,
                'error': str(e)
            }


# 全局GNN验证器实例
_gnn_verifier = None

def get_gnn_verifier():
    """获取GNN验证器单例"""
    global _gnn_verifier
    if _gnn_verifier is None:
        _gnn_verifier = GNNSignatureVerifier()
    return _gnn_verifier
