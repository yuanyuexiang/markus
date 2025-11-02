#!/usr/bin/env python3
"""
签名验证GNN训练脚本
使用关键点标注数据训练图神经网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class SignatureGNN(nn.Module):
    """签名验证图神经网络"""
    
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=128):
        super(SignatureGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, batch):
        # 图卷积层
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv3(x, edge_index)
        
        # 图级别的池化
        x = global_mean_pool(x, batch)
        
        return x


def load_keypoint_data(json_path):
    """加载关键点JSON数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_node_features(keypoints, width, height):
    """提取节点特征"""
    type_to_idx = {
        'endpoint': 0,
        'junction': 1,
        'corner': 2,
        'bifurcation': 3
    }
    
    features = []
    for kp in keypoints:
        # 归一化位置
        x_norm = kp['x'] / width
        y_norm = kp['y'] / height
        
        # one-hot编码类型
        type_onehot = [0, 0, 0, 0]
        type_idx = type_to_idx.get(kp['type'], 0)
        type_onehot[type_idx] = 1
        
        feat = [x_norm, y_norm] + type_onehot
        features.append(feat)
    
    return torch.tensor(features, dtype=torch.float)


def compute_graph_edges(keypoints, max_distance=50):
    """计算图的边"""
    edges = []
    n = len(keypoints)
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = keypoints[i]['x'], keypoints[i]['y']
            x2, y2 = keypoints[j]['x'], keypoints[j]['y']
            
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            
            if dist <= max_distance:
                edges.append([i, j])
                edges.append([j, i])  # 无向图
    
    if not edges:
        # 如果没有边,至少创建自环
        edges = [[i, i] for i in range(n)]
    
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def create_graph_from_json(json_path):
    """从JSON文件创建PyG图对象"""
    data_dict = load_keypoint_data(json_path)
    
    keypoints = data_dict['keypoints']
    width = data_dict['image_size']['width']
    height = data_dict['image_size']['height']
    
    # 提取特征和边
    x = extract_node_features(keypoints, width, height)
    edge_index = compute_graph_edges(keypoints, max_distance=50)
    
    return Data(x=x, edge_index=edge_index)


def prepare_dataset():
    """准备训练数据集"""
    print("=" * 60)
    print("📊 准备训练数据集")
    print("=" * 60)
    
    # 收集所有JSON文件
    template_files = sorted(Path(".").glob("keypoints_template_*_auto.json"))
    query_files = sorted(Path(".").glob("keypoints_query_*_auto.json"))
    
    print(f"\n找到数据文件:")
    print(f"  模板签名: {len(template_files)} 个")
    print(f"  查询签名: {len(query_files)} 个")
    
    # 创建训练样本对
    # 策略: 同一批次的template和query视为genuine pair,不同批次为forged pair
    pairs = []
    labels = []
    
    # 提取时间戳作为批次标识
    def get_timestamp(filepath):
        name = filepath.stem
        parts = name.split('_')
        # 找到类似20251029_124437的时间戳
        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():
                if i+1 < len(parts) and len(parts[i+1]) == 6 and parts[i+1].isdigit():
                    return f"{part}_{parts[i+1]}"
        return None
    
    template_by_time = {}
    for tf in template_files:
        ts = get_timestamp(tf)
        if ts:
            template_by_time[ts] = tf
    
    query_by_time = {}
    for qf in query_files:
        ts = get_timestamp(qf)
        if ts:
            query_by_time[ts] = qf
    
    # 创建genuine pairs (相同时间戳)
    genuine_count = 0
    for ts in template_by_time.keys():
        if ts in query_by_time:
            pairs.append((template_by_time[ts], query_by_time[ts]))
            labels.append(1)  # genuine
            genuine_count += 1
    
    print(f"\n生成训练对:")
    print(f"  真签名对 (Genuine): {genuine_count}")
    
    # 创建forged pairs (不同时间戳)
    forged_count = 0
    timestamps = list(template_by_time.keys())
    for i, ts1 in enumerate(timestamps):
        for ts2 in timestamps[i+1:]:
            if forged_count < genuine_count:  # 平衡数据集
                pairs.append((template_by_time[ts1], query_by_time[ts2]))
                labels.append(0)  # forged
                forged_count += 1
    
    print(f"  假签名对 (Forged): {forged_count}")
    print(f"  总计: {len(pairs)} 对")
    
    return pairs, labels


def train_siamese_gnn(pairs, labels, epochs=50):
    """训练Siamese GNN"""
    print("\n" + "=" * 60)
    print("🚀 开始训练Siamese GNN")
    print("=" * 60)
    
    # 划分训练集和测试集
    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        pairs, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(train_pairs)} 对")
    print(f"  测试集: {len(test_pairs)} 对")
    
    # 创建模型
    model = SignatureGNN(input_dim=6, hidden_dim=64, output_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练历史
    train_losses = []
    test_accs = []
    
    print(f"\n开始训练 ({epochs} epochs)...")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # 训练一个epoch
        for (template_path, query_path), label in zip(train_pairs, train_labels):
            optimizer.zero_grad()
            
            # 加载图对
            graph1 = create_graph_from_json(template_path)
            graph2 = create_graph_from_json(query_path)
            
            # 前向传播
            batch1 = torch.zeros(graph1.x.size(0), dtype=torch.long)
            batch2 = torch.zeros(graph2.x.size(0), dtype=torch.long)
            
            emb1 = model(graph1.x, graph1.edge_index, batch1)
            emb2 = model(graph2.x, graph2.edge_index, batch2)
            
            # 计算距离
            distance = F.pairwise_distance(emb1, emb2)
            
            # Contrastive loss
            label_tensor = torch.tensor([label], dtype=torch.float)
            margin = 1.0
            
            if label == 1:  # genuine
                loss = distance ** 2
            else:  # forged
                loss = torch.clamp(margin - distance, min=0.0) ** 2
            
            loss = loss.mean()
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_pairs)
        train_losses.append(avg_loss)
        
        # 测试
        model.eval()
        correct = 0
        with torch.no_grad():
            for (template_path, query_path), label in zip(test_pairs, test_labels):
                graph1 = create_graph_from_json(template_path)
                graph2 = create_graph_from_json(query_path)
                
                batch1 = torch.zeros(graph1.x.size(0), dtype=torch.long)
                batch2 = torch.zeros(graph2.x.size(0), dtype=torch.long)
                
                emb1 = model(graph1.x, graph1.edge_index, batch1)
                emb2 = model(graph2.x, graph2.edge_index, batch2)
                
                distance = F.pairwise_distance(emb1, emb2).item()
                
                # 阈值判断
                threshold = 0.5
                pred = 1 if distance < threshold else 0
                
                if pred == label:
                    correct += 1
        
        test_acc = correct / len(test_pairs)
        test_accs.append(test_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2%}")
    
    print("-" * 60)
    print(f"✅ 训练完成!")
    print(f"   最终测试准确率: {test_accs[-1]:.2%}")
    
    # 绘制训练曲线
    plot_training_curves(train_losses, test_accs)
    
    # 保存模型
    torch.save(model.state_dict(), 'signature_gnn_model.pth')
    print(f"   模型已保存: signature_gnn_model.pth")
    
    return model, test_accs[-1]


def plot_training_curves(train_losses, test_accs):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss曲线
    ax1.plot(train_losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy曲线
    ax2.plot([acc * 100 for acc in test_accs], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Test Accuracy Curve', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print(f"   训练曲线已保存: training_curves.png")
    plt.close()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🎓 签名验证GNN训练系统")
    print("=" * 60)
    
    # 检查依赖
    try:
        import torch_geometric
        print("✅ PyTorch Geometric 已安装")
    except ImportError:
        print("❌ 需要安装 PyTorch Geometric:")
        print("   pip install torch-geometric")
        print("   pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html")
        exit(1)
    
    # 准备数据
    pairs, labels = prepare_dataset()
    
    if len(pairs) == 0:
        print("\n❌ 没有找到训练数据!")
        print("   请确保已运行自动标注生成JSON文件")
        exit(1)
    
    # 训练模型
    model, final_acc = train_siamese_gnn(pairs, labels, epochs=50)
    
    print("\n" + "=" * 60)
    print("🎉 训练完成!")
    print("=" * 60)
    print(f"\n最终测试准确率: {final_acc:.2%}")
    print(f"\n生成文件:")
    print(f"  - signature_gnn_model.pth (模型权重)")
    print(f"  - training_curves.png (训练曲线)")
    print("\n下一步:")
    print(f"  1. 查看训练曲线: open training_curves.png")
    print(f"  2. 如果准确率>85%, 可以部署到后端")
    print(f"  3. 如果准确率<85%, 需要更多训练数据或调整超参数")
    print("=" * 60 + "\n")
