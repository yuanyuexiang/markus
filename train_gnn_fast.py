#!/usr/bin/env python3
"""
优化版GNN训练脚本 - 带详细进度日志
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import time


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


def load_keypoint_data(json_path):
    """加载关键点JSON数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_node_features(keypoints, width, height):
    """提取节点特征"""
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
                edges.append([j, i])
    
    if not edges:
        edges = [[i, i] for i in range(n)]
    
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def create_graph_from_json(json_path):
    """从JSON文件创建PyG图对象"""
    data_dict = load_keypoint_data(json_path)
    
    keypoints = data_dict['keypoints']
    width = data_dict['image_size']['width']
    height = data_dict['image_size']['height']
    
    x = extract_node_features(keypoints, width, height)
    edge_index = compute_graph_edges(keypoints, max_distance=50)
    
    return Data(x=x, edge_index=edge_index)


def prepare_dataset():
    """准备训练数据集 - 带详细日志"""
    print("=" * 70)
    print("📊 步骤1: 准备训练数据集")
    print("=" * 70)
    
    print("\n[1/5] 扫描JSON文件...")
    template_files = sorted(Path(".").glob("keypoints_template_*_auto.json"))
    query_files = sorted(Path(".").glob("keypoints_query_*_auto.json"))
    
    print(f"  ✓ 找到模板签名: {len(template_files)} 个")
    print(f"  ✓ 找到查询签名: {len(query_files)} 个")
    
    if len(template_files) == 0 or len(query_files) == 0:
        print("\n❌ 错误: 没有找到足够的标注数据!")
        return [], []
    
    # 提取时间戳
    print("\n[2/5] 提取时间戳...")
    def get_timestamp(filepath):
        name = filepath.stem
        parts = name.split('_')
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
    
    print(f"  ✓ 识别出 {len(template_by_time)} 个时间戳")
    
    # 创建训练对
    print("\n[3/5] 创建真签名对(Genuine pairs)...")
    pairs = []
    labels = []
    
    genuine_count = 0
    for ts in template_by_time.keys():
        if ts in query_by_time:
            pairs.append((template_by_time[ts], query_by_time[ts]))
            labels.append(1)
            genuine_count += 1
            print(f"  + Genuine pair {genuine_count}: {ts}")
    
    print(f"  ✓ 生成 {genuine_count} 个真签名对")
    
    print("\n[4/5] 创建假签名对(Forged pairs)...")
    forged_count = 0
    timestamps = list(template_by_time.keys())
    for i, ts1 in enumerate(timestamps):
        for ts2 in timestamps[i+1:]:
            if forged_count < genuine_count:
                pairs.append((template_by_time[ts1], query_by_time[ts2]))
                labels.append(0)
                forged_count += 1
                print(f"  + Forged pair {forged_count}: {ts1} vs {ts2}")
    
    print(f"  ✓ 生成 {forged_count} 个假签名对")
    
    print(f"\n[5/5] 数据集统计:")
    print(f"  - 真签名对: {genuine_count} ({genuine_count/len(pairs)*100:.1f}%)")
    print(f"  - 假签名对: {forged_count} ({forged_count/len(pairs)*100:.1f}%)")
    print(f"  - 总计: {len(pairs)} 对")
    
    return pairs, labels


def preload_all_graphs(pairs):
    """预加载所有图以加速训练 - 带进度条"""
    print("\n" + "=" * 70)
    print("📦 步骤2: 预加载所有图数据(避免训练时重复加载)")
    print("=" * 70)
    
    unique_files = set()
    for template_path, query_path in pairs:
        unique_files.add(template_path)
        unique_files.add(query_path)
    
    unique_files = sorted(unique_files)
    print(f"\n需要加载 {len(unique_files)} 个唯一图...")
    
    graph_cache = {}
    start_time = time.time()
    
    for i, filepath in enumerate(unique_files, 1):
        graph = create_graph_from_json(filepath)
        graph_cache[filepath] = graph
        
        # 每10个图显示一次进度
        if i % 5 == 0 or i == len(unique_files):
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(unique_files) - i) / rate if rate > 0 else 0
            print(f"  [{i:2d}/{len(unique_files)}] "
                  f"加载: {filepath.name[:40]:<40} "
                  f"({graph.x.size(0):4d}节点, {graph.edge_index.size(1):5d}边) "
                  f"[{rate:.1f}图/秒, ETA:{eta:.1f}秒]")
    
    elapsed = time.time() - start_time
    print(f"\n✓ 预加载完成! 耗时: {elapsed:.2f}秒")
    
    return graph_cache


def train_siamese_gnn(pairs, labels, graph_cache, epochs=20):
    """训练Siamese GNN - 带详细进度"""
    print("\n" + "=" * 70)
    print("🚀 步骤3: 开始训练Siamese GNN")
    print("=" * 70)
    
    # 划分数据
    print("\n[1/3] 划分训练集和测试集...")
    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        pairs, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"  ✓ 训练集: {len(train_pairs)} 对")
    print(f"  ✓ 测试集: {len(test_pairs)} 对")
    
    # 创建模型
    print("\n[2/3] 创建模型...")
    model = SignatureGNN(input_dim=6, hidden_dim=64, output_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"  ✓ 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    print("\n[3/3] 开始训练循环...")
    print("-" * 70)
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Train Acc':>10} | {'Test Acc':>9} | {'Time':>8}")
    print("-" * 70)
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        train_correct = 0
        
        # 训练
        for idx, ((template_path, query_path), label) in enumerate(zip(train_pairs, train_labels)):
            optimizer.zero_grad()
            
            # 从缓存获取图
            graph1 = graph_cache[template_path]
            graph2 = graph_cache[query_path]
            
            batch1 = torch.zeros(graph1.x.size(0), dtype=torch.long)
            batch2 = torch.zeros(graph2.x.size(0), dtype=torch.long)
            
            emb1 = model(graph1.x, graph1.edge_index, batch1)
            emb2 = model(graph2.x, graph2.edge_index, batch2)
            
            distance = F.pairwise_distance(emb1, emb2)
            
            label_tensor = torch.tensor([label], dtype=torch.float)
            margin = 1.0
            
            if label == 1:
                loss = distance ** 2
            else:
                loss = torch.clamp(margin - distance, min=0.0) ** 2
            
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算训练准确率
            threshold = 0.5
            pred = 1 if distance.item() < threshold else 0
            if pred == label:
                train_correct += 1
        
        avg_loss = total_loss / len(train_pairs)
        train_acc = train_correct / len(train_pairs)
        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        
        # 测试
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for (template_path, query_path), label in zip(test_pairs, test_labels):
                graph1 = graph_cache[template_path]
                graph2 = graph_cache[query_path]
                
                batch1 = torch.zeros(graph1.x.size(0), dtype=torch.long)
                batch2 = torch.zeros(graph2.x.size(0), dtype=torch.long)
                
                emb1 = model(graph1.x, graph1.edge_index, batch1)
                emb2 = model(graph2.x, graph2.edge_index, batch2)
                
                distance = F.pairwise_distance(emb1, emb2).item()
                threshold = 0.5
                pred = 1 if distance < threshold else 0
                
                if pred == label:
                    test_correct += 1
        
        test_acc = test_correct / len(test_pairs)
        test_accs.append(test_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"{epoch+1:6d} | {avg_loss:8.4f} | {train_acc:9.2%} | {test_acc:8.2%} | {epoch_time:7.2f}s")
    
    print("-" * 70)
    print(f"✅ 训练完成!")
    print(f"   最佳测试准确率: {max(test_accs):.2%}")
    print(f"   最终测试准确率: {test_accs[-1]:.2%}")
    
    # 保存
    torch.save(model.state_dict(), 'signature_gnn_model.pth')
    print(f"   模型已保存: signature_gnn_model.pth")
    
    # 绘图
    plot_training_curves(train_losses, train_accs, test_accs)
    
    return model, test_accs[-1]


def plot_training_curves(train_losses, train_accs, test_accs):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = list(range(1, len(train_losses) + 1))
    
    # Loss曲线
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy曲线
    ax2.plot(epochs, [acc * 100 for acc in train_accs], 'b-', linewidth=2, label='Train Acc')
    ax2.plot(epochs, [acc * 100 for acc in test_accs], 'g-', linewidth=2, label='Test Acc')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print(f"   训练曲线已保存: training_curves.png")
    plt.close()


if __name__ == '__main__':
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("🎓 签名验证GNN训练系统 (优化版)")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 准备数据
    pairs, labels = prepare_dataset()
    
    if len(pairs) == 0:
        print("\n❌ 没有足够的训练数据!")
        exit(1)
    
    # 预加载图
    graph_cache = preload_all_graphs(pairs)
    
    # 训练
    model, final_acc = train_siamese_gnn(pairs, labels, graph_cache, epochs=20)
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("🎉 全部完成!")
    print("=" * 70)
    print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
    print(f"最终测试准确率: {final_acc:.2%}")
    print(f"\n生成文件:")
    print(f"  - signature_gnn_model.pth")
    print(f"  - training_curves.png")
    print(f"\n下一步:")
    if final_acc > 0.85:
        print(f"  ✅ 准确率>{85}%, 可以考虑部署")
    else:
        print(f"  ⚠️  准确率<{85}%, 建议:")
        print(f"     1. 增加训练数据(更多签名样本)")
        print(f"     2. 调整超参数(hidden_dim, epochs)")
        print(f"     3. 尝试其他GNN架构(GAT, GraphSAGE)")
    print("=" * 70 + "\n")
