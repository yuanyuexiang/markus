#!/usr/bin/env python3
"""
关键点标注数据示例分析
展示如何使用标注的关键点数据
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter


def load_keypoint_data(json_path):
    """加载关键点标注数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def visualize_keypoint_distribution(data):
    """可视化关键点分布"""
    keypoints = data['keypoints']
    width = data['image_size']['width']
    height = data['image_size']['height']
    
    # 按类型分组
    type_groups = {
        'endpoint': [],
        'junction': [],
        'corner': [],
        'bifurcation': []
    }
    
    for kp in keypoints:
        kp_type = kp['type']
        if kp_type in type_groups:
            type_groups[kp_type].append((kp['x'], kp['y']))
    
    # 绘图
    plt.figure(figsize=(10, 8))
    
    colors = {
        'endpoint': 'red',
        'junction': 'green',
        'corner': 'blue',
        'bifurcation': 'cyan'
    }
    
    labels = {
        'endpoint': '端点',
        'junction': '交叉点',
        'corner': '转折点',
        'bifurcation': '分叉点'
    }
    
    for kp_type, points in type_groups.items():
        if points:
            xs, ys = zip(*points)
            plt.scatter(xs, ys, c=colors[kp_type], s=100, 
                       label=f"{labels[kp_type]} ({len(points)})",
                       alpha=0.7, edgecolors='black', linewidth=1.5)
    
    plt.xlim(0, width)
    plt.ylim(height, 0)  # Y轴反转(图像坐标系)
    plt.xlabel('X坐标', fontsize=12)
    plt.ylabel('Y坐标', fontsize=12)
    plt.title('签名关键点分布图', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = 'keypoint_distribution.png'
    plt.savefig(output_path, dpi=150)
    print(f"✅ 分布图已保存到: {output_path}")
    plt.close()


def compute_graph_edges(keypoints, max_distance=50):
    """
    根据关键点计算图的边
    连接距离小于max_distance的关键点对
    """
    edges = []
    n = len(keypoints)
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = keypoints[i]['x'], keypoints[i]['y']
            x2, y2 = keypoints[j]['x'], keypoints[j]['y']
            
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            
            if dist <= max_distance:
                edges.append((i, j))
    
    return edges


def extract_node_features(keypoints, width, height):
    """
    提取节点特征向量
    特征: [归一化x, 归一化y, 类型one-hot(4维)]
    """
    type_to_idx = {
        'endpoint': 0,
        'junction': 1,
        'corner': 2,
        'bifurcation': 3
    }
    
    features = []
    for kp in keypoints:
        # 位置特征(归一化)
        x_norm = kp['x'] / width
        y_norm = kp['y'] / height
        
        # 类型特征(one-hot)
        type_onehot = [0, 0, 0, 0]
        type_idx = type_to_idx.get(kp['type'], 0)
        type_onehot[type_idx] = 1
        
        # 合并特征
        feat = [x_norm, y_norm] + type_onehot
        features.append(feat)
    
    return np.array(features, dtype=np.float32)


def compute_graph_statistics(keypoints, edges):
    """计算图的统计特征"""
    n_nodes = len(keypoints)
    n_edges = len(edges)
    
    # 度分布
    degree = [0] * n_nodes
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    
    avg_degree = np.mean(degree)
    max_degree = np.max(degree)
    
    # 类型分布
    type_counts = Counter([kp['type'] for kp in keypoints])
    
    print("\n📊 图结构统计:")
    print(f"   节点数: {n_nodes}")
    print(f"   边数: {n_edges}")
    print(f"   平均度: {avg_degree:.2f}")
    print(f"   最大度: {max_degree}")
    print(f"   图密度: {2*n_edges/(n_nodes*(n_nodes-1)) if n_nodes > 1 else 0:.4f}")
    
    print("\n📊 关键点类型分布:")
    type_labels = {
        'endpoint': '端点',
        'junction': '交叉点',
        'corner': '转折点',
        'bifurcation': '分叉点'
    }
    for kp_type, count in type_counts.items():
        label = type_labels.get(kp_type, kp_type)
        print(f"   {label}: {count} ({count/n_nodes*100:.1f}%)")


def compare_two_signatures(data1, data2):
    """
    比较两个签名的关键点结构
    简单示例:比较关键点数量和类型分布
    """
    print("\n🔍 签名对比分析:")
    print("="*50)
    
    # 提取统计信息
    stats1 = data1['statistics']
    stats2 = data2['statistics']
    
    total1 = sum(stats1.values())
    total2 = sum(stats2.values())
    
    print(f"签名1总关键点数: {total1}")
    print(f"签名2总关键点数: {total2}")
    print(f"数量差异: {abs(total1 - total2)} ({abs(total1-total2)/max(total1,total2)*100:.1f}%)")
    
    # 类型分布相似度
    type_labels = {
        'endpoint': '端点',
        'junction': '交叉点',
        'corner': '转折点',
        'bifurcation': '分叉点'
    }
    
    print("\n类型分布对比:")
    similarity_scores = []
    for kp_type in type_labels.keys():
        count1 = stats1.get(kp_type, 0)
        count2 = stats2.get(kp_type, 0)
        
        # 归一化
        ratio1 = count1 / total1 if total1 > 0 else 0
        ratio2 = count2 / total2 if total2 > 0 else 0
        
        # 相似度(1 - 差异)
        sim = 1 - abs(ratio1 - ratio2)
        similarity_scores.append(sim)
        
        print(f"   {type_labels[kp_type]:6s}: {count1:3d} vs {count2:3d} "
              f"(相似度: {sim:.2f})")
    
    overall_sim = np.mean(similarity_scores)
    print(f"\n整体结构相似度: {overall_sim:.2f}")
    
    if overall_sim > 0.8:
        print("✅ 结论: 两个签名结构非常相似")
    elif overall_sim > 0.6:
        print("⚠️  结论: 两个签名结构有一定相似性")
    else:
        print("❌ 结论: 两个签名结构差异较大")
    
    return overall_sim


def export_for_gnn_training(data, output_path='graph_data.npz'):
    """
    导出为GNN训练格式
    PyTorch Geometric兼容格式
    """
    keypoints = data['keypoints']
    width = data['image_size']['width']
    height = data['image_size']['height']
    
    # 节点特征
    node_features = extract_node_features(keypoints, width, height)
    
    # 边
    edges = compute_graph_edges(keypoints, max_distance=50)
    edge_index = np.array(edges, dtype=np.int64).T  # [2, num_edges]
    
    # 保存
    np.savez(output_path,
             node_features=node_features,
             edge_index=edge_index,
             num_nodes=len(keypoints))
    
    print(f"\n💾 GNN训练数据已导出到: {output_path}")
    print(f"   节点特征形状: {node_features.shape}")
    print(f"   边索引形状: {edge_index.shape}")
    
    return output_path


if __name__ == '__main__':
    import sys
    
    print("="*60)
    print("🔍 关键点标注数据分析工具")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\n用法:")
        print("  单个分析: python analyze_keypoints.py <keypoint_json>")
        print("  对比分析: python analyze_keypoints.py <json1> <json2>")
        print("\n示例:")
        print("  python analyze_keypoints.py keypoints_template_20251029_150000.json")
        sys.exit(1)
    
    # 加载第一个签名
    json_path1 = sys.argv[1]
    print(f"\n📂 加载签名1: {json_path1}")
    data1 = load_keypoint_data(json_path1)
    
    print(f"   图像尺寸: {data1['image_size']['width']} x {data1['image_size']['height']}")
    print(f"   关键点数: {len(data1['keypoints'])}")
    
    # 可视化
    visualize_keypoint_distribution(data1)
    
    # 计算图结构
    edges1 = compute_graph_edges(data1['keypoints'])
    compute_graph_statistics(data1['keypoints'], edges1)
    
    # 导出GNN格式
    export_for_gnn_training(data1, 'signature1_graph.npz')
    
    # 如果提供了第二个签名,进行对比
    if len(sys.argv) >= 3:
        json_path2 = sys.argv[2]
        print(f"\n📂 加载签名2: {json_path2}")
        data2 = load_keypoint_data(json_path2)
        
        print(f"   图像尺寸: {data2['image_size']['width']} x {data2['image_size']['height']}")
        print(f"   关键点数: {len(data2['keypoints'])}")
        
        # 对比分析
        compare_two_signatures(data1, data2)
    
    print("\n" + "="*60)
    print("✅ 分析完成!")
    print("="*60 + "\n")
