"""
基于Patch和图神经网络的无人机集群异常检测模型

结合了:
1. Patch机制: 将时间序列划分为可自适应的patches
2. 图神经网络: 建模无人机之间的空间关系
3. 时空融合: 同时捕捉时间和空间依赖关系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
import numpy as np


class AdaptivePatchExtractor(nn.Module):
    """
    自适应Patch提取器
    
    功能：
    - 将多维时间序列划分为patches
    - 自适应选择patch大小和步长
    - 对每个patch提取特征
    """
    
    def __init__(self, input_dim, patch_size=10, hidden_dim=64):
        """
        初始化
        
        参数:
        - input_dim: 输入特征维度
        - patch_size: 基础patch大小
        - hidden_dim: 隐藏层维度
        """
        super(AdaptivePatchExtractor, self).__init__()
        
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        # Patch大小自适应选择网络
        self.patch_size_selector = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 输出3个patch大小选项的概率
            nn.Softmax(dim=-1)
        )
        
        # Patch特征提取器（使用1D卷积）
        self.patch_encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        )
    
    def forward(self, x, adaptive=True):
        """
        前向传播
        
        参数:
        - x: 输入时间序列 [batch, seq_len, input_dim]
        - adaptive: 是否使用自适应patch大小
        
        返回:
        - patches: patch特征 [batch, num_patches, hidden_dim]
        - patch_sizes: 每个位置的patch大小
        """
        batch_size, seq_len, input_dim = x.shape
        
        # 如果使用自适应patch大小
        if adaptive:
            # 对序列的每个时间点预测最佳patch大小
            patch_size_probs = self.patch_size_selector(x)  # [batch, seq_len, 3]
            # 三个候选patch大小: 小(5), 中(10), 大(20)
            patch_sizes = [5, 10, 20]
            # 选择概率最大的patch大小
            selected_sizes = torch.argmax(patch_size_probs, dim=-1)  # [batch, seq_len]
        else:
            selected_sizes = torch.ones(batch_size, seq_len, dtype=torch.long) * 1  # 使用中等大小
        
        # 提取patches（为简化，使用固定滑动窗口）
        patch_size = self.patch_size
        stride = patch_size // 2  # 50%重叠
        
        patches = []
        for i in range(0, seq_len - patch_size + 1, stride):
            patch = x[:, i:i+patch_size, :]  # [batch, patch_size, input_dim]
            # 转换为卷积格式 [batch, input_dim, patch_size]
            patch = patch.transpose(1, 2)
            # 提取patch特征
            patch_feat = self.patch_encoder(patch).squeeze(-1)  # [batch, hidden_dim]
            patches.append(patch_feat)
        
        if len(patches) > 0:
            patches = torch.stack(patches, dim=1)  # [batch, num_patches, hidden_dim]
        else:
            # 如果序列太短，至少返回一个patch
            patch = x.transpose(1, 2)  # [batch, input_dim, seq_len]
            patches = self.patch_encoder(patch).squeeze(-1).unsqueeze(1)
        
        return patches, selected_sizes


class SpatialGNN(nn.Module):
    """
    空间图神经网络
    
    功能：
    - 建模无人机之间的空间关系
    - 使用图注意力网络捕捉动态交互
    """
    
    def __init__(self, node_feat_dim, hidden_dim=64, num_layers=2):
        """
        初始化
        
        参数:
        - node_feat_dim: 节点特征维度
        - hidden_dim: 隐藏层维度
        - num_layers: GNN层数
        """
        super(SpatialGNN, self).__init__()
        
        self.num_layers = num_layers
        
        # 图注意力层
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(node_feat_dim, hidden_dim, heads=4, concat=True))
        
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True))
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim * 4, hidden_dim)
    
    def forward(self, x, edge_index):
        """
        前向传播
        
        参数:
        - x: 节点特征 [num_nodes, node_feat_dim]
        - edge_index: 边索引 [2, num_edges]
        
        返回:
        - node_embeddings: 节点嵌入 [num_nodes, hidden_dim]
        """
        # 通过多层GAT
        for i, gat in enumerate(self.gat_layers):
            x = gat(x, edge_index)
            x = F.elu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=0.2, training=self.training)
        
        # 输出层
        node_embeddings = self.output_layer(x)
        
        return node_embeddings


class TemporalTransformer(nn.Module):
    """
    时间Transformer
    
    功能：
    - 捕捉patches之间的时间依赖关系
    - 使用自注意力机制建模长期依赖
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2):
        """
        初始化
        
        参数:
        - input_dim: 输入特征维度
        - hidden_dim: 隐藏层维度
        - num_heads: 注意力头数
        - num_layers: Transformer层数
        """
        super(TemporalTransformer, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
        - x: 输入序列 [batch, seq_len, input_dim]
        
        返回:
        - output: 编码后的序列 [batch, seq_len, hidden_dim]
        """
        # 输入投影
        x = self.input_proj(x)  # [batch, seq_len, hidden_dim]
        
        # Transformer编码
        output = self.transformer(x)  # [batch, seq_len, hidden_dim]
        
        return output


class PatchGNNAnomalyDetector(nn.Module):
    """
    基于Patch和GNN的无人机集群异常检测模型
    
    架构：
    1. 自适应Patch提取
    2. 时间建模（Transformer）
    3. 空间建模（GNN）
    4. 时空融合
    5. 异常检测（节点级和类型识别）
    """
    
    def __init__(self, input_dim, num_drones=10, patch_size=10, hidden_dim=64, num_anomaly_types=4):
        """
        初始化
        
        参数:
        - input_dim: 输入特征维度（传感器数量）
        - num_drones: 无人机数量
        - patch_size: patch大小
        - hidden_dim: 隐藏层维度
        - num_anomaly_types: 异常类型数量（包括正常）
        """
        super(PatchGNNAnomalyDetector, self).__init__()
        
        self.input_dim = input_dim
        self.num_drones = num_drones
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_anomaly_types = num_anomaly_types
        
        # 1. 自适应Patch提取器
        self.patch_extractor = AdaptivePatchExtractor(input_dim, patch_size, hidden_dim)
        
        # 2. 时间建模
        self.temporal_model = TemporalTransformer(hidden_dim, hidden_dim, num_heads=4, num_layers=2)
        
        # 3. 空间建模
        self.spatial_model = SpatialGNN(hidden_dim, hidden_dim, num_layers=2)
        
        # 4. 时空融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 5. 异常检测头
        # 5.1 二分类：正常/异常
        self.anomaly_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # 5.2 多分类：异常类型
        self.type_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_anomaly_types)
        )
        
        # 5.3 重构头（用于无监督异常检测）
        self.reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def build_graph(self, batch_size, num_drones, device):
        """
        构建无人机集群的空间图
        
        策略：全连接图（每架无人机与其他所有无人机相连）
        
        参数:
        - batch_size: 批大小
        - num_drones: 无人机数量
        - device: 设备
        
        返回:
        - edge_index: 边索引 [2, num_edges]
        """
        # 创建全连接图
        edges = []
        for i in range(num_drones):
            for j in range(num_drones):
                if i != j:
                    edges.append([i, j])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = edge_index.to(device)
        
        return edge_index
    
    def forward(self, x, return_patches=False):
        """
        前向传播
        
        参数:
        - x: 输入数据 [batch, num_drones, seq_len, input_dim]
        - return_patches: 是否返回patch信息
        
        返回:
        - anomaly_scores: 异常分数 [batch, num_drones, 2]
        - anomaly_types: 异常类型 [batch, num_drones, num_anomaly_types]
        - reconstructions: 重构结果 [batch, num_drones, seq_len, input_dim]
        """
        batch_size, num_drones, seq_len, input_dim = x.shape
        device = x.device
        
        # 1. 对每架无人机提取patches
        all_patches = []
        all_patch_sizes = []
        
        for i in range(num_drones):
            drone_data = x[:, i, :, :]  # [batch, seq_len, input_dim]
            patches, patch_sizes = self.patch_extractor(drone_data, adaptive=True)
            all_patches.append(patches)
            all_patch_sizes.append(patch_sizes)
        
        # [num_drones, batch, num_patches, hidden_dim]
        all_patches = torch.stack(all_patches, dim=0)
        num_patches = all_patches.shape[2]
        
        # 2. 时间建模：对每架无人机的patches应用Transformer
        temporal_features = []
        for i in range(num_drones):
            drone_patches = all_patches[i]  # [batch, num_patches, hidden_dim]
            temporal_feat = self.temporal_model(drone_patches)  # [batch, num_patches, hidden_dim]
            # 使用最后一个patch的特征作为该无人机的时间特征
            temporal_feat = temporal_feat[:, -1, :]  # [batch, hidden_dim]
            temporal_features.append(temporal_feat)
        
        # [batch, num_drones, hidden_dim]
        temporal_features = torch.stack(temporal_features, dim=1)
        
        # 3. 空间建模：使用GNN建模无人机之间的关系
        # 需要将batch展平
        temporal_features_flat = temporal_features.reshape(batch_size * num_drones, self.hidden_dim)
        
        # 构建图
        edge_index = self.build_graph(batch_size, num_drones, device)
        
        # 为每个batch创建独立的图
        batch_edge_indices = []
        for b in range(batch_size):
            offset = b * num_drones
            batch_edge_index = edge_index + offset
            batch_edge_indices.append(batch_edge_index)
        
        batch_edge_index = torch.cat(batch_edge_indices, dim=1)
        
        # 应用GNN
        spatial_features_flat = self.spatial_model(temporal_features_flat, batch_edge_index)
        # [batch * num_drones, hidden_dim]
        
        # 重塑回 [batch, num_drones, hidden_dim]
        spatial_features = spatial_features_flat.reshape(batch_size, num_drones, self.hidden_dim)
        
        # 4. 时空融合
        fused_features = torch.cat([temporal_features, spatial_features], dim=-1)
        # [batch, num_drones, hidden_dim * 2]
        fused_features = self.fusion(fused_features)  # [batch, num_drones, hidden_dim]
        
        # 5. 异常检测
        # 5.1 异常分类（正常/异常）
        anomaly_scores = self.anomaly_classifier(fused_features)  # [batch, num_drones, 2]
        
        # 5.2 异常类型分类
        anomaly_types = self.type_classifier(fused_features)  # [batch, num_drones, num_anomaly_types]
        
        # 5.3 重构（用于无监督学习）
        reconstructions = self.reconstructor(fused_features)  # [batch, num_drones, input_dim]
        # 扩展到时间维度（简化：假设重构的是平均值）
        reconstructions = reconstructions.unsqueeze(2).expand(-1, -1, seq_len, -1)
        
        if return_patches:
            return anomaly_scores, anomaly_types, reconstructions, all_patches
        else:
            return anomaly_scores, anomaly_types, reconstructions


def create_model(input_dim=14, num_drones=10, patch_size=10, hidden_dim=64, num_anomaly_types=4):
    """
    创建模型的工厂函数
    
    参数:
    - input_dim: 输入特征维度
    - num_drones: 无人机数量
    - patch_size: patch大小
    - hidden_dim: 隐藏层维度
    - num_anomaly_types: 异常类型数量
    
    返回: 模型实例
    """
    model = PatchGNNAnomalyDetector(
        input_dim=input_dim,
        num_drones=num_drones,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        num_anomaly_types=num_anomaly_types
    )
    return model


if __name__ == "__main__":
    # 测试模型
    print("测试Patch-GNN异常检测模型...")
    
    # 创建模型
    model = create_model(input_dim=14, num_drones=10, patch_size=10, hidden_dim=64, num_anomaly_types=4)
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建测试数据
    batch_size = 4
    num_drones = 10
    seq_len = 50
    input_dim = 14
    
    x = torch.randn(batch_size, num_drones, seq_len, input_dim)
    
    # 前向传播
    print("\n前向传播...")
    anomaly_scores, anomaly_types, reconstructions = model(x)
    
    print(f"异常分数形状: {anomaly_scores.shape}")  # [batch, num_drones, 2]
    print(f"异常类型形状: {anomaly_types.shape}")  # [batch, num_drones, 4]
    print(f"重构结果形状: {reconstructions.shape}")  # [batch, num_drones, seq_len, input_dim]
    
    print("\n模型测试成功！")
