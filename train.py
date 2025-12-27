"""
训练脚本
用于训练基于Patch-GNN的无人机集群异常检测模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.patch_gnn_model import create_model


class DroneSwarmDataset(Dataset):
    """
    无人机集群数据集
    """
    
    def __init__(self, data_path, window_size=50, stride=25):
        """
        初始化
        
        参数:
        - data_path: 数据文件路径
        - window_size: 时间窗口大小
        - stride: 滑动步长
        """
        self.data = pd.read_csv(data_path)
        self.window_size = window_size
        self.stride = stride
        
        # 特征列（排除timestamp, drone_id, anomaly, anomaly_type）
        self.feature_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az',
                            'roll', 'pitch', 'yaw', 'battery', 'temperature', 'signal_strength']
        
        # 标准化特征
        self.mean = {}
        self.std = {}
        for col in self.feature_cols:
            self.mean[col] = self.data[col].mean()
            self.std[col] = self.data[col].std()
            self.data[col] = (self.data[col] - self.mean[col]) / (self.std[col] + 1e-8)
        
        # 异常类型映射
        self.anomaly_type_map = {
            'normal': 0,
            'position_deviation': 1,
            'sensor_malfunction_battery': 2,
            'sensor_malfunction_temperature': 2,
            'sensor_malfunction_signal_strength': 2,
            'maneuver_anomaly': 3
        }
        
        # 创建样本索引
        self.samples = self._create_samples()
    
    def _create_samples(self):
        """
        创建滑动窗口样本
        
        返回: 样本列表 [(start_idx, end_idx), ...]
        """
        samples = []
        
        # 获取每架无人机的数据索引
        drone_indices = {}
        for drone_id in range(10):
            drone_indices[drone_id] = self.data[self.data['drone_id'] == drone_id].index.tolist()
        
        # 对每架无人机创建滑动窗口
        for drone_id in range(10):
            indices = drone_indices[drone_id]
            for i in range(0, len(indices) - self.window_size + 1, self.stride):
                start_idx = i
                end_idx = i + self.window_size
                samples.append((drone_id, start_idx, end_idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        返回:
        - x: 输入特征 [num_drones, window_size, num_features]
        - y_anomaly: 异常标签 [num_drones]
        - y_type: 异常类型标签 [num_drones]
        """
        # 获取时间窗口
        drone_id, start_idx, end_idx = self.samples[idx]
        
        # 获取所有无人机在这个时间窗口的数据
        x_list = []
        y_anomaly_list = []
        y_type_list = []
        
        for d_id in range(10):
            drone_data = self.data[self.data['drone_id'] == d_id]
            window_data = drone_data.iloc[start_idx:end_idx]
            
            # 特征
            features = window_data[self.feature_cols].values  # [window_size, num_features]
            x_list.append(features)
            
            # 标签：窗口中是否有异常（任意一个时间点）
            has_anomaly = (window_data['anomaly'].values == 1).any()
            y_anomaly_list.append(1 if has_anomaly else 0)
            
            # 异常类型：使用窗口中最常见的类型
            if has_anomaly:
                anomaly_types = window_data[window_data['anomaly'] == 1]['anomaly_type'].values
                if len(anomaly_types) > 0:
                    most_common_type = pd.Series(anomaly_types).mode()[0]
                    y_type = self.anomaly_type_map.get(most_common_type, 0)
                else:
                    y_type = 0
            else:
                y_type = 0
            y_type_list.append(y_type)
        
        # 转换为tensor
        x = torch.FloatTensor(np.array(x_list))  # [num_drones, window_size, num_features]
        y_anomaly = torch.LongTensor(y_anomaly_list)  # [num_drones]
        y_type = torch.LongTensor(y_type_list)  # [num_drones]
        
        return x, y_anomaly, y_type


def train_epoch(model, dataloader, optimizer, device, criterion_anomaly, criterion_type):
    """
    训练一个epoch
    
    参数:
    - model: 模型
    - dataloader: 数据加载器
    - optimizer: 优化器
    - device: 设备
    - criterion_anomaly: 异常检测损失函数
    - criterion_type: 类型分类损失函数
    
    返回: 平均损失
    """
    model.train()
    total_loss = 0
    total_anomaly_loss = 0
    total_type_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (x, y_anomaly, y_type) in enumerate(pbar):
        # x: [batch, num_drones, window_size, num_features]
        # y_anomaly: [batch, num_drones]
        # y_type: [batch, num_drones]
        
        x = x.to(device)
        y_anomaly = y_anomaly.to(device)
        y_type = y_type.to(device)
        
        # 前向传播
        anomaly_scores, anomaly_types, reconstructions = model(x)
        # anomaly_scores: [batch, num_drones, 2]
        # anomaly_types: [batch, num_drones, num_anomaly_types]
        
        # 计算损失
        batch_size, num_drones = y_anomaly.shape
        
        # 异常检测损失
        anomaly_scores_flat = anomaly_scores.reshape(batch_size * num_drones, 2)
        y_anomaly_flat = y_anomaly.reshape(batch_size * num_drones)
        loss_anomaly = criterion_anomaly(anomaly_scores_flat, y_anomaly_flat)
        
        # 类型分类损失（只对异常样本计算）
        anomaly_mask = y_anomaly_flat == 1
        if anomaly_mask.sum() > 0:
            anomaly_types_masked = anomaly_types.reshape(batch_size * num_drones, -1)[anomaly_mask]
            y_type_masked = y_type.reshape(batch_size * num_drones)[anomaly_mask]
            loss_type = criterion_type(anomaly_types_masked, y_type_masked)
        else:
            loss_type = torch.tensor(0.0).to(device)
        
        # 总损失
        loss = loss_anomaly + 0.5 * loss_type
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录损失
        total_loss += loss.item()
        total_anomaly_loss += loss_anomaly.item()
        total_type_loss += loss_type.item() if isinstance(loss_type, torch.Tensor) else 0
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'anomaly_loss': f'{loss_anomaly.item():.4f}',
            'type_loss': f'{loss_type.item() if isinstance(loss_type, torch.Tensor) else 0:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_anomaly_loss = total_anomaly_loss / len(dataloader)
    avg_type_loss = total_type_loss / len(dataloader)
    
    return avg_loss, avg_anomaly_loss, avg_type_loss


def evaluate(model, dataloader, device, criterion_anomaly, criterion_type):
    """
    评估模型
    
    参数:
    - model: 模型
    - dataloader: 数据加载器
    - device: 设备
    - criterion_anomaly: 异常检测损失函数
    - criterion_type: 类型分类损失函数
    
    返回: 损失和指标
    """
    model.eval()
    total_loss = 0
    total_anomaly_loss = 0
    total_type_loss = 0
    
    all_y_anomaly = []
    all_pred_anomaly = []
    all_y_type = []
    all_pred_type = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for x, y_anomaly, y_type in pbar:
            x = x.to(device)
            y_anomaly = y_anomaly.to(device)
            y_type = y_type.to(device)
            
            # 前向传播
            anomaly_scores, anomaly_types, reconstructions = model(x)
            
            batch_size, num_drones = y_anomaly.shape
            
            # 计算损失
            anomaly_scores_flat = anomaly_scores.reshape(batch_size * num_drones, 2)
            y_anomaly_flat = y_anomaly.reshape(batch_size * num_drones)
            loss_anomaly = criterion_anomaly(anomaly_scores_flat, y_anomaly_flat)
            
            # 类型分类损失
            anomaly_mask = y_anomaly_flat == 1
            if anomaly_mask.sum() > 0:
                anomaly_types_masked = anomaly_types.reshape(batch_size * num_drones, -1)[anomaly_mask]
                y_type_masked = y_type.reshape(batch_size * num_drones)[anomaly_mask]
                loss_type = criterion_type(anomaly_types_masked, y_type_masked)
            else:
                loss_type = torch.tensor(0.0).to(device)
            
            loss = loss_anomaly + 0.5 * loss_type
            
            total_loss += loss.item()
            total_anomaly_loss += loss_anomaly.item()
            total_type_loss += loss_type.item() if isinstance(loss_type, torch.Tensor) else 0
            
            # 预测
            pred_anomaly = torch.argmax(anomaly_scores_flat, dim=1)
            pred_type = torch.argmax(anomaly_types.reshape(batch_size * num_drones, -1), dim=1)
            
            # 收集结果
            all_y_anomaly.extend(y_anomaly_flat.cpu().numpy())
            all_pred_anomaly.extend(pred_anomaly.cpu().numpy())
            all_y_type.extend(y_type.reshape(batch_size * num_drones).cpu().numpy())
            all_pred_type.extend(pred_type.cpu().numpy())
    
    # 计算指标
    avg_loss = total_loss / len(dataloader)
    avg_anomaly_loss = total_anomaly_loss / len(dataloader)
    avg_type_loss = total_type_loss / len(dataloader)
    
    # 异常检测指标
    accuracy = accuracy_score(all_y_anomaly, all_pred_anomaly)
    precision = precision_score(all_y_anomaly, all_pred_anomaly, average='binary', zero_division=0)
    recall = recall_score(all_y_anomaly, all_pred_anomaly, average='binary', zero_division=0)
    f1 = f1_score(all_y_anomaly, all_pred_anomaly, average='binary', zero_division=0)
    
    # 类型分类指标（只对异常样本）
    anomaly_mask = np.array(all_y_anomaly) == 1
    if anomaly_mask.sum() > 0:
        type_accuracy = accuracy_score(
            np.array(all_y_type)[anomaly_mask],
            np.array(all_pred_type)[anomaly_mask]
        )
    else:
        type_accuracy = 0.0
    
    metrics = {
        'loss': avg_loss,
        'anomaly_loss': avg_anomaly_loss,
        'type_loss': avg_type_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'type_accuracy': type_accuracy
    }
    
    return metrics


def train_model(train_data_path, test_data_path, epochs=50, batch_size=8, lr=0.001, 
                window_size=50, stride=25, device='cpu'):
    """
    训练模型
    
    参数:
    - train_data_path: 训练数据路径
    - test_data_path: 测试数据路径
    - epochs: 训练轮数
    - batch_size: 批大小
    - lr: 学习率
    - window_size: 时间窗口大小
    - stride: 滑动步长
    - device: 设备
    """
    print("="*60)
    print("开始训练Patch-GNN异常检测模型")
    print("="*60)
    
    # 创建数据集
    print("\n加载数据集...")
    train_dataset = DroneSwarmDataset(train_data_path, window_size=window_size, stride=stride)
    test_dataset = DroneSwarmDataset(test_data_path, window_size=window_size, stride=stride)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 创建模型
    print("\n创建模型...")
    input_dim = len(train_dataset.feature_cols)
    model = create_model(input_dim=input_dim, num_drones=10, patch_size=10, 
                        hidden_dim=64, num_anomaly_types=4)
    model = model.to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 损失函数
    criterion_anomaly = nn.CrossEntropyLoss()
    criterion_type = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                     patience=5, verbose=True)
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_anomaly_loss': [],
        'train_type_loss': [],
        'test_loss': [],
        'test_metrics': []
    }
    
    # 训练循环
    print("\n开始训练...")
    best_f1 = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 60)
        
        # 训练
        train_loss, train_anomaly_loss, train_type_loss = train_epoch(
            model, train_loader, optimizer, device, criterion_anomaly, criterion_type
        )
        
        # 评估
        test_metrics = evaluate(model, test_loader, device, criterion_anomaly, criterion_type)
        
        # 更新学习率
        scheduler.step(test_metrics['loss'])
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_anomaly_loss'].append(train_anomaly_loss)
        history['train_type_loss'].append(train_type_loss)
        history['test_loss'].append(test_metrics['loss'])
        history['test_metrics'].append(test_metrics)
        
        # 打印结果
        print(f"\n训练损失: {train_loss:.4f}")
        print(f"  - 异常检测损失: {train_anomaly_loss:.4f}")
        print(f"  - 类型分类损失: {train_type_loss:.4f}")
        print(f"测试损失: {test_metrics['loss']:.4f}")
        print(f"测试指标:")
        print(f"  - 准确率: {test_metrics['accuracy']:.4f}")
        print(f"  - 精确率: {test_metrics['precision']:.4f}")
        print(f"  - 召回率: {test_metrics['recall']:.4f}")
        print(f"  - F1分数: {test_metrics['f1']:.4f}")
        print(f"  - 类型准确率: {test_metrics['type_accuracy']:.4f}")
        
        # 保存最佳模型
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            os.makedirs('results', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_metrics': test_metrics,
                'history': history
            }, 'results/best_model.pth')
            print(f"\n保存最佳模型 (F1={best_f1:.4f})")
    
    # 保存训练历史
    with open('results/training_history.json', 'w') as f:
        # 转换numpy类型为Python类型
        history_serializable = {
            'train_loss': history['train_loss'],
            'train_anomaly_loss': history['train_anomaly_loss'],
            'train_type_loss': history['train_type_loss'],
            'test_loss': history['test_loss'],
            'test_metrics': [
                {k: float(v) for k, v in m.items()}
                for m in history['test_metrics']
            ]
        }
        json.dump(history_serializable, f, indent=2)
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"最佳F1分数: {best_f1:.4f}")
    print(f"模型已保存到: results/best_model.pth")
    print(f"训练历史已保存到: results/training_history.json")
    
    return model, history


def main():
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 训练模型
    model, history = train_model(
        train_data_path='data/train_data.csv',
        test_data_path='data/test_data.csv',
        epochs=50,
        batch_size=8,
        lr=0.001,
        window_size=50,
        stride=25,
        device=device
    )


if __name__ == "__main__":
    main()
