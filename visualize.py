"""
可视化脚本
用于可视化模型预测结果，对比真实值和预测值，并标注异常段
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import os
import json

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.patch_gnn_model import create_model
from train import DroneSwarmDataset


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ResultVisualizer:
    """
    结果可视化类
    """
    
    def __init__(self, model_path, test_data_path, device='cpu'):
        """
        初始化
        
        参数:
        - model_path: 模型路径
        - test_data_path: 测试数据路径
        - device: 设备
        """
        self.device = device
        self.test_data_path = test_data_path
        
        # 加载数据
        print("加载测试数据...")
        self.test_data = pd.read_csv(test_data_path)
        self.dataset = DroneSwarmDataset(test_data_path, window_size=50, stride=25)
        
        # 加载模型
        print("加载模型...")
        checkpoint = torch.load(model_path, map_location=device)
        self.model = create_model(input_dim=15, num_drones=10, patch_size=10, 
                                  hidden_dim=64, num_anomaly_types=4)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"模型加载成功 (Epoch {checkpoint['epoch']})")
        print(f"测试指标: {checkpoint['test_metrics']}")
        
        # 异常类型映射
        self.anomaly_type_names = {
            0: 'Normal',
            1: 'Position Deviation',
            2: 'Sensor Malfunction',
            3: 'Maneuver Anomaly'
        }
        
        # 颜色映射
        self.anomaly_colors = {
            0: 'green',
            1: 'red',
            2: 'orange',
            3: 'purple'
        }
    
    def predict_full_sequence(self, drone_id, start_idx=0, length=1000):
        """
        对完整序列进行预测
        
        参数:
        - drone_id: 无人机ID
        - start_idx: 起始索引
        - length: 序列长度
        
        返回: 预测结果和真实标签
        """
        print(f"\n预测无人机 {drone_id} 的序列...")
        
        # 获取该无人机的数据
        drone_data = self.test_data[self.test_data['drone_id'] == drone_id]
        end_idx = min(start_idx + length, len(drone_data))
        drone_data = drone_data.iloc[start_idx:end_idx]
        
        # 准备预测结果容器
        predictions = {
            'anomaly_pred': np.zeros(len(drone_data)),
            'anomaly_true': drone_data['anomaly'].values,
            'type_pred': np.zeros(len(drone_data)),
            'type_true': np.zeros(len(drone_data))
        }
        
        # 映射真实的异常类型
        type_map = {
            'normal': 0,
            'position_deviation': 1,
            'sensor_malfunction_battery': 2,
            'sensor_malfunction_temperature': 2,
            'sensor_malfunction_signal_strength': 2,
            'maneuver_anomaly': 3
        }
        
        for i, anomaly_type in enumerate(drone_data['anomaly_type'].values):
            predictions['type_true'][i] = type_map.get(anomaly_type, 0)
        
        # 使用滑动窗口进行预测
        window_size = 50
        stride = 1  # 使用小步长以获得更密集的预测
        
        with torch.no_grad():
            for i in range(0, len(drone_data) - window_size + 1, stride):
                # 获取所有无人机的窗口数据
                x_list = []
                for d_id in range(10):
                    d_data = self.test_data[self.test_data['drone_id'] == d_id]
                    window_data = d_data.iloc[start_idx + i:start_idx + i + window_size]
                    features = window_data[self.dataset.feature_cols].values
                    
                    # 标准化
                    for j, col in enumerate(self.dataset.feature_cols):
                        features[:, j] = (features[:, j] - self.dataset.mean[col]) / (self.dataset.std[col] + 1e-8)
                    
                    x_list.append(features)
                
                x = torch.FloatTensor(np.array(x_list)).unsqueeze(0).to(self.device)
                # x: [1, num_drones, window_size, num_features]
                
                # 预测
                anomaly_scores, anomaly_types, _ = self.model(x)
                
                # 提取目标无人机的预测
                pred_anomaly = torch.argmax(anomaly_scores[0, drone_id], dim=0).cpu().item()
                pred_type = torch.argmax(anomaly_types[0, drone_id], dim=0).cpu().item()
                
                # 将预测分配到窗口的中间点
                center_idx = i + window_size // 2
                if center_idx < len(drone_data):
                    predictions['anomaly_pred'][center_idx] = pred_anomaly
                    predictions['type_pred'][center_idx] = pred_type
        
        # 对于没有预测的点，使用最近邻插值
        for key in ['anomaly_pred', 'type_pred']:
            zero_indices = np.where(predictions[key] == 0)[0]
            if len(zero_indices) > 0 and len(zero_indices) < len(predictions[key]):
                non_zero_indices = np.where(predictions[key] != 0)[0]
                if len(non_zero_indices) > 0:
                    for idx in zero_indices:
                        # 找到最近的非零值
                        distances = np.abs(non_zero_indices - idx)
                        nearest_idx = non_zero_indices[np.argmin(distances)]
                        predictions[key][idx] = predictions[key][nearest_idx]
        
        return predictions, drone_data
    
    def plot_single_drone_results(self, drone_id, start_idx=0, length=1000, output_dir='results'):
        """
        绘制单架无人机的预测结果
        
        参数:
        - drone_id: 无人机ID
        - start_idx: 起始索引
        - length: 序列长度
        - output_dir: 输出目录
        """
        print(f"\n可视化无人机 {drone_id} 的结果...")
        
        # 预测
        predictions, drone_data = self.predict_full_sequence(drone_id, start_idx, length)
        
        # 创建图形
        fig, axes = plt.subplots(5, 1, figsize=(16, 12))
        fig.suptitle(f'Drone {drone_id} - Anomaly Detection Results', fontsize=16, fontweight='bold')
        
        time_indices = np.arange(len(drone_data))
        
        # 1. 位置特征 (x, y, z)
        ax = axes[0]
        ax.plot(time_indices, drone_data['x'].values, label='X', alpha=0.7)
        ax.plot(time_indices, drone_data['y'].values, label='Y', alpha=0.7)
        ax.plot(time_indices, drone_data['z'].values, label='Z', alpha=0.7)
        ax.set_ylabel('Position')
        ax.set_title('Position Features')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        self._add_anomaly_background(ax, predictions['anomaly_true'], time_indices)
        
        # 2. 速度特征
        ax = axes[1]
        velocity_magnitude = np.sqrt(drone_data['vx']**2 + drone_data['vy']**2 + drone_data['vz']**2)
        ax.plot(time_indices, velocity_magnitude, label='Velocity Magnitude', color='blue', alpha=0.7)
        ax.set_ylabel('Velocity')
        ax.set_title('Velocity Magnitude')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        self._add_anomaly_background(ax, predictions['anomaly_true'], time_indices)
        
        # 3. 传感器数据
        ax = axes[2]
        ax2 = ax.twinx()
        ax.plot(time_indices, drone_data['battery'].values, label='Battery', color='green', alpha=0.7)
        ax.plot(time_indices, drone_data['temperature'].values, label='Temperature', color='red', alpha=0.7)
        ax2.plot(time_indices, drone_data['signal_strength'].values, label='Signal', color='purple', alpha=0.7, linestyle='--')
        ax.set_ylabel('Battery / Temperature')
        ax2.set_ylabel('Signal Strength (dBm)')
        ax.set_title('Sensor Data')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        self._add_anomaly_background(ax, predictions['anomaly_true'], time_indices)
        
        # 4. 异常检测结果对比
        ax = axes[3]
        ax.plot(time_indices, predictions['anomaly_true'], label='Ground Truth', 
                linewidth=3, alpha=0.6, color='blue')
        ax.plot(time_indices, predictions['anomaly_pred'], label='Prediction', 
                linewidth=2, alpha=0.8, color='red', linestyle='--')
        ax.set_ylabel('Anomaly')
        ax.set_title('Anomaly Detection: Prediction vs Ground Truth')
        ax.set_ylim(-0.1, 1.1)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 5. 异常类型对比
        ax = axes[4]
        ax.plot(time_indices, predictions['type_true'], label='True Type', 
                linewidth=3, alpha=0.6, color='blue', marker='o', markersize=2)
        ax.plot(time_indices, predictions['type_pred'], label='Predicted Type', 
                linewidth=2, alpha=0.8, color='red', linestyle='--', marker='x', markersize=2)
        ax.set_ylabel('Anomaly Type')
        ax.set_xlabel('Time Step')
        ax.set_title('Anomaly Type Classification: Prediction vs Ground Truth')
        ax.set_ylim(-0.1, 3.1)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['Normal', 'Position\nDev', 'Sensor\nMal', 'Maneuver\nAnom'])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图形
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'drone_{drone_id}_results.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"图形已保存到: {output_file}")
        plt.close()
        
        # 计算并打印指标
        self._print_metrics(predictions)
    
    def _add_anomaly_background(self, ax, anomaly_labels, time_indices):
        """
        在图表背景添加异常段标注
        
        参数:
        - ax: 坐标轴
        - anomaly_labels: 异常标签数组
        - time_indices: 时间索引
        """
        # 找出异常段
        anomaly_segments = []
        in_anomaly = False
        start = None
        
        for i, is_anomaly in enumerate(anomaly_labels):
            if is_anomaly == 1 and not in_anomaly:
                start = i
                in_anomaly = True
            elif is_anomaly == 0 and in_anomaly:
                anomaly_segments.append((start, i-1))
                in_anomaly = False
        
        if in_anomaly:
            anomaly_segments.append((start, len(anomaly_labels)-1))
        
        # 添加背景色
        ylim = ax.get_ylim()
        for start, end in anomaly_segments:
            ax.axvspan(time_indices[start], time_indices[end], 
                      alpha=0.2, color='red', label='Anomaly Region' if start == anomaly_segments[0][0] else '')
    
    def _print_metrics(self, predictions):
        """
        打印评估指标
        
        参数:
        - predictions: 预测结果字典
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        y_true = predictions['anomaly_true']
        y_pred = predictions['anomaly_pred']
        
        print("\n异常检测指标:")
        print(f"  准确率: {accuracy_score(y_true, y_pred):.4f}")
        print(f"  精确率: {precision_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"  召回率: {recall_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"  F1分数: {f1_score(y_true, y_pred, zero_division=0):.4f}")
        
        print("\n混淆矩阵:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        # 类型分类指标（只对异常样本）
        anomaly_mask = y_true == 1
        if anomaly_mask.sum() > 0:
            type_true = predictions['type_true'][anomaly_mask]
            type_pred = predictions['type_pred'][anomaly_mask]
            print(f"\n异常类型分类准确率: {accuracy_score(type_true, type_pred):.4f}")
    
    def plot_all_drones(self, start_idx=0, length=500, output_dir='results'):
        """
        绘制所有无人机的结果
        
        参数:
        - start_idx: 起始索引
        - length: 序列长度
        - output_dir: 输出目录
        """
        print("\n"+"="*60)
        print("开始可视化所有无人机的结果...")
        print("="*60)
        
        for drone_id in range(10):
            self.plot_single_drone_results(drone_id, start_idx, length, output_dir)
        
        print("\n"+"="*60)
        print("可视化完成！")
        print("="*60)
        print(f"所有图形已保存到: {output_dir}/")
    
    def plot_training_history(self, history_path='results/training_history.json', output_dir='results'):
        """
        绘制训练历史
        
        参数:
        - history_path: 训练历史JSON文件路径
        - output_dir: 输出目录
        """
        print("\n绘制训练历史...")
        
        # 加载训练历史
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # 1. 总损失
        ax = axes[0, 0]
        ax.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3)
        ax.plot(epochs, history['test_loss'], label='Test Loss', marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 分项损失
        ax = axes[0, 1]
        ax.plot(epochs, history['train_anomaly_loss'], label='Anomaly Loss', marker='o', markersize=3)
        ax.plot(epochs, history['train_type_loss'], label='Type Loss', marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Component Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 准确率和F1
        ax = axes[1, 0]
        test_accuracy = [m['accuracy'] for m in history['test_metrics']]
        test_f1 = [m['f1'] for m in history['test_metrics']]
        ax.plot(epochs, test_accuracy, label='Accuracy', marker='o', markersize=3)
        ax.plot(epochs, test_f1, label='F1 Score', marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Test Accuracy and F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 精确率和召回率
        ax = axes[1, 1]
        test_precision = [m['precision'] for m in history['test_metrics']]
        test_recall = [m['recall'] for m in history['test_metrics']]
        ax.plot(epochs, test_precision, label='Precision', marker='o', markersize=3)
        ax.plot(epochs, test_recall, label='Recall', marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Test Precision and Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图形
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'training_history.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"训练历史图已保存到: {output_file}")
        plt.close()


def main():
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建可视化器
    visualizer = ResultVisualizer(
        model_path='results/best_model.pth',
        test_data_path='data/test_data.csv',
        device=device
    )
    
    # 可视化结果
    visualizer.plot_all_drones(start_idx=0, length=800, output_dir='results')
    
    # 绘制训练历史
    visualizer.plot_training_history(
        history_path='results/training_history.json',
        output_dir='results'
    )
    
    print("\n所有可视化完成！")


if __name__ == "__main__":
    main()
