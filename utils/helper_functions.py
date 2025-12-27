"""
工具函数模块
包含数据处理、评估等常用工具函数
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred):
    """
    计算评估指标
    
    参数:
    - y_true: 真实标签
    - y_pred: 预测标签
    
    返回: 指标字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0)
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """
    绘制混淆矩阵
    
    参数:
    - y_true: 真实标签
    - y_pred: 预测标签
    - labels: 类别标签
    - save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def load_drone_data(file_path, drone_id=None):
    """
    加载无人机数据
    
    参数:
    - file_path: 数据文件路径
    - drone_id: 无人机ID（可选，如果指定则只返回该无人机的数据）
    
    返回: DataFrame
    """
    data = pd.read_csv(file_path)
    
    if drone_id is not None:
        data = data[data['drone_id'] == drone_id]
    
    return data


def get_anomaly_segments(anomaly_labels):
    """
    从异常标签中提取异常段
    
    参数:
    - anomaly_labels: 异常标签数组（0表示正常，1表示异常）
    
    返回: 异常段列表 [(start, end), ...]
    """
    segments = []
    in_anomaly = False
    start = None
    
    for i, is_anomaly in enumerate(anomaly_labels):
        if is_anomaly == 1 and not in_anomaly:
            start = i
            in_anomaly = True
        elif is_anomaly == 0 and in_anomaly:
            segments.append((start, i-1))
            in_anomaly = False
    
    if in_anomaly:
        segments.append((start, len(anomaly_labels)-1))
    
    return segments


def calculate_detection_delay(true_segments, pred_segments, tolerance=10):
    """
    计算检测延迟
    
    参数:
    - true_segments: 真实异常段 [(start, end), ...]
    - pred_segments: 预测异常段 [(start, end), ...]
    - tolerance: 容忍度（时间步）
    
    返回: 平均检测延迟
    """
    delays = []
    
    for true_start, true_end in true_segments:
        min_delay = float('inf')
        
        for pred_start, pred_end in pred_segments:
            # 检查预测段是否覆盖真实段
            if pred_start <= true_end and pred_end >= true_start:
                # 计算延迟（预测开始 - 真实开始）
                delay = max(0, pred_start - true_start)
                min_delay = min(min_delay, delay)
        
        if min_delay != float('inf'):
            delays.append(min_delay)
        else:
            # 未检测到，使用异常段长度作为延迟
            delays.append(true_end - true_start + 1)
    
    return np.mean(delays) if delays else 0


def print_summary(data_path='data/drone_swarm_with_anomalies.csv'):
    """
    打印数据集摘要信息
    
    参数:
    - data_path: 数据文件路径
    """
    data = pd.read_csv(data_path)
    
    print("="*60)
    print("数据集摘要")
    print("="*60)
    
    print(f"\n总样本数: {len(data)}")
    print(f"无人机数量: {data['drone_id'].nunique()}")
    print(f"时间跨度: {data['timestamp'].iloc[0]} 到 {data['timestamp'].iloc[-1]}")
    
    if 'anomaly' in data.columns:
        print(f"\n异常样本数: {(data['anomaly'] == 1).sum()}")
        print(f"正常样本数: {(data['anomaly'] == 0).sum()}")
        print(f"异常比例: {(data['anomaly'] == 1).sum() / len(data) * 100:.2f}%")
    
    if 'anomaly_type' in data.columns:
        print("\n异常类型分布:")
        type_counts = data[data['anomaly'] == 1]['anomaly_type'].value_counts()
        for anomaly_type, count in type_counts.items():
            print(f"  - {anomaly_type}: {count}")
    
    print("\n特征统计:")
    feature_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'battery', 'temperature', 'signal_strength']
    for col in feature_cols:
        if col in data.columns:
            print(f"  - {col}: 均值={data[col].mean():.2f}, 标准差={data[col].std():.2f}")
    
    print("="*60)


def save_predictions(predictions, output_path, drone_id=None):
    """
    保存预测结果
    
    参数:
    - predictions: 预测结果字典
    - output_path: 输出路径
    - drone_id: 无人机ID（可选）
    """
    df = pd.DataFrame(predictions)
    
    if drone_id is not None:
        df['drone_id'] = drone_id
    
    df.to_csv(output_path, index=False)
    print(f"预测结果已保存到: {output_path}")


if __name__ == "__main__":
    # 打印数据集摘要
    print_summary('data/drone_swarm_with_anomalies.csv')
