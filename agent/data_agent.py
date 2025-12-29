"""
数据处理Agent模块
Data Processing Agent Module

该模块包含两个数据相关的Agent：
1. DataPreprocessingAgent: 数据预处理与可视化Agent
2. DataReplayAgent: 数据回放Agent

功能：
- 数据加载、清洗和标准化
- 数据可视化（轨迹、传感器数据、异常检测结果）
- 数据回放（支持指定时间范围和无人机）
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DataPreprocessingAgent:
    """
    数据预处理与可视化Agent
    
    功能：
    1. 加载无人机集群数据
    2. 数据清洗和标准化
    3. 特征提取和统计分析
    4. 数据可视化
    """
    
    def __init__(self, config: Dict):
        """
        初始化数据预处理Agent
        
        参数:
            config: 配置字典，包含：
                - data_dir: 数据目录
                - output_dir: 输出目录
        """
        self.config = config
        self.data_dir = config.get('data_dir', 'data')
        self.output_dir = config.get('output_dir', 'results')
        
        # 数据缓存
        self.data = None
        self.train_data = None
        self.test_data = None
        
        # 特征列定义
        self.feature_cols = [
            'x', 'y', 'z',              # 位置
            'vx', 'vy', 'vz',           # 速度
            'ax', 'ay', 'az',           # 加速度
            'roll', 'pitch', 'yaw',     # 姿态
            'battery',                   # 电池
            'temperature',               # 温度
            'signal_strength'            # 信号强度
        ]
        
        # 标准化参数
        self.normalization_params = {}
        
        print("[DataPreprocessingAgent] 数据预处理Agent已初始化")
    
    def preprocess(self, parameters: Dict) -> Dict:
        """
        执行数据预处理
        
        参数:
            parameters: 预处理参数，可包含：
                - data_file: 指定数据文件
                - normalize: 是否标准化
        
        返回:
            预处理结果字典
        """
        print("[DataPreprocessingAgent] 开始数据预处理...")
        
        try:
            # 确定数据文件
            data_file = parameters.get('data_file', 
                os.path.join(self.data_dir, 'drone_swarm_with_anomalies.csv'))
            
            # 加载数据
            if not os.path.exists(data_file):
                # 尝试使用测试数据
                data_file = os.path.join(self.data_dir, 'test_data.csv')
                if not os.path.exists(data_file):
                    return {
                        'success': False,
                        'error': f'找不到数据文件: {data_file}'
                    }
            
            print(f"[DataPreprocessingAgent] 加载数据文件: {data_file}")
            self.data = pd.read_csv(data_file)
            
            # 数据验证
            missing_cols = [col for col in self.feature_cols 
                          if col not in self.data.columns]
            if missing_cols:
                return {
                    'success': False,
                    'error': f'数据缺少必要的列: {missing_cols}'
                }
            
            # 数据清洗
            original_count = len(self.data)
            self.data = self._clean_data(self.data)
            cleaned_count = len(self.data)
            
            # 标准化
            if parameters.get('normalize', True):
                self._compute_normalization_params()
            
            # 计算统计信息
            stats = self._compute_statistics()
            
            # 加载训练和测试数据
            train_file = os.path.join(self.data_dir, 'train_data.csv')
            test_file = os.path.join(self.data_dir, 'test_data.csv')
            
            if os.path.exists(train_file):
                self.train_data = pd.read_csv(train_file)
            if os.path.exists(test_file):
                self.test_data = pd.read_csv(test_file)
            
            return {
                'success': True,
                'message': '数据预处理完成',
                'summary': f'加载 {original_count} 条记录，清洗后 {cleaned_count} 条',
                'details': {
                    '数据文件': data_file,
                    '记录数': cleaned_count,
                    '无人机数': self.data['drone_id'].nunique(),
                    '特征数': len(self.feature_cols),
                    '异常样本数': (self.data['anomaly'] == 1).sum() if 'anomaly' in self.data.columns else 'N/A',
                    '正常样本数': (self.data['anomaly'] == 0).sum() if 'anomaly' in self.data.columns else 'N/A'
                },
                'statistics': stats
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'数据预处理失败: {str(e)}'
            }
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据
        
        参数:
            data: 原始数据
        
        返回:
            清洗后的数据
        """
        # 删除空值
        data = data.dropna(subset=self.feature_cols)
        
        # 处理异常值（可选）
        # 这里使用简单的范围限制
        for col in self.feature_cols:
            if col in data.columns:
                # 使用 3 倍标准差作为异常值边界
                mean = data[col].mean()
                std = data[col].std()
                data = data[(data[col] >= mean - 3*std) & 
                           (data[col] <= mean + 3*std)]
        
        return data.reset_index(drop=True)
    
    def _compute_normalization_params(self):
        """计算标准化参数"""
        for col in self.feature_cols:
            if col in self.data.columns:
                self.normalization_params[col] = {
                    'mean': self.data[col].mean(),
                    'std': self.data[col].std()
                }
    
    def _compute_statistics(self) -> Dict:
        """计算数据统计信息"""
        stats = {}
        for col in self.feature_cols:
            if col in self.data.columns:
                stats[col] = {
                    'mean': float(self.data[col].mean()),
                    'std': float(self.data[col].std()),
                    'min': float(self.data[col].min()),
                    'max': float(self.data[col].max())
                }
        return stats
    
    def visualize(self, parameters: Dict) -> Dict:
        """
        数据可视化
        
        参数:
            parameters: 可视化参数，可包含：
                - drone_ids: 要可视化的无人机ID列表
                - features: 要可视化的特征
                - time_range: 时间范围 (start, end)
                - plot_type: 图表类型
        
        返回:
            可视化结果
        """
        print("[DataPreprocessingAgent] 生成数据可视化...")
        
        if self.data is None:
            # 尝试自动加载数据
            preprocess_result = self.preprocess({})
            if not preprocess_result.get('success'):
                return preprocess_result
        
        try:
            drone_ids = parameters.get('drone_ids', list(range(10)))
            time_range = parameters.get('time_range', None)
            plot_type = parameters.get('plot_type', 'comprehensive')
            
            os.makedirs(self.output_dir, exist_ok=True)
            
            output_files = []
            
            # 根据图表类型生成不同的可视化
            if plot_type == 'comprehensive' or plot_type == 'all':
                # 生成综合可视化
                for drone_id in drone_ids:
                    if drone_id in self.data['drone_id'].values:
                        output_file = self._plot_drone_data(drone_id, time_range)
                        if output_file:
                            output_files.append(output_file)
            
            elif plot_type == 'trajectory':
                # 生成轨迹图
                output_file = self._plot_trajectories(drone_ids, time_range)
                if output_file:
                    output_files.append(output_file)
            
            elif plot_type == 'sensors':
                # 生成传感器数据图
                output_file = self._plot_sensor_data(drone_ids, time_range)
                if output_file:
                    output_files.append(output_file)
            
            elif plot_type == 'anomaly':
                # 生成异常分布图
                output_file = self._plot_anomaly_distribution()
                if output_file:
                    output_files.append(output_file)
            
            return {
                'success': True,
                'message': f'已生成 {len(output_files)} 个可视化图表',
                'output_files': output_files,
                'details': {
                    '可视化无人机': drone_ids,
                    '图表类型': plot_type,
                    '输出目录': self.output_dir
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'可视化失败: {str(e)}'
            }
    
    def _plot_drone_data(self, drone_id: int, time_range: Optional[Tuple] = None) -> str:
        """
        绘制单架无人机的数据
        
        参数:
            drone_id: 无人机ID
            time_range: 时间范围
        
        返回:
            输出文件路径
        """
        drone_data = self.data[self.data['drone_id'] == drone_id].copy()
        
        if time_range:
            start, end = time_range
            drone_data = drone_data.iloc[start:end]
        
        if len(drone_data) == 0:
            return None
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        fig.suptitle(f'Drone {drone_id} - Flight Data Overview', fontsize=14, fontweight='bold')
        
        time_idx = range(len(drone_data))
        
        # 1. 位置数据
        ax = axes[0]
        ax.plot(time_idx, drone_data['x'].values, label='X', alpha=0.8)
        ax.plot(time_idx, drone_data['y'].values, label='Y', alpha=0.8)
        ax.plot(time_idx, drone_data['z'].values, label='Z', alpha=0.8)
        ax.set_ylabel('Position (m)')
        ax.set_title('Position Data')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        if 'anomaly' in drone_data.columns:
            self._add_anomaly_regions(ax, drone_data['anomaly'].values, time_idx)
        
        # 2. 速度数据
        ax = axes[1]
        velocity = np.sqrt(drone_data['vx']**2 + drone_data['vy']**2 + drone_data['vz']**2)
        ax.plot(time_idx, velocity.values, label='Velocity Magnitude', color='blue', alpha=0.8)
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Velocity Data')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        if 'anomaly' in drone_data.columns:
            self._add_anomaly_regions(ax, drone_data['anomaly'].values, time_idx)
        
        # 3. 传感器数据
        ax = axes[2]
        ax2 = ax.twinx()
        ax.plot(time_idx, drone_data['battery'].values, label='Battery', color='green', alpha=0.8)
        ax.plot(time_idx, drone_data['temperature'].values, label='Temperature', color='red', alpha=0.8)
        ax2.plot(time_idx, drone_data['signal_strength'].values, label='Signal', 
                color='purple', alpha=0.8, linestyle='--')
        ax.set_ylabel('Battery / Temperature')
        ax2.set_ylabel('Signal Strength (dBm)')
        ax.set_title('Sensor Data')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        if 'anomaly' in drone_data.columns:
            self._add_anomaly_regions(ax, drone_data['anomaly'].values, time_idx)
        
        # 4. 异常标签
        if 'anomaly' in drone_data.columns:
            ax = axes[3]
            ax.fill_between(time_idx, 0, drone_data['anomaly'].values, 
                           alpha=0.5, color='red', label='Anomaly')
            ax.set_ylabel('Anomaly')
            ax.set_xlabel('Time Step')
            ax.set_title('Anomaly Labels')
            ax.set_ylim(-0.1, 1.1)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = os.path.join(self.output_dir, f'drone_{drone_id}_data.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[DataPreprocessingAgent] 已保存: {output_file}")
        return output_file
    
    def _plot_trajectories(self, drone_ids: List[int], time_range: Optional[Tuple] = None) -> str:
        """
        绘制3D轨迹图
        
        参数:
            drone_ids: 无人机ID列表
            time_range: 时间范围
        
        返回:
            输出文件路径
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(drone_ids)))
        
        for i, drone_id in enumerate(drone_ids):
            drone_data = self.data[self.data['drone_id'] == drone_id]
            
            if time_range:
                start, end = time_range
                drone_data = drone_data.iloc[start:end]
            
            if len(drone_data) > 0:
                ax.plot(drone_data['x'].values, 
                       drone_data['y'].values, 
                       drone_data['z'].values, 
                       color=colors[i], alpha=0.7, label=f'Drone {drone_id}')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Drone Swarm 3D Trajectories', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        
        output_file = os.path.join(self.output_dir, 'drone_trajectories_3d.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[DataPreprocessingAgent] 已保存: {output_file}")
        return output_file
    
    def _plot_sensor_data(self, drone_ids: List[int], time_range: Optional[Tuple] = None) -> str:
        """
        绘制传感器数据对比图
        
        参数:
            drone_ids: 无人机ID列表
            time_range: 时间范围
        
        返回:
            输出文件路径
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle('Sensor Data Comparison', fontsize=14, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(drone_ids)))
        
        sensor_configs = [
            ('battery', 'Battery Level (%)', 'Battery'),
            ('temperature', 'Temperature (°C)', 'Temperature'),
            ('signal_strength', 'Signal Strength (dBm)', 'Signal')
        ]
        
        for ax_idx, (sensor, ylabel, title) in enumerate(sensor_configs):
            ax = axes[ax_idx]
            
            for i, drone_id in enumerate(drone_ids):
                drone_data = self.data[self.data['drone_id'] == drone_id]
                
                if time_range:
                    start, end = time_range
                    drone_data = drone_data.iloc[start:end]
                
                if len(drone_data) > 0:
                    time_idx = range(len(drone_data))
                    ax.plot(time_idx, drone_data[sensor].values, 
                           color=colors[i], alpha=0.6, label=f'Drone {drone_id}')
            
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(loc='upper right', ncol=5, fontsize=8)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time Step')
        
        plt.tight_layout()
        
        output_file = os.path.join(self.output_dir, 'sensor_data_comparison.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[DataPreprocessingAgent] 已保存: {output_file}")
        return output_file
    
    def _plot_anomaly_distribution(self) -> str:
        """
        绘制异常分布图
        
        返回:
            输出文件路径
        """
        if 'anomaly_type' not in self.data.columns:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Anomaly Distribution Analysis', fontsize=14, fontweight='bold')
        
        # 1. 异常类型分布
        ax = axes[0]
        anomaly_data = self.data[self.data['anomaly'] == 1]
        type_counts = anomaly_data['anomaly_type'].value_counts()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        ax.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
              colors=colors[:len(type_counts)])
        ax.set_title('Anomaly Type Distribution')
        
        # 2. 各无人机异常数量
        ax = axes[1]
        drone_anomaly_counts = anomaly_data.groupby('drone_id').size()
        ax.bar(drone_anomaly_counts.index, drone_anomaly_counts.values, 
               color='#FF6B6B', alpha=0.7)
        ax.set_xlabel('Drone ID')
        ax.set_ylabel('Anomaly Count')
        ax.set_title('Anomaly Count per Drone')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_file = os.path.join(self.output_dir, 'anomaly_distribution.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[DataPreprocessingAgent] 已保存: {output_file}")
        return output_file
    
    def _add_anomaly_regions(self, ax, anomaly_labels: np.ndarray, time_idx):
        """
        在图表中添加异常区域标记
        
        参数:
            ax: matplotlib轴对象
            anomaly_labels: 异常标签数组
            time_idx: 时间索引
        """
        # 找出异常段
        in_anomaly = False
        start = None
        
        time_idx = list(time_idx)
        
        for i, is_anomaly in enumerate(anomaly_labels):
            if is_anomaly == 1 and not in_anomaly:
                start = time_idx[i]
                in_anomaly = True
            elif is_anomaly == 0 and in_anomaly:
                ax.axvspan(start, time_idx[i-1], alpha=0.2, color='red')
                in_anomaly = False
        
        if in_anomaly:
            ax.axvspan(start, time_idx[-1], alpha=0.2, color='red')
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """获取已加载的数据"""
        return self.data
    
    def get_test_data(self) -> Optional[pd.DataFrame]:
        """获取测试数据"""
        return self.test_data


class DataReplayAgent:
    """
    数据回放Agent
    
    功能：
    1. 回放历史飞行数据
    2. 支持指定时间范围和无人机
    3. 生成回放动画
    4. 标注异常事件
    """
    
    def __init__(self, config: Dict):
        """
        初始化数据回放Agent
        
        参数:
            config: 配置字典
        """
        self.config = config
        self.data_dir = config.get('data_dir', 'data')
        self.output_dir = config.get('output_dir', 'results')
        
        self.data = None
        
        print("[DataReplayAgent] 数据回放Agent已初始化")
    
    def replay(self, parameters: Dict) -> Dict:
        """
        执行数据回放
        
        参数:
            parameters: 回放参数，可包含：
                - drone_ids: 要回放的无人机ID
                - time_range: 时间范围 (start, end)
                - speed: 回放速度
                - output_format: 输出格式
        
        返回:
            回放结果
        """
        print("[DataReplayAgent] 开始数据回放...")
        
        try:
            # 加载数据
            if self.data is None:
                data_file = os.path.join(self.data_dir, 'test_data.csv')
                if not os.path.exists(data_file):
                    data_file = os.path.join(self.data_dir, 'drone_swarm_with_anomalies.csv')
                
                if not os.path.exists(data_file):
                    return {
                        'success': False,
                        'error': '找不到数据文件'
                    }
                
                self.data = pd.read_csv(data_file)
            
            # 获取参数
            drone_ids = parameters.get('drone_ids', list(range(10)))
            time_range = parameters.get('time_range', (0, 500))
            output_format = parameters.get('output_format', 'image')
            
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 根据输出格式生成不同的回放
            if output_format == 'animation':
                output_file = self._generate_animation(drone_ids, time_range)
            else:
                output_file = self._generate_replay_frames(drone_ids, time_range)
            
            return {
                'success': True,
                'message': '数据回放完成',
                'output_file': output_file,
                'details': {
                    '回放无人机': drone_ids,
                    '时间范围': time_range,
                    '输出格式': output_format
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'数据回放失败: {str(e)}'
            }
    
    def _generate_replay_frames(self, drone_ids: List[int], 
                                 time_range: Tuple[int, int]) -> str:
        """
        生成回放帧图像
        
        参数:
            drone_ids: 无人机ID列表
            time_range: 时间范围
        
        返回:
            输出文件路径
        """
        start, end = time_range
        
        # 创建多帧图像展示
        num_frames = min(6, (end - start) // 50)  # 最多6帧
        frame_indices = np.linspace(start, end-1, num_frames, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Data Replay: Time {start} to {end}', fontsize=14, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(drone_ids)))
        
        for idx, (ax, frame_idx) in enumerate(zip(axes.flatten(), frame_indices)):
            # 获取该时刻的数据
            for i, drone_id in enumerate(drone_ids):
                drone_data = self.data[self.data['drone_id'] == drone_id]
                
                if frame_idx < len(drone_data):
                    row = drone_data.iloc[frame_idx]
                    ax.scatter(row['x'], row['y'], c=[colors[i]], 
                             s=100, label=f'D{drone_id}', edgecolors='black')
            
            ax.set_xlim(-60, 60)
            ax.set_ylim(-60, 60)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'Time Step: {frame_idx}')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            if idx == 0:
                ax.legend(loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        output_file = os.path.join(self.output_dir, 
                                   f'replay_{start}_{end}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[DataReplayAgent] 已保存: {output_file}")
        return output_file
    
    def _generate_animation(self, drone_ids: List[int], 
                            time_range: Tuple[int, int]) -> str:
        """
        生成回放动画（简化版，保存为多帧图）
        
        参数:
            drone_ids: 无人机ID列表
            time_range: 时间范围
        
        返回:
            输出文件路径
        """
        # 由于环境限制，这里生成静态的多帧图像代替动画
        return self._generate_replay_frames(drone_ids, time_range)


if __name__ == "__main__":
    # 测试数据处理Agent
    config = {
        'data_dir': 'data',
        'output_dir': 'results'
    }
    
    print("="*60)
    print("测试数据处理Agent")
    print("="*60)
    
    # 测试数据预处理Agent
    preprocess_agent = DataPreprocessingAgent(config)
    result = preprocess_agent.preprocess({})
    print(f"\n预处理结果: {result}")
    
    # 测试可视化
    viz_result = preprocess_agent.visualize({'drone_ids': [0, 1], 'plot_type': 'comprehensive'})
    print(f"\n可视化结果: {viz_result}")
    
    # 测试数据回放Agent
    replay_agent = DataReplayAgent(config)
    replay_result = replay_agent.replay({'drone_ids': [0, 1, 2], 'time_range': (0, 300)})
    print(f"\n回放结果: {replay_result}")
