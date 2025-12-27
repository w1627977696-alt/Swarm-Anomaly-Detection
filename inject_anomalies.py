"""
异常注入脚本
在正常的无人机集群数据中注入3种典型异常类型
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime

class AnomalyInjector:
    """
    无人机集群异常注入类
    
    实现三种典型异常：
    1. 位置偏离异常 (Position Deviation): 无人机偏离预定轨迹
    2. 传感器故障异常 (Sensor Malfunction): 传感器数据异常（温度、电池、信号）
    3. 机动异常 (Maneuver Anomaly): 速度和加速度突变
    """
    
    def __init__(self, data, seed=42):
        """
        初始化异常注入器
        
        参数:
        - data: 正常的无人机集群数据 (DataFrame)
        - seed: 随机种子
        """
        self.data = data.copy()
        self.seed = seed
        np.random.seed(seed)
        
        # 添加标签列
        self.data['anomaly'] = 0  # 0表示正常
        self.data['anomaly_type'] = 'normal'  # 异常类型标签
        
        # 记录异常注入的详细信息
        self.anomaly_log = []
    
    def inject_position_deviation(self, drone_id, start_idx, duration, deviation_magnitude=20):
        """
        注入位置偏离异常
        
        描述：模拟无人机由于GPS故障、风力扰动或控制系统问题导致的位置偏离
        
        参数:
        - drone_id: 目标无人机ID
        - start_idx: 异常开始的索引
        - duration: 异常持续时间（样本数）
        - deviation_magnitude: 偏离幅度（米）
        """
        print(f"\n正在注入位置偏离异常到无人机 {drone_id}...")
        
        # 获取目标无人机的数据索引
        drone_mask = self.data['drone_id'] == drone_id
        drone_indices = self.data[drone_mask].index
        
        if start_idx >= len(drone_indices) - duration:
            print(f"警告: 起始索引过大，调整为 {len(drone_indices) - duration - 1}")
            start_idx = len(drone_indices) - duration - 1
        
        # 确定异常段的实际索引
        anomaly_indices = drone_indices[start_idx:start_idx + duration]
        
        # 生成渐进的位置偏离（模拟逐渐偏离的过程）
        t = np.linspace(0, 1, duration)
        
        # X和Y方向的随机偏离
        deviation_x = deviation_magnitude * np.sin(2 * np.pi * t) * t
        deviation_y = deviation_magnitude * np.cos(2 * np.pi * t) * t
        deviation_z = deviation_magnitude * 0.3 * t  # Z方向偏离较小
        
        # 应用偏离
        self.data.loc[anomaly_indices, 'x'] += deviation_x
        self.data.loc[anomaly_indices, 'y'] += deviation_y
        self.data.loc[anomaly_indices, 'z'] += deviation_z
        
        # 标记异常
        self.data.loc[anomaly_indices, 'anomaly'] = 1
        self.data.loc[anomaly_indices, 'anomaly_type'] = 'position_deviation'
        
        # 记录日志
        log_entry = {
            'type': 'position_deviation',
            'drone_id': drone_id,
            'start_idx': start_idx,
            'duration': duration,
            'indices': anomaly_indices.tolist(),
            'description': f'无人机{drone_id}在位置偏离异常段偏离正常轨迹约{deviation_magnitude}米'
        }
        self.anomaly_log.append(log_entry)
        
        print(f"位置偏离异常已注入：无人机{drone_id}, 持续{duration}秒, 偏离幅度{deviation_magnitude}米")
    
    def inject_sensor_malfunction(self, drone_id, start_idx, duration, sensor_type='battery'):
        """
        注入传感器故障异常
        
        描述：模拟传感器读数异常，包括电池电量突降、温度异常或信号丢失
        
        参数:
        - drone_id: 目标无人机ID
        - start_idx: 异常开始的索引
        - duration: 异常持续时间（样本数）
        - sensor_type: 传感器类型 ('battery', 'temperature', 'signal_strength')
        """
        print(f"\n正在注入传感器故障异常到无人机 {drone_id} ({sensor_type})...")
        
        # 获取目标无人机的数据索引
        drone_mask = self.data['drone_id'] == drone_id
        drone_indices = self.data[drone_mask].index
        
        if start_idx >= len(drone_indices) - duration:
            start_idx = len(drone_indices) - duration - 1
        
        anomaly_indices = drone_indices[start_idx:start_idx + duration]
        
        # 根据传感器类型注入不同的异常
        if sensor_type == 'battery':
            # 电池电量突然下降
            original_values = self.data.loc[anomaly_indices, 'battery'].values
            drop_amount = np.linspace(0, 30, duration)  # 逐渐下降30%
            self.data.loc[anomaly_indices, 'battery'] = original_values - drop_amount
            self.data.loc[anomaly_indices, 'battery'] = self.data.loc[anomaly_indices, 'battery'].clip(lower=0)
            description = f'无人机{drone_id}电池电量异常下降约30%'
            
        elif sensor_type == 'temperature':
            # 温度异常升高（可能是散热问题）
            original_values = self.data.loc[anomaly_indices, 'temperature'].values
            temp_spike = 15 + 10 * np.sin(np.linspace(0, 2*np.pi, duration))
            self.data.loc[anomaly_indices, 'temperature'] = original_values + temp_spike
            description = f'无人机{drone_id}温度异常升高15-25摄氏度'
            
        elif sensor_type == 'signal_strength':
            # 信号强度突然减弱（通信干扰）
            weak_signal = -90 + np.random.normal(0, 5, duration)
            self.data.loc[anomaly_indices, 'signal_strength'] = weak_signal
            description = f'无人机{drone_id}信号强度异常减弱至-90dBm左右'
        
        # 标记异常
        self.data.loc[anomaly_indices, 'anomaly'] = 1
        self.data.loc[anomaly_indices, 'anomaly_type'] = f'sensor_malfunction_{sensor_type}'
        
        # 记录日志
        log_entry = {
            'type': f'sensor_malfunction_{sensor_type}',
            'drone_id': drone_id,
            'start_idx': start_idx,
            'duration': duration,
            'indices': anomaly_indices.tolist(),
            'description': description
        }
        self.anomaly_log.append(log_entry)
        
        print(f"传感器故障异常已注入：{description}, 持续{duration}秒")
    
    def inject_maneuver_anomaly(self, drone_id, start_idx, duration, intensity=3.0):
        """
        注入机动异常
        
        描述：模拟无人机异常机动，如突然加速、急转弯或振荡
        
        参数:
        - drone_id: 目标无人机ID
        - start_idx: 异常开始的索引
        - duration: 异常持续时间（样本数）
        - intensity: 异常强度倍数
        """
        print(f"\n正在注入机动异常到无人机 {drone_id}...")
        
        # 获取目标无人机的数据索引
        drone_mask = self.data['drone_id'] == drone_id
        drone_indices = self.data[drone_mask].index
        
        if start_idx >= len(drone_indices) - duration:
            start_idx = len(drone_indices) - duration - 1
        
        anomaly_indices = drone_indices[start_idx:start_idx + duration]
        
        # 生成振荡式的速度和加速度变化
        t = np.linspace(0, 4*np.pi, duration)
        
        # 速度异常：添加高频振荡
        oscillation_vx = intensity * np.sin(t)
        oscillation_vy = intensity * np.cos(t)
        oscillation_vz = intensity * 0.5 * np.sin(2*t)
        
        self.data.loc[anomaly_indices, 'vx'] += oscillation_vx
        self.data.loc[anomaly_indices, 'vy'] += oscillation_vy
        self.data.loc[anomaly_indices, 'vz'] += oscillation_vz
        
        # 加速度异常：更剧烈的变化
        oscillation_ax = intensity * 2 * np.cos(t)
        oscillation_ay = intensity * 2 * np.sin(t)
        oscillation_az = intensity * np.cos(2*t)
        
        self.data.loc[anomaly_indices, 'ax'] += oscillation_ax
        self.data.loc[anomaly_indices, 'ay'] += oscillation_ay
        self.data.loc[anomaly_indices, 'az'] += oscillation_az
        
        # 姿态角也会受到影响
        self.data.loc[anomaly_indices, 'roll'] += intensity * 10 * np.sin(t)
        self.data.loc[anomaly_indices, 'pitch'] += intensity * 5 * np.cos(t)
        
        # 标记异常
        self.data.loc[anomaly_indices, 'anomaly'] = 1
        self.data.loc[anomaly_indices, 'anomaly_type'] = 'maneuver_anomaly'
        
        # 记录日志
        log_entry = {
            'type': 'maneuver_anomaly',
            'drone_id': drone_id,
            'start_idx': start_idx,
            'duration': duration,
            'indices': anomaly_indices.tolist(),
            'description': f'无人机{drone_id}出现异常机动，速度和加速度产生{intensity}倍强度的振荡'
        }
        self.anomaly_log.append(log_entry)
        
        print(f"机动异常已注入：无人机{drone_id}, 持续{duration}秒, 强度{intensity}倍")
    
    def inject_all_anomalies(self):
        """
        批量注入所有类型的异常
        
        策略：
        - 在不同的无人机上注入不同类型的异常
        - 在不同的时间段注入，避免重叠
        - 每种异常类型注入多个实例
        """
        print("="*60)
        print("开始批量注入异常...")
        print("="*60)
        
        # 获取每架无人机的样本数
        samples_per_drone = len(self.data[self.data['drone_id'] == 0])
        
        # 1. 位置偏离异常 - 在无人机0, 3, 7上注入
        self.inject_position_deviation(drone_id=0, start_idx=1000, duration=300, deviation_magnitude=20)
        self.inject_position_deviation(drone_id=3, start_idx=2000, duration=400, deviation_magnitude=25)
        self.inject_position_deviation(drone_id=7, start_idx=1500, duration=350, deviation_magnitude=30)
        
        # 2. 传感器故障异常 - 在无人机1, 4, 8上注入不同传感器故障
        self.inject_sensor_malfunction(drone_id=1, start_idx=1200, duration=250, sensor_type='battery')
        self.inject_sensor_malfunction(drone_id=4, start_idx=2200, duration=300, sensor_type='temperature')
        self.inject_sensor_malfunction(drone_id=8, start_idx=1800, duration=280, sensor_type='signal_strength')
        
        # 3. 机动异常 - 在无人机2, 5, 9上注入
        self.inject_maneuver_anomaly(drone_id=2, start_idx=1400, duration=200, intensity=3.0)
        self.inject_maneuver_anomaly(drone_id=5, start_idx=2400, duration=250, intensity=3.5)
        self.inject_maneuver_anomaly(drone_id=9, start_idx=1600, duration=220, intensity=4.0)
        
        print("\n" + "="*60)
        print("异常注入完成！")
        print("="*60)
        print(f"\n总共注入了 {len(self.anomaly_log)} 个异常段")
        print(f"异常样本数: {(self.data['anomaly'] == 1).sum()}")
        print(f"正常样本数: {(self.data['anomaly'] == 0).sum()}")
        print(f"异常比例: {(self.data['anomaly'] == 1).sum() / len(self.data) * 100:.2f}%")
    
    def save_data_and_log(self, output_dir='data'):
        """
        保存带有异常的数据和异常注入日志
        
        参数:
        - output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存完整数据
        data_file = os.path.join(output_dir, 'drone_swarm_with_anomalies.csv')
        self.data.to_csv(data_file, index=False)
        print(f"\n带异常的数据已保存到: {data_file}")
        
        # 保存异常日志
        log_file = os.path.join(output_dir, 'anomaly_injection_log.txt')
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("无人机集群异常注入日志\n")
            f.write("="*80 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总异常数: {len(self.anomaly_log)}\n")
            f.write(f"异常样本数: {(self.data['anomaly'] == 1).sum()}\n")
            f.write(f"正常样本数: {(self.data['anomaly'] == 0).sum()}\n\n")
            
            f.write("="*80 + "\n")
            f.write("异常类型说明\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. 位置偏离异常 (Position Deviation)\n")
            f.write("   - 描述: 无人机偏离预定飞行轨迹\n")
            f.write("   - 原因: GPS故障、风力扰动、控制系统问题\n")
            f.write("   - 特征: x, y, z坐标出现渐进式偏离\n\n")
            
            f.write("2. 传感器故障异常 (Sensor Malfunction)\n")
            f.write("   - 描述: 传感器读数异常\n")
            f.write("   - 子类型:\n")
            f.write("     a) 电池异常: 电量突然大幅下降\n")
            f.write("     b) 温度异常: 温度异常升高（散热问题）\n")
            f.write("     c) 信号异常: 信号强度突然减弱（通信干扰）\n\n")
            
            f.write("3. 机动异常 (Maneuver Anomaly)\n")
            f.write("   - 描述: 速度和加速度出现异常变化\n")
            f.write("   - 原因: 控制算法错误、电机故障、传感器干扰\n")
            f.write("   - 特征: vx, vy, vz, ax, ay, az出现高频振荡\n\n")
            
            f.write("="*80 + "\n")
            f.write("详细异常注入记录\n")
            f.write("="*80 + "\n\n")
            
            for i, log in enumerate(self.anomaly_log, 1):
                f.write(f"异常 #{i}\n")
                f.write(f"  类型: {log['type']}\n")
                f.write(f"  无人机ID: {log['drone_id']}\n")
                f.write(f"  起始索引: {log['start_idx']}\n")
                f.write(f"  持续时间: {log['duration']} 秒\n")
                f.write(f"  描述: {log['description']}\n")
                f.write(f"  数据索引范围: {log['indices'][0]} - {log['indices'][-1]}\n")
                f.write("\n")
        
        print(f"异常注入日志已保存到: {log_file}")
        
        return self.data
    
    def split_train_test(self, train_ratio=0.7, output_dir='data'):
        """
        划分训练集和测试集
        
        参数:
        - train_ratio: 训练集比例
        - output_dir: 输出目录
        
        返回: train_data, test_data
        """
        print(f"\n划分训练集和测试集 (训练集比例: {train_ratio})...")
        
        # 按时间顺序划分
        total_samples = len(self.data)
        split_idx = int(total_samples * train_ratio)
        
        train_data = self.data.iloc[:split_idx].copy()
        test_data = self.data.iloc[split_idx:].copy()
        
        # 保存训练集和测试集
        os.makedirs(output_dir, exist_ok=True)
        
        train_file = os.path.join(output_dir, 'train_data.csv')
        test_file = os.path.join(output_dir, 'test_data.csv')
        
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        
        print(f"训练集保存到: {train_file}")
        print(f"  - 样本数: {len(train_data)}")
        print(f"  - 异常样本数: {(train_data['anomaly'] == 1).sum()}")
        print(f"  - 正常样本数: {(train_data['anomaly'] == 0).sum()}")
        
        print(f"\n测试集保存到: {test_file}")
        print(f"  - 样本数: {len(test_data)}")
        print(f"  - 异常样本数: {(test_data['anomaly'] == 1).sum()}")
        print(f"  - 正常样本数: {(test_data['anomaly'] == 0).sum()}")
        
        return train_data, test_data


def main():
    """主函数：加载正常数据并注入异常"""
    # 加载正常数据
    print("加载正常无人机集群数据...")
    normal_data = pd.read_csv('data/drone_swarm_normal.csv')
    print(f"数据形状: {normal_data.shape}")
    
    # 创建异常注入器
    injector = AnomalyInjector(normal_data, seed=42)
    
    # 注入所有异常
    injector.inject_all_anomalies()
    
    # 保存数据和日志
    injector.save_data_and_log(output_dir='data')
    
    # 划分训练集和测试集
    train_data, test_data = injector.split_train_test(train_ratio=0.7, output_dir='data')
    
    print("\n" + "="*60)
    print("数据准备完成！")
    print("="*60)
    print("\n生成的文件:")
    print("  1. data/drone_swarm_normal.csv - 正常数据")
    print("  2. data/drone_swarm_with_anomalies.csv - 带异常的完整数据")
    print("  3. data/train_data.csv - 训练集")
    print("  4. data/test_data.csv - 测试集")
    print("  5. data/anomaly_injection_log.txt - 异常注入日志")


if __name__ == "__main__":
    main()
