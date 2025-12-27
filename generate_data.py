"""
无人机集群数据生成器
用于生成包含10架无人机的模拟数据，每架无人机包含多维时间序列数据
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

class DroneSwarmDataGenerator:
    """
    无人机集群数据生成类
    
    生成的特征包括：
    - timestamp: 时间戳
    - drone_id: 无人机编号 (0-9)
    - x, y, z: 三维位置坐标 (米)
    - vx, vy, vz: 三维速度 (米/秒)
    - ax, ay, az: 三维加速度 (米/秒²)
    - roll, pitch, yaw: 姿态角 (度)
    - battery: 电池电量 (百分比)
    - temperature: 温度 (摄氏度)
    - signal_strength: 信号强度 (dBm)
    """
    
    def __init__(self, num_drones=10, duration_minutes=60, sampling_rate=1):
        """
        初始化数据生成器
        
        参数:
        - num_drones: 无人机数量
        - duration_minutes: 数据持续时间（分钟）
        - sampling_rate: 采样率（Hz）
        """
        self.num_drones = num_drones
        self.duration_minutes = duration_minutes
        self.sampling_rate = sampling_rate
        self.num_samples = duration_minutes * 60 * sampling_rate
        
    def generate_circular_formation(self, t, drone_id, radius=50, height=50):
        """
        生成圆形编队飞行轨迹
        
        参数:
        - t: 时间索引
        - drone_id: 无人机ID
        - radius: 圆形编队半径
        - height: 飞行高度
        
        返回: x, y, z 坐标
        """
        # 每架无人机在圆周上均匀分布
        angle_offset = (2 * np.pi * drone_id) / self.num_drones
        # 添加时间相关的旋转
        angular_velocity = 0.01  # 角速度
        angle = angle_offset + angular_velocity * t
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height + 5 * np.sin(0.02 * t)  # 轻微的高度变化
        
        return x, y, z
    
    def calculate_velocity(self, positions):
        """
        根据位置计算速度
        
        参数:
        - positions: 位置数组 (n, 3)
        
        返回: 速度数组 (n, 3)
        """
        velocities = np.zeros_like(positions)
        velocities[1:] = np.diff(positions, axis=0) * self.sampling_rate
        velocities[0] = velocities[1]  # 第一个点使用第二个点的速度
        return velocities
    
    def calculate_acceleration(self, velocities):
        """
        根据速度计算加速度
        
        参数:
        - velocities: 速度数组 (n, 3)
        
        返回: 加速度数组 (n, 3)
        """
        accelerations = np.zeros_like(velocities)
        accelerations[1:] = np.diff(velocities, axis=0) * self.sampling_rate
        accelerations[0] = accelerations[1]
        return accelerations
    
    def generate_attitude(self, velocities):
        """
        根据速度生成姿态角
        
        参数:
        - velocities: 速度数组 (n, 3)
        
        返回: roll, pitch, yaw 数组
        """
        # 计算偏航角（根据水平速度方向）
        yaw = np.arctan2(velocities[:, 1], velocities[:, 0]) * 180 / np.pi
        
        # 俯仰角（根据垂直和水平速度）
        horizontal_speed = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
        pitch = np.arctan2(velocities[:, 2], horizontal_speed) * 180 / np.pi
        
        # 滚转角（模拟转弯时的倾斜）
        roll = np.zeros(len(velocities))
        for i in range(1, len(velocities)):
            # 根据偏航角变化率估计滚转
            yaw_rate = (yaw[i] - yaw[i-1]) * self.sampling_rate
            roll[i] = np.clip(yaw_rate * 5, -30, 30)  # 限制在±30度
        
        return roll, pitch, yaw
    
    def generate_battery_level(self, t_array):
        """
        生成电池电量数据（线性下降 + 噪声）
        
        参数:
        - t_array: 时间数组
        
        返回: 电池电量数组
        """
        # 从100%线性下降到80%
        battery = 100 - (20 * t_array / len(t_array))
        # 添加小的随机波动
        noise = np.random.normal(0, 0.5, len(t_array))
        battery = battery + noise
        return np.clip(battery, 0, 100)
    
    def generate_temperature(self, t_array):
        """
        生成温度数据（随时间缓慢上升 + 噪声）
        
        参数:
        - t_array: 时间数组
        
        返回: 温度数组
        """
        # 基础温度 + 缓慢上升 + 周期性变化 + 噪声
        base_temp = 25
        rising = 5 * t_array / len(t_array)
        periodic = 2 * np.sin(2 * np.pi * t_array / (len(t_array) / 5))
        noise = np.random.normal(0, 0.3, len(t_array))
        temperature = base_temp + rising + periodic + noise
        return temperature
    
    def generate_signal_strength(self, positions):
        """
        生成信号强度数据（基于与中心的距离）
        
        参数:
        - positions: 位置数组 (n, 3)
        
        返回: 信号强度数组 (dBm)
        """
        # 计算与原点的距离
        distances = np.sqrt(np.sum(positions**2, axis=1))
        # 信号强度随距离衰减（简化的自由空间路径损耗模型）
        signal = -40 - 20 * np.log10(distances + 1)  # +1 避免log(0)
        # 添加噪声
        noise = np.random.normal(0, 2, len(signal))
        signal = signal + noise
        return np.clip(signal, -100, -20)
    
    def generate_single_drone_data(self, drone_id, start_time):
        """
        生成单架无人机的数据
        
        参数:
        - drone_id: 无人机ID
        - start_time: 开始时间
        
        返回: DataFrame包含该无人机的所有数据
        """
        # 生成时间序列
        time_array = np.arange(self.num_samples)
        timestamps = [start_time + timedelta(seconds=i/self.sampling_rate) 
                     for i in range(self.num_samples)]
        
        # 生成位置数据
        positions = np.array([self.generate_circular_formation(t, drone_id) 
                            for t in time_array])
        
        # 添加小的随机扰动，使轨迹更真实
        noise = np.random.normal(0, 0.5, positions.shape)
        positions = positions + noise
        
        # 计算速度和加速度
        velocities = self.calculate_velocity(positions)
        accelerations = self.calculate_acceleration(velocities)
        
        # 生成姿态角
        roll, pitch, yaw = self.generate_attitude(velocities)
        
        # 生成其他传感器数据
        battery = self.generate_battery_level(time_array)
        temperature = self.generate_temperature(time_array)
        signal_strength = self.generate_signal_strength(positions)
        
        # 创建DataFrame
        data = {
            'timestamp': timestamps,
            'drone_id': drone_id,
            'x': positions[:, 0],
            'y': positions[:, 1],
            'z': positions[:, 2],
            'vx': velocities[:, 0],
            'vy': velocities[:, 1],
            'vz': velocities[:, 2],
            'ax': accelerations[:, 0],
            'ay': accelerations[:, 1],
            'az': accelerations[:, 2],
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'battery': battery,
            'temperature': temperature,
            'signal_strength': signal_strength
        }
        
        return pd.DataFrame(data)
    
    def generate_swarm_data(self, output_dir='data'):
        """
        生成整个无人机集群的数据
        
        参数:
        - output_dir: 输出目录
        
        返回: 包含所有无人机数据的DataFrame
        """
        print(f"开始生成{self.num_drones}架无人机的集群数据...")
        print(f"数据持续时间: {self.duration_minutes}分钟")
        print(f"采样率: {self.sampling_rate} Hz")
        print(f"总样本数: {self.num_samples}")
        
        start_time = datetime(2024, 1, 1, 10, 0, 0)
        
        all_data = []
        for drone_id in range(self.num_drones):
            print(f"生成无人机 {drone_id} 的数据...")
            drone_data = self.generate_single_drone_data(drone_id, start_time)
            all_data.append(drone_data)
        
        # 合并所有无人机的数据
        swarm_data = pd.concat(all_data, ignore_index=True)
        
        # 按时间戳和无人机ID排序
        swarm_data = swarm_data.sort_values(['timestamp', 'drone_id']).reset_index(drop=True)
        
        # 保存数据
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'drone_swarm_normal.csv')
        swarm_data.to_csv(output_file, index=False)
        print(f"\n数据已保存到: {output_file}")
        print(f"数据形状: {swarm_data.shape}")
        print(f"\n数据预览:")
        print(swarm_data.head(20))
        print(f"\n数据统计信息:")
        print(swarm_data.describe())
        
        return swarm_data


def main():
    """主函数：生成无人机集群数据"""
    # 设置随机种子以保证可复现性
    np.random.seed(42)
    
    # 创建数据生成器
    # 生成60分钟的数据，采样率1Hz（每秒1个样本）
    generator = DroneSwarmDataGenerator(
        num_drones=10,
        duration_minutes=60,
        sampling_rate=1
    )
    
    # 生成数据
    swarm_data = generator.generate_swarm_data(output_dir='data')
    
    print("\n数据生成完成！")
    print(f"特征维度: {len(swarm_data.columns)}")
    print(f"特征列表: {list(swarm_data.columns)}")


if __name__ == "__main__":
    main()
