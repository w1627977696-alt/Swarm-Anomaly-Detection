"""
异常检测Agent模块
Anomaly Detection Agent Module

该模块实现了基于Patch-GNN模型的无人机集群异常检测功能。

功能：
1. 加载训练好的Patch-GNN模型
2. 对无人机集群数据进行异常检测
3. 识别异常类型
4. 生成检测结果报告
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.patch_gnn_model import create_model


class AnomalyDetectionAgent:
    """
    异常检测Agent
    
    功能：
    1. 使用Patch-GNN模型进行异常检测
    2. 识别异常类型（位置偏离、传感器故障、机动异常）
    3. 计算异常置信度
    4. 提供详细的检测结果
    """
    
    def __init__(self, config: Dict):
        """
        初始化异常检测Agent
        
        参数:
            config: 配置字典，包含：
                - model_path: 模型文件路径
                - data_dir: 数据目录
                - device: 计算设备
        """
        self.config = config
        self.model_path = config.get('model_path', 'results/best_model.pth')
        self.data_dir = config.get('data_dir', 'data')
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型参数
        self.input_dim = 15
        self.num_drones = 10
        self.patch_size = 10
        self.hidden_dim = 64
        self.num_anomaly_types = 4
        self.window_size = 50
        
        # 特征列
        self.feature_cols = [
            'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az',
            'roll', 'pitch', 'yaw', 'battery', 'temperature', 'signal_strength'
        ]
        
        # 异常类型映射
        self.anomaly_type_names = {
            0: 'normal',
            1: 'position_deviation',
            2: 'sensor_malfunction',
            3: 'maneuver_anomaly'
        }
        
        self.anomaly_type_names_cn = {
            0: '正常',
            1: '位置偏离异常',
            2: '传感器故障异常',
            3: '机动异常'
        }
        
        # 模型和数据缓存
        self.model = None
        self.data = None
        self.normalization_params = {}
        
        print(f"[AnomalyDetectionAgent] 异常检测Agent已初始化")
        print(f"[AnomalyDetectionAgent] 设备: {self.device}")
    
    def _load_model(self) -> bool:
        """
        加载模型
        
        返回:
            是否加载成功
        """
        if self.model is not None:
            return True
        
        try:
            if os.path.exists(self.model_path):
                print(f"[AnomalyDetectionAgent] 加载模型: {self.model_path}")
                
                self.model = create_model(
                    input_dim=self.input_dim,
                    num_drones=self.num_drones,
                    patch_size=self.patch_size,
                    hidden_dim=self.hidden_dim,
                    num_anomaly_types=self.num_anomaly_types
                )
                
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model = self.model.to(self.device)
                self.model.eval()
                
                print(f"[AnomalyDetectionAgent] 模型加载成功")
                return True
            else:
                print(f"[AnomalyDetectionAgent] 模型文件不存在，使用新初始化的模型")
                self.model = create_model(
                    input_dim=self.input_dim,
                    num_drones=self.num_drones,
                    patch_size=self.patch_size,
                    hidden_dim=self.hidden_dim,
                    num_anomaly_types=self.num_anomaly_types
                )
                self.model = self.model.to(self.device)
                self.model.eval()
                return True
                
        except Exception as e:
            print(f"[AnomalyDetectionAgent] 模型加载失败: {e}")
            return False
    
    def _load_data(self, data_file: Optional[str] = None) -> bool:
        """
        加载数据
        
        参数:
            data_file: 数据文件路径
        
        返回:
            是否加载成功
        """
        if data_file is None:
            data_file = os.path.join(self.data_dir, 'test_data.csv')
            if not os.path.exists(data_file):
                data_file = os.path.join(self.data_dir, 'drone_swarm_with_anomalies.csv')
        
        try:
            print(f"[AnomalyDetectionAgent] 加载数据: {data_file}")
            self.data = pd.read_csv(data_file)
            
            # 计算标准化参数
            for col in self.feature_cols:
                if col in self.data.columns:
                    self.normalization_params[col] = {
                        'mean': self.data[col].mean(),
                        'std': self.data[col].std()
                    }
            
            print(f"[AnomalyDetectionAgent] 数据加载成功: {len(self.data)} 条记录")
            return True
            
        except Exception as e:
            print(f"[AnomalyDetectionAgent] 数据加载失败: {e}")
            return False
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        标准化特征
        
        参数:
            features: 原始特征数组 [seq_len, num_features]
        
        返回:
            标准化后的特征
        """
        normalized = features.copy()
        for i, col in enumerate(self.feature_cols):
            if col in self.normalization_params:
                mean = self.normalization_params[col]['mean']
                std = self.normalization_params[col]['std']
                normalized[:, i] = (features[:, i] - mean) / (std + 1e-8)
        return normalized
    
    def detect(self, parameters: Dict) -> Dict:
        """
        执行异常检测
        
        参数:
            parameters: 检测参数，可包含：
                - drone_ids: 要检测的无人机ID列表
                - time_range: 时间范围 (start, end)
                - threshold: 异常判断阈值
                - data_file: 数据文件路径
        
        返回:
            检测结果字典
        """
        print("[AnomalyDetectionAgent] 开始异常检测...")
        
        try:
            # 加载模型
            if not self._load_model():
                return {
                    'success': False,
                    'error': '模型加载失败'
                }
            
            # 加载数据
            data_file = parameters.get('data_file')
            if self.data is None or data_file:
                if not self._load_data(data_file):
                    return {
                        'success': False,
                        'error': '数据加载失败'
                    }
            
            # 获取参数
            drone_ids = parameters.get('drone_ids', list(range(self.num_drones)))
            time_range = parameters.get('time_range', None)
            threshold = parameters.get('threshold', 0.5)
            
            # 执行检测
            results = self._run_detection(drone_ids, time_range, threshold)
            
            # 生成摘要
            anomaly_count = sum(1 for r in results['anomalies'] if r['is_anomaly'])
            
            return {
                'success': True,
                'message': '异常检测完成',
                'summary': f'共检测 {len(drone_ids)} 架无人机，发现 {anomaly_count} 个异常',
                'anomalies': results['anomalies'],
                'detection_summary': results['summary'],
                'details': {
                    '检测无人机': drone_ids,
                    '时间范围': time_range if time_range else '全部',
                    '异常阈值': threshold,
                    '发现异常数': anomaly_count,
                    '正常数': len(drone_ids) - anomaly_count
                }
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'异常检测失败: {str(e)}'
            }
    
    def _run_detection(self, drone_ids: List[int], 
                       time_range: Optional[tuple], 
                       threshold: float) -> Dict:
        """
        运行检测算法
        
        参数:
            drone_ids: 无人机ID列表
            time_range: 时间范围
            threshold: 异常阈值
        
        返回:
            检测结果
        """
        anomalies = []
        summary = {
            'total_drones': len(drone_ids),
            'anomalous_drones': 0,
            'anomaly_types': {},
            'details': []
        }
        
        # 获取时间范围内的数据
        if time_range:
            start, end = time_range
        else:
            # 使用前500个时间步进行检测
            start, end = 0, min(500, len(self.data[self.data['drone_id'] == 0]))
        
        # 准备输入数据
        with torch.no_grad():
            for window_start in range(start, end - self.window_size + 1, self.window_size):
                x_list = []
                
                for drone_id in range(self.num_drones):
                    drone_data = self.data[self.data['drone_id'] == drone_id]
                    
                    if window_start + self.window_size <= len(drone_data):
                        window_data = drone_data.iloc[window_start:window_start + self.window_size]
                        features = window_data[self.feature_cols].values
                        features = self._normalize_features(features)
                        x_list.append(features)
                    else:
                        # 使用零填充
                        x_list.append(np.zeros((self.window_size, self.input_dim)))
                
                # 转换为tensor
                x = torch.FloatTensor(np.array(x_list)).unsqueeze(0).to(self.device)
                
                # 模型预测
                anomaly_scores, anomaly_types, _ = self.model(x)
                
                # 解析结果
                for drone_id in drone_ids:
                    if drone_id < self.num_drones:
                        score = torch.softmax(anomaly_scores[0, drone_id], dim=0)
                        anomaly_prob = score[1].item()  # 异常概率
                        
                        type_probs = torch.softmax(anomaly_types[0, drone_id], dim=0)
                        predicted_type = torch.argmax(type_probs).item()
                        type_confidence = type_probs[predicted_type].item()
                        
                        is_anomaly = anomaly_prob > threshold
                        
                        # 获取真实标签（如果有）
                        true_anomaly = None
                        true_type = None
                        if 'anomaly' in self.data.columns:
                            drone_data = self.data[self.data['drone_id'] == drone_id]
                            window_labels = drone_data.iloc[window_start:window_start + self.window_size]['anomaly'].values
                            true_anomaly = int(window_labels.any())
                            
                            if 'anomaly_type' in drone_data.columns and true_anomaly:
                                anomaly_rows = drone_data.iloc[window_start:window_start + self.window_size]
                                anomaly_types_in_window = anomaly_rows[anomaly_rows['anomaly'] == 1]['anomaly_type'].values
                                if len(anomaly_types_in_window) > 0:
                                    true_type = anomaly_types_in_window[0]
        
        # 汇总每架无人机的检测结果
        for drone_id in drone_ids:
            drone_result = self._analyze_drone(drone_id, start, end, threshold)
            anomalies.append(drone_result)
            
            if drone_result['is_anomaly']:
                summary['anomalous_drones'] += 1
                anomaly_type = drone_result['anomaly_type']
                summary['anomaly_types'][anomaly_type] = \
                    summary['anomaly_types'].get(anomaly_type, 0) + 1
                summary['details'].append({
                    'drone_id': drone_id,
                    'type': anomaly_type,
                    'confidence': drone_result['confidence']
                })
        
        return {
            'anomalies': anomalies,
            'summary': summary
        }
    
    def _analyze_drone(self, drone_id: int, start: int, end: int, 
                       threshold: float) -> Dict:
        """
        分析单架无人机的异常情况
        
        参数:
            drone_id: 无人机ID
            start: 起始索引
            end: 结束索引
            threshold: 异常阈值
        
        返回:
            无人机异常分析结果
        """
        drone_data = self.data[self.data['drone_id'] == drone_id]
        
        if len(drone_data) == 0:
            return {
                'drone_id': drone_id,
                'is_anomaly': False,
                'anomaly_type': 'normal',
                'anomaly_type_cn': '正常',
                'confidence': 0.0,
                'details': '无数据'
            }
        
        # 使用滑动窗口进行检测
        anomaly_scores_list = []
        type_predictions = []
        
        with torch.no_grad():
            for window_start in range(start, min(end - self.window_size + 1, len(drone_data) - self.window_size + 1), 
                                      self.window_size // 2):
                x_list = []
                
                for d_id in range(self.num_drones):
                    d_data = self.data[self.data['drone_id'] == d_id]
                    
                    if window_start + self.window_size <= len(d_data):
                        window_data = d_data.iloc[window_start:window_start + self.window_size]
                        features = window_data[self.feature_cols].values
                        features = self._normalize_features(features)
                        x_list.append(features)
                    else:
                        x_list.append(np.zeros((self.window_size, self.input_dim)))
                
                x = torch.FloatTensor(np.array(x_list)).unsqueeze(0).to(self.device)
                
                anomaly_scores, anomaly_types, _ = self.model(x)
                
                score = torch.softmax(anomaly_scores[0, drone_id], dim=0)
                anomaly_prob = score[1].item()
                anomaly_scores_list.append(anomaly_prob)
                
                type_probs = torch.softmax(anomaly_types[0, drone_id], dim=0)
                predicted_type = torch.argmax(type_probs).item()
                type_predictions.append((predicted_type, type_probs[predicted_type].item()))
        
        if not anomaly_scores_list:
            return {
                'drone_id': drone_id,
                'is_anomaly': False,
                'anomaly_type': 'normal',
                'anomaly_type_cn': '正常',
                'confidence': 0.0,
                'details': '数据不足'
            }
        
        # 计算平均异常分数
        avg_anomaly_score = np.mean(anomaly_scores_list)
        max_anomaly_score = np.max(anomaly_scores_list)
        
        # 判断是否异常
        is_anomaly = max_anomaly_score > threshold
        
        # 确定异常类型
        if is_anomaly:
            # 使用出现最多的非正常类型
            type_counts = {}
            for t, conf in type_predictions:
                if t != 0:  # 排除正常类型
                    type_counts[t] = type_counts.get(t, 0) + 1
            
            if type_counts:
                predicted_type = max(type_counts, key=type_counts.get)
            else:
                predicted_type = 1  # 默认位置偏离
            
            # 计算该类型的平均置信度
            type_confidences = [conf for t, conf in type_predictions if t == predicted_type]
            avg_confidence = np.mean(type_confidences) if type_confidences else 0.5
        else:
            predicted_type = 0
            avg_confidence = 1 - avg_anomaly_score
        
        # 获取真实标签用于验证
        true_label = None
        if 'anomaly' in drone_data.columns:
            segment = drone_data.iloc[start:end]
            if len(segment) > 0:
                true_label = segment['anomaly'].max()
        
        # 生成详细描述
        if is_anomaly:
            details = self._generate_anomaly_details(drone_id, predicted_type, 
                                                     avg_anomaly_score, drone_data.iloc[start:end])
        else:
            details = '无异常'
        
        return {
            'drone_id': drone_id,
            'is_anomaly': is_anomaly,
            'anomaly_type': self.anomaly_type_names.get(predicted_type, 'unknown'),
            'anomaly_type_cn': self.anomaly_type_names_cn.get(predicted_type, '未知'),
            'confidence': float(max_anomaly_score),
            'avg_score': float(avg_anomaly_score),
            'true_label': true_label,
            'predicted_type_id': predicted_type,
            'details': details
        }
    
    def _generate_anomaly_details(self, drone_id: int, anomaly_type: int, 
                                  score: float, segment: pd.DataFrame) -> str:
        """
        生成异常详细描述
        
        参数:
            drone_id: 无人机ID
            anomaly_type: 异常类型
            score: 异常分数
            segment: 数据片段
        
        返回:
            详细描述字符串
        """
        type_name = self.anomaly_type_names_cn.get(anomaly_type, '未知')
        
        if anomaly_type == 1:  # 位置偏离
            pos_std = segment[['x', 'y', 'z']].std().mean()
            details = f"检测到{type_name}，位置波动标准差: {pos_std:.2f}m"
        elif anomaly_type == 2:  # 传感器故障
            battery_change = segment['battery'].diff().abs().mean()
            temp_change = segment['temperature'].diff().abs().mean()
            signal_change = segment['signal_strength'].diff().abs().mean()
            details = f"检测到{type_name}，电池变化: {battery_change:.2f}%，温度变化: {temp_change:.2f}°C"
        elif anomaly_type == 3:  # 机动异常
            velocity = np.sqrt(segment['vx']**2 + segment['vy']**2 + segment['vz']**2)
            vel_std = velocity.std()
            details = f"检测到{type_name}，速度波动标准差: {vel_std:.2f}m/s"
        else:
            details = f"检测到异常，类型: {type_name}"
        
        return details
    
    def get_anomaly_info(self, anomaly_type: int) -> Dict:
        """
        获取异常类型的详细信息
        
        参数:
            anomaly_type: 异常类型ID
        
        返回:
            异常类型信息字典
        """
        info = {
            0: {
                'name': '正常',
                'name_en': 'Normal',
                'description': '无人机运行正常，无异常行为',
                'severity': '无',
                'causes': []
            },
            1: {
                'name': '位置偏离异常',
                'name_en': 'Position Deviation',
                'description': '无人机偏离预定飞行轨迹',
                'severity': '中等至严重',
                'causes': ['GPS故障', '风力扰动', '控制系统问题', '导航系统异常']
            },
            2: {
                'name': '传感器故障异常',
                'name_en': 'Sensor Malfunction',
                'description': '传感器读数出现异常',
                'severity': '中等',
                'causes': ['传感器损坏', '电池老化', '温度过高', '通信干扰']
            },
            3: {
                'name': '机动异常',
                'name_en': 'Maneuver Anomaly',
                'description': '速度和加速度出现异常变化',
                'severity': '中等至严重',
                'causes': ['控制算法错误', '电机故障', '传感器干扰', '结构损伤']
            }
        }
        return info.get(anomaly_type, info[0])


if __name__ == "__main__":
    # 测试异常检测Agent
    config = {
        'model_path': 'results/best_model.pth',
        'data_dir': 'data',
        'device': 'cpu'
    }
    
    print("="*60)
    print("测试异常检测Agent")
    print("="*60)
    
    agent = AnomalyDetectionAgent(config)
    
    # 测试检测
    result = agent.detect({
        'drone_ids': [0, 1, 2],
        'threshold': 0.5
    })
    
    print(f"\n检测结果:")
    print(f"成功: {result.get('success')}")
    print(f"摘要: {result.get('summary')}")
    
    if 'anomalies' in result:
        print("\n异常详情:")
        for anomaly in result['anomalies']:
            print(f"  无人机{anomaly['drone_id']}: "
                  f"异常={anomaly['is_anomaly']}, "
                  f"类型={anomaly['anomaly_type_cn']}, "
                  f"置信度={anomaly['confidence']:.2f}")
