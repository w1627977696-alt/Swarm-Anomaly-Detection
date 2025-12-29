# 无人机集群异常检测与影响评估Agent系统
# Drone Swarm Anomaly Detection & Impact Assessment Agent System

[English](#english) | [中文](#chinese)

---

<a name="chinese"></a>
## 中文文档

### 项目简介

本项目实现了基于深度学习的无人机集群异常检测系统，结合了Patch机制和图神经网络(GNN)技术，能够有效识别无人机集群中的异常行为及其类型。

**🆕 v2.0新增：Agent智能运维系统**

本项目现已升级为Agent形式的智能运维平台，支持通过自然语言交互执行各种分析任务：

- 🤖 **自然语言交互**：使用中文或英文与系统对话
- 📊 **数据预处理与可视化**：加载和展示无人机飞行数据
- 🔄 **数据回放**：回放历史飞行数据
- 🔍 **异常检测**：基于Patch-GNN模型的智能异常检测
- 📈 **影响评估**：基于RAG技术的异常影响分析
- 📝 **报告生成**：生成结构化的综合分析报告

### 快速开始Agent系统

```bash
# 安装依赖
pip install -r requirements.txt

# 启动Agent系统
python main_agent.py

# 然后输入自然语言指令，例如：
>>> 检测所有无人机的异常
>>> 评估异常影响
>>> 生成综合报告
>>> 帮助
```

详细使用说明请参阅 [AGENT_DOCUMENTATION.md](AGENT_DOCUMENTATION.md)

### 主要特性

1. **模拟数据生成**：自动生成包含10架无人机的集群飞行数据，每架无人机包含多维时间序列特征
2. **智能异常注入**：支持3种典型异常类型的注入，并提供详细的注入日志
3. **先进的检测模型**：结合Patch自适应选择和图神经网络的深度学习模型
4. **完整的可视化**：提供预测结果与真实值的对比可视化，并标注异常段
5. **🆕 Agent智能运维**：自然语言交互的智能运维平台
6. **🆕 RAG影响评估**：基于知识库的异常影响分析
7. **🆕 综合报告生成**：自动生成结构化分析报告

### 系统架构

```
Swarm-Anomaly-Detection/
├── agent/                         # 🆕 Agent模块
│   ├── __init__.py               # Agent包初始化
│   ├── core.py                   # Agent核心模块
│   ├── natural_language_processor.py  # 自然语言处理
│   ├── data_agent.py             # 数据处理Agent
│   ├── detection_agent.py        # 异常检测Agent
│   ├── assessment_agent.py       # 影响评估Agent
│   └── report_agent.py           # 报告生成Agent
├── knowledge_base/                # 🆕 RAG知识库
│   ├── __init__.py               # 知识库初始化
│   └── knowledge_base.py         # 知识库实现
├── data/                          # 数据目录
│   ├── drone_swarm_normal.csv     # 正常数据
│   ├── drone_swarm_with_anomalies.csv  # 带异常的数据
│   ├── train_data.csv             # 训练集
│   ├── test_data.csv              # 测试集
│   └── anomaly_injection_log.txt  # 异常注入日志
├── models/                        # 模型目录
│   └── patch_gnn_model.py         # Patch-GNN模型实现
├── results/                       # 结果目录
│   ├── best_model.pth             # 最佳模型
│   ├── training_history.json      # 训练历史
│   ├── training_history.png       # 训练历史图
│   └── drone_*_results.png        # 各无人机的检测结果
├── main_agent.py                  # 🆕 Agent系统入口
├── generate_data.py               # 数据生成脚本
├── inject_anomalies.py            # 异常注入脚本
├── train.py                       # 训练脚本
├── visualize.py                   # 可视化脚本
├── AGENT_DOCUMENTATION.md         # 🆕 Agent详细文档
└── requirements.txt               # 依赖包列表
```

### 数据特征

每架无人机包含以下15个特征维度的时间序列数据：

| 特征 | 描述 | 单位 |
|------|------|------|
| x, y, z | 三维位置坐标 | 米 |
| vx, vy, vz | 三维速度 | 米/秒 |
| ax, ay, az | 三维加速度 | 米/秒² |
| roll, pitch, yaw | 姿态角 | 度 |
| battery | 电池电量 | 百分比 |
| temperature | 温度 | 摄氏度 |
| signal_strength | 信号强度 | dBm |

### 异常类型说明

系统支持检测以下3种典型异常：

#### 1. 位置偏离异常 (Position Deviation)
- **描述**：无人机偏离预定飞行轨迹
- **原因**：GPS故障、风力扰动、控制系统问题
- **特征表现**：x, y, z坐标出现渐进式偏离，偏离幅度可达20-30米
- **注入位置**：无人机0, 3, 7

#### 2. 传感器故障异常 (Sensor Malfunction)
- **描述**：传感器读数异常
- **子类型**：
  - **电池异常**：电量突然大幅下降约30%
  - **温度异常**：温度异常升高15-25摄氏度（散热问题）
  - **信号异常**：信号强度突然减弱至-90dBm左右（通信干扰）
- **注入位置**：无人机1（电池）, 4（温度）, 8（信号）

#### 3. 机动异常 (Maneuver Anomaly)
- **描述**：速度和加速度出现异常变化
- **原因**：控制算法错误、电机故障、传感器干扰
- **特征表现**：vx, vy, vz, ax, ay, az出现高频振荡，振幅为正常值的3-4倍
- **注入位置**：无人机2, 5, 9

### 模型架构

本系统采用创新的**Patch-GNN**架构：

1. **自适应Patch提取器**
   - 将时间序列划分为可变长度的patches
   - 自适应选择最优patch大小（5, 10, 20）
   - 使用1D卷积提取patch特征

2. **时间建模模块**
   - 使用Transformer编码器捕捉patches之间的时间依赖
   - 多头注意力机制建模长期依赖关系

3. **空间建模模块**
   - 使用图注意力网络(GAT)建模无人机之间的空间关系
   - 全连接图结构捕捉集群内的相互影响

4. **时空融合层**
   - 融合时间和空间特征
   - 生成综合的无人机状态表示

5. **异常检测头**
   - **二分类头**：判断是否异常
   - **多分类头**：识别异常类型
   - **重构头**：支持无监督异常检测

### 安装与使用

#### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/w1627977696-alt/Swarm-Anomaly-Detection.git
cd Swarm-Anomaly-Detection

# 安装依赖
pip install -r requirements.txt
```

#### 2. 生成数据

```bash
# 生成正常无人机集群数据
python generate_data.py

# 注入异常并划分训练/测试集
python inject_anomalies.py
```

生成的数据：
- `data/drone_swarm_normal.csv`: 正常数据（36,000条记录）
- `data/drone_swarm_with_anomalies.csv`: 带异常的完整数据
- `data/train_data.csv`: 训练集（70%）
- `data/test_data.csv`: 测试集（30%）
- `data/anomaly_injection_log.txt`: 详细的异常注入日志

#### 3. 训练模型

```bash
python train.py
```

训练参数：
- Epochs: 50
- Batch Size: 8
- Learning Rate: 0.001
- Window Size: 50
- Stride: 25

训练输出：
- `results/best_model.pth`: 最佳模型权重
- `results/training_history.json`: 训练历史记录

#### 4. 可视化结果

```bash
python visualize.py
```

生成的可视化：
- `results/drone_*_results.png`: 每架无人机的检测结果（10张）
- `results/training_history.png`: 训练历史曲线

每张结果图包含5个子图：
1. **位置特征**：x, y, z坐标随时间变化
2. **速度幅值**：速度大小随时间变化
3. **传感器数据**：电池、温度、信号强度
4. **异常检测**：预测值vs真实值对比
5. **异常类型**：类型分类预测vs真实标签

### 性能指标

在测试集上的典型性能：

| 指标 | 数值 |
|------|------|
| 准确率 (Accuracy) | ~90% |
| 精确率 (Precision) | ~85% |
| 召回率 (Recall) | ~88% |
| F1分数 (F1 Score) | ~86% |
| 类型分类准确率 | ~80% |

### 代码结构详解

#### generate_data.py
- `DroneSwarmDataGenerator`: 数据生成器类
  - `generate_circular_formation()`: 生成圆形编队轨迹
  - `calculate_velocity()`: 计算速度
  - `calculate_acceleration()`: 计算加速度
  - `generate_attitude()`: 生成姿态角
  - `generate_battery_level()`: 生成电池数据
  - `generate_temperature()`: 生成温度数据
  - `generate_signal_strength()`: 生成信号强度

#### inject_anomalies.py
- `AnomalyInjector`: 异常注入器类
  - `inject_position_deviation()`: 注入位置偏离异常
  - `inject_sensor_malfunction()`: 注入传感器故障
  - `inject_maneuver_anomaly()`: 注入机动异常
  - `split_train_test()`: 划分训练/测试集

#### models/patch_gnn_model.py
- `AdaptivePatchExtractor`: 自适应Patch提取器
- `TemporalTransformer`: 时间Transformer模块
- `SpatialGNN`: 空间图神经网络模块
- `PatchGNNAnomalyDetector`: 完整的异常检测模型

#### train.py
- `DroneSwarmDataset`: PyTorch数据集类
- `train_epoch()`: 训练一个epoch
- `evaluate()`: 评估模型
- `train_model()`: 完整训练流程

#### visualize.py
- `ResultVisualizer`: 结果可视化类
  - `predict_full_sequence()`: 对完整序列预测
  - `plot_single_drone_results()`: 绘制单架无人机结果
  - `plot_all_drones()`: 绘制所有无人机结果
  - `plot_training_history()`: 绘制训练历史

### 技术亮点

1. **自适应Patch选择**：根据时间序列的局部特征自动选择最优patch大小
2. **图神经网络**：充分利用无人机之间的空间关系进行异常检测
3. **多任务学习**：同时进行异常检测和类型分类
4. **端到端训练**：所有模块联合训练，优化整体性能
5. **可解释性**：通过可视化清晰展示异常段和检测结果

### 未来改进方向

1. 增加更多异常类型（如通信中断、碰撞风险等）
2. 实现在线实时检测功能
3. 添加异常定位和追溯功能
4. 支持可变数量的无人机集群
5. 集成强化学习进行异常响应

### 贡献指南

欢迎提交Issue和Pull Request！

### 许可证

MIT License

---

<a name="english"></a>
## English Documentation

### Project Overview

This project implements a deep learning-based anomaly detection system for drone swarms, combining Patch mechanisms and Graph Neural Networks (GNN) to effectively identify anomalous behaviors and their types in drone swarms.

### Key Features

1. **Simulated Data Generation**: Automatically generates flight data for a swarm of 10 drones with multi-dimensional time series features
2. **Intelligent Anomaly Injection**: Supports injection of 3 typical anomaly types with detailed logging
3. **Advanced Detection Model**: Deep learning model combining adaptive Patch selection and Graph Neural Networks
4. **Comprehensive Visualization**: Provides visualization comparing predictions with ground truth and annotates anomalous segments

### System Architecture

```
Swarm-Anomaly-Detection/
├── data/                          # Data directory
│   ├── drone_swarm_normal.csv     # Normal data
│   ├── drone_swarm_with_anomalies.csv  # Data with anomalies
│   ├── train_data.csv             # Training set
│   ├── test_data.csv              # Test set
│   └── anomaly_injection_log.txt  # Anomaly injection log
├── models/                        # Model directory
│   └── patch_gnn_model.py         # Patch-GNN model implementation
├── results/                       # Results directory
│   ├── best_model.pth             # Best model
│   ├── training_history.json      # Training history
│   ├── training_history.png       # Training history plot
│   └── drone_*_results.png        # Detection results for each drone
├── generate_data.py               # Data generation script
├── inject_anomalies.py            # Anomaly injection script
├── train.py                       # Training script
├── visualize.py                   # Visualization script
└── requirements.txt               # Dependencies
```

### Data Features

Each drone contains 15-dimensional time series data:

| Feature | Description | Unit |
|---------|-------------|------|
| x, y, z | 3D position coordinates | meters |
| vx, vy, vz | 3D velocity | m/s |
| ax, ay, az | 3D acceleration | m/s² |
| roll, pitch, yaw | Attitude angles | degrees |
| battery | Battery level | percentage |
| temperature | Temperature | Celsius |
| signal_strength | Signal strength | dBm |

### Anomaly Types

The system detects 3 typical anomaly types:

#### 1. Position Deviation
- **Description**: Drone deviates from planned trajectory
- **Causes**: GPS failure, wind disturbance, control system issues
- **Manifestation**: Gradual deviation in x, y, z coordinates (20-30m)
- **Injection**: Drones 0, 3, 7

#### 2. Sensor Malfunction
- **Description**: Abnormal sensor readings
- **Subtypes**:
  - **Battery**: Sudden ~30% drop
  - **Temperature**: 15-25°C increase (cooling issue)
  - **Signal**: Drop to ~-90dBm (communication interference)
- **Injection**: Drones 1 (battery), 4 (temp), 8 (signal)

#### 3. Maneuver Anomaly
- **Description**: Abnormal velocity and acceleration changes
- **Causes**: Control algorithm errors, motor failure, sensor interference
- **Manifestation**: High-frequency oscillations in vx, vy, vz, ax, ay, az (3-4x normal)
- **Injection**: Drones 2, 5, 9

### Model Architecture

The system uses an innovative **Patch-GNN** architecture:

1. **Adaptive Patch Extractor**
   - Divides time series into variable-length patches
   - Adaptively selects optimal patch size (5, 10, 20)
   - Extracts patch features using 1D convolution

2. **Temporal Modeling Module**
   - Uses Transformer encoder to capture temporal dependencies
   - Multi-head attention for long-term dependencies

3. **Spatial Modeling Module**
   - Uses Graph Attention Network (GAT) for spatial relationships
   - Fully connected graph captures swarm interactions

4. **Spatio-Temporal Fusion Layer**
   - Fuses temporal and spatial features
   - Generates comprehensive drone state representation

5. **Anomaly Detection Heads**
   - **Binary classifier**: Normal/Anomaly detection
   - **Multi-class classifier**: Anomaly type identification
   - **Reconstructor**: Supports unsupervised detection

### Installation and Usage

#### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/w1627977696-alt/Swarm-Anomaly-Detection.git
cd Swarm-Anomaly-Detection

# Install dependencies
pip install -r requirements.txt
```

#### 2. Generate Data

```bash
# Generate normal drone swarm data
python generate_data.py

# Inject anomalies and split train/test sets
python inject_anomalies.py
```

Generated data:
- `data/drone_swarm_normal.csv`: Normal data (36,000 records)
- `data/drone_swarm_with_anomalies.csv`: Complete data with anomalies
- `data/train_data.csv`: Training set (70%)
- `data/test_data.csv`: Test set (30%)
- `data/anomaly_injection_log.txt`: Detailed injection log

#### 3. Train Model

```bash
python train.py
```

Training parameters:
- Epochs: 50
- Batch Size: 8
- Learning Rate: 0.001
- Window Size: 50
- Stride: 25

Training outputs:
- `results/best_model.pth`: Best model weights
- `results/training_history.json`: Training history

#### 4. Visualize Results

```bash
python visualize.py
```

Generated visualizations:
- `results/drone_*_results.png`: Detection results for each drone (10 plots)
- `results/training_history.png`: Training history curves

Each result plot contains 5 subplots:
1. **Position Features**: x, y, z coordinates over time
2. **Velocity Magnitude**: Speed magnitude over time
3. **Sensor Data**: Battery, temperature, signal strength
4. **Anomaly Detection**: Prediction vs ground truth comparison
5. **Anomaly Type**: Type classification prediction vs true labels

### Performance Metrics

Typical performance on test set:

| Metric | Value |
|--------|-------|
| Accuracy | ~90% |
| Precision | ~85% |
| Recall | ~88% |
| F1 Score | ~86% |
| Type Accuracy | ~80% |

### Technical Highlights

1. **Adaptive Patch Selection**: Automatically selects optimal patch size based on local time series features
2. **Graph Neural Networks**: Fully utilizes spatial relationships between drones
3. **Multi-task Learning**: Simultaneous anomaly detection and type classification
4. **End-to-End Training**: All modules trained jointly for optimal performance
5. **Interpretability**: Clear visualization of anomalous segments and detection results

### Future Improvements

1. Add more anomaly types (communication loss, collision risk, etc.)
2. Implement online real-time detection
3. Add anomaly localization and tracing
4. Support variable number of drones
5. Integrate reinforcement learning for anomaly response

### Contributing

Issues and Pull Requests are welcome!

### License

MIT License

---

## Contact

For questions and suggestions, please open an issue or contact the repository owner.
