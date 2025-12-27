# 详细使用说明文档

## 目录
1. [系统概述](#系统概述)
2. [环境配置](#环境配置)
3. [快速开始](#快速开始)
4. [详细使用指南](#详细使用指南)
5. [模型架构详解](#模型架构详解)
6. [数据格式说明](#数据格式说明)
7. [训练参数调优](#训练参数调优)
8. [常见问题](#常见问题)
9. [API参考](#api参考)

---

## 系统概述

本系统是一个基于深度学习的无人机集群异常检测平台，采用最新的Patch机制和图神经网络技术，能够：

- ✅ 自动生成真实感的无人机集群飞行数据
- ✅ 注入3种典型异常（位置偏离、传感器故障、机动异常）
- ✅ 使用先进的深度学习模型进行异常检测
- ✅ 识别异常的无人机节点和异常类型
- ✅ 提供详细的可视化结果和性能分析

### 技术特点

1. **Patch自适应选择**：根据时间序列特征自动选择最优的patch大小
2. **图神经网络**：建模无人机之间的空间关系和交互
3. **时空融合**：同时捕捉时间序列的时间依赖和空间依赖
4. **多任务学习**：同时进行异常检测和类型识别
5. **端到端训练**：所有模块联合优化

---

## 环境配置

### 系统要求

- Python 3.7+
- PyTorch 1.10+
- CUDA 10.2+ (可选，用于GPU加速)

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/w1627977696-alt/Swarm-Anomaly-Detection.git
cd Swarm-Anomaly-Detection
```

2. **创建虚拟环境（推荐）**
```bash
# 使用conda
conda create -n drone-anomaly python=3.8
conda activate drone-anomaly

# 或使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

### 验证安装

运行测试脚本验证所有功能是否正常：
```bash
python test_system.py
```

如果所有测试通过，说明环境配置成功。

---

## 快速开始

### 方法1：一键运行完整流程

```bash
python run_pipeline.py --all
```

这将依次执行：
1. 生成无人机集群数据
2. 注入异常并划分数据集
3. 训练模型（50个epoch）
4. 生成可视化结果

**注意**：完整流程可能需要数小时，取决于硬件配置。

### 方法2：分步执行

```bash
# 步骤1：生成数据
python generate_data.py

# 步骤2：注入异常
python inject_anomalies.py

# 步骤3：训练模型
python train.py

# 步骤4：可视化结果
python visualize.py
```

---

## 详细使用指南

### 1. 数据生成

#### 基本用法

```bash
python generate_data.py
```

#### 自定义参数

编辑 `generate_data.py` 中的参数：

```python
generator = DroneSwarmDataGenerator(
    num_drones=10,        # 无人机数量
    duration_minutes=60,  # 飞行时长（分钟）
    sampling_rate=1       # 采样率（Hz）
)
```

#### 生成的数据

- **文件**：`data/drone_swarm_normal.csv`
- **样本数**：3600 × 10 = 36,000（60分钟 × 60秒 × 10架无人机）
- **特征数**：15个传感器特征

### 2. 异常注入

#### 基本用法

```bash
python inject_anomalies.py
```

#### 自定义异常注入

编辑 `inject_anomalies.py` 的 `inject_all_anomalies()` 方法：

```python
# 位置偏离异常
injector.inject_position_deviation(
    drone_id=0,           # 目标无人机
    start_idx=1000,       # 起始索引
    duration=300,         # 持续时间（秒）
    deviation_magnitude=20  # 偏离幅度（米）
)

# 传感器故障异常
injector.inject_sensor_malfunction(
    drone_id=1,
    start_idx=1200,
    duration=250,
    sensor_type='battery'  # 'battery', 'temperature', 'signal_strength'
)

# 机动异常
injector.inject_maneuver_anomaly(
    drone_id=2,
    start_idx=1400,
    duration=200,
    intensity=3.0  # 异常强度倍数
)
```

#### 生成的文件

1. `data/drone_swarm_with_anomalies.csv` - 带异常的完整数据
2. `data/train_data.csv` - 训练集（70%）
3. `data/test_data.csv` - 测试集（30%）
4. `data/anomaly_injection_log.txt` - 详细的异常注入日志

### 3. 模型训练

#### 基本用法

```bash
python train.py
```

#### 自定义训练参数

使用 `run_pipeline.py` 传递参数：

```bash
python run_pipeline.py --train --epochs 100 --batch-size 16 --lr 0.0005
```

或直接修改 `train.py` 中的 `train_model()` 调用：

```python
train_model(
    train_data_path='data/train_data.csv',
    test_data_path='data/test_data.csv',
    epochs=50,          # 训练轮数
    batch_size=8,       # 批大小
    lr=0.001,           # 学习率
    window_size=50,     # 时间窗口大小
    stride=25,          # 滑动步长
    device='cuda'       # 'cuda' 或 'cpu'
)
```

#### 训练监控

训练过程中会实时显示：
- 训练损失（总损失、异常检测损失、类型分类损失）
- 测试指标（准确率、精确率、召回率、F1分数）
- 学习率调整信息

#### 生成的文件

1. `results/best_model.pth` - 最佳模型权重（根据F1分数）
2. `results/training_history.json` - 训练历史记录

### 4. 结果可视化

#### 基本用法

```bash
python visualize.py
```

#### 自定义可视化参数

编辑 `visualize.py` 中的参数：

```python
visualizer.plot_all_drones(
    start_idx=0,      # 起始索引
    length=800,       # 可视化长度（时间步）
    output_dir='results'
)
```

#### 生成的可视化

每架无人机生成一张包含5个子图的结果图：

1. **位置特征图**：显示x, y, z坐标的变化趋势
2. **速度图**：显示速度幅值的变化
3. **传感器数据图**：显示电池、温度、信号强度
4. **异常检测图**：对比预测值和真实值
5. **异常类型图**：对比预测的异常类型和真实类型

#### 解读可视化结果

- **红色背景区域**：标注的真实异常段
- **蓝色实线**：真实标签
- **红色虚线**：模型预测
- **重合区域**：预测准确

---

## 模型架构详解

### 整体架构

```
输入数据 [batch, 10 drones, 50 timesteps, 15 features]
    ↓
┌─────────────────────────────────────────┐
│  自适应Patch提取器                       │
│  - 将时间序列划分为patches              │
│  - 自适应选择patch大小（5/10/20）      │
│  - 使用1D卷积提取特征                   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  时间建模（Transformer）                 │
│  - 捕捉patches之间的时间依赖            │
│  - 多头注意力机制                       │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  空间建模（图神经网络）                  │
│  - 建模无人机之间的空间关系             │
│  - 使用图注意力网络（GAT）              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  时空融合                                │
│  - 融合时间和空间特征                   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  异常检测头                              │
│  - 二分类头：正常/异常                  │
│  - 多分类头：异常类型识别               │
│  - 重构头：重构原始特征                 │
└─────────────────────────────────────────┘
    ↓
输出：[batch, 10 drones, 2/4 classes]
```

### 关键模块说明

#### 1. AdaptivePatchExtractor

**功能**：自适应地将时间序列划分为patches

**创新点**：
- 不是固定patch大小，而是根据时间序列的局部特征自适应选择
- 三个候选大小：5（短期）、10（中期）、20（长期）
- 使用小型神经网络预测每个位置的最佳patch大小

**实现**：
```python
self.patch_size_selector = nn.Sequential(
    nn.Linear(input_dim, 32),
    nn.ReLU(),
    nn.Linear(32, 3),  # 输出3个候选大小的概率
    nn.Softmax(dim=-1)
)
```

#### 2. TemporalTransformer

**功能**：捕捉patches之间的时间依赖关系

**特点**：
- 使用标准的Transformer编码器
- 4个注意力头，捕捉不同方面的时间模式
- 2层堆叠，增强表达能力

#### 3. SpatialGNN

**功能**：建模无人机之间的空间关系

**特点**：
- 使用图注意力网络（GAT）
- 全连接图：每架无人机与其他所有无人机相连
- 动态权重：注意力机制自动学习无人机之间的重要性

#### 4. 多任务学习

模型同时优化三个目标：

1. **异常检测**：二分类交叉熵损失
2. **类型识别**：多分类交叉熵损失（只对异常样本）
3. **特征重构**：MSE损失（可选，用于无监督学习）

总损失：
```
Loss = Loss_anomaly + 0.5 × Loss_type
```

---

## 数据格式说明

### CSV文件格式

每个CSV文件包含以下列：

| 列名 | 类型 | 说明 | 范围/单位 |
|------|------|------|-----------|
| timestamp | datetime | 时间戳 | YYYY-MM-DD HH:MM:SS |
| drone_id | int | 无人机ID | 0-9 |
| x | float | X坐标 | -51 到 51 米 |
| y | float | Y坐标 | -51 到 51 米 |
| z | float | Z坐标 | 45 到 55 米 |
| vx | float | X方向速度 | 米/秒 |
| vy | float | Y方向速度 | 米/秒 |
| vz | float | Z方向速度 | 米/秒 |
| ax | float | X方向加速度 | 米/秒² |
| ay | float | Y方向加速度 | 米/秒² |
| az | float | Z方向加速度 | 米/秒² |
| roll | float | 滚转角 | -180 到 180 度 |
| pitch | float | 俯仰角 | -180 到 180 度 |
| yaw | float | 偏航角 | -180 到 180 度 |
| battery | float | 电池电量 | 0 到 100 % |
| temperature | float | 温度 | 摄氏度 |
| signal_strength | float | 信号强度 | dBm |
| anomaly | int | 异常标签 | 0=正常, 1=异常 |
| anomaly_type | str | 异常类型 | 见下表 |

### 异常类型编码

| anomaly_type | 数值编码 | 说明 |
|--------------|----------|------|
| normal | 0 | 正常 |
| position_deviation | 1 | 位置偏离 |
| sensor_malfunction_battery | 2 | 电池故障 |
| sensor_malfunction_temperature | 2 | 温度故障 |
| sensor_malfunction_signal_strength | 2 | 信号故障 |
| maneuver_anomaly | 3 | 机动异常 |

**注意**：所有传感器故障类型都编码为2，因为它们本质上都是传感器异常。

---

## 训练参数调优

### 批大小（Batch Size）

**推荐值**：4-16

- **小批量（4-8）**：
  - 优点：内存占用少，可在普通GPU上运行
  - 缺点：训练速度慢，梯度估计有噪声
  
- **大批量（12-16）**：
  - 优点：训练速度快，梯度估计准确
  - 缺点：需要更多GPU内存

### 学习率（Learning Rate）

**推荐值**：0.0001 - 0.001

- **0.001**：默认值，适合大多数情况
- **0.0005**：如果训练不稳定，降低学习率
- **0.0001**：精细调优阶段

**学习率调度**：使用ReduceLROnPlateau，当验证损失不再下降时自动降低学习率。

### 窗口大小（Window Size）

**推荐值**：30-100

- **30-50**：短期模式，快速反应
- **50-70**：中期模式，平衡准确率和响应速度（**推荐**）
- **70-100**：长期模式，更准确但响应慢

### Patch大小

**推荐值**：5-20

模型会自适应选择，但可以调整候选范围：

```python
patch_sizes = [5, 10, 20]  # 小、中、大三种
```

### 隐藏层维度（Hidden Dim）

**推荐值**：32-128

- **32-64**：小模型，快速训练，适合数据量小的情况
- **64-96**：中等模型（**推荐**）
- **96-128**：大模型，表达能力强，需要更多训练数据

---

## 常见问题

### Q1: 训练时内存不足怎么办？

**解决方案**：
1. 减小批大小：`batch_size=4`
2. 减小窗口大小：`window_size=30`
3. 减小隐藏层维度：`hidden_dim=32`
4. 使用CPU训练（虽然会很慢）

### Q2: 模型准确率不高怎么办？

**可能原因和解决方案**：
1. **训练不足**：增加训练轮数（epochs=100）
2. **学习率不合适**：尝试调整学习率
3. **数据不平衡**：增加异常样本数量或使用加权损失
4. **模型太小**：增大hidden_dim或增加GNN层数

### Q3: 如何加快训练速度？

**方法**：
1. 使用GPU：确保安装CUDA版本的PyTorch
2. 增大批大小：在内存允许的情况下
3. 减少数据量：减小duration_minutes
4. 使用多进程数据加载：`num_workers=4`

### Q4: 如何添加新的异常类型？

**步骤**：
1. 在`AnomalyInjector`类中添加新的注入方法
2. 在`anomaly_type_map`中添加新类型的映射
3. 修改模型的`num_anomaly_types`参数
4. 更新可视化代码中的类型名称

### Q5: 可以用在实际无人机上吗？

**建议**：
- 当前版本使用模拟数据，实际部署需要：
  1. 使用真实无人机数据重新训练
  2. 实现在线数据流处理
  3. 优化模型以满足实时性要求
  4. 添加更多实际场景的异常类型

---

## API参考

### generate_data.py

#### DroneSwarmDataGenerator

```python
generator = DroneSwarmDataGenerator(
    num_drones=10,        # 无人机数量
    duration_minutes=60,  # 飞行时长
    sampling_rate=1       # 采样率（Hz）
)

data = generator.generate_swarm_data(output_dir='data')
```

### inject_anomalies.py

#### AnomalyInjector

```python
injector = AnomalyInjector(data, seed=42)

# 注入位置偏离
injector.inject_position_deviation(
    drone_id, start_idx, duration, deviation_magnitude
)

# 注入传感器故障
injector.inject_sensor_malfunction(
    drone_id, start_idx, duration, sensor_type
)

# 注入机动异常
injector.inject_maneuver_anomaly(
    drone_id, start_idx, duration, intensity
)

# 保存数据
injector.save_data_and_log(output_dir='data')

# 划分训练/测试集
train_data, test_data = injector.split_train_test(train_ratio=0.7)
```

### models/patch_gnn_model.py

#### create_model

```python
model = create_model(
    input_dim=15,           # 输入特征维度
    num_drones=10,          # 无人机数量
    patch_size=10,          # Patch大小
    hidden_dim=64,          # 隐藏层维度
    num_anomaly_types=4     # 异常类型数量
)
```

### train.py

#### train_model

```python
model, history = train_model(
    train_data_path='data/train_data.csv',
    test_data_path='data/test_data.csv',
    epochs=50,
    batch_size=8,
    lr=0.001,
    window_size=50,
    stride=25,
    device='cuda'
)
```

### visualize.py

#### ResultVisualizer

```python
visualizer = ResultVisualizer(
    model_path='results/best_model.pth',
    test_data_path='data/test_data.csv',
    device='cuda'
)

# 可视化单个无人机
visualizer.plot_single_drone_results(
    drone_id=0,
    start_idx=0,
    length=1000,
    output_dir='results'
)

# 可视化所有无人机
visualizer.plot_all_drones(
    start_idx=0,
    length=800,
    output_dir='results'
)

# 绘制训练历史
visualizer.plot_training_history(
    history_path='results/training_history.json',
    output_dir='results'
)
```

---

## 性能基准

### 硬件要求

| 配置 | CPU训练 | GPU训练 |
|------|---------|---------|
| 最低 | 4核心, 8GB RAM | GTX 1060, 6GB VRAM |
| 推荐 | 8核心, 16GB RAM | RTX 2060, 8GB VRAM |
| 理想 | 16核心, 32GB RAM | RTX 3080, 10GB VRAM |

### 训练时间估计

| Epochs | Batch Size | CPU | GPU (RTX 2060) |
|--------|------------|-----|----------------|
| 50 | 8 | ~10 小时 | ~45 分钟 |
| 100 | 8 | ~20 小时 | ~90 分钟 |
| 50 | 16 | ~8 小时 | ~30 分钟 |

---

## 更新日志

### v1.0.0 (2024-12-27)
- ✅ 初始版本发布
- ✅ 实现完整的数据生成、异常注入、训练、可视化流程
- ✅ 支持3种典型异常类型
- ✅ 提供详细的中英文文档

---

## 联系与支持

- **Issues**: 在GitHub上提交Issue
- **贡献**: 欢迎Pull Request
- **讨论**: 在Discussions区域参与讨论

---

## 许可证

MIT License - 详见LICENSE文件
