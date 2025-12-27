# 快速入门指南 / Quick Start Guide

## 中文版

### 5分钟快速开始

1. **安装依赖**
```bash
pip install numpy pandas torch torch-geometric matplotlib seaborn scikit-learn tqdm
```

2. **测试系统**
```bash
python test_system.py
```
如果看到"所有测试通过"，说明系统正常。

3. **生成数据**
```bash
python generate_data.py
python inject_anomalies.py
```
这将生成10架无人机60分钟的飞行数据，并注入9个异常段。

4. **查看数据**
```python
import pandas as pd
data = pd.read_csv('data/drone_swarm_with_anomalies.csv')
print(data.head())
print(f"异常样本数: {(data['anomaly'] == 1).sum()}")
```

5. **训练模型**（可选，需要较长时间）
```bash
python train.py
```
或者使用更少的epoch快速测试：
```bash
python run_pipeline.py --train --epochs 5
```

6. **查看结果**
生成的文件：
- `data/train_data.csv` - 训练数据
- `data/test_data.csv` - 测试数据  
- `data/anomaly_injection_log.txt` - 异常详细说明
- `results/best_model.pth` - 训练的模型（如果运行了训练）

---

## English Version

### 5-Minute Quick Start

1. **Install Dependencies**
```bash
pip install numpy pandas torch torch-geometric matplotlib seaborn scikit-learn tqdm
```

2. **Test System**
```bash
python test_system.py
```
If you see "All tests passed", the system is working correctly.

3. **Generate Data**
```bash
python generate_data.py
python inject_anomalies.py
```
This generates 60 minutes of flight data for 10 drones with 9 anomaly segments.

4. **Explore Data**
```python
import pandas as pd
data = pd.read_csv('data/drone_swarm_with_anomalies.csv')
print(data.head())
print(f"Anomaly samples: {(data['anomaly'] == 1).sum()}")
```

5. **Train Model** (Optional, takes time)
```bash
python train.py
```
Or quick test with fewer epochs:
```bash
python run_pipeline.py --train --epochs 5
```

6. **Check Results**
Generated files:
- `data/train_data.csv` - Training data
- `data/test_data.csv` - Test data
- `data/anomaly_injection_log.txt` - Detailed anomaly info
- `results/best_model.pth` - Trained model (if training was run)

---

## 常见使用场景 / Common Use Cases

### 场景1：只想看看数据
```bash
python generate_data.py
python inject_anomalies.py
# 然后用pandas或Excel查看CSV文件
```

### 场景2：完整运行（需要数小时）
```bash
python run_pipeline.py --all
```

### 场景3：快速测试模型（5个epoch）
```bash
python generate_data.py
python inject_anomalies.py
python run_pipeline.py --train --epochs 5
```

### 场景4：只测试功能
```bash
python test_system.py
```

---

## Scenario 1: Just Want to See Data
```bash
python generate_data.py
python inject_anomalies.py
# Then view CSV files with pandas or Excel
```

## Scenario 2: Full Pipeline (Takes Hours)
```bash
python run_pipeline.py --all
```

## Scenario 3: Quick Model Test (5 epochs)
```bash
python generate_data.py
python inject_anomalies.py
python run_pipeline.py --train --epochs 5
```

## Scenario 4: Just Test Functionality
```bash
python test_system.py
```

---

## 项目结构 / Project Structure

```
Swarm-Anomaly-Detection/
├── data/                      # 数据文件 / Data files
├── models/                    # 模型代码 / Model code
├── utils/                     # 工具函数 / Utility functions
├── results/                   # 结果输出 / Results output
├── generate_data.py          # 生成数据 / Generate data
├── inject_anomalies.py       # 注入异常 / Inject anomalies
├── train.py                  # 训练模型 / Train model
├── visualize.py              # 可视化 / Visualize results
├── run_pipeline.py           # 一键运行 / One-click run
├── test_system.py            # 测试系统 / Test system
├── README.md                 # 项目说明 / Project overview
├── DOCUMENTATION_CN.md       # 详细文档 / Detailed docs
└── requirements.txt          # 依赖列表 / Dependencies
```

---

## 帮助 / Help

遇到问题？查看：
- 详细文档：`DOCUMENTATION_CN.md`
- 测试系统：`python test_system.py`
- GitHub Issues

Having trouble? Check:
- Detailed docs: `DOCUMENTATION_CN.md`
- Test system: `python test_system.py`
- GitHub Issues
