"""
快速测试脚本
演示系统的基本功能，不进行完整训练
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.patch_gnn_model import create_model
from train import DroneSwarmDataset
import os

def test_data_loading():
    """测试数据加载"""
    print("\n" + "="*60)
    print("测试 1: 数据加载")
    print("="*60)
    
    # 检查数据文件是否存在
    files_to_check = [
        'data/drone_swarm_normal.csv',
        'data/drone_swarm_with_anomalies.csv',
        'data/train_data.csv',
        'data/test_data.csv'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            print(f"✓ {file_path}: {data.shape[0]} 行, {data.shape[1]} 列")
        else:
            print(f"✗ {file_path}: 文件不存在")
    
    return True


def test_dataset():
    """测试数据集类"""
    print("\n" + "="*60)
    print("测试 2: PyTorch数据集")
    print("="*60)
    
    try:
        dataset = DroneSwarmDataset('data/train_data.csv', window_size=50, stride=25)
        print(f"✓ 数据集创建成功")
        print(f"  - 样本数: {len(dataset)}")
        print(f"  - 特征维度: {len(dataset.feature_cols)}")
        
        # 获取一个样本
        x, y_anomaly, y_type = dataset[0]
        print(f"  - 样本形状: x={x.shape}, y_anomaly={y_anomaly.shape}, y_type={y_type.shape}")
        print(f"  - 异常样本数: {y_anomaly.sum().item()}")
        
        return True
    except Exception as e:
        print(f"✗ 数据集测试失败: {e}")
        return False


def test_model():
    """测试模型"""
    print("\n" + "="*60)
    print("测试 3: Patch-GNN模型")
    print("="*60)
    
    try:
        # 创建模型
        model = create_model(input_dim=15, num_drones=10, patch_size=10, 
                           hidden_dim=64, num_anomaly_types=4)
        print(f"✓ 模型创建成功")
        print(f"  - 参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        batch_size = 2
        num_drones = 10
        seq_len = 50
        input_dim = 15
        
        x = torch.randn(batch_size, num_drones, seq_len, input_dim)
        anomaly_scores, anomaly_types, reconstructions = model(x)
        
        print(f"✓ 前向传播成功")
        print(f"  - 异常分数形状: {anomaly_scores.shape}")
        print(f"  - 异常类型形状: {anomaly_types.shape}")
        print(f"  - 重构结果形状: {reconstructions.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_prediction():
    """测试简单预测"""
    print("\n" + "="*60)
    print("测试 4: 模型预测（使用随机权重）")
    print("="*60)
    
    try:
        # 加载一小部分测试数据
        test_data = pd.read_csv('data/test_data.csv')
        drone_0_data = test_data[test_data['drone_id'] == 0].head(100)
        
        print(f"✓ 加载测试数据: {len(drone_0_data)} 个样本")
        
        # 创建模型
        model = create_model(input_dim=15, num_drones=10, patch_size=10, 
                           hidden_dim=64, num_anomaly_types=4)
        model.eval()
        
        # 准备输入（使用滑动窗口）
        feature_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az',
                       'roll', 'pitch', 'yaw', 'battery', 'temperature', 'signal_strength']
        
        window_size = 50
        if len(drone_0_data) >= window_size:
            # 获取所有无人机的窗口数据
            x_list = []
            for drone_id in range(10):
                drone_data = test_data[test_data['drone_id'] == drone_id].head(window_size)
                features = drone_data[feature_cols].values
                
                # 简单标准化
                features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
                x_list.append(features)
            
            x = torch.FloatTensor(np.array(x_list)).unsqueeze(0)  # [1, 10, 50, 15]
            
            # 预测
            with torch.no_grad():
                anomaly_scores, anomaly_types, _ = model(x)
            
            # 提取预测结果
            pred_anomaly = torch.argmax(anomaly_scores[0], dim=1).numpy()
            pred_types = torch.argmax(anomaly_types[0], dim=1).numpy()
            
            print(f"✓ 预测成功")
            print(f"  - 预测为异常的无人机: {np.where(pred_anomaly == 1)[0].tolist()}")
            print(f"  - 异常类型分布: {np.bincount(pred_types)}")
            print(f"\n  注意: 这是使用随机初始化权重的预测，仅用于测试功能")
            
        return True
    except Exception as e:
        print(f"✗ 预测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization_setup():
    """测试可视化功能"""
    print("\n" + "="*60)
    print("测试 5: 可视化功能")
    print("="*60)
    
    try:
        # 创建简单的测试图
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        
        # 生成一些测试数据
        x = np.arange(100)
        y_true = np.zeros(100)
        y_true[30:50] = 1  # 模拟异常段
        y_pred = np.zeros(100)
        y_pred[28:52] = 1  # 模拟预测（有轻微偏移）
        
        ax.plot(x, y_true, label='Ground Truth', linewidth=3, alpha=0.6)
        ax.plot(x, y_pred, label='Prediction', linewidth=2, linestyle='--', alpha=0.8)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Anomaly')
        ax.set_title('Test Visualization: Anomaly Detection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加异常段背景
        ax.axvspan(30, 50, alpha=0.2, color='red', label='Anomaly Region')
        
        # 保存图形
        os.makedirs('results', exist_ok=True)
        output_file = 'results/test_visualization.png'
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 可视化功能正常")
        print(f"  - 测试图已保存到: {output_file}")
        
        return True
    except Exception as e:
        print(f"✗ 可视化测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("="*60)
    print("无人机集群异常检测系统 - 功能测试")
    print("="*60)
    
    results = []
    
    # 运行所有测试
    results.append(("数据加载", test_data_loading()))
    results.append(("数据集类", test_dataset()))
    results.append(("模型创建", test_model()))
    results.append(("模型预测", test_simple_prediction()))
    results.append(("可视化", test_visualization_setup()))
    
    # 总结测试结果
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n所有测试通过！系统功能正常。")
        print("\n后续步骤:")
        print("  1. 运行 'python train.py' 进行完整训练（需要较长时间）")
        print("  2. 运行 'python visualize.py' 查看训练结果")
        print("  3. 或者运行 'python run_pipeline.py --all' 一键执行完整流程")
    else:
        print("\n部分测试失败，请检查错误信息。")
    
    print("="*60)


if __name__ == "__main__":
    main()
