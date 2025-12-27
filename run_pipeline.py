"""
主运行脚本 / Main Pipeline Script
用于一键运行整个无人机集群异常检测流程

运行流程：
1. 生成正常数据
2. 注入异常
3. 训练模型
4. 可视化结果
"""

import os
import sys
import argparse
import torch

def run_data_generation():
    """运行数据生成"""
    print("\n" + "="*80)
    print("步骤 1/4: 生成无人机集群数据")
    print("="*80)
    
    from generate_data import main as generate_main
    generate_main()
    
    print("\n✓ 数据生成完成")


def run_anomaly_injection():
    """运行异常注入"""
    print("\n" + "="*80)
    print("步骤 2/4: 注入异常并划分数据集")
    print("="*80)
    
    from inject_anomalies import main as inject_main
    inject_main()
    
    print("\n✓ 异常注入完成")


def run_training(epochs=50, batch_size=8, lr=0.001):
    """运行模型训练"""
    print("\n" + "="*80)
    print("步骤 3/4: 训练Patch-GNN异常检测模型")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    from train import train_model
    train_model(
        train_data_path='data/train_data.csv',
        test_data_path='data/test_data.csv',
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        window_size=50,
        stride=25,
        device=device
    )
    
    print("\n✓ 模型训练完成")


def run_visualization():
    """运行结果可视化"""
    print("\n" + "="*80)
    print("步骤 4/4: 可视化检测结果")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from visualize import ResultVisualizer
    visualizer = ResultVisualizer(
        model_path='results/best_model.pth',
        test_data_path='data/test_data.csv',
        device=device
    )
    
    # 可视化所有无人机
    visualizer.plot_all_drones(start_idx=0, length=800, output_dir='results')
    
    # 绘制训练历史
    visualizer.plot_training_history(
        history_path='results/training_history.json',
        output_dir='results'
    )
    
    print("\n✓ 可视化完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='无人机集群异常检测系统 - 完整流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行完整流程
  python run_pipeline.py --all
  
  # 只生成数据
  python run_pipeline.py --generate
  
  # 只训练模型（需要先有数据）
  python run_pipeline.py --train --epochs 30 --batch-size 16
  
  # 只可视化（需要先训练模型）
  python run_pipeline.py --visualize
        """
    )
    
    # 运行选项
    parser.add_argument('--all', action='store_true', help='运行完整流程（数据生成+异常注入+训练+可视化）')
    parser.add_argument('--generate', action='store_true', help='只运行数据生成')
    parser.add_argument('--inject', action='store_true', help='只运行异常注入')
    parser.add_argument('--train', action='store_true', help='只运行模型训练')
    parser.add_argument('--visualize', action='store_true', help='只运行可视化')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数 (默认: 50)')
    parser.add_argument('--batch-size', type=int, default=8, help='批大小 (默认: 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率 (默认: 0.001)')
    
    args = parser.parse_args()
    
    # 如果没有指定任何选项，默认运行完整流程
    if not any([args.all, args.generate, args.inject, args.train, args.visualize]):
        args.all = True
    
    print("="*80)
    print("无人机集群异常检测系统")
    print("Drone Swarm Anomaly Detection System")
    print("="*80)
    
    try:
        if args.all:
            # 运行完整流程
            run_data_generation()
            run_anomaly_injection()
            run_training(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
            run_visualization()
        else:
            # 运行指定步骤
            if args.generate:
                run_data_generation()
            
            if args.inject:
                run_anomaly_injection()
            
            if args.train:
                run_training(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
            
            if args.visualize:
                run_visualization()
        
        print("\n" + "="*80)
        print("✓ 所有任务完成！")
        print("="*80)
        print("\n生成的文件:")
        print("  数据文件:")
        print("    - data/drone_swarm_normal.csv")
        print("    - data/drone_swarm_with_anomalies.csv")
        print("    - data/train_data.csv")
        print("    - data/test_data.csv")
        print("    - data/anomaly_injection_log.txt")
        print("\n  模型文件:")
        print("    - results/best_model.pth")
        print("    - results/training_history.json")
        print("\n  可视化文件:")
        print("    - results/training_history.png")
        print("    - results/drone_0_results.png")
        print("    - results/drone_1_results.png")
        print("    - ... (共10个无人机的结果图)")
        print("\n查看 README.md 了解更多信息")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
