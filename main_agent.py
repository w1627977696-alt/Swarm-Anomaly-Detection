#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机集群异常检测Agent系统 - 主入口
UAV Swarm Anomaly Detection Agent System - Main Entry

这是Agent系统的主入口文件，提供：
1. 交互式命令行界面
2. 批量命令执行
3. 配置管理

使用方法：
    # 启动交互式会话
    python main_agent.py
    
    # 指定配置
    python main_agent.py --config config.json
    
    # 执行单个命令
    python main_agent.py --command "检测所有无人机的异常"
    
    # 批量执行命令
    python main_agent.py --batch commands.txt

Author: AI Agent
Version: 1.0.0
"""

import os
import sys
import argparse
import json
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.core import DroneSwarmAgent, create_agent


def parse_arguments():
    """
    解析命令行参数
    
    返回:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='无人机集群异常检测Agent系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 启动交互式会话
  python main_agent.py
  
  # 使用自定义配置
  python main_agent.py --config my_config.json
  
  # 执行单个命令
  python main_agent.py --command "检测所有无人机的异常"
  
  # 批量执行命令
  python main_agent.py --batch commands.txt
  
  # 直接生成报告
  python main_agent.py --command "生成综合报告"

支持的命令示例:
  - "检测异常" / "detect anomaly"
  - "可视化数据" / "visualize data"
  - "数据回放" / "replay data"
  - "影响评估" / "impact assessment"
  - "生成报告" / "generate report"
  - "帮助" / "help"
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='配置文件路径 (JSON格式)'
    )
    
    parser.add_argument(
        '--command', '-cmd',
        type=str,
        default=None,
        help='执行单个命令'
    )
    
    parser.add_argument(
        '--batch', '-b',
        type=str,
        default=None,
        help='批量命令文件路径（每行一个命令）'
    )
    
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='data',
        help='数据目录路径 (默认: data)'
    )
    
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        default='results/best_model.pth',
        help='模型文件路径 (默认: results/best_model.pth)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results',
        help='输出目录路径 (默认: results)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出模式'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    加载配置文件
    
    参数:
        config_path: 配置文件路径
    
    返回:
        配置字典
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def run_batch_commands(agent: DroneSwarmAgent, batch_file: str) -> list:
    """
    批量执行命令
    
    参数:
        agent: Agent实例
        batch_file: 命令文件路径
    
    返回:
        执行结果列表
    """
    results = []
    
    with open(batch_file, 'r', encoding='utf-8') as f:
        commands = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"\n开始批量执行 {len(commands)} 个命令...")
    print("="*60)
    
    for i, cmd in enumerate(commands, 1):
        print(f"\n[{i}/{len(commands)}] 执行: {cmd}")
        print("-"*40)
        result = agent.process_input(cmd)
        results.append({'command': cmd, 'result': result})
        print(result)
    
    print("\n" + "="*60)
    print(f"批量执行完成，共 {len(commands)} 个命令")
    print("="*60)
    
    return results


def print_welcome():
    """打印欢迎信息"""
    welcome = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║           无人机集群异常检测与影响评估Agent系统                                ║
║           UAV Swarm Anomaly Detection & Impact Assessment Agent              ║
║                                                                              ║
║           版本: 1.0.0                                                        ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  功能特性:                                                                   ║
║  ✓ 自然语言交互 - 使用中文或英文与系统对话                                    ║
║  ✓ 数据预处理与可视化 - 加载和展示无人机飞行数据                               ║
║  ✓ 数据回放 - 回放历史飞行数据                                               ║
║  ✓ 异常检测 - 基于Patch-GNN模型的智能异常检测                                 ║
║  ✓ 影响评估 - 基于RAG技术的异常影响分析                                       ║
║  ✓ 报告生成 - 生成结构化的综合分析报告                                        ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  快速开始:                                                                   ║
║  • 输入 "帮助" 或 "help" 获取使用说明                                         ║
║  • 输入 "状态" 查看系统状态                                                   ║
║  • 输入 "退出" 或 "quit" 退出系统                                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(welcome)


def main():
    """主函数"""
    args = parse_arguments()
    
    # 打印欢迎信息（仅在交互模式下）
    if not args.command and not args.batch:
        print_welcome()
    
    # 构建配置
    config = {
        'data_dir': args.data_dir,
        'model_path': args.model_path,
        'output_dir': args.output_dir,
        'knowledge_base_path': 'knowledge_base'
    }
    
    # 加载配置文件（如果指定）
    if args.config:
        file_config = load_config(args.config)
        config.update(file_config)
    
    # 创建Agent
    agent = create_agent(config)
    
    try:
        # 执行单个命令
        if args.command:
            print(f"\n执行命令: {args.command}")
            print("="*60)
            result = agent.process_input(args.command)
            print(result)
            return
        
        # 批量执行命令
        if args.batch:
            if not os.path.exists(args.batch):
                print(f"错误: 批量命令文件不存在: {args.batch}")
                sys.exit(1)
            run_batch_commands(agent, args.batch)
            return
        
        # 交互式会话
        agent.run_interactive()
        
    except KeyboardInterrupt:
        print("\n\n检测到中断信号，正在退出...")
    except Exception as e:
        print(f"\n发生错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    print("\n感谢使用无人机集群异常检测Agent系统！")


if __name__ == "__main__":
    main()
