"""
Agent核心模块
Core Module for Drone Swarm Agent System

该模块实现了无人机集群异常检测Agent系统的核心功能。
作为系统的主入口，协调各个子Agent完成用户请求的任务。

主要组件：
1. DroneSwarmAgent: 主Agent类，负责任务调度和协调
2. AgentState: Agent状态管理
3. TaskExecutor: 任务执行器
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .natural_language_processor import NaturalLanguageProcessor, TaskType


class AgentState(Enum):
    """
    Agent状态枚举
    定义Agent的各种运行状态
    """
    IDLE = "idle"                   # 空闲状态
    PROCESSING = "processing"       # 处理中
    WAITING_INPUT = "waiting"       # 等待输入
    ERROR = "error"                 # 错误状态
    COMPLETED = "completed"         # 完成状态


# 配置常量
# 当置信度低于此阈值时，返回"无法理解"的响应
CONFIDENCE_THRESHOLD_UNCLEAR = 0.3


class AgentContext:
    """
    Agent上下文类
    
    管理Agent运行时的上下文信息，包括：
    - 当前任务状态
    - 已检测到的异常
    - 影响评估结果
    - 对话历史
    """
    
    def __init__(self):
        """初始化上下文"""
        self.state = AgentState.IDLE
        self.current_task = None
        self.detected_anomalies = []
        self.impact_assessments = []
        self.conversation_history = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_loaded = False
        self.model_loaded = False
        self.results = {}
    
    def update_state(self, new_state: AgentState):
        """更新Agent状态"""
        self.state = new_state
    
    def add_anomaly(self, anomaly: Dict):
        """添加检测到的异常"""
        self.detected_anomalies.append(anomaly)
    
    def add_assessment(self, assessment: Dict):
        """添加影响评估结果"""
        self.impact_assessments.append(assessment)
    
    def add_conversation(self, role: str, content: str):
        """添加对话历史"""
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'role': role,
            'content': content
        })
    
    def get_summary(self) -> Dict:
        """获取上下文摘要"""
        return {
            'session_id': self.session_id,
            'state': self.state.value,
            'data_loaded': self.data_loaded,
            'model_loaded': self.model_loaded,
            'anomalies_count': len(self.detected_anomalies),
            'assessments_count': len(self.impact_assessments),
            'conversation_length': len(self.conversation_history)
        }


class DroneSwarmAgent:
    """
    无人机集群异常检测主Agent
    
    功能：
    1. 接收并解析用户的自然语言输入
    2. 根据任务类型调度相应的子Agent
    3. 协调各子Agent之间的数据传递
    4. 管理系统状态和上下文
    5. 返回执行结果给用户
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化主Agent
        
        参数:
            config: 配置字典，可包含以下键：
                - data_dir: 数据目录路径
                - model_path: 模型文件路径
                - knowledge_base_path: 知识库路径
                - output_dir: 输出目录
        """
        # 默认配置
        self.config = {
            'data_dir': 'data',
            'model_path': 'results/best_model.pth',
            'knowledge_base_path': 'knowledge_base',
            'output_dir': 'results'
        }
        if config:
            self.config.update(config)
        
        # 初始化上下文
        self.context = AgentContext()
        
        # 初始化自然语言处理器
        self.nlp = NaturalLanguageProcessor()
        
        # 子Agent将在需要时延迟初始化
        self._data_agent = None
        self._replay_agent = None
        self._detection_agent = None
        self._assessment_agent = None
        self._report_agent = None
        
        print("="*60)
        print("无人机集群异常检测Agent系统已启动")
        print("Drone Swarm Anomaly Detection Agent System Started")
        print("="*60)
        print(f"会话ID: {self.context.session_id}")
        print(f"数据目录: {self.config['data_dir']}")
        print(f"模型路径: {self.config['model_path']}")
        print("输入 '帮助' 或 'help' 获取使用说明")
        print("="*60)
    
    @property
    def data_agent(self):
        """延迟初始化数据处理Agent"""
        if self._data_agent is None:
            from .data_agent import DataPreprocessingAgent
            self._data_agent = DataPreprocessingAgent(self.config)
        return self._data_agent
    
    @property
    def replay_agent(self):
        """延迟初始化数据回放Agent"""
        if self._replay_agent is None:
            from .data_agent import DataReplayAgent
            self._replay_agent = DataReplayAgent(self.config)
        return self._replay_agent
    
    @property
    def detection_agent(self):
        """延迟初始化异常检测Agent"""
        if self._detection_agent is None:
            from .detection_agent import AnomalyDetectionAgent
            self._detection_agent = AnomalyDetectionAgent(self.config)
        return self._detection_agent
    
    @property
    def assessment_agent(self):
        """延迟初始化影响评估Agent"""
        if self._assessment_agent is None:
            from .assessment_agent import ImpactAssessmentAgent
            self._assessment_agent = ImpactAssessmentAgent(self.config)
        return self._assessment_agent
    
    @property
    def report_agent(self):
        """延迟初始化报告生成Agent"""
        if self._report_agent is None:
            from .report_agent import ReportGenerationAgent
            self._report_agent = ReportGenerationAgent(self.config)
        return self._report_agent
    
    def process_input(self, user_input: str) -> str:
        """
        处理用户输入
        
        参数:
            user_input: 用户输入的自然语言字符串
        
        返回:
            处理结果字符串
        """
        # 记录用户输入
        self.context.add_conversation('user', user_input)
        
        # 更新状态
        self.context.update_state(AgentState.PROCESSING)
        
        try:
            # 解析用户输入
            parsed = self.nlp.parse(user_input)
            task_type = parsed['task_type']
            parameters = parsed['parameters']
            confidence = parsed['confidence']
            
            # 根据任务类型执行相应操作
            if task_type == TaskType.HELP:
                response = self._handle_help()
            elif task_type == TaskType.STATUS:
                response = self._handle_status()
            elif task_type == TaskType.UNKNOWN:
                response = self._handle_unknown(user_input, confidence)
            else:
                response = self._execute_task(task_type, parameters)
            
            # 记录响应
            self.context.add_conversation('assistant', response)
            
            # 更新状态
            self.context.update_state(AgentState.IDLE)
            
            return response
            
        except Exception as e:
            self.context.update_state(AgentState.ERROR)
            error_msg = f"处理请求时发生错误: {str(e)}"
            self.context.add_conversation('assistant', error_msg)
            return error_msg
    
    def _execute_task(self, task_type: TaskType, parameters: Dict) -> str:
        """
        执行具体任务
        
        参数:
            task_type: 任务类型
            parameters: 任务参数
        
        返回:
            执行结果字符串
        """
        # 数据预处理
        if task_type == TaskType.DATA_PREPROCESS:
            result = self.data_agent.preprocess(parameters)
            self.context.data_loaded = result.get('success', False)
            return self._format_result("数据预处理", result)
        
        # 数据可视化
        elif task_type == TaskType.DATA_VISUALIZE:
            if not self.context.data_loaded:
                # 自动先加载数据
                self.data_agent.preprocess({})
                self.context.data_loaded = True
            result = self.data_agent.visualize(parameters)
            return self._format_result("数据可视化", result)
        
        # 数据回放
        elif task_type == TaskType.DATA_REPLAY:
            if not self.context.data_loaded:
                self.data_agent.preprocess({})
                self.context.data_loaded = True
            result = self.replay_agent.replay(parameters)
            return self._format_result("数据回放", result)
        
        # 异常检测
        elif task_type in [TaskType.ANOMALY_DETECT, TaskType.ANOMALY_ANALYSIS]:
            if not self.context.data_loaded:
                self.data_agent.preprocess({})
                self.context.data_loaded = True
            result = self.detection_agent.detect(parameters)
            # 保存检测结果到上下文
            if 'anomalies' in result:
                for anomaly in result['anomalies']:
                    self.context.add_anomaly(anomaly)
            self.context.results['detection'] = result
            return self._format_result("异常检测", result)
        
        # 影响评估
        elif task_type in [TaskType.IMPACT_ASSESS, TaskType.RISK_EVALUATE]:
            # 如果没有检测结果，先进行检测
            if not self.context.detected_anomalies:
                detection_result = self.detection_agent.detect({})
                if 'anomalies' in detection_result:
                    for anomaly in detection_result['anomalies']:
                        self.context.add_anomaly(anomaly)
                self.context.results['detection'] = detection_result
            
            # 进行影响评估
            result = self.assessment_agent.assess(
                self.context.detected_anomalies, 
                parameters
            )
            # 保存评估结果到上下文
            if 'assessments' in result:
                for assessment in result['assessments']:
                    self.context.add_assessment(assessment)
            self.context.results['assessment'] = result
            return self._format_result("影响评估", result)
        
        # 报告生成
        elif task_type == TaskType.GENERATE_REPORT:
            # 如果没有检测和评估结果，先进行
            if not self.context.detected_anomalies:
                detection_result = self.detection_agent.detect({})
                if 'anomalies' in detection_result:
                    for anomaly in detection_result['anomalies']:
                        self.context.add_anomaly(anomaly)
                self.context.results['detection'] = detection_result
            
            if not self.context.impact_assessments:
                assessment_result = self.assessment_agent.assess(
                    self.context.detected_anomalies, 
                    {}
                )
                if 'assessments' in assessment_result:
                    for assessment in assessment_result['assessments']:
                        self.context.add_assessment(assessment)
                self.context.results['assessment'] = assessment_result
            
            # 生成报告
            result = self.report_agent.generate(
                self.context.results.get('detection', {}),
                self.context.results.get('assessment', {}),
                parameters
            )
            self.context.results['report'] = result
            return self._format_result("报告生成", result)
        
        # 结果导出
        elif task_type == TaskType.EXPORT_RESULTS:
            result = self._export_results(parameters)
            return self._format_result("结果导出", result)
        
        else:
            return f"暂不支持的任务类型: {task_type.value}"
    
    def _format_result(self, task_name: str, result: Dict) -> str:
        """
        格式化执行结果
        
        参数:
            task_name: 任务名称
            result: 结果字典
        
        返回:
            格式化后的结果字符串
        """
        output_lines = [
            f"\n{'='*60}",
            f"【{task_name}完成】",
            f"{'='*60}"
        ]
        
        if result.get('success', True):
            output_lines.append("✓ 执行成功")
            
            if 'message' in result:
                output_lines.append(f"\n{result['message']}")
            
            if 'summary' in result:
                output_lines.append(f"\n摘要: {result['summary']}")
            
            if 'output_file' in result:
                output_lines.append(f"\n输出文件: {result['output_file']}")
            
            if 'details' in result:
                output_lines.append("\n详细信息:")
                for key, value in result['details'].items():
                    output_lines.append(f"  • {key}: {value}")
        else:
            output_lines.append(f"✗ 执行失败: {result.get('error', '未知错误')}")
        
        output_lines.append("="*60)
        
        return "\n".join(output_lines)
    
    def _handle_help(self) -> str:
        """处理帮助请求"""
        return self.nlp.get_help_message()
    
    def _handle_status(self) -> str:
        """处理状态查询请求"""
        summary = self.context.get_summary()
        
        status_lines = [
            "\n" + "="*60,
            "【系统状态】",
            "="*60,
            f"会话ID: {summary['session_id']}",
            f"当前状态: {summary['state']}",
            f"数据已加载: {'是' if summary['data_loaded'] else '否'}",
            f"模型已加载: {'是' if summary['model_loaded'] else '否'}",
            f"已检测异常数: {summary['anomalies_count']}",
            f"已完成评估数: {summary['assessments_count']}",
            f"对话轮数: {summary['conversation_length']}",
            "="*60
        ]
        
        return "\n".join(status_lines)
    
    def _handle_unknown(self, user_input: str, confidence: float) -> str:
        """处理无法识别的输入"""
        if confidence > CONFIDENCE_THRESHOLD_UNCLEAR:
            return f"您的请求 '{user_input}' 不太明确，请尝试更具体的描述。\n输入 '帮助' 查看支持的功能。"
        else:
            return f"抱歉，我无法理解您的请求: '{user_input}'\n请输入 '帮助' 查看支持的功能和示例指令。"
    
    def _export_results(self, parameters: Dict) -> Dict:
        """
        导出结果
        
        参数:
            parameters: 导出参数
        
        返回:
            导出结果
        """
        output_dir = self.config.get('output_dir', 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备导出数据
        export_data = {
            'session_id': self.context.session_id,
            'export_time': datetime.now().isoformat(),
            'detected_anomalies': self.context.detected_anomalies,
            'impact_assessments': self.context.impact_assessments,
            'results': self.context.results
        }
        
        # 导出为JSON文件
        output_file = os.path.join(output_dir, f'export_{self.context.session_id}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return {
            'success': True,
            'message': '结果已成功导出',
            'output_file': output_file
        }
    
    def run_interactive(self):
        """
        运行交互式会话
        
        启动一个交互式命令行界面，允许用户持续输入指令
        """
        print("\n开始交互式会话，输入 '退出' 或 'quit' 结束")
        print("-"*60)
        
        while True:
            try:
                user_input = input("\n>>> 请输入指令: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['退出', 'quit', 'exit', 'q']:
                    print("\n感谢使用，再见！")
                    break
                
                response = self.process_input(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n检测到中断，正在退出...")
                break
            except Exception as e:
                print(f"\n发生错误: {e}")
                continue
    
    def execute_batch(self, commands: List[str]) -> List[str]:
        """
        批量执行命令
        
        参数:
            commands: 命令列表
        
        返回:
            执行结果列表
        """
        results = []
        for cmd in commands:
            result = self.process_input(cmd)
            results.append(result)
        return results


# 便捷函数
def create_agent(config: Optional[Dict] = None) -> DroneSwarmAgent:
    """
    创建Agent实例的工厂函数
    
    参数:
        config: 配置字典
    
    返回:
        DroneSwarmAgent实例
    """
    return DroneSwarmAgent(config)


if __name__ == "__main__":
    # 创建并运行Agent
    agent = create_agent()
    agent.run_interactive()
