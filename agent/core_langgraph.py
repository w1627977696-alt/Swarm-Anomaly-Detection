"""
无人机集群异常检测Agent系统 - LangGraph 实现
UAV Swarm Anomaly Detection Agent System - LangGraph Implementation

本模块使用 LangGraph 框架实现多 Agent 协作系统。
LangGraph 提供了状态管理和 Agent 编排功能。

主要特性:
1. 使用 StateGraph 进行 Agent 流程编排
2. 状态化的 Agent 交互
3. 可视化的工作流
4. 灵活的条件路由

Author: AI Agent
Version: 2.0.0 (LangGraph)
"""

import os
import sys
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
import operator

# LangGraph 核心组件
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# LangChain 核心组件
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_base.knowledge_base_langchain import DroneAnomalyKnowledgeBase


class AgentState(TypedDict):
    """
    Agent 状态定义
    
    使用 TypedDict 定义状态结构，便于类型检查和编辑器支持。
    """
    # 消息历史
    messages: Annotated[List[BaseMessage], operator.add]
    
    # 任务信息
    task_type: Optional[str]
    task_parameters: Optional[Dict[str, Any]]
    
    # 数据加载状态
    data_loaded: bool
    data_path: Optional[str]
    
    # 检测结果
    detected_anomalies: List[Dict[str, Any]]
    
    # 评估结果
    impact_assessments: List[Dict[str, Any]]
    
    # 报告内容
    report: Optional[str]
    
    # 中间结果
    intermediate_results: Dict[str, Any]
    
    # 下一步操作
    next_action: Optional[str]


class DroneSwarmAgentGraph:
    """
    无人机集群异常检测 Agent 系统 - LangGraph 实现
    
    使用 LangGraph 的 StateGraph 实现多 Agent 协作系统。
    各个 Agent 作为节点，通过状态传递协同工作。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化 Agent 图
        
        参数:
            config: 配置字典
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
        
        # 初始化知识库
        print("[AgentGraph] 初始化知识库...")
        self.knowledge_base = DroneAnomalyKnowledgeBase(
            self.config.get('knowledge_base_path', 'knowledge_base')
        )
        
        # 创建 Agent 图
        print("[AgentGraph] 构建 Agent 工作流...")
        self.graph = self._build_graph()
        
        # 编译图
        self.app = self.graph.compile()
        
        print("[AgentGraph] Agent 系统已就绪")
        print("="*60)
    
    def _build_graph(self) -> StateGraph:
        """
        构建 Agent 工作流图
        
        返回:
            StateGraph 实例
        """
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 添加节点 (各个 Agent)
        workflow.add_node("router", self._router_agent)
        workflow.add_node("data_loader", self._data_loader_agent)
        workflow.add_node("anomaly_detector", self._anomaly_detector_agent)
        workflow.add_node("impact_assessor", self._impact_assessor_agent)
        workflow.add_node("report_generator", self._report_generator_agent)
        workflow.add_node("responder", self._responder_agent)
        
        # 设置入口点
        workflow.set_entry_point("router")
        
        # 添加条件边 (根据 next_action 路由)
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "load_data": "data_loader",
                "detect_anomaly": "anomaly_detector",
                "assess_impact": "impact_assessor",
                "generate_report": "report_generator",
                "respond": "responder",
                "end": END
            }
        )
        
        # 各节点完成后返回 router 进行下一步决策
        workflow.add_edge("data_loader", "router")
        workflow.add_edge("anomaly_detector", "router")
        workflow.add_edge("impact_assessor", "router")
        workflow.add_edge("report_generator", "router")
        workflow.add_edge("responder", END)
        
        return workflow
    
    def _router_agent(self, state: AgentState) -> AgentState:
        """
        路由 Agent - 决定下一步操作
        
        参数:
            state: 当前状态
        
        返回:
            更新后的状态
        """
        # 获取最后一条用户消息
        user_messages = [msg for msg in state['messages'] if isinstance(msg, HumanMessage)]
        
        if not user_messages:
            state['next_action'] = 'respond'
            return state
        
        last_message = user_messages[-1].content.lower()
        
        # 简单的意图识别
        if any(kw in last_message for kw in ['加载', '读取', 'load', 'data']):
            if not state.get('data_loaded', False):
                state['next_action'] = 'load_data'
            else:
                state['next_action'] = 'respond'
        
        elif any(kw in last_message for kw in ['检测', '异常', 'detect', 'anomaly']):
            if not state.get('data_loaded', False):
                state['next_action'] = 'load_data'
            elif not state.get('detected_anomalies'):
                state['next_action'] = 'detect_anomaly'
            else:
                state['next_action'] = 'respond'
        
        elif any(kw in last_message for kw in ['评估', '影响', 'assess', 'impact']):
            if not state.get('detected_anomalies'):
                state['next_action'] = 'detect_anomaly'
            elif not state.get('impact_assessments'):
                state['next_action'] = 'assess_impact'
            else:
                state['next_action'] = 'respond'
        
        elif any(kw in last_message for kw in ['报告', '生成', 'report', 'generate']):
            if not state.get('detected_anomalies'):
                state['next_action'] = 'detect_anomaly'
            elif not state.get('impact_assessments'):
                state['next_action'] = 'assess_impact'
            elif not state.get('report'):
                state['next_action'] = 'generate_report'
            else:
                state['next_action'] = 'respond'
        
        elif any(kw in last_message for kw in ['帮助', 'help', '状态', 'status']):
            state['next_action'] = 'respond'
        
        elif any(kw in last_message for kw in ['退出', 'quit', 'exit']):
            state['next_action'] = 'end'
        
        else:
            state['next_action'] = 'respond'
        
        return state
    
    def _route_decision(self, state: AgentState) -> str:
        """
        路由决策函数
        
        参数:
            state: 当前状态
        
        返回:
            下一个节点名称
        """
        return state.get('next_action', 'respond')
    
    def _data_loader_agent(self, state: AgentState) -> AgentState:
        """
        数据加载 Agent
        
        参数:
            state: 当前状态
        
        返回:
            更新后的状态
        """
        print("[DataLoader] 正在加载数据...")
        
        # 模拟数据加载
        data_path = os.path.join(self.config['data_dir'], 'test_data.csv')
        
        if os.path.exists(data_path):
            state['data_loaded'] = True
            state['data_path'] = data_path
            state['messages'].append(
                AIMessage(content=f"✓ 数据已成功加载: {data_path}")
            )
        else:
            state['data_loaded'] = False
            state['messages'].append(
                AIMessage(content=f"✗ 数据文件不存在: {data_path}\n请先运行数据生成脚本")
            )
        
        return state
    
    def _anomaly_detector_agent(self, state: AgentState) -> AgentState:
        """
        异常检测 Agent
        
        参数:
            state: 当前状态
        
        返回:
            更新后的状态
        """
        print("[AnomalyDetector] 正在执行异常检测...")
        
        # 这里应该调用实际的异常检测模型
        # 目前使用模拟数据
        
        anomalies = [
            {
                'drone_id': 0,
                'anomaly_type': 'position_deviation',
                'severity': 'high',
                'confidence': 0.92,
                'timestamp': '2024-12-29 10:15:30',
                'description': '无人机0检测到位置偏离异常'
            },
            {
                'drone_id': 1,
                'anomaly_type': 'sensor_malfunction_battery',
                'severity': 'medium',
                'confidence': 0.85,
                'timestamp': '2024-12-29 10:18:45',
                'description': '无人机1检测到电池传感器异常'
            }
        ]
        
        state['detected_anomalies'] = anomalies
        
        # 构建响应消息
        message = f"✓ 异常检测完成\n"
        message += f"检测到 {len(anomalies)} 个异常:\n"
        for i, anomaly in enumerate(anomalies, 1):
            message += f"  {i}. 无人机{anomaly['drone_id']}: {anomaly['description']}\n"
        
        state['messages'].append(AIMessage(content=message))
        
        return state
    
    def _impact_assessor_agent(self, state: AgentState) -> AgentState:
        """
        影响评估 Agent - 基于 RAG
        
        参数:
            state: 当前状态
        
        返回:
            更新后的状态
        """
        print("[ImpactAssessor] 正在评估异常影响...")
        
        anomalies = state.get('detected_anomalies', [])
        if not anomalies:
            state['messages'].append(
                AIMessage(content="✗ 没有可评估的异常")
            )
            return state
        
        assessments = []
        
        for anomaly in anomalies:
            # 使用 RAG 检索相关知识
            anomaly_type = anomaly.get('anomaly_type', '')
            
            # 搜索相关知识
            knowledge_results = self.knowledge_base.search_all(
                anomaly_type, 
                top_k=1
            )
            
            # 获取相似案例
            similar_cases = self.knowledge_base.get_similar_cases(
                anomaly.get('description', ''),
                top_k=2
            )
            
            # 构建评估
            assessment = {
                'drone_id': anomaly.get('drone_id'),
                'anomaly_type': anomaly_type,
                'severity': anomaly.get('severity'),
                'impact_level': self._determine_impact_level(anomaly),
                'risk_factors': self._extract_risk_factors(knowledge_results),
                'similar_cases': similar_cases,
                'recommendations': self._generate_recommendations(
                    anomaly, 
                    knowledge_results
                )
            }
            
            assessments.append(assessment)
        
        state['impact_assessments'] = assessments
        
        # 构建响应消息
        message = f"✓ 影响评估完成\n"
        message += f"已评估 {len(assessments)} 个异常的影响:\n"
        for i, assessment in enumerate(assessments, 1):
            message += f"  {i}. 无人机{assessment['drone_id']}: "
            message += f"影响等级={assessment['impact_level']}, "
            message += f"严重程度={assessment['severity']}\n"
        
        state['messages'].append(AIMessage(content=message))
        
        return state
    
    def _determine_impact_level(self, anomaly: Dict) -> str:
        """确定影响等级"""
        severity = anomaly.get('severity', 'low')
        confidence = anomaly.get('confidence', 0.5)
        
        if severity == 'high' and confidence > 0.8:
            return '严重'
        elif severity in ['high', 'medium']:
            return '中等'
        else:
            return '轻微'
    
    def _extract_risk_factors(self, knowledge_results: Dict) -> List[str]:
        """从知识库结果中提取风险因素"""
        factors = []
        
        for collection, results in knowledge_results.items():
            if results and len(results) > 0:
                doc, score = results[0]
                metadata = doc.metadata
                
                if 'characteristics' in metadata:
                    factors.append(f"特征: {metadata['characteristics']}")
        
        return factors[:3]  # 最多返回 3 个
    
    def _generate_recommendations(
        self, 
        anomaly: Dict, 
        knowledge_results: Dict
    ) -> List[str]:
        """生成处置建议"""
        recommendations = [
            "持续监控异常无人机状态",
            "记录详细的异常日志",
            "准备应急响应方案"
        ]
        
        severity = anomaly.get('severity', 'low')
        if severity == 'high':
            recommendations.insert(0, "立即启动应急预案")
            recommendations.insert(1, "考虑让异常无人机返航")
        
        return recommendations
    
    def _report_generator_agent(self, state: AgentState) -> AgentState:
        """
        报告生成 Agent
        
        参数:
            state: 当前状态
        
        返回:
            更新后的状态
        """
        print("[ReportGenerator] 正在生成综合报告...")
        
        anomalies = state.get('detected_anomalies', [])
        assessments = state.get('impact_assessments', [])
        
        if not anomalies:
            state['messages'].append(
                AIMessage(content="✗ 没有可生成报告的数据")
            )
            return state
        
        # 生成报告
        report = self._build_report(anomalies, assessments)
        state['report'] = report
        
        # 保存报告
        output_dir = self.config.get('output_dir', 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path = os.path.join(output_dir, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        state['messages'].append(
            AIMessage(content=f"✓ 报告已生成: {report_path}")
        )
        
        return state
    
    def _build_report(
        self, 
        anomalies: List[Dict], 
        assessments: List[Dict]
    ) -> str:
        """构建报告内容"""
        lines = [
            "# 无人机集群异常检测与影响评估报告",
            f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## 1. 执行摘要\n",
            f"- 检测到 {len(anomalies)} 个异常",
            f"- 完成 {len(assessments)} 项影响评估\n",
            "## 2. 异常检测详情\n"
        ]
        
        for i, anomaly in enumerate(anomalies, 1):
            lines.append(f"### 2.{i} 异常 #{i}")
            lines.append(f"- **无人机ID**: {anomaly.get('drone_id')}")
            lines.append(f"- **异常类型**: {anomaly.get('anomaly_type')}")
            lines.append(f"- **严重程度**: {anomaly.get('severity')}")
            lines.append(f"- **置信度**: {anomaly.get('confidence', 0):.2%}")
            lines.append(f"- **描述**: {anomaly.get('description', '无')}\n")
        
        lines.append("## 3. 影响评估\n")
        
        for i, assessment in enumerate(assessments, 1):
            lines.append(f"### 3.{i} 评估 #{i}")
            lines.append(f"- **无人机ID**: {assessment.get('drone_id')}")
            lines.append(f"- **影响等级**: {assessment.get('impact_level')}")
            lines.append(f"- **处置建议**:")
            for rec in assessment.get('recommendations', []):
                lines.append(f"  - {rec}")
            lines.append("")
        
        lines.append("## 4. 总结\n")
        lines.append("请根据以上分析采取适当的应对措施。\n")
        
        return "\n".join(lines)
    
    def _responder_agent(self, state: AgentState) -> AgentState:
        """
        响应 Agent - 生成用户响应
        
        参数:
            state: 当前状态
        
        返回:
            更新后的状态
        """
        # 获取最后一条用户消息
        user_messages = [msg for msg in state['messages'] if isinstance(msg, HumanMessage)]
        
        if not user_messages:
            return state
        
        last_message = user_messages[-1].content.lower()
        
        # 根据消息内容生成响应
        if any(kw in last_message for kw in ['帮助', 'help']):
            response = self._get_help_message()
        elif any(kw in last_message for kw in ['状态', 'status']):
            response = self._get_status_message(state)
        else:
            response = "我已理解您的请求。请输入具体指令或输入 '帮助' 查看可用命令。"
        
        state['messages'].append(AIMessage(content=response))
        state['next_action'] = 'end'
        
        return state
    
    def _get_help_message(self) -> str:
        """获取帮助信息"""
        return """
=== 无人机集群异常检测 Agent 系统 ===

支持的命令:
1. 加载数据 / load data - 加载无人机集群数据
2. 检测异常 / detect anomaly - 执行异常检测
3. 评估影响 / assess impact - 评估异常影响
4. 生成报告 / generate report - 生成综合分析报告
5. 状态 / status - 查看系统状态
6. 帮助 / help - 显示此帮助信息
7. 退出 / quit - 退出系统

示例:
>>> 检测所有无人机的异常
>>> 评估异常影响
>>> 生成综合报告
"""
    
    def _get_status_message(self, state: AgentState) -> str:
        """获取状态信息"""
        return f"""
=== 系统状态 ===
- 数据已加载: {'是' if state.get('data_loaded', False) else '否'}
- 已检测异常: {len(state.get('detected_anomalies', []))}
- 已完成评估: {len(state.get('impact_assessments', []))}
- 报告已生成: {'是' if state.get('report') else '否'}
"""
    
    def process_message(self, message: str) -> str:
        """
        处理用户消息
        
        参数:
            message: 用户消息
        
        返回:
            系统响应
        """
        # 初始化状态
        initial_state = AgentState(
            messages=[HumanMessage(content=message)],
            task_type=None,
            task_parameters=None,
            data_loaded=False,
            data_path=None,
            detected_anomalies=[],
            impact_assessments=[],
            report=None,
            intermediate_results={},
            next_action=None
        )
        
        # 运行工作流
        result = self.app.invoke(initial_state)
        
        # 提取 AI 响应
        ai_messages = [msg.content for msg in result['messages'] 
                      if isinstance(msg, AIMessage)]
        
        return "\n".join(ai_messages) if ai_messages else "处理完成"
    
    def run_interactive(self):
        """运行交互式会话"""
        print("\n" + "="*60)
        print("无人机集群异常检测 Agent 系统 (LangGraph)")
        print("="*60)
        print("输入 '帮助' 获取使用说明")
        print("输入 '退出' 结束会话\n")
        
        while True:
            try:
                user_input = input(">>> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['退出', 'quit', 'exit', 'q']:
                    print("\n再见！")
                    break
                
                response = self.process_message(user_input)
                print(f"\n{response}\n")
                
            except KeyboardInterrupt:
                print("\n\n检测到中断，正在退出...")
                break
            except Exception as e:
                print(f"\n发生错误: {e}\n")
                continue


# 便捷函数
def create_agent(config: Optional[Dict] = None) -> DroneSwarmAgentGraph:
    """
    创建 Agent 实例
    
    参数:
        config: 配置字典
    
    返回:
        DroneSwarmAgentGraph 实例
    """
    return DroneSwarmAgentGraph(config)


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("测试 LangGraph Agent 系统")
    print("="*60)
    
    # 创建 Agent
    agent = create_agent()
    
    # 测试消息处理
    test_messages = [
        "帮助",
        "检测异常",
        "状态"
    ]
    
    for msg in test_messages:
        print(f"\n>>> {msg}")
        response = agent.process_message(msg)
        print(response)
        print("-"*60)
