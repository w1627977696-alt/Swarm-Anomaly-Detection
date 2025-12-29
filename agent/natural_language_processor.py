"""
自然语言处理模块
Natural Language Processing Module

该模块负责解析用户的自然语言输入，将其转换为系统可执行的任务指令。
支持中文和英文两种语言的输入。

功能：
1. 意图识别：识别用户想要执行的任务类型
2. 实体提取：提取任务相关的参数（如无人机ID、时间范围等）
3. 任务分发：将解析后的任务分发给相应的Agent
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


class TaskType(Enum):
    """
    任务类型枚举
    定义系统支持的所有任务类型
    """
    # 数据预处理相关
    DATA_PREPROCESS = "data_preprocess"         # 数据预处理
    DATA_VISUALIZE = "data_visualize"           # 数据可视化
    DATA_REPLAY = "data_replay"                 # 数据回放
    
    # 异常检测相关
    ANOMALY_DETECT = "anomaly_detect"           # 异常检测
    ANOMALY_ANALYSIS = "anomaly_analysis"       # 异常分析
    
    # 影响评估相关
    IMPACT_ASSESS = "impact_assess"             # 影响评估
    RISK_EVALUATE = "risk_evaluate"             # 风险评估
    
    # 报告生成相关
    GENERATE_REPORT = "generate_report"         # 生成报告
    EXPORT_RESULTS = "export_results"           # 导出结果
    
    # 系统相关
    HELP = "help"                               # 帮助信息
    STATUS = "status"                           # 系统状态
    UNKNOWN = "unknown"                         # 未知任务


class NaturalLanguageProcessor:
    """
    自然语言处理器
    
    负责解析用户输入的自然语言指令，提取意图和参数。
    支持中文和英文两种语言。
    """
    
    def __init__(self):
        """
        初始化自然语言处理器
        
        设置关键词匹配规则和参数提取模式
        """
        # 定义任务关键词映射（中文）
        self.task_keywords_cn = {
            TaskType.DATA_PREPROCESS: [
                '预处理', '数据处理', '清洗数据', '处理数据', '加载数据', 
                '准备数据', '导入数据', '读取数据'
            ],
            TaskType.DATA_VISUALIZE: [
                '可视化', '显示图', '画图', '绘图', '展示', '查看图表',
                '显示数据', '数据展示', '图表'
            ],
            TaskType.DATA_REPLAY: [
                '回放', '重放', '播放数据', '数据回放', '历史回放',
                '查看历史', '回顾', '重现'
            ],
            TaskType.ANOMALY_DETECT: [
                '检测', '异常检测', '发现异常', '检查异常', '诊断',
                '分析异常', '寻找异常', '检测问题'
            ],
            TaskType.ANOMALY_ANALYSIS: [
                '分析异常', '异常分析', '详细分析', '深入分析',
                '异常详情', '异常信息'
            ],
            TaskType.IMPACT_ASSESS: [
                '影响评估', '评估影响', '影响分析', '后果评估',
                '严重程度', '影响范围'
            ],
            TaskType.RISK_EVALUATE: [
                '风险评估', '风险分析', '安全评估', '危险程度',
                '风险等级', '安全分析'
            ],
            TaskType.GENERATE_REPORT: [
                '生成报告', '报告', '生成总结', '总结', '汇总',
                '输出报告', '创建报告', '综合报告'
            ],
            TaskType.EXPORT_RESULTS: [
                '导出', '保存结果', '导出结果', '输出结果', '存储'
            ],
            TaskType.HELP: [
                '帮助', '怎么用', '使用说明', '功能介绍', '有什么功能',
                '能做什么', '指令'
            ],
            TaskType.STATUS: [
                '状态', '系统状态', '当前状态', '运行状态'
            ]
        }
        
        # 定义任务关键词映射（英文）
        self.task_keywords_en = {
            TaskType.DATA_PREPROCESS: [
                'preprocess', 'process data', 'clean data', 'load data',
                'prepare data', 'import data', 'read data'
            ],
            TaskType.DATA_VISUALIZE: [
                'visualize', 'plot', 'draw', 'show', 'display',
                'chart', 'graph', 'visualization'
            ],
            TaskType.DATA_REPLAY: [
                'replay', 'playback', 'play data', 'history', 'review',
                'recreate', 'rerun'
            ],
            TaskType.ANOMALY_DETECT: [
                'detect', 'find anomaly', 'anomaly detection', 'diagnose',
                'check anomaly', 'search anomaly'
            ],
            TaskType.ANOMALY_ANALYSIS: [
                'analyze anomaly', 'anomaly analysis', 'detailed analysis',
                'anomaly detail', 'anomaly information'
            ],
            TaskType.IMPACT_ASSESS: [
                'impact assessment', 'assess impact', 'impact analysis',
                'consequence', 'severity', 'impact scope'
            ],
            TaskType.RISK_EVALUATE: [
                'risk assessment', 'risk analysis', 'safety assessment',
                'risk level', 'safety analysis', 'danger level'
            ],
            TaskType.GENERATE_REPORT: [
                'generate report', 'report', 'summary', 'summarize',
                'create report', 'output report', 'comprehensive report'
            ],
            TaskType.EXPORT_RESULTS: [
                'export', 'save results', 'output results', 'store'
            ],
            TaskType.HELP: [
                'help', 'how to', 'usage', 'instruction', 'what can',
                'function', 'command'
            ],
            TaskType.STATUS: [
                'status', 'system status', 'current status', 'running status'
            ]
        }
        
        # 参数提取正则表达式
        self.param_patterns = {
            # 无人机ID提取
            'drone_id': [
                r'无人机\s*(\d+)',
                r'drone\s*(\d+)',
                r'UAV\s*(\d+)',
                r'第\s*(\d+)\s*架',
                r'#(\d+)',
                r'ID\s*(\d+)'
            ],
            # 时间范围提取
            'time_range': [
                r'从\s*(\d+)\s*到\s*(\d+)',
                r'(\d+)\s*-\s*(\d+)\s*秒',
                r'(\d+)\s*to\s*(\d+)',
                r'from\s*(\d+)\s*to\s*(\d+)'
            ],
            # 异常类型提取
            'anomaly_type': [
                r'位置偏离|position deviation',
                r'传感器故障|sensor malfunction',
                r'机动异常|maneuver anomaly',
                r'电池异常|battery anomaly',
                r'温度异常|temperature anomaly',
                r'信号异常|signal anomaly'
            ],
            # 严重程度
            'severity': [
                r'轻微|轻度|低|minor|low|slight',
                r'中等|中度|中|moderate|medium',
                r'严重|高|重度|severe|high|critical'
            ]
        }
    
    def parse(self, user_input: str) -> Dict[str, Any]:
        """
        解析用户输入
        
        参数:
            user_input: 用户输入的自然语言字符串
        
        返回:
            解析结果字典，包含：
            - task_type: 任务类型
            - parameters: 提取的参数
            - confidence: 置信度
            - raw_input: 原始输入
        """
        # 预处理输入
        processed_input = self._preprocess_input(user_input)
        
        # 识别任务类型
        task_type, confidence = self._identify_task(processed_input)
        
        # 提取参数
        parameters = self._extract_parameters(processed_input, task_type)
        
        return {
            'task_type': task_type,
            'parameters': parameters,
            'confidence': confidence,
            'raw_input': user_input
        }
    
    def _preprocess_input(self, text: str) -> str:
        """
        预处理输入文本
        
        参数:
            text: 原始文本
        
        返回:
            预处理后的文本
        """
        # 转换为小写（英文部分）
        text = text.lower()
        # 去除多余空白
        text = ' '.join(text.split())
        # 替换常见的同义词和缩写
        replacements = {
            'uav': '无人机',
            '飞机': '无人机',
            '飞行器': '无人机',
            '查一下': '检测',
            '看看': '查看',
            '做个': '生成',
            '出个': '生成'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _identify_task(self, text: str) -> Tuple[TaskType, float]:
        """
        识别任务类型
        
        参数:
            text: 预处理后的文本
        
        返回:
            (任务类型, 置信度)
        """
        max_score = 0
        best_task = TaskType.UNKNOWN
        
        # 计算每种任务的匹配分数
        for task_type, keywords in self.task_keywords_cn.items():
            score = self._calculate_match_score(text, keywords)
            if score > max_score:
                max_score = score
                best_task = task_type
        
        # 也检查英文关键词
        for task_type, keywords in self.task_keywords_en.items():
            score = self._calculate_match_score(text, keywords)
            if score > max_score:
                max_score = score
                best_task = task_type
        
        # 计算置信度（0-1范围）
        confidence = min(max_score / 2.0, 1.0)  # 假设匹配2个关键词为最高置信度
        
        return best_task, confidence
    
    def _calculate_match_score(self, text: str, keywords: List[str]) -> float:
        """
        计算关键词匹配分数
        
        参数:
            text: 输入文本
            keywords: 关键词列表
        
        返回:
            匹配分数
        """
        score = 0.0
        for keyword in keywords:
            if keyword in text:
                # 根据关键词长度给予不同权重
                score += 1.0 + (len(keyword) / 10.0)
        return score
    
    def _extract_parameters(self, text: str, task_type: TaskType) -> Dict[str, Any]:
        """
        从文本中提取参数
        
        参数:
            text: 输入文本
            task_type: 任务类型
        
        返回:
            参数字典
        """
        params = {}
        
        # 提取无人机ID
        drone_ids = self._extract_drone_ids(text)
        if drone_ids:
            params['drone_ids'] = drone_ids
        
        # 提取时间范围
        time_range = self._extract_time_range(text)
        if time_range:
            params['time_range'] = time_range
        
        # 提取异常类型
        anomaly_types = self._extract_anomaly_types(text)
        if anomaly_types:
            params['anomaly_types'] = anomaly_types
        
        # 提取严重程度
        severity = self._extract_severity(text)
        if severity:
            params['severity'] = severity
        
        # 根据任务类型添加默认参数
        params = self._add_default_parameters(params, task_type)
        
        return params
    
    def _extract_drone_ids(self, text: str) -> List[int]:
        """
        提取无人机ID
        
        参数:
            text: 输入文本
        
        返回:
            无人机ID列表
        """
        drone_ids = []
        for pattern in self.param_patterns['drone_id']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    drone_id = int(match)
                    if 0 <= drone_id < 10 and drone_id not in drone_ids:
                        drone_ids.append(drone_id)
                except ValueError:
                    continue
        return drone_ids
    
    def _extract_time_range(self, text: str) -> Optional[Tuple[int, int]]:
        """
        提取时间范围
        
        参数:
            text: 输入文本
        
        返回:
            时间范围元组 (start, end) 或 None
        """
        for pattern in self.param_patterns['time_range']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    start = int(match.group(1))
                    end = int(match.group(2))
                    return (start, end)
                except (ValueError, IndexError):
                    continue
        return None
    
    def _extract_anomaly_types(self, text: str) -> List[str]:
        """
        提取异常类型
        
        参数:
            text: 输入文本
        
        返回:
            异常类型列表
        """
        anomaly_types = []
        type_mapping = {
            '位置偏离': 'position_deviation',
            'position deviation': 'position_deviation',
            '传感器故障': 'sensor_malfunction',
            'sensor malfunction': 'sensor_malfunction',
            '机动异常': 'maneuver_anomaly',
            'maneuver anomaly': 'maneuver_anomaly',
            '电池异常': 'sensor_malfunction_battery',
            'battery anomaly': 'sensor_malfunction_battery',
            '温度异常': 'sensor_malfunction_temperature',
            'temperature anomaly': 'sensor_malfunction_temperature',
            '信号异常': 'sensor_malfunction_signal_strength',
            'signal anomaly': 'sensor_malfunction_signal_strength'
        }
        
        text_lower = text.lower()
        for key, value in type_mapping.items():
            if key.lower() in text_lower and value not in anomaly_types:
                anomaly_types.append(value)
        
        return anomaly_types
    
    def _extract_severity(self, text: str) -> Optional[str]:
        """
        提取严重程度
        
        参数:
            text: 输入文本
        
        返回:
            严重程度字符串或None
        """
        text_lower = text.lower()
        
        # 高严重度
        high_keywords = ['严重', '高', '重度', 'severe', 'high', 'critical']
        for keyword in high_keywords:
            if keyword in text_lower:
                return 'high'
        
        # 中等严重度
        medium_keywords = ['中等', '中度', '中', 'moderate', 'medium']
        for keyword in medium_keywords:
            if keyword in text_lower:
                return 'medium'
        
        # 低严重度
        low_keywords = ['轻微', '轻度', '低', 'minor', 'low', 'slight']
        for keyword in low_keywords:
            if keyword in text_lower:
                return 'low'
        
        return None
    
    def _add_default_parameters(self, params: Dict[str, Any], 
                                task_type: TaskType) -> Dict[str, Any]:
        """
        添加默认参数
        
        参数:
            params: 已提取的参数
            task_type: 任务类型
        
        返回:
            添加默认值后的参数字典
        """
        defaults = {
            TaskType.DATA_VISUALIZE: {
                'output_format': 'png',
                'show_legend': True
            },
            TaskType.ANOMALY_DETECT: {
                'threshold': 0.5,
                'window_size': 50
            },
            TaskType.GENERATE_REPORT: {
                'format': 'markdown',
                'include_charts': True
            }
        }
        
        if task_type in defaults:
            for key, value in defaults[task_type].items():
                if key not in params:
                    params[key] = value
        
        return params
    
    def get_task_description(self, task_type: TaskType) -> str:
        """
        获取任务类型的描述
        
        参数:
            task_type: 任务类型
        
        返回:
            任务描述字符串
        """
        descriptions = {
            TaskType.DATA_PREPROCESS: "数据预处理：加载、清洗和标准化无人机集群数据",
            TaskType.DATA_VISUALIZE: "数据可视化：生成无人机飞行数据和异常检测结果的图表",
            TaskType.DATA_REPLAY: "数据回放：回放历史飞行数据，支持指定时间范围和无人机",
            TaskType.ANOMALY_DETECT: "异常检测：使用Patch-GNN模型检测无人机集群中的异常行为",
            TaskType.ANOMALY_ANALYSIS: "异常分析：深入分析检测到的异常，提供详细信息",
            TaskType.IMPACT_ASSESS: "影响评估：基于RAG技术评估异常对无人机集群的影响",
            TaskType.RISK_EVALUATE: "风险评估：评估异常的风险等级和潜在后果",
            TaskType.GENERATE_REPORT: "报告生成：生成包含异常检测和影响评估结果的综合报告",
            TaskType.EXPORT_RESULTS: "结果导出：导出分析结果到指定格式的文件",
            TaskType.HELP: "帮助：显示系统使用说明和支持的功能",
            TaskType.STATUS: "状态：显示系统当前运行状态",
            TaskType.UNKNOWN: "未知任务：无法识别的指令"
        }
        return descriptions.get(task_type, "未知任务类型")
    
    def get_help_message(self) -> str:
        """
        获取帮助信息
        
        返回:
            帮助信息字符串
        """
        help_text = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                    无人机集群异常检测Agent系统 - 使用帮助                         ║
╚════════════════════════════════════════════════════════════════════════════════╝

【支持的功能】

1. 数据预处理与可视化
   示例指令：
   • "加载数据并进行预处理"
   • "可视化无人机0的飞行轨迹"
   • "显示所有无人机的传感器数据"

2. 数据回放
   示例指令：
   • "回放无人机3从1000到1500秒的数据"
   • "回放异常发生时的历史数据"

3. 异常检测
   示例指令：
   • "检测所有无人机的异常"
   • "对无人机0进行异常检测"
   • "分析无人机集群的异常行为"

4. 异常影响评估
   示例指令：
   • "评估检测到的异常影响"
   • "分析位置偏离异常的严重程度"
   • "进行风险评估"

5. 报告生成
   示例指令：
   • "生成综合报告"
   • "生成无人机0的异常分析报告"
   • "导出检测结果"

【参数说明】

• 无人机ID：可以使用 "无人机0"、"drone 1"、"#2" 等格式
• 时间范围：可以使用 "从1000到1500"、"1000-1500秒" 等格式
• 异常类型：支持 "位置偏离"、"传感器故障"、"机动异常" 等

【提示】

• 系统支持中文和英文混合输入
• 如果指令不够明确，系统会询问更多信息
• 输入 "状态" 可以查看当前系统状态

════════════════════════════════════════════════════════════════════════════════
        """
        return help_text


if __name__ == "__main__":
    # 测试自然语言处理器
    processor = NaturalLanguageProcessor()
    
    # 测试用例
    test_inputs = [
        "检测无人机0的异常",
        "生成综合报告",
        "可视化所有无人机的飞行数据",
        "评估位置偏离异常的影响",
        "回放无人机3从1000到1500秒的数据",
        "帮助",
        "detect anomaly in drone 2",
        "generate comprehensive report"
    ]
    
    print("="*60)
    print("自然语言处理器测试")
    print("="*60)
    
    for test_input in test_inputs:
        result = processor.parse(test_input)
        print(f"\n输入: {test_input}")
        print(f"任务类型: {result['task_type'].value}")
        print(f"置信度: {result['confidence']:.2f}")
        print(f"参数: {result['parameters']}")
    
    print("\n" + "="*60)
    print(processor.get_help_message())
