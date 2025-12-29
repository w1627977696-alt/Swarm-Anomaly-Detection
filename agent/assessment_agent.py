"""
异常影响评估Agent模块
Impact Assessment Agent Module

该模块实现了基于RAG技术的无人机集群异常影响评估功能。
通过检索知识库中的相关知识，结合异常检测结果，进行综合影响评估。

功能：
1. 基于RAG的影响评估
2. 严重程度分级
3. 风险分析
4. 处置建议生成
"""

import os
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_base.knowledge_base import DroneAnomalyKnowledgeBase


class ImpactAssessmentAgent:
    """
    异常影响评估Agent
    
    基于RAG技术的影响评估Agent，通过检索知识库中的相关知识，
    结合异常检测结果进行综合评估。
    
    功能：
    1. 异常严重程度评估
    2. 影响范围分析
    3. 风险等级判定
    4. 处置方案建议
    """
    
    def __init__(self, config: Dict):
        """
        初始化影响评估Agent
        
        参数:
            config: 配置字典，包含：
                - knowledge_base_path: 知识库路径
        """
        self.config = config
        kb_path = config.get('knowledge_base_path', 'knowledge_base')
        
        # 初始化知识库
        self.knowledge_base = DroneAnomalyKnowledgeBase(kb_path)
        
        # 评估历史
        self.assessment_history = []
        
        print("[ImpactAssessmentAgent] 影响评估Agent已初始化")
        print(f"[ImpactAssessmentAgent] 知识库路径: {kb_path}")
    
    def assess(self, anomalies: List[Dict], parameters: Dict) -> Dict:
        """
        执行影响评估
        
        参数:
            anomalies: 异常检测结果列表
            parameters: 评估参数
        
        返回:
            评估结果字典
        """
        print("[ImpactAssessmentAgent] 开始异常影响评估...")
        
        if not anomalies:
            return {
                'success': True,
                'message': '未提供异常数据，无法进行评估',
                'assessments': []
            }
        
        try:
            assessments = []
            overall_assessment = {
                'total_anomalies': 0,
                'severity_distribution': {'low': 0, 'medium': 0, 'high': 0},
                'affected_drones': [],
                'overall_risk_level': 'low',
                'recommendations': []
            }
            
            # 对每个异常进行评估
            for anomaly in anomalies:
                if anomaly.get('is_anomaly', False):
                    assessment = self._assess_single_anomaly(anomaly)
                    assessments.append(assessment)
                    
                    # 更新总体统计
                    overall_assessment['total_anomalies'] += 1
                    overall_assessment['affected_drones'].append(anomaly.get('drone_id'))
                    
                    severity = assessment.get('severity_level', 'medium')
                    overall_assessment['severity_distribution'][severity] = \
                        overall_assessment['severity_distribution'].get(severity, 0) + 1
            
            # 计算整体风险等级
            overall_assessment['overall_risk_level'] = \
                self._calculate_overall_risk(overall_assessment)
            
            # 生成总体建议
            overall_assessment['recommendations'] = \
                self._generate_overall_recommendations(overall_assessment, assessments)
            
            # 获取影响模式
            affected_count = len(overall_assessment['affected_drones'])
            impact_pattern = self.knowledge_base.get_impact_pattern(affected_count)
            overall_assessment['impact_pattern'] = impact_pattern.get('title', '未知')
            overall_assessment['impact_pattern_en'] = impact_pattern.get('title_en', 'Unknown')
            overall_assessment['propagation_risk'] = impact_pattern.get('propagation_risk', 'unknown')
            
            # 保存评估历史
            self.assessment_history.append({
                'timestamp': datetime.now().isoformat(),
                'assessments': assessments,
                'overall': overall_assessment
            })
            
            return {
                'success': True,
                'message': '影响评估完成',
                'summary': f'评估了 {len(assessments)} 个异常，'
                          f'总体风险等级: {overall_assessment["overall_risk_level"]}',
                'assessments': assessments,
                'overall_assessment': overall_assessment,
                'details': {
                    '评估异常数': len(assessments),
                    '受影响无人机': overall_assessment['affected_drones'],
                    '影响模式': overall_assessment['impact_pattern'],
                    '整体风险': overall_assessment['overall_risk_level']
                }
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'影响评估失败: {str(e)}'
            }
    
    def _assess_single_anomaly(self, anomaly: Dict) -> Dict:
        """
        评估单个异常的影响
        
        参数:
            anomaly: 异常信息字典
        
        返回:
            单个异常的评估结果
        """
        drone_id = anomaly.get('drone_id', -1)
        anomaly_type = anomaly.get('anomaly_type', 'unknown')
        confidence = anomaly.get('confidence', 0.5)
        
        # 使用RAG检索相关知识
        # 1. 获取异常类型信息
        type_info = self.knowledge_base.get_anomaly_info(anomaly_type)
        
        # 2. 获取严重程度
        severity_info = self.knowledge_base.get_severity_level(confidence)
        severity_level = severity_info.get('level', 'medium')
        
        # 3. 检索相关历史案例
        similar_cases = self.knowledge_base.get_similar_cases(anomaly_type, top_k=2)
        
        # 4. 综合搜索相关知识
        query = f"{anomaly_type} {anomaly.get('anomaly_type_cn', '')}"
        related_knowledge = self.knowledge_base.search_all(query, top_k=2)
        
        # 构建评估结果
        assessment = {
            'drone_id': drone_id,
            'anomaly_type': anomaly_type,
            'anomaly_type_cn': anomaly.get('anomaly_type_cn', '未知'),
            'confidence': confidence,
            
            # 严重程度评估
            'severity_level': severity_level,
            'severity_info': {
                'title': severity_info.get('title', ''),
                'description': severity_info.get('description', ''),
                'urgency': severity_info.get('urgency', '中'),
                'recommended_action': severity_info.get('recommended_action', '')
            },
            
            # 异常类型详情
            'type_info': {
                'description': type_info.get('description', '') if type_info else '',
                'characteristics': type_info.get('characteristics', []) if type_info else [],
                'typical_causes': type_info.get('typical_causes', []) if type_info else []
            },
            
            # 相关历史案例
            'similar_cases': [
                {
                    'case_id': case.get('case_id', ''),
                    'title': case.get('title', ''),
                    'resolution': case.get('resolution', []),
                    'lessons_learned': case.get('lessons_learned', [])
                }
                for case in similar_cases
            ],
            
            # 风险评估
            'risk_assessment': self._assess_risk(anomaly, type_info, severity_info),
            
            # 处置建议
            'recommendations': self._generate_recommendations(
                anomaly, type_info, severity_info, similar_cases
            )
        }
        
        return assessment
    
    def _assess_risk(self, anomaly: Dict, type_info: Optional[Dict], 
                     severity_info: Dict) -> Dict:
        """
        评估风险
        
        参数:
            anomaly: 异常信息
            type_info: 异常类型信息
            severity_info: 严重程度信息
        
        返回:
            风险评估结果
        """
        confidence = anomaly.get('confidence', 0.5)
        severity_level = severity_info.get('level', 'medium')
        
        # 计算风险分数
        severity_weights = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
        risk_score = confidence * severity_weights.get(severity_level, 0.6)
        
        # 确定风险等级
        if risk_score < 0.3:
            risk_level = 'low'
            risk_level_cn = '低风险'
        elif risk_score < 0.6:
            risk_level = 'medium'
            risk_level_cn = '中等风险'
        else:
            risk_level = 'high'
            risk_level_cn = '高风险'
        
        # 确定潜在后果
        potential_consequences = []
        if type_info:
            anomaly_type = type_info.get('type', '')
            if 'position' in anomaly_type:
                potential_consequences = [
                    '可能偏离预定航线',
                    '存在碰撞风险',
                    '任务执行可能受影响'
                ]
            elif 'sensor' in anomaly_type:
                if 'battery' in anomaly_type:
                    potential_consequences = [
                        '续航时间可能不足',
                        '可能需要提前返航',
                        '任务可能无法完成'
                    ]
                elif 'temperature' in anomaly_type:
                    potential_consequences = [
                        '设备可能过热损坏',
                        '性能可能下降',
                        '安全风险增加'
                    ]
                elif 'signal' in anomaly_type:
                    potential_consequences = [
                        '通信可能中断',
                        '可能失去控制',
                        '协调能力下降'
                    ]
            elif 'maneuver' in anomaly_type:
                potential_consequences = [
                    '飞行稳定性下降',
                    '碰撞风险增加',
                    '任务精度受影响'
                ]
        
        return {
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'risk_level_cn': risk_level_cn,
            'potential_consequences': potential_consequences
        }
    
    def _generate_recommendations(self, anomaly: Dict, type_info: Optional[Dict],
                                   severity_info: Dict, similar_cases: List[Dict]) -> List[Dict]:
        """
        生成处置建议
        
        参数:
            anomaly: 异常信息
            type_info: 异常类型信息
            severity_info: 严重程度信息
            similar_cases: 相似历史案例
        
        返回:
            建议列表
        """
        recommendations = []
        severity_level = severity_info.get('level', 'medium')
        
        # 基于严重程度的通用建议
        if severity_level == 'high':
            recommendations.append({
                'priority': 'urgent',
                'priority_cn': '紧急',
                'action': '立即采取干预措施',
                'action_en': 'Take immediate intervention',
                'details': '考虑将受影响的无人机从任务中撤出或执行紧急降落'
            })
        elif severity_level == 'medium':
            recommendations.append({
                'priority': 'important',
                'priority_cn': '重要',
                'action': '加强监控并准备干预',
                'action_en': 'Increase monitoring and prepare intervention',
                'details': '密切关注异常发展，准备备用方案'
            })
        else:
            recommendations.append({
                'priority': 'normal',
                'priority_cn': '一般',
                'action': '持续监控',
                'action_en': 'Continue monitoring',
                'details': '记录异常情况，观察是否自行恢复'
            })
        
        # 基于异常类型的特定建议
        if type_info:
            anomaly_type = type_info.get('type', '')
            
            if 'position' in anomaly_type:
                recommendations.append({
                    'priority': 'important',
                    'priority_cn': '重要',
                    'action': '校验定位系统',
                    'action_en': 'Verify positioning system',
                    'details': '检查GPS信号质量，考虑启用备用定位方式'
                })
            elif 'battery' in anomaly_type:
                recommendations.append({
                    'priority': 'important',
                    'priority_cn': '重要',
                    'action': '规划返航',
                    'action_en': 'Plan return flight',
                    'details': '评估剩余电量，必要时提前返航'
                })
            elif 'temperature' in anomaly_type:
                recommendations.append({
                    'priority': 'important',
                    'priority_cn': '重要',
                    'action': '降低负载',
                    'action_en': 'Reduce load',
                    'details': '降低飞行速度或任务强度，让系统散热'
                })
            elif 'signal' in anomaly_type:
                recommendations.append({
                    'priority': 'urgent',
                    'priority_cn': '紧急',
                    'action': '切换通信频段',
                    'action_en': 'Switch communication channel',
                    'details': '尝试切换到备用通信频段，或调整天线方向'
                })
            elif 'maneuver' in anomaly_type:
                recommendations.append({
                    'priority': 'urgent',
                    'priority_cn': '紧急',
                    'action': '检查动力系统',
                    'action_en': 'Check propulsion system',
                    'details': '检查电机和螺旋桨状态，考虑降落检修'
                })
        
        # 从历史案例中提取建议
        for case in similar_cases[:1]:  # 只取最相关的案例
            resolutions = case.get('resolution', [])
            if resolutions:
                recommendations.append({
                    'priority': 'reference',
                    'priority_cn': '参考',
                    'action': f'参考案例 {case.get("case_id", "")}',
                    'action_en': f'Refer to case {case.get("case_id", "")}',
                    'details': resolutions[0] if resolutions else ''
                })
        
        return recommendations
    
    def _calculate_overall_risk(self, overall: Dict) -> str:
        """
        计算整体风险等级
        
        参数:
            overall: 总体评估数据
        
        返回:
            整体风险等级
        """
        dist = overall.get('severity_distribution', {})
        
        high_count = dist.get('high', 0)
        medium_count = dist.get('medium', 0)
        total = overall.get('total_anomalies', 0)
        
        if high_count >= 2 or (high_count >= 1 and total >= 3):
            return 'high'
        elif high_count >= 1 or medium_count >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _generate_overall_recommendations(self, overall: Dict, 
                                          assessments: List[Dict]) -> List[str]:
        """
        生成整体建议
        
        参数:
            overall: 总体评估数据
            assessments: 各异常的评估结果
        
        返回:
            建议列表
        """
        recommendations = []
        risk_level = overall.get('overall_risk_level', 'medium')
        affected_count = len(overall.get('affected_drones', []))
        
        # 基于整体风险的建议
        if risk_level == 'high':
            recommendations.append("建议暂停任务执行，对受影响的无人机进行全面检查")
            recommendations.append("考虑重新规划任务，避开异常无人机")
        elif risk_level == 'medium':
            recommendations.append("建议加强监控频率，准备应急预案")
            recommendations.append("评估任务是否可以在当前状态下安全完成")
        else:
            recommendations.append("当前异常风险可控，继续监控即可")
        
        # 基于受影响范围的建议
        if affected_count >= 3:
            recommendations.append("多架无人机受影响，建议检查是否存在共同原因")
            recommendations.append("考虑是否存在环境因素（如电磁干扰、恶劣天气）")
        
        # 添加后续行动建议
        recommendations.append("建议在任务结束后进行详细的异常分析和设备检查")
        
        return recommendations
    
    def get_assessment_summary(self) -> Dict:
        """
        获取评估历史摘要
        
        返回:
            评估历史摘要
        """
        if not self.assessment_history:
            return {'message': '暂无评估历史'}
        
        return {
            'total_assessments': len(self.assessment_history),
            'latest_assessment': self.assessment_history[-1] if self.assessment_history else None
        }


if __name__ == "__main__":
    # 测试影响评估Agent
    config = {
        'knowledge_base_path': 'knowledge_base'
    }
    
    print("="*60)
    print("测试异常影响评估Agent")
    print("="*60)
    
    agent = ImpactAssessmentAgent(config)
    
    # 模拟异常检测结果
    test_anomalies = [
        {
            'drone_id': 0,
            'is_anomaly': True,
            'anomaly_type': 'position_deviation',
            'anomaly_type_cn': '位置偏离异常',
            'confidence': 0.85,
            'details': '位置偏离约25米'
        },
        {
            'drone_id': 1,
            'is_anomaly': True,
            'anomaly_type': 'sensor_malfunction_battery',
            'anomaly_type_cn': '电池传感器异常',
            'confidence': 0.72,
            'details': '电量下降约30%'
        },
        {
            'drone_id': 2,
            'is_anomaly': False,
            'anomaly_type': 'normal',
            'anomaly_type_cn': '正常',
            'confidence': 0.1
        }
    ]
    
    # 执行评估
    result = agent.assess(test_anomalies, {})
    
    print(f"\n评估结果:")
    print(f"成功: {result.get('success')}")
    print(f"摘要: {result.get('summary')}")
    
    if 'overall_assessment' in result:
        overall = result['overall_assessment']
        print(f"\n整体评估:")
        print(f"  总异常数: {overall.get('total_anomalies')}")
        print(f"  影响模式: {overall.get('impact_pattern')}")
        print(f"  整体风险: {overall.get('overall_risk_level')}")
        
        print(f"\n整体建议:")
        for rec in overall.get('recommendations', []):
            print(f"  • {rec}")
    
    if 'assessments' in result:
        print(f"\n详细评估:")
        for assess in result['assessments']:
            print(f"\n  无人机{assess['drone_id']}:")
            print(f"    异常类型: {assess['anomaly_type_cn']}")
            print(f"    严重程度: {assess['severity_info']['title']}")
            print(f"    风险等级: {assess['risk_assessment']['risk_level_cn']}")
            
            print(f"    处置建议:")
            for rec in assess.get('recommendations', [])[:2]:
                print(f"      [{rec['priority_cn']}] {rec['action']}")
