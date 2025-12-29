"""
报告生成Agent模块
Report Generation Agent Module

该模块实现了综合报告生成功能。
基于异常检测和影响评估结果，生成结构化的分析报告。

功能：
1. 生成结构化报告
2. 包含异常检测结果
3. 包含影响评估结果
4. 提供处置方案建议
5. 支持多种输出格式
"""

import os
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ReportGenerationAgent:
    """
    报告生成Agent
    
    负责生成综合分析报告，整合异常检测和影响评估的结果。
    
    功能：
    1. 生成Markdown格式报告
    2. 生成结构化JSON报告
    3. 包含执行摘要、详细分析和建议
    """
    
    def __init__(self, config: Dict):
        """
        初始化报告生成Agent
        
        参数:
            config: 配置字典，包含：
                - output_dir: 输出目录
        """
        self.config = config
        self.output_dir = config.get('output_dir', 'results')
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("[ReportGenerationAgent] 报告生成Agent已初始化")
    
    def generate(self, detection_results: Dict, assessment_results: Dict, 
                parameters: Dict) -> Dict:
        """
        生成综合报告
        
        参数:
            detection_results: 异常检测结果
            assessment_results: 影响评估结果
            parameters: 报告参数
        
        返回:
            报告生成结果
        """
        print("[ReportGenerationAgent] 开始生成综合报告...")
        
        try:
            # 获取报告参数
            report_format = parameters.get('format', 'markdown')
            include_charts = parameters.get('include_charts', True)
            
            # 生成报告内容
            report_content = self._build_report(detection_results, assessment_results)
            
            # 根据格式输出
            if report_format == 'markdown':
                output_file = self._save_markdown_report(report_content)
            elif report_format == 'json':
                output_file = self._save_json_report(report_content)
            else:
                output_file = self._save_markdown_report(report_content)
            
            return {
                'success': True,
                'message': '综合报告生成完成',
                'output_file': output_file,
                'summary': f'报告已保存至 {output_file}',
                'report_content': report_content,
                'details': {
                    '报告格式': report_format,
                    '输出路径': output_file,
                    '生成时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'报告生成失败: {str(e)}'
            }
    
    def _build_report(self, detection_results: Dict, 
                     assessment_results: Dict) -> Dict:
        """
        构建报告内容
        
        参数:
            detection_results: 检测结果
            assessment_results: 评估结果
        
        返回:
            报告内容字典
        """
        report = {
            'metadata': {
                'title': '无人机集群异常检测与影响评估综合报告',
                'title_en': 'UAV Swarm Anomaly Detection and Impact Assessment Report',
                'generated_at': datetime.now().isoformat(),
                'version': '1.0'
            },
            'executive_summary': self._build_executive_summary(
                detection_results, assessment_results
            ),
            'detection_analysis': self._build_detection_analysis(detection_results),
            'impact_assessment': self._build_impact_assessment(assessment_results),
            'recommendations': self._build_recommendations(
                detection_results, assessment_results
            ),
            'conclusion': self._build_conclusion(detection_results, assessment_results)
        }
        
        return report
    
    def _build_executive_summary(self, detection: Dict, assessment: Dict) -> Dict:
        """构建执行摘要"""
        
        # 提取关键数据
        anomaly_count = 0
        affected_drones = []
        
        if 'anomalies' in detection:
            for a in detection['anomalies']:
                if a.get('is_anomaly'):
                    anomaly_count += 1
                    affected_drones.append(a.get('drone_id'))
        
        overall_risk = 'unknown'
        if 'overall_assessment' in assessment:
            overall_risk = assessment['overall_assessment'].get('overall_risk_level', 'unknown')
        
        return {
            'overview': f'本报告对无人机集群进行了异常检测与影响评估分析。'
                       f'共检测发现 {anomaly_count} 个异常，'
                       f'涉及 {len(affected_drones)} 架无人机，'
                       f'整体风险等级为 {self._translate_risk(overall_risk)}。',
            'key_findings': [
                f'发现异常总数：{anomaly_count}',
                f'受影响无人机：{affected_drones if affected_drones else "无"}',
                f'整体风险等级：{self._translate_risk(overall_risk)}',
                f'需要立即关注：{"是" if overall_risk == "high" else "否"}'
            ],
            'anomaly_count': anomaly_count,
            'affected_drones': affected_drones,
            'overall_risk': overall_risk,
            'overall_risk_cn': self._translate_risk(overall_risk)
        }
    
    def _build_detection_analysis(self, detection: Dict) -> Dict:
        """构建异常检测分析部分"""
        
        analysis = {
            'title': '异常检测分析',
            'summary': detection.get('summary', ''),
            'anomalies': [],
            'statistics': {
                'total_drones_checked': 0,
                'anomalies_found': 0,
                'anomaly_types': {}
            }
        }
        
        if 'anomalies' in detection:
            for anomaly in detection['anomalies']:
                is_anomaly = anomaly.get('is_anomaly', False)
                
                analysis['statistics']['total_drones_checked'] += 1
                
                if is_anomaly:
                    analysis['statistics']['anomalies_found'] += 1
                    
                    anomaly_type = anomaly.get('anomaly_type', 'unknown')
                    analysis['statistics']['anomaly_types'][anomaly_type] = \
                        analysis['statistics']['anomaly_types'].get(anomaly_type, 0) + 1
                    
                    analysis['anomalies'].append({
                        'drone_id': anomaly.get('drone_id'),
                        'anomaly_type': anomaly_type,
                        'anomaly_type_cn': anomaly.get('anomaly_type_cn', '未知'),
                        'confidence': anomaly.get('confidence', 0),
                        'details': anomaly.get('details', '')
                    })
        
        return analysis
    
    def _build_impact_assessment(self, assessment: Dict) -> Dict:
        """构建影响评估部分"""
        
        impact = {
            'title': '异常影响评估',
            'summary': assessment.get('summary', ''),
            'overall': {},
            'detailed_assessments': []
        }
        
        if 'overall_assessment' in assessment:
            overall = assessment['overall_assessment']
            impact['overall'] = {
                'total_anomalies': overall.get('total_anomalies', 0),
                'severity_distribution': overall.get('severity_distribution', {}),
                'impact_pattern': overall.get('impact_pattern', ''),
                'impact_pattern_en': overall.get('impact_pattern_en', ''),
                'overall_risk_level': overall.get('overall_risk_level', ''),
                'overall_risk_cn': self._translate_risk(overall.get('overall_risk_level', '')),
                'propagation_risk': overall.get('propagation_risk', '')
            }
        
        if 'assessments' in assessment:
            for a in assessment['assessments']:
                impact['detailed_assessments'].append({
                    'drone_id': a.get('drone_id'),
                    'anomaly_type': a.get('anomaly_type'),
                    'anomaly_type_cn': a.get('anomaly_type_cn'),
                    'severity': a.get('severity_info', {}).get('title', ''),
                    'risk_level': a.get('risk_assessment', {}).get('risk_level_cn', ''),
                    'risk_score': a.get('risk_assessment', {}).get('risk_score', 0),
                    'potential_consequences': a.get('risk_assessment', {}).get(
                        'potential_consequences', []
                    ),
                    'similar_cases': [
                        case.get('title', '') for case in a.get('similar_cases', [])
                    ]
                })
        
        return impact
    
    def _build_recommendations(self, detection: Dict, assessment: Dict) -> Dict:
        """构建处置建议部分"""
        
        recommendations = {
            'title': '处置方案建议',
            'immediate_actions': [],
            'short_term_actions': [],
            'long_term_actions': [],
            'per_drone_recommendations': []
        }
        
        # 从整体评估中提取建议
        if 'overall_assessment' in assessment:
            overall_recs = assessment['overall_assessment'].get('recommendations', [])
            for rec in overall_recs:
                recommendations['immediate_actions'].append(rec)
        
        # 从各异常评估中提取建议
        if 'assessments' in assessment:
            for a in assessment['assessments']:
                drone_id = a.get('drone_id')
                drone_recs = []
                
                for rec in a.get('recommendations', []):
                    priority = rec.get('priority', 'normal')
                    action = rec.get('action', '')
                    details = rec.get('details', '')
                    
                    if priority == 'urgent':
                        recommendations['immediate_actions'].append(
                            f"[无人机{drone_id}] {action}: {details}"
                        )
                    elif priority == 'important':
                        recommendations['short_term_actions'].append(
                            f"[无人机{drone_id}] {action}: {details}"
                        )
                    
                    drone_recs.append({
                        'priority': rec.get('priority_cn', '一般'),
                        'action': action,
                        'details': details
                    })
                
                if drone_recs:
                    recommendations['per_drone_recommendations'].append({
                        'drone_id': drone_id,
                        'recommendations': drone_recs
                    })
        
        # 添加通用长期建议
        recommendations['long_term_actions'] = [
            '建立更完善的异常监测机制',
            '定期进行设备检查和维护',
            '更新和优化异常检测模型',
            '记录本次异常供后续分析'
        ]
        
        return recommendations
    
    def _build_conclusion(self, detection: Dict, assessment: Dict) -> Dict:
        """构建结论部分"""
        
        conclusion = {
            'title': '总结',
            'summary': '',
            'next_steps': []
        }
        
        # 构建总结
        anomaly_count = 0
        if 'anomalies' in detection:
            anomaly_count = sum(1 for a in detection['anomalies'] if a.get('is_anomaly'))
        
        risk_level = ''
        if 'overall_assessment' in assessment:
            risk_level = assessment['overall_assessment'].get('overall_risk_level', '')
        
        if anomaly_count == 0:
            conclusion['summary'] = '本次检测未发现明显异常，无人机集群运行状态良好。'
            conclusion['next_steps'] = [
                '继续保持日常监控',
                '定期进行设备检查'
            ]
        elif risk_level == 'high':
            conclusion['summary'] = (
                f'本次检测发现 {anomaly_count} 个异常，整体风险等级为高。'
                f'建议立即采取措施处理受影响的无人机，防止异常扩散或导致事故。'
            )
            conclusion['next_steps'] = [
                '立即执行紧急处置方案',
                '暂停非必要的任务执行',
                '对受影响无人机进行全面检查',
                '分析异常根本原因并采取预防措施'
            ]
        elif risk_level == 'medium':
            conclusion['summary'] = (
                f'本次检测发现 {anomaly_count} 个异常，整体风险等级为中等。'
                f'建议密切关注异常发展，准备干预措施。'
            )
            conclusion['next_steps'] = [
                '加强对异常无人机的监控',
                '准备应急干预方案',
                '在任务结束后进行详细检查',
                '评估是否需要调整任务计划'
            ]
        else:
            conclusion['summary'] = (
                f'本次检测发现 {anomaly_count} 个轻微异常，整体风险等级为低。'
                f'异常在可控范围内，可继续正常任务执行。'
            )
            conclusion['next_steps'] = [
                '继续监控异常状态',
                '记录异常情况供后续分析',
                '在任务结束后进行常规检查'
            ]
        
        return conclusion
    
    def _translate_risk(self, risk: str) -> str:
        """翻译风险等级"""
        translations = {
            'low': '低',
            'medium': '中等',
            'high': '高',
            'critical': '严重',
            'unknown': '未知'
        }
        return translations.get(risk, risk)
    
    def _save_markdown_report(self, report: Dict) -> str:
        """
        保存Markdown格式报告
        
        参数:
            report: 报告内容
        
        返回:
            输出文件路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.output_dir, f'report_{timestamp}.md')
        
        md_content = self._format_markdown(report)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"[ReportGenerationAgent] Markdown报告已保存: {output_file}")
        return output_file
    
    def _format_markdown(self, report: Dict) -> str:
        """
        格式化为Markdown
        
        参数:
            report: 报告内容
        
        返回:
            Markdown格式字符串
        """
        lines = []
        
        # 标题
        meta = report.get('metadata', {})
        lines.append(f"# {meta.get('title', '异常检测报告')}")
        lines.append(f"**{meta.get('title_en', '')}**")
        lines.append("")
        lines.append(f"生成时间：{meta.get('generated_at', '')}")
        lines.append(f"版本：{meta.get('version', '1.0')}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # 执行摘要
        exec_summary = report.get('executive_summary', {})
        lines.append("## 1. 执行摘要")
        lines.append("")
        lines.append(exec_summary.get('overview', ''))
        lines.append("")
        lines.append("### 关键发现")
        lines.append("")
        for finding in exec_summary.get('key_findings', []):
            lines.append(f"- {finding}")
        lines.append("")
        
        # 异常检测分析
        detection = report.get('detection_analysis', {})
        lines.append("## 2. 异常检测分析")
        lines.append("")
        lines.append(f"**摘要**：{detection.get('summary', '')}")
        lines.append("")
        
        stats = detection.get('statistics', {})
        lines.append("### 2.1 检测统计")
        lines.append("")
        lines.append(f"- 检测无人机数：{stats.get('total_drones_checked', 0)}")
        lines.append(f"- 发现异常数：{stats.get('anomalies_found', 0)}")
        lines.append("")
        
        anomaly_types = stats.get('anomaly_types', {})
        if anomaly_types:
            lines.append("异常类型分布：")
            lines.append("")
            lines.append("| 异常类型 | 数量 |")
            lines.append("|---------|------|")
            for t, c in anomaly_types.items():
                lines.append(f"| {t} | {c} |")
            lines.append("")
        
        anomalies = detection.get('anomalies', [])
        if anomalies:
            lines.append("### 2.2 异常详情")
            lines.append("")
            lines.append("| 无人机ID | 异常类型 | 置信度 | 详情 |")
            lines.append("|---------|---------|--------|------|")
            for a in anomalies:
                lines.append(
                    f"| {a.get('drone_id', '')} | {a.get('anomaly_type_cn', '')} | "
                    f"{a.get('confidence', 0):.2%} | {a.get('details', '')} |"
                )
            lines.append("")
        
        # 影响评估
        impact = report.get('impact_assessment', {})
        lines.append("## 3. 异常影响评估")
        lines.append("")
        
        overall = impact.get('overall', {})
        if overall:
            lines.append("### 3.1 整体评估")
            lines.append("")
            lines.append(f"- 总异常数：{overall.get('total_anomalies', 0)}")
            lines.append(f"- 影响模式：{overall.get('impact_pattern', '')}")
            lines.append(f"- 传播风险：{overall.get('propagation_risk', '')}")
            lines.append(f"- **整体风险等级：{overall.get('overall_risk_cn', '')}**")
            lines.append("")
            
            severity_dist = overall.get('severity_distribution', {})
            if severity_dist:
                lines.append("严重程度分布：")
                lines.append("")
                lines.append("| 严重程度 | 数量 |")
                lines.append("|---------|------|")
                for s, c in severity_dist.items():
                    lines.append(f"| {self._translate_risk(s)} | {c} |")
                lines.append("")
        
        detailed = impact.get('detailed_assessments', [])
        if detailed:
            lines.append("### 3.2 详细影响分析")
            lines.append("")
            for d in detailed:
                lines.append(f"#### 无人机 {d.get('drone_id', '')}")
                lines.append("")
                lines.append(f"- 异常类型：{d.get('anomaly_type_cn', '')}")
                lines.append(f"- 严重程度：{d.get('severity', '')}")
                lines.append(f"- 风险等级：{d.get('risk_level', '')}")
                lines.append(f"- 风险分数：{d.get('risk_score', 0):.2f}")
                lines.append("")
                
                consequences = d.get('potential_consequences', [])
                if consequences:
                    lines.append("潜在后果：")
                    for c in consequences:
                        lines.append(f"  - {c}")
                    lines.append("")
                
                cases = d.get('similar_cases', [])
                if cases:
                    lines.append(f"相似案例：{', '.join(cases)}")
                    lines.append("")
        
        # 处置建议
        recs = report.get('recommendations', {})
        lines.append("## 4. 处置方案建议")
        lines.append("")
        
        immediate = recs.get('immediate_actions', [])
        if immediate:
            lines.append("### 4.1 立即行动")
            lines.append("")
            for r in immediate:
                lines.append(f"- ⚠️ {r}")
            lines.append("")
        
        short_term = recs.get('short_term_actions', [])
        if short_term:
            lines.append("### 4.2 短期措施")
            lines.append("")
            for r in short_term:
                lines.append(f"- {r}")
            lines.append("")
        
        long_term = recs.get('long_term_actions', [])
        if long_term:
            lines.append("### 4.3 长期建议")
            lines.append("")
            for r in long_term:
                lines.append(f"- {r}")
            lines.append("")
        
        per_drone = recs.get('per_drone_recommendations', [])
        if per_drone:
            lines.append("### 4.4 各无人机具体建议")
            lines.append("")
            for pd in per_drone:
                lines.append(f"**无人机 {pd.get('drone_id', '')}**：")
                for rec in pd.get('recommendations', []):
                    lines.append(f"  - [{rec.get('priority', '')}] {rec.get('action', '')}")
                lines.append("")
        
        # 结论
        conclusion = report.get('conclusion', {})
        lines.append("## 5. 总结")
        lines.append("")
        lines.append(conclusion.get('summary', ''))
        lines.append("")
        lines.append("### 后续步骤")
        lines.append("")
        for step in conclusion.get('next_steps', []):
            lines.append(f"1. {step}")
        lines.append("")
        
        # 页脚
        lines.append("---")
        lines.append("")
        lines.append("*本报告由无人机集群异常检测Agent系统自动生成*")
        
        return '\n'.join(lines)
    
    def _save_json_report(self, report: Dict) -> str:
        """
        保存JSON格式报告
        
        参数:
            report: 报告内容
        
        返回:
            输出文件路径
        """
        import json
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.output_dir, f'report_{timestamp}.json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"[ReportGenerationAgent] JSON报告已保存: {output_file}")
        return output_file


if __name__ == "__main__":
    # 测试报告生成Agent
    config = {
        'output_dir': 'results'
    }
    
    print("="*60)
    print("测试报告生成Agent")
    print("="*60)
    
    agent = ReportGenerationAgent(config)
    
    # 模拟检测结果
    detection_results = {
        'success': True,
        'summary': '检测了10架无人机，发现2个异常',
        'anomalies': [
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
    }
    
    # 模拟评估结果
    assessment_results = {
        'success': True,
        'summary': '评估了2个异常，整体风险等级：中等',
        'overall_assessment': {
            'total_anomalies': 2,
            'severity_distribution': {'low': 0, 'medium': 2, 'high': 0},
            'impact_pattern': '局部传播模式',
            'impact_pattern_en': 'Local Propagation',
            'overall_risk_level': 'medium',
            'propagation_risk': 'medium',
            'recommendations': [
                '建议加强监控频率，准备应急预案',
                '评估任务是否可以在当前状态下安全完成'
            ]
        },
        'assessments': [
            {
                'drone_id': 0,
                'anomaly_type': 'position_deviation',
                'anomaly_type_cn': '位置偏离异常',
                'severity_info': {'title': '中等严重度'},
                'risk_assessment': {
                    'risk_level_cn': '中等风险',
                    'risk_score': 0.51,
                    'potential_consequences': ['可能偏离预定航线', '存在碰撞风险']
                },
                'recommendations': [
                    {'priority_cn': '重要', 'action': '校验定位系统', 'details': '检查GPS信号'}
                ],
                'similar_cases': [{'title': 'GPS干扰导致的位置偏离'}]
            },
            {
                'drone_id': 1,
                'anomaly_type': 'sensor_malfunction_battery',
                'anomaly_type_cn': '电池传感器异常',
                'severity_info': {'title': '中等严重度'},
                'risk_assessment': {
                    'risk_level_cn': '中等风险',
                    'risk_score': 0.43,
                    'potential_consequences': ['续航时间可能不足', '可能需要提前返航']
                },
                'recommendations': [
                    {'priority_cn': '重要', 'action': '规划返航', 'details': '评估剩余电量'}
                ],
                'similar_cases': [{'title': '电池老化导致电量异常下降'}]
            }
        ]
    }
    
    # 生成报告
    result = agent.generate(detection_results, assessment_results, {'format': 'markdown'})
    
    print(f"\n报告生成结果:")
    print(f"成功: {result.get('success')}")
    print(f"输出文件: {result.get('output_file')}")
