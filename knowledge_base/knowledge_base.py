"""
无人机集群异常知识库
UAV Swarm Anomaly Knowledge Base

本模块实现了基于RAG（检索增强生成）技术的知识库系统。
包含无人机异常类型、严重程度、影响模式和历史案例等知识。

功能：
1. 知识库管理（添加、更新、检索）
2. 向量化存储和相似度搜索
3. 知识检索和推理
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


class KnowledgeBase:
    """
    知识库基类
    
    提供知识存储和检索的基础功能
    """
    
    def __init__(self, name: str, save_path: Optional[str] = None):
        """
        初始化知识库
        
        参数:
            name: 知识库名称
            save_path: 保存路径
        """
        self.name = name
        self.save_path = save_path
        self.knowledge = []
        self.embeddings = []
        
        # 加载已有知识
        if save_path and os.path.exists(save_path):
            self.load()
    
    def add_knowledge(self, item: Dict):
        """
        添加知识条目
        
        参数:
            item: 知识条目字典
        """
        item['id'] = len(self.knowledge)
        item['created_at'] = datetime.now().isoformat()
        self.knowledge.append(item)
        
        # 计算简单的文本嵌入（基于关键词）
        embedding = self._compute_embedding(item)
        self.embeddings.append(embedding)
    
    def _compute_embedding(self, item: Dict) -> np.ndarray:
        """
        计算知识条目的嵌入向量
        
        使用简单的词袋模型，实际应用中可替换为预训练模型
        
        参数:
            item: 知识条目
        
        返回:
            嵌入向量
        """
        # 提取文本内容
        text_fields = ['title', 'description', 'keywords', 'type', 'category']
        text = ' '.join(str(item.get(f, '')) for f in text_fields)
        
        # 简单的词袋表示
        keywords = [
            '位置', '偏离', 'position', 'deviation',
            '传感器', '故障', 'sensor', 'malfunction',
            '机动', '速度', '加速度', 'maneuver', 'velocity',
            '电池', 'battery', '温度', 'temperature', '信号', 'signal',
            '严重', '中等', '轻微', 'severe', 'moderate', 'minor',
            '碰撞', '失控', '通信', '干扰', 'collision', 'communication',
            'GPS', '导航', '控制', '电机'
        ]
        
        embedding = np.zeros(len(keywords))
        for i, kw in enumerate(keywords):
            if kw.lower() in text.lower():
                embedding[i] = 1.0
        
        # 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        搜索相关知识
        
        参数:
            query: 查询字符串
            top_k: 返回结果数量
        
        返回:
            相关知识条目列表
        """
        if not self.knowledge:
            return []
        
        # 计算查询嵌入
        query_item = {'description': query, 'keywords': query}
        query_embedding = self._compute_embedding(query_item)
        
        # 计算相似度
        similarities = []
        for i, emb in enumerate(self.embeddings):
            sim = np.dot(query_embedding, emb)
            similarities.append((i, sim))
        
        # 排序并返回top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, sim in similarities[:top_k]:
            item = self.knowledge[idx].copy()
            item['similarity'] = float(sim)
            results.append(item)
        
        return results
    
    def get_by_type(self, knowledge_type: str) -> List[Dict]:
        """
        按类型获取知识
        
        参数:
            knowledge_type: 知识类型
        
        返回:
            匹配的知识条目列表
        """
        return [k for k in self.knowledge if k.get('type') == knowledge_type]
    
    def save(self):
        """保存知识库到文件"""
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'name': self.name,
                    'knowledge': self.knowledge,
                    'embeddings': [emb.tolist() for emb in self.embeddings]
                }, f, ensure_ascii=False, indent=2)
    
    def load(self):
        """从文件加载知识库"""
        if self.save_path and os.path.exists(self.save_path):
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.knowledge = data.get('knowledge', [])
                self.embeddings = [np.array(e) for e in data.get('embeddings', [])]


class AnomalyTypeKnowledge(KnowledgeBase):
    """
    异常类型知识库
    
    存储无人机集群中各种异常类型的定义、特征和识别方法
    """
    
    def __init__(self, save_path: Optional[str] = None):
        super().__init__('anomaly_types', save_path)
        
        # 初始化默认知识
        if not self.knowledge:
            self._initialize_default_knowledge()
    
    def _initialize_default_knowledge(self):
        """初始化默认异常类型知识"""
        
        # 位置偏离异常
        self.add_knowledge({
            'type': 'position_deviation',
            'title': '位置偏离异常',
            'title_en': 'Position Deviation Anomaly',
            'description': '无人机偏离预定飞行轨迹，可能表现为渐进式偏离或突然偏离',
            'keywords': '位置,偏离,轨迹,GPS,导航',
            'characteristics': [
                'x, y, z坐标出现异常变化',
                '偏离幅度可达20-30米',
                '通常呈渐进式发展'
            ],
            'detection_features': ['position', 'trajectory', 'deviation_distance'],
            'typical_causes': [
                'GPS信号干扰或故障',
                '强风或恶劣天气',
                '导航系统异常',
                '飞行控制器故障'
            ],
            'severity_range': 'medium_to_high',
            'category': 'navigation'
        })
        
        # 传感器故障 - 电池异常
        self.add_knowledge({
            'type': 'sensor_malfunction_battery',
            'title': '电池传感器异常',
            'title_en': 'Battery Sensor Malfunction',
            'description': '电池电量读数异常，可能表现为突然下降或读数不稳定',
            'keywords': '电池,传感器,电量,下降,故障',
            'characteristics': [
                '电量突然大幅下降（约30%）',
                '读数不稳定或跳动',
                '与实际电量不符'
            ],
            'detection_features': ['battery', 'voltage', 'current'],
            'typical_causes': [
                '电池老化或损坏',
                '电量传感器故障',
                '电池连接不良',
                '极端温度影响'
            ],
            'severity_range': 'medium',
            'category': 'sensor'
        })
        
        # 传感器故障 - 温度异常
        self.add_knowledge({
            'type': 'sensor_malfunction_temperature',
            'title': '温度传感器异常',
            'title_en': 'Temperature Sensor Malfunction',
            'description': '温度读数异常升高，可能表示散热问题或传感器故障',
            'keywords': '温度,传感器,过热,散热,故障',
            'characteristics': [
                '温度异常升高15-25摄氏度',
                '持续性高温报警',
                '与环境温度不符'
            ],
            'detection_features': ['temperature', 'heat'],
            'typical_causes': [
                '散热系统故障',
                '电机过载',
                '温度传感器故障',
                '环境温度过高'
            ],
            'severity_range': 'medium_to_high',
            'category': 'sensor'
        })
        
        # 传感器故障 - 信号异常
        self.add_knowledge({
            'type': 'sensor_malfunction_signal',
            'title': '信号传感器异常',
            'title_en': 'Signal Sensor Malfunction',
            'description': '通信信号强度异常减弱，可能导致失联风险',
            'keywords': '信号,通信,干扰,失联,天线',
            'characteristics': [
                '信号强度降至-90dBm左右',
                '信号波动剧烈',
                '通信延迟增加'
            ],
            'detection_features': ['signal_strength', 'communication'],
            'typical_causes': [
                '通信干扰',
                '天线故障',
                '距离超出范围',
                '电磁环境复杂'
            ],
            'severity_range': 'high',
            'category': 'communication'
        })
        
        # 机动异常
        self.add_knowledge({
            'type': 'maneuver_anomaly',
            'title': '机动异常',
            'title_en': 'Maneuver Anomaly',
            'description': '速度和加速度出现异常变化，表现为不正常的振荡或突变',
            'keywords': '机动,速度,加速度,振荡,控制',
            'characteristics': [
                '速度出现高频振荡',
                '加速度异常波动',
                '姿态角不稳定',
                '振幅为正常值的3-4倍'
            ],
            'detection_features': ['velocity', 'acceleration', 'attitude'],
            'typical_causes': [
                '飞行控制算法错误',
                '电机故障或不平衡',
                '传感器数据干扰',
                '结构损伤'
            ],
            'severity_range': 'medium_to_high',
            'category': 'control'
        })


class SeverityKnowledge(KnowledgeBase):
    """
    严重程度知识库
    
    存储异常严重程度的评估标准和分级方法
    """
    
    def __init__(self, save_path: Optional[str] = None):
        super().__init__('severity', save_path)
        
        if not self.knowledge:
            self._initialize_default_knowledge()
    
    def _initialize_default_knowledge(self):
        """初始化默认严重程度知识"""
        
        # 低严重度
        self.add_knowledge({
            'type': 'severity_level',
            'level': 'low',
            'title': '低严重度',
            'title_en': 'Low Severity',
            'description': '异常对任务执行影响较小，可通过自动调整恢复',
            'keywords': '轻微,低,可恢复,自动,调整',
            'score_range': [0.0, 0.3],
            'characteristics': [
                '单一传感器轻微偏差',
                '可自动校正',
                '不影响任务继续执行',
                '无安全风险'
            ],
            'recommended_action': '继续监控，记录日志',
            'urgency': '低',
            'category': 'severity'
        })
        
        # 中等严重度
        self.add_knowledge({
            'type': 'severity_level',
            'level': 'medium',
            'title': '中等严重度',
            'title_en': 'Medium Severity',
            'description': '异常需要人工关注，可能影响任务效率',
            'keywords': '中等,关注,效率,干预,监控',
            'score_range': [0.3, 0.7],
            'characteristics': [
                '多个参数出现异常',
                '可能影响任务效率',
                '需要人工监控',
                '存在潜在安全风险'
            ],
            'recommended_action': '加强监控，准备干预措施',
            'urgency': '中',
            'category': 'severity'
        })
        
        # 高严重度
        self.add_knowledge({
            'type': 'severity_level',
            'level': 'high',
            'title': '高严重度',
            'title_en': 'High Severity',
            'description': '异常严重影响任务执行，需要立即干预',
            'keywords': '严重,紧急,立即,干预,危险',
            'score_range': [0.7, 1.0],
            'characteristics': [
                '关键系统故障',
                '严重偏离预定轨迹',
                '可能导致碰撞或坠落',
                '需要紧急干预'
            ],
            'recommended_action': '立即采取紧急措施，可能需要召回无人机',
            'urgency': '高',
            'category': 'severity'
        })


class ImpactPatternKnowledge(KnowledgeBase):
    """
    影响模式知识库
    
    存储不同异常类型对无人机集群的影响模式
    """
    
    def __init__(self, save_path: Optional[str] = None):
        super().__init__('impact_patterns', save_path)
        
        if not self.knowledge:
            self._initialize_default_knowledge()
    
    def _initialize_default_knowledge(self):
        """初始化默认影响模式知识"""
        
        # 单机影响
        self.add_knowledge({
            'type': 'impact_pattern',
            'pattern': 'single_drone',
            'title': '单机影响模式',
            'title_en': 'Single Drone Impact',
            'description': '异常仅影响单架无人机，不波及其他成员',
            'keywords': '单机,孤立,独立,局部',
            'affected_scope': 'single',
            'propagation_risk': 'low',
            'characteristics': [
                '异常局限于单架无人机',
                '其他无人机正常运行',
                '可通过隔离受影响无人机处理'
            ],
            'mitigation_strategies': [
                '将异常无人机从编队中隔离',
                '重新分配任务给其他无人机',
                '引导异常无人机返回或降落'
            ],
            'category': 'impact'
        })
        
        # 局部传播
        self.add_knowledge({
            'type': 'impact_pattern',
            'pattern': 'local_propagation',
            'title': '局部传播模式',
            'title_en': 'Local Propagation Impact',
            'description': '异常影响邻近的几架无人机',
            'keywords': '局部,传播,邻近,扩散',
            'affected_scope': 'local',
            'propagation_risk': 'medium',
            'characteristics': [
                '异常可能影响2-3架邻近无人机',
                '通常由位置或通信问题引起',
                '需要及时干预防止扩散'
            ],
            'mitigation_strategies': [
                '调整邻近无人机的位置',
                '增加安全距离',
                '临时重新配置编队'
            ],
            'category': 'impact'
        })
        
        # 全局影响
        self.add_knowledge({
            'type': 'impact_pattern',
            'pattern': 'global_impact',
            'title': '全局影响模式',
            'title_en': 'Global Impact',
            'description': '异常影响整个无人机集群的运行',
            'keywords': '全局,整体,集群,系统',
            'affected_scope': 'global',
            'propagation_risk': 'high',
            'characteristics': [
                '影响整个集群的协调',
                '可能导致任务失败',
                '需要紧急全局干预'
            ],
            'mitigation_strategies': [
                '暂停任务执行',
                '全体无人机进入安全模式',
                '重新建立通信和控制'
            ],
            'category': 'impact'
        })
        
        # 碰撞风险
        self.add_knowledge({
            'type': 'impact_pattern',
            'pattern': 'collision_risk',
            'title': '碰撞风险模式',
            'title_en': 'Collision Risk Pattern',
            'description': '异常导致无人机间碰撞风险增加',
            'keywords': '碰撞,风险,避障,安全距离',
            'affected_scope': 'safety',
            'propagation_risk': 'critical',
            'characteristics': [
                '无人机间距离过近',
                '轨迹出现交叉',
                '避障系统可能失效'
            ],
            'mitigation_strategies': [
                '立即增加安全距离',
                '启动紧急避障',
                '如必要，执行紧急降落'
            ],
            'category': 'safety'
        })


class HistoricalCaseKnowledge(KnowledgeBase):
    """
    历史案例知识库
    
    存储历史异常事件及其处理经验
    """
    
    def __init__(self, save_path: Optional[str] = None):
        super().__init__('historical_cases', save_path)
        
        if not self.knowledge:
            self._initialize_default_knowledge()
    
    def _initialize_default_knowledge(self):
        """初始化默认历史案例"""
        
        # 案例1：GPS干扰导致位置偏离
        self.add_knowledge({
            'type': 'historical_case',
            'case_id': 'CASE_001',
            'title': 'GPS干扰导致的位置偏离',
            'title_en': 'Position Deviation due to GPS Interference',
            'description': '在城市环境飞行时，GPS信号受到建筑物反射干扰，导致多架无人机位置偏离',
            'keywords': 'GPS,干扰,城市,建筑,位置偏离',
            'anomaly_type': 'position_deviation',
            'severity': 'medium',
            'affected_drones': [0, 3, 7],
            'root_cause': 'GPS多径效应和信号遮挡',
            'impact': '任务效率降低30%，需要人工干预重新校准',
            'resolution': [
                '启用备用定位系统（视觉或惯导）',
                '调整飞行高度避开干扰区域',
                '增加定位数据滤波强度'
            ],
            'lessons_learned': [
                '城市环境需要多源定位融合',
                '提前规划避开GPS盲区',
                '建立位置异常快速响应机制'
            ],
            'date': '2024-03-15',
            'category': 'case'
        })
        
        # 案例2：电池老化导致电量异常
        self.add_knowledge({
            'type': 'historical_case',
            'case_id': 'CASE_002',
            'title': '电池老化导致电量异常下降',
            'title_en': 'Battery Anomaly due to Aging',
            'description': '长期使用的电池出现老化，导致电量读数突然下降和续航减少',
            'keywords': '电池,老化,电量,续航,维护',
            'anomaly_type': 'sensor_malfunction_battery',
            'severity': 'medium',
            'affected_drones': [1, 4],
            'root_cause': '电池循环次数过多，内阻增大',
            'impact': '续航时间减少40%，任务提前终止',
            'resolution': [
                '更换老化电池',
                '建立电池健康监测系统',
                '设置电量预警阈值'
            ],
            'lessons_learned': [
                '建立电池使用寿命跟踪系统',
                '定期进行电池容量测试',
                '制定电池更换标准'
            ],
            'date': '2024-02-20',
            'category': 'case'
        })
        
        # 案例3：电机故障导致机动异常
        self.add_knowledge({
            'type': 'historical_case',
            'case_id': 'CASE_003',
            'title': '电机故障导致飞行震荡',
            'title_en': 'Flight Oscillation due to Motor Failure',
            'description': '单个电机性能下降导致无人机出现周期性震荡和姿态不稳',
            'keywords': '电机,故障,震荡,姿态,控制',
            'anomaly_type': 'maneuver_anomaly',
            'severity': 'high',
            'affected_drones': [2],
            'root_cause': '电机轴承磨损，转速不稳定',
            'impact': '无人机被紧急召回，任务中断',
            'resolution': [
                '紧急降落并更换电机',
                '飞控自动补偿模式',
                '其他无人机接管任务'
            ],
            'lessons_learned': [
                '建立电机振动监测机制',
                '定期进行电机性能测试',
                '配备冗余电机方案'
            ],
            'date': '2024-01-10',
            'category': 'case'
        })
        
        # 案例4：通信干扰导致失联
        self.add_knowledge({
            'type': 'historical_case',
            'case_id': 'CASE_004',
            'title': '电磁干扰导致通信中断',
            'title_en': 'Communication Loss due to EMI',
            'description': '在工业区执行任务时，电磁干扰导致部分无人机通信中断',
            'keywords': '通信,干扰,电磁,失联,工业区',
            'anomaly_type': 'sensor_malfunction_signal',
            'severity': 'high',
            'affected_drones': [5, 8],
            'root_cause': '工业设备产生的电磁干扰',
            'impact': '两架无人机进入自主返航模式',
            'resolution': [
                '切换到备用通信频段',
                '启动自主返航程序',
                '避开干扰区域重新规划路线'
            ],
            'lessons_learned': [
                '任务前进行电磁环境评估',
                '配置多频段通信能力',
                '建立失联后自主处理机制'
            ],
            'date': '2024-04-05',
            'category': 'case'
        })


class DroneAnomalyKnowledgeBase:
    """
    无人机异常知识库管理器
    
    统一管理所有子知识库，提供综合检索能力
    """
    
    def __init__(self, base_path: str = 'knowledge_base'):
        """
        初始化知识库管理器
        
        参数:
            base_path: 知识库文件存储路径
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        
        # 初始化各子知识库
        self.anomaly_types = AnomalyTypeKnowledge(
            os.path.join(base_path, 'anomaly_types.json')
        )
        self.severity = SeverityKnowledge(
            os.path.join(base_path, 'severity.json')
        )
        self.impact_patterns = ImpactPatternKnowledge(
            os.path.join(base_path, 'impact_patterns.json')
        )
        self.historical_cases = HistoricalCaseKnowledge(
            os.path.join(base_path, 'historical_cases.json')
        )
        
        print(f"[KnowledgeBase] 知识库已初始化")
        print(f"  - 异常类型: {len(self.anomaly_types.knowledge)} 条")
        print(f"  - 严重程度: {len(self.severity.knowledge)} 条")
        print(f"  - 影响模式: {len(self.impact_patterns.knowledge)} 条")
        print(f"  - 历史案例: {len(self.historical_cases.knowledge)} 条")
    
    def search_all(self, query: str, top_k: int = 3) -> Dict[str, List[Dict]]:
        """
        在所有知识库中搜索
        
        参数:
            query: 查询字符串
            top_k: 每个知识库返回的结果数
        
        返回:
            各知识库的搜索结果
        """
        return {
            'anomaly_types': self.anomaly_types.search(query, top_k),
            'severity': self.severity.search(query, top_k),
            'impact_patterns': self.impact_patterns.search(query, top_k),
            'historical_cases': self.historical_cases.search(query, top_k)
        }
    
    def get_anomaly_info(self, anomaly_type: str) -> Optional[Dict]:
        """
        获取异常类型信息
        
        参数:
            anomaly_type: 异常类型标识
        
        返回:
            异常类型信息
        """
        results = self.anomaly_types.get_by_type(anomaly_type)
        if results:
            return results[0]
        
        # 模糊匹配
        for item in self.anomaly_types.knowledge:
            if anomaly_type.lower() in item.get('type', '').lower():
                return item
        
        return None
    
    def get_severity_level(self, score: float) -> Dict:
        """
        根据异常分数获取严重程度
        
        参数:
            score: 异常分数 (0-1)
        
        返回:
            严重程度信息
        """
        for item in self.severity.knowledge:
            score_range = item.get('score_range', [0, 1])
            if score_range[0] <= score < score_range[1]:
                return item
        
        # 默认返回最高级别
        return self.severity.knowledge[-1] if self.severity.knowledge else {}
    
    def get_similar_cases(self, anomaly_type: str, top_k: int = 3) -> List[Dict]:
        """
        获取相似历史案例
        
        参数:
            anomaly_type: 异常类型
            top_k: 返回案例数
        
        返回:
            相似案例列表
        """
        cases = self.historical_cases.search(anomaly_type, top_k)
        
        # 优先返回类型匹配的案例
        matching_cases = [c for c in cases 
                        if c.get('anomaly_type', '').lower() in anomaly_type.lower()
                        or anomaly_type.lower() in c.get('anomaly_type', '').lower()]
        
        if matching_cases:
            return matching_cases
        return cases
    
    def get_impact_pattern(self, affected_count: int) -> Dict:
        """
        根据受影响无人机数量获取影响模式
        
        参数:
            affected_count: 受影响的无人机数量
        
        返回:
            影响模式信息
        """
        if affected_count == 1:
            pattern = 'single_drone'
        elif affected_count <= 3:
            pattern = 'local_propagation'
        else:
            pattern = 'global_impact'
        
        for item in self.impact_patterns.knowledge:
            if item.get('pattern') == pattern:
                return item
        
        return {}
    
    def save_all(self):
        """保存所有知识库"""
        self.anomaly_types.save()
        self.severity.save()
        self.impact_patterns.save()
        self.historical_cases.save()


if __name__ == "__main__":
    # 测试知识库
    print("="*60)
    print("测试无人机异常知识库")
    print("="*60)
    
    kb = DroneAnomalyKnowledgeBase('knowledge_base')
    
    # 测试搜索
    print("\n搜索 '位置偏离':")
    results = kb.search_all('位置偏离', top_k=2)
    for category, items in results.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  - {item.get('title', 'N/A')} (相似度: {item.get('similarity', 0):.2f})")
    
    # 测试获取异常信息
    print("\n获取 'position_deviation' 异常信息:")
    info = kb.get_anomaly_info('position_deviation')
    if info:
        print(f"  标题: {info.get('title')}")
        print(f"  描述: {info.get('description')}")
    
    # 测试获取严重程度
    print("\n获取异常分数 0.8 的严重程度:")
    severity = kb.get_severity_level(0.8)
    print(f"  级别: {severity.get('title')}")
    print(f"  建议: {severity.get('recommended_action')}")
    
    # 保存知识库
    kb.save_all()
    print("\n知识库已保存")
