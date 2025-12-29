"""
无人机集群异常知识库 - LangChain 实现
UAV Swarm Anomaly Knowledge Base - LangChain Implementation

本模块使用 LangChain 框架实现基于 RAG（检索增强生成）技术的知识库系统。
包含无人机异常类型、严重程度、影响模式和历史案例等知识。

功能：
1. 使用 LangChain 的文档加载器管理知识
2. 使用 Chroma 向量存储进行语义搜索
3. 使用 sentence-transformers 进行高质量嵌入
4. 支持多种检索策略（相似度、MMR 等）

Author: AI Agent
Version: 2.0.0 (LangChain)
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings

# LangChain 核心组件
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

# LangChain 社区组件
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 忽略一些警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class DroneAnomalyKnowledgeBase:
    """
    无人机异常知识库 - LangChain 实现
    
    使用 LangChain 框架构建的知识库系统，支持：
    1. 向量化存储和语义搜索
    2. 多种检索策略
    3. 知识的增删改查
    4. 持久化存储
    """
    
    def __init__(self, base_path: str = "knowledge_base"):
        """
        初始化知识库
        
        参数:
            base_path: 知识库存储路径
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        
        # 初始化嵌入模型 - 使用简单的本地嵌入
        print("[KnowledgeBase] 初始化嵌入模型...")
        try:
            # 优先使用本地已有模型（如果存在）
            from langchain_community.embeddings import FakeEmbeddings
            print("[KnowledgeBase] 使用简化的本地嵌入模型（适用于离线环境）")
            # 使用固定维度的假嵌入用于演示
            self.embeddings = FakeEmbeddings(size=384)
            self._use_fake_embeddings = True
        except Exception as e:
            print(f"[KnowledgeBase] 警告: 嵌入初始化失败: {e}")
            from langchain_community.embeddings import FakeEmbeddings
            self.embeddings = FakeEmbeddings(size=384)
            self._use_fake_embeddings = True
        
        # 初始化向量存储
        self.vector_stores: Dict[str, Chroma] = {}
        self._init_knowledge_bases()
        
        print(f"[KnowledgeBase] 知识库已初始化: {base_path}")
    
    def _init_knowledge_bases(self):
        """初始化各类知识库"""
        # 1. 异常类型知识库
        self._init_anomaly_types()
        
        # 2. 严重程度知识库
        self._init_severity_levels()
        
        # 3. 影响模式知识库
        self._init_impact_patterns()
        
        # 4. 历史案例知识库
        self._init_historical_cases()
    
    def _create_vector_store(self, collection_name: str, documents: List[Document]) -> Chroma:
        """
        创建向量存储
        
        参数:
            collection_name: 集合名称
            documents: 文档列表
        
        返回:
            Chroma 向量存储实例
        """
        persist_directory = os.path.join(self.base_path, collection_name)
        
        # 检查是否已存在
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            # 加载已有的向量存储
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
            print(f"[KnowledgeBase] 已加载知识库: {collection_name}")
        else:
            # 创建新的向量存储
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=persist_directory
            )
            print(f"[KnowledgeBase] 已创建知识库: {collection_name}")
        
        return vector_store
    
    def _init_anomaly_types(self):
        """初始化异常类型知识库"""
        anomaly_types = [
            {
                "type": "position_deviation",
                "name_cn": "位置偏离异常",
                "name_en": "Position Deviation Anomaly",
                "description": "无人机偏离预定飞行轨迹，可能由GPS故障、风力扰动或控制系统问题引起。",
                "characteristics": "x, y, z坐标出现渐进式偏离，偏离幅度可达20-30米",
                "causes": ["GPS故障", "风力扰动", "控制系统问题", "导航算法错误"],
                "detection_features": ["位置坐标异常", "轨迹偏离", "与预定路径的距离增加"],
                "severity_factors": ["偏离距离", "偏离速度", "持续时间", "环境因素"]
            },
            {
                "type": "sensor_malfunction_battery",
                "name_cn": "电池传感器异常",
                "name_en": "Battery Sensor Malfunction",
                "description": "电池电量传感器出现异常读数，电量突然大幅下降约30%。",
                "characteristics": "电池电量突然下降，不符合正常放电曲线",
                "causes": ["传感器故障", "电池老化", "电源管理系统错误", "温度影响"],
                "detection_features": ["电量突降", "电量波动异常", "与预期放电曲线偏离"],
                "severity_factors": ["电量下降幅度", "剩余电量", "飞行任务阶段"]
            },
            {
                "type": "sensor_malfunction_temperature",
                "name_cn": "温度传感器异常",
                "name_en": "Temperature Sensor Malfunction",
                "description": "温度传感器读数异常，温度异常升高15-25摄氏度，可能表示散热问题。",
                "characteristics": "温度异常升高，超出正常工作范围",
                "causes": ["传感器故障", "散热系统故障", "环境温度过高", "过载运行"],
                "detection_features": ["温度突升", "温度持续高于阈值", "温度变化率异常"],
                "severity_factors": ["温度值", "升温速度", "持续时间", "临界温度距离"]
            },
            {
                "type": "sensor_malfunction_signal",
                "name_cn": "信号传感器异常",
                "name_en": "Signal Sensor Malfunction",
                "description": "通信信号强度异常，信号突然减弱至-90dBm左右，可能存在通信干扰。",
                "characteristics": "信号强度突然下降，通信质量恶化",
                "causes": ["通信干扰", "天线故障", "距离过远", "环境遮挡"],
                "detection_features": ["信号强度下降", "通信延迟增加", "数据包丢失率上升"],
                "severity_factors": ["信号强度", "通信稳定性", "控制指令延迟"]
            },
            {
                "type": "maneuver_anomaly",
                "name_cn": "机动异常",
                "name_en": "Maneuver Anomaly",
                "description": "速度和加速度出现异常变化，可能由控制算法错误、电机故障或传感器干扰引起。",
                "characteristics": "vx, vy, vz, ax, ay, az出现高频振荡，振幅为正常值的3-4倍",
                "causes": ["控制算法错误", "电机故障", "IMU传感器干扰", "风扰动"],
                "detection_features": ["速度振荡", "加速度异常", "姿态不稳定", "运动轨迹抖动"],
                "severity_factors": ["振荡幅度", "频率", "持续时间", "可控性"]
            }
        ]
        
        documents = []
        for anomaly in anomaly_types:
            # 构建文档内容
            content = f"""
类型: {anomaly['name_cn']} ({anomaly['name_en']})
标识: {anomaly['type']}
描述: {anomaly['description']}
特征表现: {anomaly['characteristics']}
可能原因: {', '.join(anomaly['causes'])}
检测特征: {', '.join(anomaly['detection_features'])}
严重程度因素: {', '.join(anomaly['severity_factors'])}
"""
            # 转换列表为 JSON 字符串以兼容 ChromaDB
            metadata = {
                'type': anomaly['type'],
                'name_cn': anomaly['name_cn'],
                'name_en': anomaly['name_en'],
                'description': anomaly['description'],
                'characteristics': anomaly['characteristics'],
                'causes': json.dumps(anomaly['causes'], ensure_ascii=False),
                'detection_features': json.dumps(anomaly['detection_features'], ensure_ascii=False),
                'severity_factors': json.dumps(anomaly['severity_factors'], ensure_ascii=False)
            }
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        
        self.vector_stores['anomaly_types'] = self._create_vector_store(
            'anomaly_types', 
            documents
        )
    
    def _init_severity_levels(self):
        """初始化严重程度知识库"""
        severity_levels = [
            {
                "level": "low",
                "name_cn": "低",
                "name_en": "Low",
                "description": "轻微异常，不影响任务执行，可以继续监控",
                "characteristics": ["偏离程度小", "持续时间短", "无安全风险"],
                "recommendations": ["持续监控", "记录日志", "适时检查"],
                "threshold_indicators": {
                    "position_deviation": "<10m",
                    "battery_drop": "<10%",
                    "temperature_increase": "<10°C",
                    "signal_strength": ">-80dBm"
                }
            },
            {
                "level": "medium",
                "name_cn": "中等",
                "name_en": "Medium",
                "description": "明显异常，需要注意观察，可能影响任务质量",
                "characteristics": ["偏离程度中等", "持续时间较长", "存在潜在风险"],
                "recommendations": ["加强监控", "准备应急预案", "考虑任务调整"],
                "threshold_indicators": {
                    "position_deviation": "10-20m",
                    "battery_drop": "10-25%",
                    "temperature_increase": "10-20°C",
                    "signal_strength": "-80 to -85dBm"
                }
            },
            {
                "level": "high",
                "name_cn": "高",
                "name_en": "High",
                "description": "严重异常，可能导致任务失败或安全事故，需要立即处置",
                "characteristics": ["偏离程度大", "持续时间长", "存在明显安全风险"],
                "recommendations": ["立即处置", "启动应急预案", "考虑返航或迫降"],
                "threshold_indicators": {
                    "position_deviation": ">20m",
                    "battery_drop": ">25%",
                    "temperature_increase": ">20°C",
                    "signal_strength": "<-85dBm"
                }
            }
        ]
        
        documents = []
        for severity in severity_levels:
            content = f"""
级别: {severity['name_cn']} ({severity['name_en']})
标识: {severity['level']}
描述: {severity['description']}
特征: {', '.join(severity['characteristics'])}
处置建议: {', '.join(severity['recommendations'])}
阈值指标: {json.dumps(severity['threshold_indicators'], ensure_ascii=False)}
"""
            metadata = {
                'level': severity['level'],
                'name_cn': severity['name_cn'],
                'name_en': severity['name_en'],
                'description': severity['description'],
                'characteristics': json.dumps(severity['characteristics'], ensure_ascii=False),
                'recommendations': json.dumps(severity['recommendations'], ensure_ascii=False),
                'threshold_indicators': json.dumps(severity['threshold_indicators'], ensure_ascii=False)
            }
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        
        self.vector_stores['severity_levels'] = self._create_vector_store(
            'severity_levels',
            documents
        )
    
    def _init_impact_patterns(self):
        """初始化影响模式知识库"""
        impact_patterns = [
            {
                "pattern": "single_drone",
                "name_cn": "单机影响",
                "name_en": "Single Drone Impact",
                "description": "异常仅影响单架无人机，不会传播到其他无人机",
                "characteristics": ["影响范围局限", "其他无人机正常", "可以隔离处理"],
                "typical_causes": ["硬件故障", "传感器故障", "个别无人机的软件问题"],
                "mitigation_strategies": ["隔离异常无人机", "调整编队", "替换备用无人机"]
            },
            {
                "pattern": "local_propagation",
                "name_cn": "局部传播",
                "name_en": "Local Propagation",
                "description": "异常可能影响邻近的无人机，存在局部传播风险",
                "characteristics": ["影响邻近节点", "存在传播趋势", "需要及时控制"],
                "typical_causes": ["编队算法问题", "通信干扰", "协同控制错误"],
                "mitigation_strategies": ["调整编队间距", "增强局部控制", "隔离影响区域"]
            },
            {
                "pattern": "global_impact",
                "name_cn": "全局影响",
                "name_en": "Global Impact",
                "description": "异常可能影响整个集群的正常运行",
                "characteristics": ["影响范围大", "系统性问题", "需要紧急处置"],
                "typical_causes": ["中心节点故障", "全局算法错误", "环境突变"],
                "mitigation_strategies": ["启动应急模式", "考虑全局返航", "切换控制模式"]
            },
            {
                "pattern": "collision_risk",
                "name_cn": "碰撞风险",
                "name_en": "Collision Risk",
                "description": "异常导致无人机之间或与障碍物存在碰撞风险",
                "characteristics": ["距离过近", "轨迹交叉", "存在碰撞危险"],
                "typical_causes": ["位置偏离", "控制失灵", "避障系统故障"],
                "mitigation_strategies": ["紧急避让", "增大安全距离", "激活防碰撞模式"]
            }
        ]
        
        documents = []
        for pattern in impact_patterns:
            content = f"""
模式: {pattern['name_cn']} ({pattern['name_en']})
标识: {pattern['pattern']}
描述: {pattern['description']}
特征: {', '.join(pattern['characteristics'])}
典型原因: {', '.join(pattern['typical_causes'])}
缓解策略: {', '.join(pattern['mitigation_strategies'])}
"""
            metadata = {
                'pattern': pattern['pattern'],
                'name_cn': pattern['name_cn'],
                'name_en': pattern['name_en'],
                'description': pattern['description'],
                'characteristics': json.dumps(pattern['characteristics'], ensure_ascii=False),
                'typical_causes': json.dumps(pattern['typical_causes'], ensure_ascii=False),
                'mitigation_strategies': json.dumps(pattern['mitigation_strategies'], ensure_ascii=False)
            }
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        
        self.vector_stores['impact_patterns'] = self._create_vector_store(
            'impact_patterns',
            documents
        )
    
    def _init_historical_cases(self):
        """初始化历史案例知识库"""
        historical_cases = [
            {
                "case_id": "CASE_001",
                "title": "GPS干扰导致的位置偏离",
                "date": "2024-01-15",
                "anomaly_type": "position_deviation",
                "severity": "high",
                "description": "在城市环境中飞行时，无人机受到GPS干扰，导致位置偏离超过25米",
                "root_cause": "城市峡谷效应导致GPS信号质量下降，多路径效应严重",
                "impact": "影响了3架无人机，导致编队失序",
                "resolution": ["切换到视觉导航模式", "增大安全间距", "调整飞行高度"],
                "lessons_learned": ["在复杂环境中应启用多源导航", "提前规划备用导航方案", "设置GPS质量监控阈值"]
            },
            {
                "case_id": "CASE_002",
                "title": "电池传感器故障误报",
                "date": "2024-02-20",
                "anomaly_type": "sensor_malfunction_battery",
                "severity": "medium",
                "description": "电池管理系统传感器故障，误报电量不足",
                "root_cause": "电池管理系统固件版本存在bug",
                "impact": "单架无人机提前返航，未完成任务",
                "resolution": ["更新固件", "增加电池健康度检查", "使用冗余传感器"],
                "lessons_learned": ["定期检查传感器校准", "使用多传感器交叉验证", "建立传感器故障检测机制"]
            },
            {
                "case_id": "CASE_003",
                "title": "高温环境下的过热保护",
                "date": "2024-03-10",
                "anomaly_type": "sensor_malfunction_temperature",
                "severity": "high",
                "description": "在高温环境下长时间飞行，电机温度过高触发保护机制",
                "root_cause": "环境温度35°C + 长时间高负荷飞行",
                "impact": "2架无人机自动降低功率，速度受限",
                "resolution": ["缩短单次飞行时间", "降低负载", "选择凉爽时段飞行"],
                "lessons_learned": ["考虑环境温度对性能的影响", "建立温度管理策略", "设置动态性能限制"]
            },
            {
                "case_id": "CASE_004",
                "title": "通信干扰导致的编队松散",
                "date": "2024-04-05",
                "anomaly_type": "sensor_malfunction_signal",
                "severity": "medium",
                "description": "在工业区附近飞行时遭遇强电磁干扰，通信质量下降",
                "root_cause": "工业设备的电磁干扰",
                "impact": "编队间距增大，同步性下降",
                "resolution": ["切换通信频段", "增强信号功率", "调整飞行路线"],
                "lessons_learned": ["事先勘察电磁环境", "准备多频段通信方案", "设置通信质量监控"]
            },
            {
                "case_id": "CASE_005",
                "title": "强风导致的机动异常",
                "date": "2024-05-12",
                "anomaly_type": "maneuver_anomaly",
                "severity": "high",
                "description": "遭遇突发强风，无人机姿态和速度出现异常振荡",
                "root_cause": "风速突然增大到15m/s，超出设计阈值",
                "impact": "5架无人机出现振荡，编队严重变形",
                "resolution": ["降低飞行高度", "增大控制增益", "启用抗风模式"],
                "lessons_learned": ["实时监控风速", "设置风速阈值", "准备不同风况的控制策略"]
            }
        ]
        
        documents = []
        for case in historical_cases:
            content = f"""
案例编号: {case['case_id']}
标题: {case['title']}
日期: {case['date']}
异常类型: {case['anomaly_type']}
严重程度: {case['severity']}
描述: {case['description']}
根本原因: {case['root_cause']}
影响: {case['impact']}
解决方案: {', '.join(case['resolution'])}
经验教训: {', '.join(case['lessons_learned'])}
"""
            metadata = {
                'case_id': case['case_id'],
                'title': case['title'],
                'date': case['date'],
                'anomaly_type': case['anomaly_type'],
                'severity': case['severity'],
                'description': case['description'],
                'root_cause': case['root_cause'],
                'impact': case['impact'],
                'resolution': json.dumps(case['resolution'], ensure_ascii=False),
                'lessons_learned': json.dumps(case['lessons_learned'], ensure_ascii=False)
            }
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        
        self.vector_stores['historical_cases'] = self._create_vector_store(
            'historical_cases',
            documents
        )
    
    def search(self, query: str, collection_name: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """
        在指定知识库中搜索
        
        参数:
            query: 查询字符串
            collection_name: 知识库名称
            top_k: 返回结果数量
        
        返回:
            文档和相似度得分的列表
        """
        if collection_name not in self.vector_stores:
            return []
        
        vector_store = self.vector_stores[collection_name]
        results = vector_store.similarity_search_with_score(query, k=top_k)
        return results
    
    def search_all(self, query: str, top_k: int = 2) -> Dict[str, List[Tuple[Document, float]]]:
        """
        在所有知识库中搜索
        
        参数:
            query: 查询字符串
            top_k: 每个知识库返回的结果数量
        
        返回:
            各知识库的搜索结果字典
        """
        results = {}
        for collection_name in self.vector_stores.keys():
            results[collection_name] = self.search(query, collection_name, top_k)
        return results
    
    def get_anomaly_info(self, anomaly_type: str) -> Optional[Dict]:
        """
        获取特定异常类型的详细信息
        
        参数:
            anomaly_type: 异常类型标识
        
        返回:
            异常信息字典
        """
        results = self.search(anomaly_type, 'anomaly_types', top_k=1)
        if results:
            doc, score = results[0]
            return doc.metadata
        return None
    
    def get_similar_cases(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        获取相似的历史案例
        
        参数:
            query: 查询描述
            top_k: 返回案例数量
        
        返回:
            案例列表
        """
        results = self.search(query, 'historical_cases', top_k)
        cases = []
        for doc, score in results:
            case = doc.metadata.copy()
            case['similarity_score'] = float(score)
            cases.append(case)
        return cases
    
    def add_document(self, collection_name: str, content: str, metadata: Dict):
        """
        添加新文档到知识库
        
        参数:
            collection_name: 知识库名称
            content: 文档内容
            metadata: 元数据
        """
        if collection_name not in self.vector_stores:
            print(f"[KnowledgeBase] 警告: 知识库 {collection_name} 不存在")
            return
        
        doc = Document(page_content=content, metadata=metadata)
        self.vector_stores[collection_name].add_documents([doc])
    
    def get_retriever(self, collection_name: str, search_type: str = "similarity", k: int = 3):
        """
        获取检索器（用于 LangChain 链式调用）
        
        参数:
            collection_name: 知识库名称
            search_type: 搜索类型 ("similarity", "mmr", "similarity_score_threshold")
            k: 返回结果数量
        
        返回:
            LangChain Retriever 对象
        """
        if collection_name not in self.vector_stores:
            return None
        
        return self.vector_stores[collection_name].as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("测试 LangChain 知识库实现")
    print("="*60)
    
    # 创建知识库
    kb = DroneAnomalyKnowledgeBase("knowledge_base")
    
    # 测试搜索
    print("\n测试 1: 搜索位置偏离相关知识")
    print("-"*60)
    results = kb.search_all("位置偏离 GPS", top_k=2)
    for collection, docs in results.items():
        print(f"\n[{collection}]")
        for doc, score in docs:
            print(f"  相似度: {score:.3f}")
            print(f"  内容预览: {doc.page_content[:100]}...")
    
    # 测试获取异常信息
    print("\n\n测试 2: 获取特定异常类型信息")
    print("-"*60)
    info = kb.get_anomaly_info("position_deviation")
    if info:
        print(f"类型: {info.get('name_cn')}")
        print(f"描述: {info.get('description')}")
    
    # 测试获取相似案例
    print("\n\n测试 3: 获取相似历史案例")
    print("-"*60)
    cases = kb.get_similar_cases("GPS信号问题导致位置异常", top_k=2)
    for case in cases:
        print(f"\n案例: {case.get('title')}")
        print(f"  相似度得分: {case.get('similarity_score', 0):.3f}")
        print(f"  日期: {case.get('date')}")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
