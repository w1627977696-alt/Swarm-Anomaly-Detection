# 无人机集群异常检测Agent系统使用文档

## UAV Swarm Anomaly Detection Agent System User Guide

[中文](#中文文档) | [English](#english-documentation)

---

## 🆕 v2.0 重大更新

本系统已升级至 v2.0，采用 **LangChain** 和 **LangGraph** 框架重构：

### 主要改进

1. **LangGraph Agent 编排**: 使用 StateGraph 实现状态化的多 Agent 协作
2. **LangChain RAG 系统**: 基于 ChromaDB 向量存储的知识检索增强生成
3. **语义搜索**: 使用 sentence-transformers 实现高质量的语义理解
4. **可扩展架构**: 易于集成新的 Agent 和知识源

### 技术栈

- **Agent 框架**: LangGraph 0.2.0+
- **RAG 框架**: LangChain 0.3.0+
- **向量存储**: ChromaDB 0.5.0+
- **嵌入模型**: Sentence-Transformers 2.2.0+

### 文档索引

- **技术实现详情**: 参见 [LANGCHAIN_IMPLEMENTATION.md](LANGCHAIN_IMPLEMENTATION.md)
- **快速开始**: 参见下方使用指南
- **API 参考**: 参见第7节

---

## 中文文档

### 1. 系统概述

无人机集群异常检测Agent系统是一个基于 **LangChain/LangGraph** 框架的智能运维平台。该系统使用 StateGraph 进行 Agent 编排，通过 RAG 技术进行知识检索，实现无人机集群的异常检测、影响评估和报告生成。

#### 1.1 主要功能

| 功能模块 | 描述 | 实现方式 |
|---------|------|---------|
| 路由决策 | 分析用户输入，决定执行流程 | LangGraph Router Agent |
| 数据加载 | 加载无人机集群数据 | LangGraph Data Loader Agent |
| 异常检测 | 使用Patch-GNN模型检测异常 | LangGraph Detector Agent |
| 影响评估 | 基于RAG评估异常影响和风险 | LangGraph Assessor Agent + ChromaDB RAG |
| 报告生成 | 生成结构化分析报告 | LangGraph Reporter Agent |
| 用户响应 | 格式化输出给用户 | LangGraph Responder Agent |

#### 1.2 LangGraph 工作流架构

```
┌─────────────────────────────────────────────────────────────┐
│                         用户输入                             │
│                    Natural Language                         │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   LangGraph StateGraph                      │
│                                                             │
│  ┌──────────┐    条件路由    ┌──────────┐                  │
│  │  Router  │───────────────▶│Data Load │                  │
│  │  Agent   │                │  Agent   │                  │
│  └────┬─────┘                └────┬─────┘                  │
│       │                           │                        │
│       │    ┌──────────┐    ┌──────▼─────┐                 │
│       ├───▶│ Detector │◀───│  Assessor  │                 │
│       │    │  Agent   │    │   Agent    │                 │
│       │    └────┬─────┘    └────┬───────┘                 │
│       │         │               │                          │
│       │    ┌────▼─────┐    ┌────▼─────┐                   │
│       ├───▶│ Reporter │    │Responder │                   │
│       │    │  Agent   │    │  Agent   │                   │
│       │    └──────────┘    └────┬─────┘                   │
│       │                          │                         │
│       └──────────────────────────┴────▶ END               │
│                                                            │
│  状态管理: AgentState (TypedDict)                          │
│  • messages: List[BaseMessage]                            │
│  • data_loaded: bool                                      │
│  • detected_anomalies: List[Dict]                         │
│  • impact_assessments: List[Dict]                         │
│  • report: Optional[str]                                  │
│  • next_action: str                                       │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   LangChain RAG 知识库                       │
│                      (ChromaDB)                             │
│                                                             │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐ │
│  │ Anomaly Types  │  │ Severity Level │  │   Impact     │ │
│  │   Knowledge    │  │   Knowledge    │  │  Patterns    │ │
│  └────────────────┘  └────────────────┘  └──────────────┘ │
│                                                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │            Historical Cases Knowledge              │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  语义检索: Sentence-Transformers Embeddings                │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    知识库 (RAG)                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ 异常类型 │ │ 严重程度 │ │ 影响模式 │ │ 历史案例 │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 2. 快速开始

#### 2.1 安装依赖

```bash
pip install -r requirements.txt
```

#### 2.2 启动Agent系统

```bash
# 启动交互式会话
python main_agent.py

# 或者使用别名
python -m agent.core
```

#### 2.3 基本交互

启动后，您可以使用自然语言与系统交互：

```
>>> 请输入指令: 帮助

>>> 请输入指令: 检测所有无人机的异常

>>> 请输入指令: 评估检测到的异常影响

>>> 请输入指令: 生成综合报告

>>> 请输入指令: 退出
```

### 3. 详细使用指南

#### 3.1 数据预处理与可视化

**支持的指令示例：**

```
# 加载数据
"加载数据并进行预处理"
"读取无人机集群数据"

# 可视化
"可视化无人机0的飞行数据"
"显示所有无人机的传感器数据"
"画出3D轨迹图"
"展示异常分布"
```

**参数说明：**
- `drone_ids`: 指定无人机ID，如 "无人机0和1" 或 "drone 0, 1, 2"
- `plot_type`: 图表类型，支持 "comprehensive"（综合）、"trajectory"（轨迹）、"sensors"（传感器）、"anomaly"（异常分布）

#### 3.2 数据回放

**支持的指令示例：**

```
"回放无人机3从1000到1500秒的数据"
"回放所有无人机的历史数据"
"重放异常发生时的飞行数据"
```

**参数说明：**
- `time_range`: 时间范围，如 "从1000到1500" 或 "1000-1500秒"
- `drone_ids`: 指定要回放的无人机

#### 3.3 异常检测

**支持的指令示例：**

```
"检测所有无人机的异常"
"对无人机0进行异常检测"
"分析无人机集群的异常行为"
"检测位置偏离异常"
```

**参数说明：**
- `drone_ids`: 要检测的无人机列表
- `threshold`: 异常判断阈值（默认0.5）
- `anomaly_types`: 关注的异常类型

**异常类型说明：**

| 类型 | 中文名称 | 特征 |
|-----|---------|------|
| `position_deviation` | 位置偏离异常 | 无人机偏离预定轨迹 |
| `sensor_malfunction_battery` | 电池传感器异常 | 电量读数异常 |
| `sensor_malfunction_temperature` | 温度传感器异常 | 温度读数异常 |
| `sensor_malfunction_signal_strength` | 信号传感器异常 | 通信信号异常 |
| `maneuver_anomaly` | 机动异常 | 速度和加速度异常 |

#### 3.4 异常影响评估

**支持的指令示例：**

```
"评估检测到的异常影响"
"分析位置偏离异常的严重程度"
"进行风险评估"
"评估异常的影响范围"
```

**评估内容：**
1. **严重程度评估**：低/中等/高
2. **影响范围分析**：单机/局部/全局
3. **风险等级判定**：基于置信度和严重程度
4. **潜在后果预测**：根据异常类型推断可能后果
5. **历史案例参考**：检索相似案例和处理经验

#### 3.5 报告生成

**支持的指令示例：**

```
"生成综合报告"
"生成无人机0的异常分析报告"
"导出检测结果"
"生成Markdown格式的报告"
```

**报告内容结构：**

```
1. 执行摘要
   - 关键发现
   - 异常统计
   - 整体风险

2. 异常检测分析
   - 检测统计
   - 异常详情表

3. 异常影响评估
   - 整体评估
   - 各无人机详细分析
   - 潜在后果

4. 处置方案建议
   - 立即行动
   - 短期措施
   - 长期建议

5. 总结
   - 结论
   - 后续步骤
```

### 4. RAG知识库

#### 4.1 知识库结构

系统内置了四类知识库，用于支持影响评估：

**1. 异常类型知识库 (AnomalyTypeKnowledge)**
- 各类异常的定义和特征
- 检测特征和识别方法
- 典型原因分析

**2. 严重程度知识库 (SeverityKnowledge)**
- 严重程度分级标准
- 各级别的特征和处理建议
- 风险阈值定义

**3. 影响模式知识库 (ImpactPatternKnowledge)**
- 单机影响模式
- 局部传播模式
- 全局影响模式
- 碰撞风险模式

**4. 历史案例知识库 (HistoricalCaseKnowledge)**
- 历史异常事件记录
- 处理方法和经验
- 教训总结

#### 4.2 知识检索

系统使用简化的向量检索方法进行相关知识匹配：

```python
# 知识检索示例
from knowledge_base import DroneAnomalyKnowledgeBase

kb = DroneAnomalyKnowledgeBase('knowledge_base')

# 搜索相关知识
results = kb.search_all('位置偏离', top_k=3)

# 获取异常信息
info = kb.get_anomaly_info('position_deviation')

# 获取相似案例
cases = kb.get_similar_cases('sensor_malfunction', top_k=2)
```

#### 4.3 扩展知识库

可以通过以下方式添加新知识：

```python
# 添加新的历史案例
kb.historical_cases.add_knowledge({
    'type': 'historical_case',
    'case_id': 'CASE_NEW',
    'title': '新的异常案例',
    'description': '案例描述...',
    'anomaly_type': 'position_deviation',
    'severity': 'high',
    'resolution': ['解决方案1', '解决方案2'],
    'lessons_learned': ['经验1', '经验2']
})

# 保存知识库
kb.save_all()
```

### 5. 命令行参数

```bash
python main_agent.py [选项]

选项:
  -c, --config FILE     配置文件路径（JSON格式）
  -cmd, --command CMD   执行单个命令
  -b, --batch FILE      批量命令文件路径
  -d, --data-dir DIR    数据目录路径（默认: data）
  -m, --model-path PATH 模型文件路径（默认: results/best_model.pth）
  -o, --output-dir DIR  输出目录路径（默认: results）
  -v, --verbose         详细输出模式
```

**使用示例：**

```bash
# 执行单个命令
python main_agent.py --command "检测异常并生成报告"

# 批量执行命令
python main_agent.py --batch commands.txt

# 使用自定义配置
python main_agent.py --config my_config.json
```

### 6. 配置文件

配置文件使用JSON格式：

```json
{
    "data_dir": "data",
    "model_path": "results/best_model.pth",
    "output_dir": "results",
    "knowledge_base_path": "knowledge_base"
}
```

### 7. API参考

#### 7.1 DroneSwarmAgent

```python
from agent import DroneSwarmAgent

# 创建Agent
agent = DroneSwarmAgent(config)

# 处理单个输入
result = agent.process_input("检测异常")

# 批量执行
results = agent.execute_batch(["检测异常", "生成报告"])

# 交互式运行
agent.run_interactive()
```

#### 7.2 单独使用各Agent

```python
# 异常检测
from agent import AnomalyDetectionAgent
detector = AnomalyDetectionAgent(config)
result = detector.detect({'drone_ids': [0, 1, 2]})

# 影响评估
from agent import ImpactAssessmentAgent
assessor = ImpactAssessmentAgent(config)
result = assessor.assess(anomalies, {})

# 报告生成
from agent import ReportGenerationAgent
reporter = ReportGenerationAgent(config)
result = reporter.generate(detection_results, assessment_results, {})
```

### 8. 常见问题

**Q1: 模型文件不存在怎么办？**

A: 系统会自动使用新初始化的模型。为获得更好的检测效果，建议先运行训练：
```bash
python train.py
```

**Q2: 如何添加新的异常类型？**

A: 需要修改以下文件：
1. `knowledge_base/knowledge_base.py` - 添加异常类型知识
2. `agent/detection_agent.py` - 添加检测逻辑
3. `models/patch_gnn_model.py` - 如需要，调整模型输出

**Q3: 系统不理解我的指令怎么办？**

A: 尝试使用更明确的表达，或参考帮助信息中的示例指令：
```
>>> 请输入指令: 帮助
```

---

## English Documentation

### 1. System Overview

The UAV Swarm Anomaly Detection Agent System is an intelligent operation and maintenance platform based on natural language interaction. The system integrates anomaly detection, impact assessment, and report generation for UAV swarms into a unified Agent framework.

### 2. Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the agent system
python main_agent.py
```

### 3. Basic Commands

```
# Help
>>> help

# Anomaly Detection
>>> detect anomaly for all drones

# Impact Assessment
>>> assess the impact of detected anomalies

# Report Generation
>>> generate comprehensive report

# Exit
>>> quit
```

### 4. Key Features

1. **Natural Language Processing**: Supports both Chinese and English input
2. **Data Processing**: Load, clean, normalize, and visualize UAV data
3. **Anomaly Detection**: Patch-GNN based intelligent detection
4. **Impact Assessment**: RAG-based impact analysis
5. **Report Generation**: Structured analysis reports

### 5. Knowledge Base

The RAG-based knowledge base includes:
- Anomaly type definitions
- Severity level standards
- Impact patterns
- Historical cases

### 6. API Reference

```python
from agent import DroneSwarmAgent

# Create agent
agent = DroneSwarmAgent(config)

# Process input
result = agent.process_input("detect anomaly")

# Batch execute
results = agent.execute_batch(["detect anomaly", "generate report"])
```

---

## 更新日志 / Changelog

### v1.0.0 (2024-12-29)
- ✅ 初始版本发布
- ✅ 实现自然语言交互框架
- ✅ 实现数据预处理与可视化Agent
- ✅ 实现数据回放Agent
- ✅ 实现异常检测Agent
- ✅ 实现基于RAG的影响评估Agent
- ✅ 实现综合报告生成Agent
- ✅ 构建异常知识库

---

## 许可证 / License

MIT License
