# LangChain/LangGraph 技术实现文档

## Technical Implementation with LangChain/LangGraph

本文档详细说明了项目中 LangChain 和 LangGraph 的实现细节。

---

## 目录

1. [架构概述](#架构概述)
2. [LangChain RAG 实现](#langchain-rag-实现)
3. [LangGraph Agent 编排](#langgraph-agent-编排)
4. [集成说明](#集成说明)
5. [配置选项](#配置选项)
6. [最佳实践](#最佳实践)
7. [故障排除](#故障排除)

---

## 架构概述

### 为什么选择 LangChain/LangGraph?

1. **成熟的框架**: LangChain 是业界领先的 LLM 应用框架
2. **RAG 支持**: 内置向量存储、检索器等 RAG 组件
3. **状态管理**: LangGraph 提供强大的状态化 Agent 编排
4. **社区支持**: 活跃的社区和丰富的文档
5. **可扩展性**: 易于集成新的模型和工具

### 技术栈版本

```
langchain >= 0.3.0
langchain-core >= 0.3.0
langchain-community >= 0.3.0
langgraph >= 0.2.0
chromadb >= 0.5.0
sentence-transformers >= 2.2.0
```

---

## LangChain RAG 实现

### 1. 知识库设计

#### 文件位置
`knowledge_base/knowledge_base_langchain.py`

#### 核心组件

```python
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
```

#### 知识库结构

```
DroneAnomalyKnowledgeBase
│
├── Vector Stores (Chroma)
│   ├── anomaly_types      # 异常类型知识
│   ├── severity_levels    # 严重程度知识
│   ├── impact_patterns    # 影响模式知识
│   └── historical_cases   # 历史案例知识
│
├── Embeddings (Sentence-Transformers/Fake)
│   └── 384-dimensional vectors
│
└── Persistence
    └── knowledge_base/ directory
        ├── anomaly_types/
        ├── severity_levels/
        ├── impact_patterns/
        └── historical_cases/
```

### 2. 向量存储实现

#### ChromaDB 配置

```python
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=self.embeddings,
    collection_name=collection_name,
    persist_directory=persist_directory
)
```

**优势**:
- 本地持久化存储
- 无需外部服务
- 快速相似度搜索
- 支持元数据过滤

#### 嵌入策略

**生产环境** (需要网络):
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```

**离线环境**:
```python
from langchain_community.embeddings import FakeEmbeddings

embeddings = FakeEmbeddings(size=384)
```

### 3. 文档结构

每个知识条目包含：

```python
Document(
    page_content="""
    标题: XXX
    描述: XXX
    特征: XXX
    ...
    """,
    metadata={
        'type': 'xxx',
        'name_cn': 'xxx',
        'name_en': 'xxx',
        # 注意: 列表需要转为 JSON 字符串
        'characteristics': json.dumps([...])
    }
)
```

**重要**: ChromaDB 元数据只支持标量类型 (str, int, float, bool)，列表需要序列化为 JSON。

### 4. 检索方法

#### 相似度搜索

```python
results = vector_store.similarity_search_with_score(
    query="位置偏离",
    k=3
)
# 返回: List[Tuple[Document, float]]
```

#### MMR 搜索 (多样性)

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)
docs = retriever.get_relevant_documents("GPS故障")
```

#### 阈值过滤

```python
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7}
)
```

### 5. 知识库初始化流程

```
1. 初始化嵌入模型
   ├─ 尝试加载 Sentence-Transformers
   └─ 失败则使用 FakeEmbeddings

2. 创建/加载向量存储
   ├─ 检查持久化目录
   ├─ 存在则加载
   └─ 不存在则创建并初始化

3. 初始化各知识库
   ├─ 异常类型 (5条)
   ├─ 严重程度 (3条)
   ├─ 影响模式 (4条)
   └─ 历史案例 (5条)

4. 准备检索器
   └─ 为每个集合创建 Retriever
```

---

## LangGraph Agent 编排

### 1. StateGraph 架构

#### 文件位置
`agent/core_langgraph.py`

#### 状态定义

```python
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    # 使用 Annotated 和 operator.add 实现消息累积
    messages: Annotated[List[BaseMessage], operator.add]
    
    # 任务相关
    task_type: Optional[str]
    task_parameters: Optional[Dict[str, Any]]
    
    # 状态标记
    data_loaded: bool
    detected_anomalies: List[Dict[str, Any]]
    impact_assessments: List[Dict[str, Any]]
    report: Optional[str]
    
    # 路由控制
    next_action: Optional[str]
```

**关键点**:
- `Annotated[List, operator.add]`: 消息自动累积，不会覆盖
- TypedDict 提供类型检查
- 状态在各节点间传递

### 2. 工作流图

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                         ▼
                    ┌─────────┐
              ┌────▶│ Router  │◀────┐
              │     └────┬────┘     │
              │          │          │
              │          ▼          │
              │   Conditional Edges │
              │   ┌──┬──┬──┬──┬──┐ │
              │   │  │  │  │  │  │ │
          ┌───┴───▼──▼──▼──▼──▼──▼─┴───┐
          │  Data  Anomaly  Impact       │
          │ Loader Detector Assessor     │
          │         │       │            │
          │    Reporter  Responder       │
          └─────────┬───────┬──────────┬─┘
                    │       │          │
                    └───────┴──────────┼──▶ END
```

### 3. Agent 节点实现

#### Router Agent (路由器)

```python
def _router_agent(self, state: AgentState) -> AgentState:
    """
    分析用户输入，决定下一步操作
    """
    last_message = state['messages'][-1].content.lower()
    
    # 简单的关键词匹配
    if '检测' in last_message:
        if not state.get('data_loaded'):
            state['next_action'] = 'load_data'
        else:
            state['next_action'] = 'detect_anomaly'
    # ... 其他路由逻辑
    
    return state
```

#### Data Loader Agent

```python
def _data_loader_agent(self, state: AgentState) -> AgentState:
    """
    加载数据
    """
    data_path = os.path.join(self.config['data_dir'], 'test_data.csv')
    
    if os.path.exists(data_path):
        state['data_loaded'] = True
        state['messages'].append(
            AIMessage(content=f"✓ 数据已加载: {data_path}")
        )
    
    return state
```

#### Impact Assessor Agent (RAG集成)

```python
def _impact_assessor_agent(self, state: AgentState) -> AgentState:
    """
    使用 RAG 评估影响
    """
    anomalies = state.get('detected_anomalies', [])
    
    for anomaly in anomalies:
        # 1. 检索相关知识
        knowledge = self.knowledge_base.search_all(
            anomaly['anomaly_type'], 
            top_k=1
        )
        
        # 2. 获取相似案例
        similar_cases = self.knowledge_base.get_similar_cases(
            anomaly['description'],
            top_k=2
        )
        
        # 3. 基于知识生成评估
        assessment = {
            'drone_id': anomaly['drone_id'],
            'impact_level': self._determine_impact(anomaly, knowledge),
            'recommendations': self._generate_recommendations(
                anomaly, knowledge
            ),
            'similar_cases': similar_cases
        }
        
        state['impact_assessments'].append(assessment)
    
    return state
```

### 4. 条件路由

```python
def _route_decision(self, state: AgentState) -> str:
    """
    根据 next_action 决定下一个节点
    """
    return state.get('next_action', 'respond')

# 在图构建时配置
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
```

### 5. 图编译和执行

```python
# 编译图
self.app = self.graph.compile()

# 执行
result = self.app.invoke(initial_state)
```

---

## 集成说明

### 1. 集成现有检测模型

在 `_anomaly_detector_agent` 中集成 Patch-GNN:

```python
from models.patch_gnn_model import create_model
import torch

def _anomaly_detector_agent(self, state: AgentState) -> AgentState:
    # 加载模型
    model = create_model(...)
    model.load_state_dict(torch.load(self.config['model_path']))
    model.eval()
    
    # 加载数据
    data = self._load_test_data(state['data_path'])
    
    # 推理
    with torch.no_grad():
        predictions = model(data)
    
    # 后处理
    anomalies = self._process_predictions(predictions)
    state['detected_anomalies'] = anomalies
    
    return state
```

### 2. 集成 LLM (可选)

如果需要使用大语言模型增强响应:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 初始化 LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是无人机集群异常分析专家。"),
    ("human", "{input}")
])

# 创建链
chain = prompt | llm | StrOutputParser()

# 在 Agent 中使用
response = chain.invoke({"input": user_message})
```

### 3. 数据流

```
用户输入
  │
  ├─▶ Router Agent
  │     │
  │     ├─▶ 解析意图
  │     └─▶ 设置 next_action
  │
  ├─▶ Data Loader
  │     └─▶ 加载 CSV 数据
  │
  ├─▶ Anomaly Detector
  │     ├─▶ 加载 Patch-GNN 模型
  │     ├─▶ 执行推理
  │     └─▶ 提取异常
  │
  ├─▶ Impact Assessor
  │     ├─▶ 查询 RAG 知识库
  │     │     ├─▶ 异常类型知识
  │     │     ├─▶ 严重程度知识
  │     │     ├─▶ 影响模式知识
  │     │     └─▶ 历史案例
  │     ├─▶ 计算影响等级
  │     └─▶ 生成建议
  │
  ├─▶ Report Generator
  │     ├─▶ 汇总检测结果
  │     ├─▶ 汇总评估结果
  │     └─▶ 生成 Markdown 报告
  │
  └─▶ Responder
        └─▶ 格式化输出给用户
```

---

## 配置选项

### 1. 环境变量

创建 `.env` 文件:

```bash
# 数据路径
DATA_DIR=data
MODEL_PATH=results/best_model.pth
OUTPUT_DIR=results
KNOWLEDGE_BASE_PATH=knowledge_base

# LLM 配置 (可选)
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# 嵌入模型配置
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

# ChromaDB 配置
CHROMA_PERSIST_DIR=knowledge_base
```

### 2. 配置字典

```python
config = {
    'data_dir': 'data',
    'model_path': 'results/best_model.pth',
    'output_dir': 'results',
    'knowledge_base_path': 'knowledge_base',
    
    # Agent 配置
    'agent_type': 'langgraph',  # 或 'legacy'
    
    # RAG 配置
    'rag': {
        'embedding_model': 'all-MiniLM-L6-v2',
        'vector_store': 'chroma',
        'search_type': 'similarity',  # 或 'mmr'
        'top_k': 3
    },
    
    # 检测配置
    'detection': {
        'window_size': 50,
        'stride': 25,
        'threshold': 0.5
    }
}
```

---

## 最佳实践

### 1. 知识库管理

```python
# 定期更新知识库
kb = DroneAnomalyKnowledgeBase(kb_path)

# 添加新案例
new_case = {
    'case_id': 'CASE_006',
    'title': '新案例',
    'description': '...',
    # ...
}

kb.add_document(
    'historical_cases',
    content=format_case(new_case),
    metadata=new_case
)

# 重建索引（如果需要）
kb.vector_stores['historical_cases'].persist()
```

### 2. Agent 状态持久化

```python
# 保存状态用于后续恢复
import pickle

def save_state(state: AgentState, path: str):
    with open(path, 'wb') as f:
        pickle.dump(state, f)

def load_state(path: str) -> AgentState:
    with open(path, 'rb') as f:
        return pickle.load(f)
```

### 3. 性能优化

```python
# 1. 批量检索
queries = [anomaly['description'] for anomaly in anomalies]
results = kb.batch_search(queries, top_k=2)

# 2. 缓存嵌入
from functools import lru_cache

@lru_cache(maxsize=128)
def get_embedding(text: str):
    return embeddings.embed_query(text)

# 3. 异步处理
import asyncio

async def process_anomalies_async(anomalies):
    tasks = [assess_anomaly(a) for a in anomalies]
    return await asyncio.gather(*tasks)
```

### 4. 错误处理

```python
def _anomaly_detector_agent(self, state: AgentState) -> AgentState:
    try:
        # 检测逻辑
        anomalies = self._detect(state['data_path'])
        state['detected_anomalies'] = anomalies
        state['messages'].append(
            AIMessage(content=f"✓ 检测完成: {len(anomalies)}个异常")
        )
    except FileNotFoundError:
        state['messages'].append(
            AIMessage(content="✗ 错误: 数据文件不存在")
        )
    except Exception as e:
        state['messages'].append(
            AIMessage(content=f"✗ 错误: {str(e)}")
        )
    
    return state
```

---

## 故障排除

### 问题 1: ChromaDB 元数据错误

**错误**:
```
ValueError: Expected metadata value to be a str, int, float, bool, or None, got [...] which is a list
```

**解决**:
```python
# 将列表转为 JSON 字符串
import json

metadata = {
    'causes': json.dumps(causes_list, ensure_ascii=False)
}
```

### 问题 2: 嵌入模型下载失败

**错误**:
```
Failed to resolve 'huggingface.co'
```

**解决**:
```python
# 使用 FakeEmbeddings 用于离线环境
from langchain_community.embeddings import FakeEmbeddings

embeddings = FakeEmbeddings(size=384)
```

### 问题 3: 状态未持久化

**问题**: 每次调用 `process_message` 状态都重置

**解决**:
```python
class DroneSwarmAgentGraph:
    def __init__(self, config):
        self.persistent_state = None
    
    def process_message(self, message: str) -> str:
        # 使用或创建持久状态
        if self.persistent_state is None:
            initial_state = self._create_initial_state(message)
        else:
            # 添加新消息到现有状态
            initial_state = self.persistent_state
            initial_state['messages'].append(
                HumanMessage(content=message)
            )
        
        result = self.app.invoke(initial_state)
        self.persistent_state = result
        
        return self._extract_response(result)
```

### 问题 4: 内存占用过高

**解决**:
```python
# 1. 限制消息历史长度
def _trim_messages(state: AgentState) -> AgentState:
    if len(state['messages']) > 20:
        state['messages'] = state['messages'][-20:]
    return state

# 2. 定期清理向量存储
kb.vector_stores['collection'].delete(
    where={"timestamp": {"$lt": cutoff_time}}
)

# 3. 使用更小的嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)
```

---

## 参考资源

- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [ChromaDB 文档](https://docs.trychroma.com/)
- [Sentence-Transformers](https://www.sbert.net/)

---

## 更新日志

### v2.0.0 (2024-12-29)
- ✅ 初始 LangChain/LangGraph 实现
- ✅ ChromaDB 向量存储集成
- ✅ StateGraph Agent 编排
- ✅ RAG 知识库系统
- ✅ 离线模式支持

---

**作者**: AI Agent  
**许可**: MIT License
