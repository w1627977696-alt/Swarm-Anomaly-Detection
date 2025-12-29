# LangChain/LangGraph 版本说明

## 正确的版本信息

本项目使用以下 LangChain 和 LangGraph 版本：

### 主要框架

| 包名 | 版本 | 说明 |
|------|------|------|
| langchain | 1.2.0 | 主框架，1.X 最新稳定版 |
| langgraph | 1.0.5 | Agent 编排，1.X 最新稳定版 |

### 依赖包

| 包名 | 版本范围 | 说明 |
|------|---------|------|
| langchain-core | >=1.2.0,<2.0.0 | 核心组件，由 langchain 自动安装 |
| langchain-community | >=0.4.0,<1.0.0 | 社区集成，**注意：没有 1.X 版本** |
| langgraph-checkpoint | >=3.0.0,<4.0.0 | LangGraph 检查点，由 langgraph 自动安装 |
| langgraph-prebuilt | >=1.0.0,<2.0.0 | LangGraph 预构建组件，由 langgraph 自动安装 |

## 重要说明

### 1. langchain-community 的版本

**langchain-community 没有 1.X 版本！**

- 最新版本是 **0.4.1**
- 这是正常的，因为 langchain-community 使用不同的版本号体系
- 它与 LangChain 1.2.0 完全兼容

### 2. 自动管理的依赖

安装 `langchain==1.2.0` 和 `langgraph==1.0.5` 时，pip 会自动安装兼容的版本：

```bash
pip install langchain==1.2.0 langgraph==1.0.5
```

会自动安装：
- langchain-core (通常是 1.2.5 或更高)
- langgraph-checkpoint (通常是 3.0.1 或更高)
- langgraph-prebuilt (通常是 1.0.5 或更高)

### 3. 不需要的包

以下包已从 requirements.txt 中移除或标记为可选：

- ❌ **langchain-openai**: 代码中未使用
- 🔄 **openai, tiktoken**: 标记为可选，需要时取消注释
- 🔄 **pypdf, python-docx**: 标记为可选，需要时取消注释

## 验证安装

运行以下命令验证版本：

```bash
python -c "
import langchain
import langgraph
import langchain_core
import langchain_community

print(f'LangChain: {langchain.__version__}')
print(f'LangGraph: {langgraph.__version__}')
print(f'LangChain-Core: {langchain_core.__version__}')
print(f'LangChain-Community: {langchain_community.__version__}')
"
```

期望输出：
```
LangChain: 1.2.0
LangGraph: 1.0.5
LangChain-Core: 1.2.x (x >= 0)
LangChain-Community: 0.4.x (x >= 0)
```

## API 兼容性

所有使用的 API 都与 LangChain 1.2.0 和 LangGraph 1.0.5 完全兼容：

### LangChain Core APIs ✅
- `from langchain_core.documents import Document`
- `from langchain_core.embeddings import Embeddings`
- `from langchain_core.vectorstores import VectorStore`
- `from langchain_core.messages import BaseMessage, HumanMessage, AIMessage`
- `from langchain_core.prompts import ChatPromptTemplate`
- `from langchain_core.output_parsers import StrOutputParser`

### LangChain Community APIs ✅
- `from langchain_community.embeddings import FakeEmbeddings, HuggingFaceEmbeddings`
- `from langchain_community.vectorstores import Chroma`

### LangGraph APIs ✅
- `from langgraph.graph import StateGraph, END`
- `from langgraph.prebuilt import ToolNode`

## 更新历史

- **2024-12-29**: 修正版本号，langchain-community 使用 0.4.x 而非 1.0.x
- **2024-12-29**: 更新到 LangChain 1.2.0 和 LangGraph 1.0.5
- **2024-12-29**: 标记 openai 相关包为可选依赖

## 参考文档

- [LangChain 1.2.0 发布说明](https://python.langchain.com/docs/changelog)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [LangChain Community](https://python.langchain.com/docs/integrations/platforms/)
