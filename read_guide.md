# Read Guide - 快速上手 How to Fix Your Context

## 项目概述

本项目是一个**上下文工程 (Context Engineering)** 教学项目，来源于 Drew Breunig 的博文 [How to Fix Your Context](https://www.dbreunig.com/2025/06/26/how-to-fix-your-context.html)，由 LangChain 团队维护。项目通过 6 个 Jupyter Notebook 演示了 6 种常见的上下文工程技术，所有实现基于 **LangGraph** 框架。

**核心问题**：LLM 的上下文窗口并非"越长越好"——随着输入长度增长，模型性能会退化（即 **Context Rot**），表现为：

| 失效模式 | 含义 |
|---------|------|
| Context Poisoning | 幻觉进入上下文并被反复引用 |
| Context Distraction | 上下文过长导致模型偏离训练知识 |
| Context Confusion | 冗余内容影响响应质量 |
| Context Clash | 上下文中出现矛盾信息 |

---

## 项目结构

```
how_to_fix_your_context/
├── README.md                              # 项目文档
├── requirements.txt                       # Python 依赖
├── read_guide.md                          # 本文件
└── notebooks/
    ├── utils.py                           # 工具函数（Rich 格式化输出）
    ├── 01-rag.ipynb                       # 技术1: 检索增强生成
    ├── 02-tool-loadout.ipynb              # 技术2: 动态工具选择
    ├── 03-context-quarantine.ipynb        # 技术3: 上下文隔离（多Agent）
    ├── 04-context-pruning.ipynb           # 技术4: 上下文裁剪
    ├── 05-context-summarization.ipynb     # 技术5: 上下文摘要
    └── 06-context-offloading.ipynb        # 技术6: 上下文卸载
```

---

## 架构概览

### LangGraph StateGraph 核心模式

所有 Notebook 共享同一套 LangGraph 架构模式：

```
┌─────────┐     ┌──────────┐     ┌────────────────┐     ┌──────────┐
│  START   │────>│ llm_call │────>│ conditional_edge│────>│   END    │
└─────────┘     └──────────┘     └────────────────┘     └──────────┘
                     ^                    │
                     │              tool_calls?
                     │                    │ Yes
                     │            ┌───────v──────┐
                     └────────────│  tool_node   │
                                  └──────────────┘
```

- **Nodes（节点）**：接收当前 State，执行逻辑，返回 State 更新
- **Edges（边）**：连接节点，支持线性、条件分支、循环
- **State（状态）**：节点间共享的数据结构（基于 `MessagesState`）

### 执行流程（通用）

```
1. 环境配置 → 设置 API Key
2. 数据准备 → 加载文档（RAG 场景）
3. 向量存储 → 创建 Embedding + 存储
4. 工具定义 → 定义/绑定工具
5. 图定义   → 创建 StateGraph，添加节点和边
6. 编译     → graph.compile()，可选传入 store/checkpointer
7. 执行     → graph.invoke() 传入初始消息
8. 展示     → utils.py 格式化输出结果
```

---

## 类结构

```
MessagesState (LangGraph 内置基类, 管理 messages 列表)
│
├── 直接使用 ─────────── 01-rag.ipynb
│
├── ToolLoadoutState ──── 02-tool-loadout.ipynb
│   └── + tools_by_name: Dict[str, Any]    # 跟踪已选工具
│
├── State ─────────────── 04-context-pruning.ipynb, 05-context-summarization.ipynb
│   └── + summary: str                      # 存储压缩/摘要后的内容
│
└── ScratchpadState ───── 06-context-offloading.ipynb
    └── + scratchpad: str (Pydantic Field)  # 临时草稿本

Pydantic BaseModel（工具参数定义）
├── WriteToScratchpad ─── 06-context-offloading.ipynb
└── ReadFromScratchpad ── 06-context-offloading.ipynb
```

---

## 6 大技术详解目录

### 1. RAG - 检索增强生成
**Notebook**: `notebooks/01-rag.ipynb`

| 项目 | 内容 |
|------|------|
| 核心思想 | 只检索与问题相关的信息加入上下文 |
| 关键技术 | RecursiveCharacterTextSplitter 分块、OpenAI Embedding、向量存储 + 检索工具 |
| LLM | Claude Sonnet（主推理）|
| Token 消耗 | ~25k tokens（基线） |
| 解决的问题 | 避免把全部文档塞入上下文 |

**技术要点**：
- 文档加载 → 文本分块 → Embedding → 向量存储 → 语义检索 → 作为工具绑定到 Agent

---

### 2. Tool Loadout - 动态工具选择
**Notebook**: `notebooks/02-tool-loadout.ipynb`

| 项目 | 内容 |
|------|------|
| 核心思想 | 只将语义相关的工具定义加入上下文，而非全部工具 |
| 关键技术 | 工具描述向量化、InMemoryStore 索引、语义搜索选 Top-5 工具 |
| 使用场景 | Python math 库所有函数作为工具，按需选取 |
| 解决的问题 | Context Confusion（工具描述重叠导致的混淆） |

**技术要点**：
- 工具注册表（UUID 映射）→ 向量化工具描述 → 按用户查询语义搜索 → 动态绑定 Top-5 工具
- 使用 `ToolLoadoutState` 跟踪每轮选择的工具

---

### 3. Context Quarantine - 上下文隔离
**Notebook**: `notebooks/03-context-quarantine.ipynb`

| 项目 | 内容 |
|------|------|
| 核心思想 | 每个 Agent 独立上下文窗口，互不干扰 |
| 关键技术 | LangGraph Supervisor 架构、多 Agent 协作、工具级 Handoff |
| Agent 组成 | Supervisor（调度）+ Math Expert（计算）+ Research Expert（搜索） |
| 解决的问题 | Context Clash（不同任务信息冲突） |

**技术要点**：
- Supervisor 决定任务分发 → 专家 Agent 在隔离上下文中执行 → 结果汇总
- 每个 Agent 有自己的系统提示词和专属工具集

---

### 4. Context Pruning - 上下文裁剪
**Notebook**: `notebooks/04-context-pruning.ipynb`

| 项目 | 内容 |
|------|------|
| 核心思想 | 用小模型剔除检索结果中与问题无关的内容 |
| 关键技术 | GPT-4o-mini 做裁剪、基于原始用户问题过滤、State 中增加 summary 字段 |
| 性能提升 | 同样查询：25k → 11k tokens（减少 56%） |
| 解决的问题 | Context Distraction + Context Confusion |

**技术要点**：
- 检索结果 → GPT-4o-mini 根据用户原始问题裁剪 → 只保留相关段落 → 压缩存入 State
- 裁剪在 tool_node 之后、llm_call 之前完成

---

### 5. Context Summarization - 上下文摘要
**Notebook**: `notebooks/05-context-summarization.ipynb`

| 项目 | 内容 |
|------|------|
| 核心思想 | 将冗长的检索内容浓缩为精炼摘要 |
| 关键技术 | GPT-4o-mini 做摘要、目标压缩比 50-70%、保留所有关键信息 |
| 与 Pruning 的区别 | Pruning 删除无关内容；Summarization 压缩所有内容 |
| 解决的问题 | Context Distraction（内容相关但过于冗长） |

**技术要点**：
- 适用场景：检索到的内容都相关，但太长太啰嗦
- 摘要 Prompt 强调保留关键数据点和事实，去除冗余表述

---

### 6. Context Offloading - 上下文卸载
**Notebook**: `notebooks/06-context-offloading.ipynb`

| 项目 | 内容 |
|------|------|
| 核心思想 | 将信息存储到上下文窗口之外，按需读取 |
| 关键技术 | ScratchpadState、InMemoryStore、InMemorySaver checkpointer |
| 两种模式 | Session Scratchpad（会话内临时存储）、Persistent Memory（跨线程持久存储） |
| 解决的问题 | Context Rot（长时间累积导致的退化） |

**技术要点**：
- **Scratchpad 模式**：`ScratchpadState.scratchpad` 字段 + `WriteToScratchpad` / `ReadFromScratchpad` 工具
- **Memory 模式**：`InMemoryStore` 命名空间 + Key-Value 存储，跨 thread_id 可访问
- 类似 ChatGPT Memory 和 Anthropic 多 Agent 研究员的实现思路

---

## 关键依赖

| 依赖 | 作用 |
|------|------|
| `langgraph >= 0.5.4` | 核心：StateGraph 工作流编排 |
| `langchain >= 0.3.0` | LLM 应用框架、工具绑定 |
| `langchain-anthropic >= 0.3.0` | Claude Sonnet 接入 |
| `langchain-openai >= 0.2.0` | OpenAI Embedding + GPT-4o-mini |
| `langgraph_supervisor >= 0.0.27` | 多 Agent Supervisor 模式 |
| `langchain-tavily >= 0.2.10` | Web 搜索工具（Notebook 06） |
| `pydantic >= 2.0.0` | 数据验证 + 工具参数定义 |
| `rich >= 14.0.0` | 终端格式化输出 |

## 环境变量

```bash
export OPENAI_API_KEY="..."       # OpenAI Embedding + GPT-4o-mini
export ANTHROPIC_API_KEY="..."    # Claude Sonnet 主模型
export TAVILY_API_KEY="..."       # Web 搜索（仅 Notebook 06 需要）
```

---

## 快速上手

```bash
# 1. 克隆并进入项目
git clone https://github.com/langchain-ai/how_to_fix_your_context
cd how_to_fix_your_context

# 2. 创建虚拟环境
uv venv && source .venv/bin/activate

# 3. 安装依赖
uv pip install -r requirements.txt

# 4. 设置环境变量
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# 5. 启动 Jupyter
jupyter notebook notebooks/
```

**推荐阅读顺序**：`01-rag` → `04-context-pruning` → `05-context-summarization` → `02-tool-loadout` → `03-context-quarantine` → `06-context-offloading`

理由：先掌握 RAG 基线，然后学习两种上下文压缩手段（Pruning/Summarization），再看工具层面优化（Tool Loadout），接着理解多 Agent 隔离（Quarantine），最后学习外部存储（Offloading）。

---

## 技术点速查索引

| 技术点 | Notebook | 相关类/函数 |
|--------|----------|-------------|
| 文档分块 | 01 | `RecursiveCharacterTextSplitter` |
| 向量存储与检索 | 01, 02 | `InMemoryVectorStore`, `init_embeddings` |
| 自定义 State | 02, 04, 05, 06 | `ToolLoadoutState`, `State`, `ScratchpadState` |
| 条件边路由 | 01-06 | `tools_condition` / 自定义条件函数 |
| 动态工具绑定 | 02 | `InMemoryStore(index=...)`, `llm.bind_tools()` |
| 多 Agent Supervisor | 03 | `create_supervisor()` |
| LLM 辅助裁剪 | 04 | GPT-4o-mini + pruning prompt |
| LLM 辅助摘要 | 05 | GPT-4o-mini + summarization prompt |
| 会话内草稿本 | 06 | `ScratchpadState`, Pydantic 工具模型 |
| 跨线程持久记忆 | 06 | `InMemoryStore`, `InMemorySaver` |
| Rich 格式化输出 | all | `notebooks/utils.py` |
