# SmartRAG — 多 Agent 智能问答与视觉识别系统

基于 **LangGraph** 构建的多 Agent 状态机系统，支持 Supervisor 意图路由、RAG + Web 双路并行检索、自评审闭环，并扩展了 YOLO 视觉识别与天气查询能力。

---

## ✨ 核心特性

- **多 Agent 协作**：基于 LangGraph 的 StateGraph，6 个节点 + 5 条动态路由，实现任务编排与状态管理
- **混合检索**：Dense（语义）+ Sparse（BM25 关键词）双路召回，Qdrant 内部 RRF 融合
- **多查询扩展**：LLM 改写用户 query 为 N 个语义变体，线程池并行检索，MD5 去重 + 二次精排
- **Cross-Encoder 重排序**：基于 `BAAI/bge-reranker-base` 对 Top-20 候选精排到 Top-5
- **自评审闭环**：Critique Agent 审核生成结果，不通过则携带反馈打回重写，最多 N 轮防死循环
- **路由纠错环**：单路检索未命中时，自动补跑另一条链路（RAG ↔ Web）
- **多模态扩展**：YOLO11n 从 PyTorch → ONNX → TensorRT 部署，支持图片目标检测
- **流式输出**：FastAPI + SSE 实现逐 token 打字机效果，前端实时展示 Agent 执行进度
- **对话记忆**：基于 SQLite Checkpoint 持久化多轮对话历史

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────┐
│  React 前端 (ChatPage.tsx)                       │
│  图片上传 / SSE 流式接收 / 打字机渲染              │
└────────────────┬────────────────────────────────┘
                 │ POST /api/upload  (图片)
                 │ POST /api/query/stream  (问答)
┌────────────────▼────────────────────────────────┐
│  FastAPI 后端 (app/api.py)                       │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│  LangGraph 多 Agent 工作流 (app/graph/workflow.py)│
│                                                 │
│   START → supervisor ─┬─→ rag_node      ─┐      │
│                       ├─→ web_node      ─┼─→ synthesis_node ─→ critique_node ─→ END
│                       ├─→ both_node     ─┤        ↑                    │
│                       ├─→ weather_node  ─┤        └──── REVISE ────────┘
│                       └─→ detection_node─┘                              │
│                              │                                          │
│                         (路由纠错环)                                     │
└─────────────────────────────────────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┬────────────┐
    ▼            ▼            ▼            ▼
 Qdrant      智谱 GLM      Tavily      YOLO + TensorRT
(向量库)    (LLM+Embed)   (Web搜索)     (视觉识别)
```

---

## 🛠️ 技术栈

| 层级       | 技术                                                 |
| ---------- | ---------------------------------------------------- |
| 前端       | React 19 + TypeScript + Vite + TailwindCSS           |
| 后端       | Python 3.10 + FastAPI + Uvicorn + SSE                |
| Agent 编排 | LangGraph + LangChain                                |
| 大模型     | 智谱 GLM-4-Air（对话）+ embedding-3（向量化）        |
| 向量库     | Qdrant Cloud（混合检索：Dense 1024维 + Sparse BM25） |
| 重排序     | sentence-transformers + BAAI/bge-reranker-base       |
| 网络搜索   | Tavily API                                           |
| 视觉识别   | YOLO11n + ONNX Runtime + TensorRT 10.9               |
| 持久化     | SQLite（对话记忆）+ 本地文件（图片）                 |

---

## 📂 项目结构

```
agent/
├── app/
│   ├── api.py                    # FastAPI 路由、SSE 接口、图片上传
│   ├── config.py                 # 环境变量配置
│   ├── llm.py                    # LLM 工厂
│   ├── graph/
│   │   └── workflow.py           # LangGraph 状态机、节点、路由
│   ├── agents/                   # 各类 Agent（RAG/Web/Synthesis/Critique）
│   ├── rag/
│   │   ├── ingestion.py          # 文档解析、切分、双路向量化
│   │   ├── retriever.py          # 混合检索 + 多查询扩展 + 重排
│   │   └── reranker.py           # Cross-Encoder 重排序器
│   └── tools/
│       ├── registry.py           # 工具 Schema 注册
│       ├── weather_tool.py       # Open-Meteo 天气查询
│       └── vision_tool.py        # YOLO 目标检测
├── frontend/
│   └── src/
│       ├── pages/ChatPage.tsx    # 聊天界面
│       └── lib/api.ts            # 后端 API 封装
├── data/                         # SQLite checkpoints + 评估数据
├── docs/                         # 知识库文档
├── uploads/                      # 用户上传的图片
└── scripts/                      # 导入、评估脚本
```

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+
- Node.js 18+
- NVIDIA GPU（可选，用于 YOLO TensorRT 加速）

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入以下 Key：

```env
# 智谱 GLM
ZHIPUAI_API_KEY=your_key
ZHIPU_MODEL=glm-4-air
ZHIPU_EMBEDDING_MODEL=embedding-3

# Qdrant Cloud
QDRANT_URL=https://xxx.cloud.qdrant.io:6333
QDRANT_API_KEY=your_key
QDRANT_COLLECTION=smartrag_hybrid

# Tavily 搜索
TAVILY_API_KEY=your_key

# HuggingFace 缓存（可选，建议迁移到非 C 盘）
HF_HOME=D:\hf_cache
HF_ENDPOINT=https://hf-mirror.com
```

### 3. 启动后端

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.api:app --reload --port 8000
```

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

访问 `http://localhost:5173` 即可使用。

### 5. 导入知识库文档

```bash
python -c "from app.rag.ingestion import ingest_file; ingest_file('docs/company_docs.txt')"
```

---

## 📊 性能实测

在 RTX 4060 Laptop GPU 上的性能对比：

| 模型格式               | 单图推理耗时 | FPS      | 相对加速比 |
| ---------------------- | ------------ | -------- | ---------- |
| PyTorch (.pt)          | 25.32 ms     | 39.5     | 1.00×      |
| ONNX (.onnx)           | 21.58 ms     | 46.3     | 1.17×      |
| **TensorRT (.engine)** | **18.71 ms** | **53.4** | **1.35×**  |

RAG 检索链路优化：

- 混合检索（Dense + Sparse BM25）实现关键词与语义的双重召回
- Header-aware 切分策略将 **recall@k 从 0.50 提升至 1.00**
- 多查询扩展使用线程池并行，将 3 路串行检索从 4 秒降至 1.5 秒

---

## 🔧 我做了什么改造

本项目基于开源 LangGraph 多 Agent 框架进行二次开发，主要改造如下：

### 功能扩展

- ✅ 新增 **YOLO11n 视觉检测节点**，完成 PyTorch → ONNX → TensorRT 部署链路
- ✅ 新增 **天气查询工具**，接入 Open-Meteo API，支持从自然语言中提取城市名
- ✅ 前端新增 **图片上传与预览**，后端新增 `/api/upload` 接口
- ✅ 打通"图片上传 → 后端落盘 → YOLO 推理 → LLM 综合生成"全链路

### RAG 优化

- ✅ 实现 **多查询扩展**：LLM 改写 + 线程池并行 + MD5 去重 + 二次精排
- ✅ 修复 **分数覆盖 bug**：合并时保留首次命中 + 用原始 query 二次精排

### 问题排查

独立排查并解决 5+ 个工程问题：

1. **TensorRT DLL 缺失**：`nvinfer_10.dll` 加载失败，用 Dependencies 工具分析后安装 `nvidia-cudnn-cu12`
2. **LangGraph 数据流断链**：新增节点后未加入 `last_state` 追踪列表，导致前端报"工作流未返回结果"
3. **Function Calling 路由误判**：`glm-4-flash` 把内部业务问题误判为 Web 搜索，通过强化 Prompt + 代码兜底解决
4. **多查询分数覆盖**：合并去重时低分文档覆盖高分文档，导致弱命中误判
5. **HuggingFace 缓存迁移**：`HF_HOME` 迁移后 reranker 检测不到，需修改 `_model_is_cached` 优先读取环境变量

---

## 📝 License

本项目仅供学习交流使用。

---

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Qdrant](https://github.com/qdrant/qdrant)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [BAAI BGE-Reranker](https://github.com/FlagOpen/FlagEmbedding)