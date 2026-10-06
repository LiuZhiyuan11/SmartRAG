# OmniRAG 项目结构与功能扩展教程

本文按当前仓库结构介绍一次查询如何经过前端、FastAPI 和 LangGraph，并以“用户整句询问天气”为例，说明如何让监督者提取城市、传递工作流状态并异步调用天气工具。

## 1. 项目目录导览

```text
app/
├── api.py                 # FastAPI 路由、请求模型、SSE 接口
├── config.py              # 环境变量与应用配置
├── graph/
│   └── workflow.py        # LangGraph 状态、节点、路由和工作流入口
├── agents/                # RAG、Web、综合回答、评审 Agent
├── rag/                   # 文档导入、检索、重排、评估
├── tools/
│   ├── registry.py        # 路由工具 schema 与 LangChain 工具
│   ├── weather_tool.py    # Open-Meteo 地理编码与天气查询
│   └── vision_tool.py     # 图片目标检测
└── mcp/
    └── server.py          # 对外提供 MCP 工具
frontend/
└── src/
    ├── lib/api.ts         # 浏览器调用后端 API、解析 SSE
    └── pages/ChatPage.tsx # 聊天界面
tests/                     # pytest 测试
docs/                      # 项目说明和开发文档
scripts/                   # 导入、数据准备等命令行脚本
data/                      # checkpoint、评估数据和运行数据
```

常见修改位置：

| 要修改的内容 | 主要文件 |
| --- | --- |
| HTTP 请求字段或 API 行为 | `app/api.py` |
| 查询如何分配给知识库、Web 或其他能力 | `app/graph/workflow.py`、`app/tools/registry.py` |
| 某个工具如何访问外部服务 | `app/tools/` |
| 最终答案的组织方式 | `app/agents/synthesis_agent.py` |
| 浏览器请求格式或 SSE 展示 | `frontend/src/lib/api.ts`、`frontend/src/pages/ChatPage.tsx` |
| 功能回归测试 | `tests/` |

## 2. 一条聊天请求经过哪些代码

以网页端发送“永州今天天气怎么样”为例：

1. `ChatPage.tsx` 收集用户输入，并调用 `frontend/src/lib/api.ts` 的 `streamQuery`。
2. `streamQuery` 把整句放进 JSON 的 `query` 字段，发送到 `/api/query/stream`。前端不需要先把城市拆出来。
3. `app/api.py` 的 `query_stream` 接收请求，并把 `query` 交给 `stream_query`。
4. `app/graph/workflow.py` 的 `stream_query` 建立初始状态并启动 LangGraph。
5. `supervisor_node` 读取整句和对话历史，调用带有 `ROUTING_TOOLS` 的 LLM，决定路由和工具参数。
6. LangGraph 根据 `TOOL_TO_ROUTE` 把天气请求送到 `weather_node`。
7. 天气节点调用 `registry.py` 中的天气工具；天气结果进入 `weather_context`。
8. `synthesis_node` 将问题和天气上下文交给 `SynthesisAgent` 生成回答，SSE 最终事件返回给前端。

这里有两种不同的“工具”概念：

- `ROUTING_TOOLS` 是给监督者 LLM 使用的工具描述。它让 LLM 返回诸如工具名称和参数的结构化决定；它本身不负责执行天气查询。
- 用 `@tool` 装饰的 `get_weather` 是可执行的 LangChain `StructuredTool`。它应通过 `.invoke(...)` 或 `.ainvoke(...)` 执行，不能像普通 Python 函数一样直接调用。

## 3. 让监督者从整句提取城市

不要在前端要求用户额外提交一个城市字段，也不要用简单字符串切割从整句话猜城市。把天气工具的路由 schema 定义成需要 `city` 的结构化参数，让监督者从原始问题和对话历史中抽取：

```python
{
    "name": "get_weather",
    "description": "查询指定城市的当前天气。city 必须是城市名称；从用户问题或对话历史中提取，不要把整句问题作为城市。",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，例如永州、上海",
            },
        },
        "required": ["city"],
    },
}
```

因此，用户输入“永州今天天气怎么样”时，期望监督者返回的工具调用近似于：

```json
{
  "name": "get_weather",
  "args": {
    "city": "永州"
  }
}
```

用户只问“今天天气怎么样”时，监督者不应凭空猜城市。可以先尝试从传入的对话历史中解析城市；如果仍没有明确城市，就让工作流返回“请提供要查询的城市”，不要请求地理编码服务。

## 4. 把提取的城市放进 LangGraph state

监督者的 LLM 只返回工具调用，不会自动修改 LangGraph state。需要在 `supervisor_node` 中读取工具参数，再把城市写进节点返回的 state。

当前工作区的 `app/graph/workflow.py` 已经在 `AgentState` 和 `_initial_state` 中声明并初始化 `weather_city`，不需要重复添加。下面列出这两个位置，是为了说明新增 state 字段时需要同时维护的地方：

```python
class AgentState(TypedDict):
    # 其他字段……
    weather_city: str
    weather_context: str


def _initial_state(query: str) -> AgentState:
    return {
        # 其他初始字段……
        "weather_city": "",
        "weather_context": "",
    }
```

然后在 `supervisor_node` 提取参数。以下是核心逻辑示意；保留项目中现有的路由、日志和其他 state 字段：

```python
tool_call = tool_calls[0]
tool_name = tool_call["name"]
tool_args = tool_call.get("args") or {}

weather_city = state.get("weather_city", "")
if tool_name == "get_weather":
    weather_city = str(tool_args.get("city") or "").strip()

return {
    **state,
    "route": route,
    "direct_answer": direct,
    "weather_city": weather_city,
}
```

`**state` 会保留已有状态；新字段必须同时出现在 `AgentState`、初始 state 和节点返回值中，才能沿 LangGraph 节点传递。

## 5. 在天气节点中异步执行工具

天气工具和其底层 Open-Meteo 函数都是异步的，不需要用 `asyncio.to_thread`。使用 `StructuredTool.ainvoke`，参数名必须与 `@tool` 函数的输入 schema 一致：

```python
city = state.get("weather_city", "").strip()
if not city:
    context = "请提供要查询天气的城市。"
else:
    result = await get_weather.ainvoke({"city": city})
    context = json.dumps(result, ensure_ascii=False)
```

`get_weather` 的底层函数返回字典，而 `weather_context` 会作为提示词文本传给综合 Agent，因此示例将结果序列化为 JSON 字符串。实际节点还应记录耗时，并按现有工作流格式返回：

```python
return {
    **state,
    "weather_context": context,
    "weather_ran": True,
}
```

关键点：

- 正确：`await get_weather.ainvoke({"city": city})`
- 不正确：`await asyncio.to_thread(get_weather, state["query"])`
- 也不正确：`get_weather(city)`，因为被 `@tool` 装饰后，它是 `StructuredTool` 而不是普通可调用函数。

如果选择直接调用 `app.tools.weather_tool.get_weather(city)`，则应调用底层异步函数，而不是同时把它当作 `StructuredTool` 使用。建议在同一条工作流里选定一种调用方式；当前项目已采用 LangChain 工具包装，使用 `.ainvoke` 更一致。

## 6. 检查图的连线与答案生成

确认 `build_graph` 中已注册天气节点，并且 supervisor 的条件路由包含目标节点：

```python
graph.add_node("weather_node", weather_node)
# supervisor 的条件路由需要包含 "weather_node"
graph.add_edge("weather_node", "synthesis_node")
```

同时确认：

- `TOOL_TO_ROUTE["get_weather"]` 的值与 `route_supervisor` 识别的路由值一致。
- `SynthesisAgent` 的提示词和 `arun_stream` 输入包含 `weather_context`。
- `synthesis_node` 调用 `arun_stream` 时，参数顺序与 `SynthesisAgent.arun_stream` 定义一致。参数很多时可考虑改成具名参数，减少后续插入新上下文导致的位置错乱。
- 如果要在最终 API 响应中单独展示天气原始数据，还需要把字段加入 `_extract_result`、API 响应模型和前端类型；如果只展示综合后的回答，则不必额外暴露它。

## 7. 常见错误排查

### `'StructuredTool' object is not callable`

原因：把 `@tool` 装饰后的对象当普通函数调用。检查调用处，异步执行用 `.ainvoke({...})`，同步执行用 `.invoke({...})`。

### 天气服务提示找不到地点

原因：传入的是整句问题，而不是城市名；或监督者没有从问题 / 历史中提取出城市。检查日志中监督者的工具参数和 state 的 `weather_city`，不要只检查最终异常。

### 工具参数名称错误

原因：schema 声明 `city`，调用时却传 `query` 或 `location`。确保 `ROUTING_TOOLS` 的参数名、`@tool` 函数签名和 `.ainvoke` 字典键完全一致。

### 综合节点参数数量或顺序错误

原因：`SynthesisAgent.arun_stream` 签名改变了，但工作流仍按旧的位置参数调用。比较 `app/agents/synthesis_agent.py` 的签名与 `workflow.py` 中的调用。

## 8. 本地运行与验证

在仓库根目录启动后端：

```powershell
uvicorn app.api:app --reload --port 8000
```

另开终端启动前端：

```powershell
Set-Location frontend
npm run dev
```

验证顺序建议如下：

1. 查询“永州今天天气怎么样”，检查监督者选择 `get_weather`，并返回城市参数 `永州`。
2. 检查 `weather_node` 收到的 `weather_city` 是城市名，而不是整句查询。
3. 检查天气节点通过 `.ainvoke({"city": city})` 得到天气结果，再由综合节点生成回答。
4. 查询“今天天气怎么样”，确认缺少城市时会询问用户，而不是猜测或把整句送去地理编码。
5. 在 `tests/` 为参数提取和缺少城市的分支添加单元测试；测试中 mock 天气工具，避免依赖真实网络服务。
