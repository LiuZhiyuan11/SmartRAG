"""
OmniRAG — 工具注册中心
──────────────────────────
统一管理 Function Calling 与 MCP 工具的 schema 定义。

设计原则：
  - 一份 schema，两处使用：workflow 监督者用 bind_tools 做路由决策，
    MCP server 用 mcp.types.Tool 暴露给外部宿主。
  - 路由决策型工具（workflow 内部）：LLM 选择工具 → 映射到 LangGraph 路由
  - 执行型工具（MCP 对外）：full_query / evaluate 宿主 LLM 自己做不到的能力

路由工具与执行工具的边界：
  - rag_search / web_search / parallel_search / direct_answer
    → workflow 内部路由决策，不在此实际执行检索（执行在 LangGraph 各节点）
  - full_query / evaluate
    → MCP 对外暴露，宿主 LLM 调用后由本服务跑完整管道
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
import asyncio

# ── 路由决策工具 schema（OpenAI function calling 格式）──────────────────────
# LangChain bind_tools 接受此 dict 列表，GLM-4-Flash 会返回 tool_calls。

ROUTING_TOOLS: list[dict[str, Any]] = [
    {
        "name": "rag_search",
        "description": (
            "搜索内部知识库。本系统知识库只含用户上传的文档"
            "（项目报告、论文、笔记、README）。当查询涉及这些文档中的内容"
            "——如报告数据、文档结论、论文观点——必须使用本工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要检索的问题"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "搜索实时网络获取外部信息。适用于最新信息、当前事件、新闻、"
            "外部产品或 AI 模型、公司动态、公众人物、通用技术概念等"
            "知识库中没有的内容。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "网络搜索查询"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "parallel_search",
        "description": (
            "RAG 知识库与 Web 网络双路并行检索。适用于需要同时结合"
            "用户上传文档和外部实时信息的研究型问题。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "研究问题"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "direct_answer",
        "description": (
            "直接回答，不检索。仅限纯问候、寒暄、简单算术等不依赖任何"
            "外部信息的问题（如'你好'、'1+1'）。任何需要事实依据的问题——"
            "包括文档内容、实时信息（天气、新闻、股价、赛事）、具体数据——"
            "都不得使用本工具，必须选择检索工具。禁止用自身知识编造事实。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "问题"},
            },
            "required": ["query"],
        },
    },

{
        "name": "get_weather",
        "description": (
            "查询指定城市的当前天气，包括天气状况、气温、体感温度、"
            "湿度、风速和降水量。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string",
                         "description": "查询指定城市的当前天气。city 必须是城市名称；从用户问题或对话历史中提取，不要把整句问题作为城市。"},
            },
            "required": ["city"],
        },
    },

{
        "name": "object_detection",
        "description": (
            "使用 YOLO11n 模型进行目标检测，识别图像中的物体并返回检测结果。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "图像文件路径或 URL，例如：/path/to/image.jpg 或 https://example.com/image.jpg"},
            },
            "required": ["image_path"],
        },
    },

]


# ── 路由工具名 → LangGraph route 值的映射 ────────────────────────────────────
# workflow 的 route_supervisor 据 route 值选择走哪个节点。

TOOL_TO_ROUTE: dict[str, str] = {
    "rag_search": "rag_agent",
    "web_search": "web_agent",
    "parallel_search": "both",
    "get_weather": "weather_node",
    "object_detection": "detection_node",
    "direct_answer": "synthesis",  # direct_answer=True，走 synthesis 直接回答
}

# direct_answer 工具需要额外标记 direct_answer=True
DIRECT_ANSWER_TOOLS = {"direct_answer"}


# ── MCP 执行型工具 schema（mcp.types.Tool 格式）─────────────────────────────
# 仅 MCP server 使用，workflow 不参与。full_query / evaluate 是宿主 LLM
# 自己做不到的完整能力，必须由本服务执行。
# parallel_search / direct_answer 不对外暴露：
#   - parallel_search：宿主可分别调 rag_search + web_search
#   - direct_answer：宿主用自身 LLM 直接回答
#
# 注意：本函数只负责 schema 定义；工具分发由 mcp/server.py 的
# call_tool() 显式 if-elif 完成（4 个工具，无动态注册需求）。


def get_mcp_tools() -> list:
    """返回 mcp.types.Tool 列表。延迟导入避免 workflow 依赖 mcp 包。"""
    from mcp.types import Tool

    # 从 ROUTING_TOOLS 取 rag_search / web_search，补可选参数
    routing_by_name = {t["name"]: t for t in ROUTING_TOOLS}
    tools: list[Tool] = []

    rag = routing_by_name["rag_search"]
    rag_schema = {**rag["parameters"]}
    rag_schema["properties"]["top_k"] = {
        "type": "integer",
        "default": 5,
        "description": "返回的结果数量",
    }
    tools.append(Tool(
        name="rag_search",
        description=rag["description"],
        inputSchema=rag_schema,
    ))

    web = routing_by_name["web_search"]
    web_schema = {**web["parameters"]}
    web_schema["properties"]["max_results"] = {
        "type": "integer",
        "default": 5,
        "description": "最大结果数",
    }
    tools.append(Tool(
        name="web_search",
        description=web["description"],
        inputSchema=web_schema,
    ))

    # 天气查询工具
    tools.append(Tool(
        name="get_weather",
        description=(
            "查询指定城市的当前天气，包括天气状况、气温、体感温度、"
            "湿度、风速和降水量等信息。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称，例如：北京、Shanghai",
                },
            },
            "required": ["location"],
        },
    ))

    # 识物工具
    tools.append(Tool(
        name="object_detection",
        description=(
            "使用 YOLO11n 模型进行目标检测，识别图像中的物体并返回检测结果。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "图像文件路径或 URL，例如：/path/to/image.jpg 或 https://example.com/image.jpg",
                },
            },
            "required": ["image_path"],
        },
    ))

    # 执行型工具（仅 MCP 对外）
    tools.append(Tool(
        name="full_query",
        description=(
            "运行 OmniRAG 完整多 Agent 工作流：监督者路由 → RAG/Web 检索 → "
            "综合生成 → 评审修订。用于需要高质量、结构化答案的研究型查询，"
            "通常耗时 20-60 秒。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "研究问题"},
                "thread_id": {
                    "type": "string",
                    "description": "可选：对话线程 ID（用于多轮记忆）",
                },
                "max_iterations": {
                    "type": "integer",
                    "default": 2,
                    "description": "评审最大修订轮数（1=不修订，2=质量与速度平衡）",
                },
            },
            "required": ["query"],
        },
    ))
    tools.append(Tool(
        name="evaluate",
        description=(
            "对知识库跑一次 RAGAS 评估：用黄金集样本过完整 Agent 管道，"
            "计算 faithfulness / answer_relevancy / context_precision / "
            "context_recall 四个指标。3-10 分钟。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "k": {"type": "integer", "default": 10, "description": "检索 top-k"},
            },
        },
    ))
    return tools

@tool
async def get_weather(city: str) -> dict[Any, Any]:
    """查询指定城市的天气"""
    from app.tools import weather_tool
    return await weather_tool.get_weather(city)

@tool
async def object_detection(image_path: str) -> str:
    """使用 YOLO11n 模型进行目标检测"""
    from app.tools import vision_tool
    return await asyncio.to_thread(vision_tool.vision_service.detect, image_path,)