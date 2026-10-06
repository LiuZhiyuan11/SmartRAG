"""
OmniRAG — MCP 服务器
─────────────────────
自定义 Model Context Protocol 服务器，工具 schema 由 app.tools.registry 统一管理：
  • rag_search   — 混合检索知识库（Dense+Sparse RRF + Cross-Encoder 重排序）
  • web_search   — Tavily 实时网络搜索（代理绕过 + 双重保障）
  • weather      — Open-Meteo 当前天气查询（无需 API Key）
  • full_query   — 跑完整多 Agent 工作流（监督者→检索→综合→评审）
  • evaluate     — 对黄金集跑 RAGAS 4 维评估（faithfulness / answer_relevancy / context_precision / context_recall）

schema 与 workflow 监督者的 Function Calling 工具共享同一份定义（registry.py），
实现一处定义、两处使用：内部 bind_tools 路由 + 外部 MCP 宿主调用。

只保留宿主 LLM 本身做不到的外部能力：
  - summarise_docs / extract_entities / calculate 已移除：
    这些是 LLM 提示词包一层，宿主直接调用 LLM 更快，做成 MCP 工具纯浪费往返。

独立运行：python -m app.mcp.server
"""

import logging
import os
import sys
from typing import Any


def _ensure_pywin32_dlls():
    """Windows 下 pywin32 的 DLL 目录（pywin32_system32）默认不在 sys.path 中。

    兼容两种安装方式：
    - 常规 pip 安装：pywintypes 已可直接导入，本函数无副作用
    - vendored .libs 场景：把 pywin32_system32 加入 sys.path，避免 ImportError
    """
    if sys.platform != "win32":
        return
    for entry in list(sys.path):
        dll_dir = os.path.join(entry, "pywin32_system32")
        if os.path.isdir(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
            except (OSError, ValueError):
                pass
            sys.path.insert(0, dll_dir)
            return


_ensure_pywin32_dlls()

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent

from app.tools.registry import get_mcp_tools

logger = logging.getLogger(__name__)

app = Server("omnirag-mcp")


# ── 工具处理器 ─────────────────────────────────────────────────────────────────
# 工具 schema 从 registry 导入，与 workflow 监督者的 Function Calling 共享。

@app.list_tools()
async def list_tools() -> list:
    return get_mcp_tools()


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "rag_search":
        return await _rag_search(**arguments)
    elif name == "web_search":
        return await _web_search(**arguments)
    elif name == "get_weather":
        return await _weather(**arguments)
    elif name == "full_query":
        return await _full_query(**arguments)
    elif name == "evaluate":
        return await _evaluate(**arguments)
    else:
        return [TextContent(type="text", text=f"未知工具: {name}")]


# ── 实现 ───────────────────────────────────────────────────────────────────────

async def _rag_search(query: str, top_k: int = 5) -> list[TextContent]:
    try:
        from app.agents.rag_agent import create_retriever
        from app.config import settings

        # 复用 rag_agent 的共享检索器工厂（同一套参数逻辑 + lru_cache 单例）。
        # 候选池取 top_k × 2（最少为默认 top_k），重排后再截取 top_k，
        # 保证 Cross-Encoder 有足够候选对比，不因为用户传小 top_k 而浪费重排序。
        retriever = create_retriever(
            top_k=max(top_k * 2, settings.retrieval_top_k),
            reranker_top_n=top_k,
        )
        docs = retriever.invoke(query)
        if not docs:
            return [TextContent(type="text", text="未找到相关文档。")]

        results = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            score = meta.get("rerank_score", "N/A")
            results.append(
                f"[结果 {i}] 来源: {meta.get('source','未知')} | "
                f"页码: {meta.get('page','?')} | "
                f"类型: {meta.get('content_type','text')} | "
                f"重排序分数: {score}\n\n{doc.page_content}"
            )
        return [TextContent(type="text", text="\n\n---\n\n".join(results))]
    except Exception as e:
        logger.error("rag_search 错误: %s", e)
        return [TextContent(type="text", text=f"搜索错误: {e}")]


async def _web_search(query: str, max_results: int = 5) -> list[TextContent]:
    """复用 web_agent.py 中已修过代理问题的 tavily_search_robust，
    避免两处各写一份 Tavily 调用导致其中一处又被代理卡死。"""
    import asyncio

    try:
        from app.agents.web_agent import tavily_search_robust

        raw = await asyncio.to_thread(
            tavily_search_robust, query, max_results=max_results
        )
        if not raw:
            return [TextContent(type="text", text="无网络搜索结果。")]
        results = []
        for r in raw:
            results.append(
                f"**{r.get('title','无标题')}**\n"
                f"URL: {r.get('url','')}\n"
                f"{r.get('content','')}"
            )
        return [TextContent(type="text", text="\n\n---\n\n".join(results))]
    except Exception as e:
        logger.error("web_search 错误: %s", e)
        return [TextContent(type="text", text=f"网络搜索错误: {e}")]


async def _weather(location: str) -> list[TextContent]:
    """通过 Open-Meteo 地理编码与天气 API 查询城市当前天气，无需 API Key。"""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            geo_response = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={
                    "name": location,
                    "count": 1,
                    "language": "zh",
                    "format": "json",
                },
            )
            geo_response.raise_for_status()
            locations = geo_response.json().get("results", [])
            if not locations:
                return [TextContent(type="text", text=f"未找到地点：{location}")]

            place = locations[0]
            weather_response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": (
                        "temperature_2m,relative_humidity_2m,apparent_temperature,"
                        "precipitation,weather_code,wind_speed_10m"
                    ),
                    "timezone": "auto",
                },
            )
            weather_response.raise_for_status()
            current = weather_response.json()["current"]
    except httpx.HTTPError as e:
        logger.error("weather 查询错误: %s", e)
        return [TextContent(type="text", text=f"天气查询失败：{e}")]

    description = _weather_description(current["weather_code"])
    place_name = ", ".join(
        part for part in (place.get("name"), place.get("admin1"), place.get("country")) if part
    )
    text = (
        f"{place_name} 当前天气（{current['time']}）\n"
        f"天气：{description}\n"
        f"气温：{current['temperature_2m']}°C（体感 {current['apparent_temperature']}°C）\n"
        f"湿度：{current['relative_humidity_2m']}%\n"
        f"风速：{current['wind_speed_10m']} km/h\n"
        f"降水量：{current['precipitation']} mm"
    )
    return [TextContent(type="text", text=text)]


def _weather_description(code: int) -> str:
    """将 Open-Meteo WMO 天气代码转换为中文描述。"""
    if code == 0:
        return "晴"
    if code in (1, 2, 3):
        return {1: "大致晴朗", 2: "局部多云", 3: "阴"}[code]
    if code in (45, 48):
        return "有雾"
    if code in (51, 53, 55, 56, 57):
        return "毛毛雨"
    if code in (61, 63, 65, 66, 67):
        return "雨"
    if code in (71, 73, 75, 77):
        return "降雪"
    if code in (80, 81, 82):
        return "阵雨"
    if code in (85, 86):
        return "阵雪"
    if code in (95, 96, 99):
        return "雷暴"
    return f"未知天气（代码 {code}）"

async def detect_objects(image_path: str) -> list[TextContent]:
    """使用 YOLO11n 模型进行目标检测"""
    try:
        from app.tools.vision_tool import vision_service
        result = await vision_service.detect(image_path)
        return [TextContent(type="text", text=result)]
    except Exception as e:
        logger.error("object_detection 错误: %s", e)
        return [TextContent(type="text", text=f"目标检测失败: {e}")]


async def _full_query(
    query: str,
    thread_id: str = "",
    max_iterations: int = 2,
) -> list[TextContent]:
    """运行完整多 Agent 工作流：监督者路由 → 检索 → 综合 → 评审。
    用独立的 MCP checkpoint 数据库（mcp_checkpoints.db），避免与用户对话/评估冲突。
    """
    import asyncio
    import uuid

    from app.config import settings
    from app.graph.workflow import arun_query

    db_path = os.path.join(settings.data_dir, "mcp_checkpoints.db")
    tid = thread_id or f"mcp-{uuid.uuid4().hex[:8]}"
    try:
        result = await arun_query(
            query,
            thread_id=tid,
            max_iterations=max_iterations,
            db_path=db_path,
        )
        answer = result.get("final_answer") or result.get("draft_answer", "（无答案）")
        route = result.get("route", "")
        iterations = result.get("iterations", 0)
        critique = result.get("critique", "")
        summary = (
            f"路由: {route} | 评审轮数: {iterations}"
            f"{' | 评审: ' + critique if critique else ''}"
            f" | thread_id: {tid}\n\n"
        )
        return [TextContent(type="text", text=summary + answer)]
    except Exception as e:
        logger.error("full_query 错误: %s", e)
        return [TextContent(type="text", text=f"查询失败: {e}")]


async def _evaluate(k: int = 10) -> list[TextContent]:
    """跑 RAGAS 4 维评估（同步长任务，用 asyncio.to_thread 不阻塞 MCP 心跳）。"""
    import asyncio

    from app.rag.evaluation import run_evaluation

    try:
        result = await asyncio.to_thread(
            run_evaluation,
            samples=None,
            k=k,
        )
        if result.get("error"):
            return [TextContent(type="text", text=f"评估失败: {result['error']}")]

        m = result["metrics"]
        lines = [
            f"## 评估结果（{result.get('sample_count', 0)} 条样本, RAGAS）",
            f"- Faithfulness: {_fmt_metric(m.get('faithfulness'))}",
            f"- Answer Relevancy: {_fmt_metric(m.get('answer_relevancy'))}",
            f"- Context Precision: {_fmt_metric(m.get('context_precision'))}",
            f"- Context Recall: {_fmt_metric(m.get('context_recall'))}",
            f"- 摘要: {result.get('summary', '')}",
            f"- 报告: {result.get('report_path', '')}",
            "",
            "## 单条样本",
        ]
        for s in result.get("samples", []):
            lines.append(
                f"Q: {s.get('query','')[:80]}\n"
                f"   faith={_fmt_metric(s.get('faithfulness'))} "
                f"rel={_fmt_metric(s.get('answer_relevancy'))} "
                f"cprec={_fmt_metric(s.get('context_precision'))} "
                f"crec={_fmt_metric(s.get('context_recall'))}"
            )
        return [TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        logger.error("evaluate 错误: %s", e)
        return [TextContent(type="text", text=f"评估失败: {e}")]


def _fmt_metric(v):
    if v is None:
        return "N/A"
    return f"{v:.3f}"


# ── 入口点 ─────────────────────────────────────────────────────────────────────

async def run_server():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_server())
