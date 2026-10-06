"""
OmniRAG — 多 Agent LangGraph 工作流
─────────────────────────────────────────
监督者 Agent 路由 → RAG / Web / 并行双路 → 综合 → 评审（反馈闭环）

核心设计：
- 双路检索（RAG + Web）通过 asyncio.gather 真正并行执行
- 评审 Agent 的 REVISE 反馈会传回综合 Agent，形成真正的评估-修订闭环
- 对话历史进入监督者与综合提示词，checkpoint 记忆真正生效
- 支持流式事件输出（SSE），前端可实时展示 Agent 执行过程
"""

import asyncio
import logging
import os
import time
import uuid
from functools import partial
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agents.critique_agent import create_critique_agent
from app.agents.rag_agent import create_rag_agent
from app.agents.synthesis_agent import create_synthesis_agent
from app.agents.web_agent import create_web_agent
from app.config import settings
from app.llm import get_llm
from app.tools.registry import (
    DIRECT_ANSWER_TOOLS,
    ROUTING_TOOLS,
    TOOL_TO_ROUTE,
)
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# ── 状态模式 ──────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    image_path: str  # 用于 object_detection 节点，需前端或用户提供
    rag_context: str
    web_context: str
    weather_city: str  # 用于 get_weather 节点，需前端或用户提供
    weather_context: str
    detection_context: str
    draft_answer: str
    final_answer: str
    critique: str
    iterations: int
    route: str
    direct_answer: bool  # 简单常识问题直接回答，不走检索
    # 路由环（检索后纠错）：链路未命中 / 链路已执行标记
    rag_empty: bool  # RAG 未命中（空结果或弱命中）
    web_empty: bool  # Web 未命中（搜索失败或无结果）
    weather_empty: bool  # Weather 未命中（查询失败或无结果）
    detection_empty: bool # 检测到未命中，对方链路未跑 → 补路
    rag_ran: bool    # RAG 链路已执行（防 rag↔web 互补死循环）
    web_ran: bool    # Web 链路已执行
    weather_ran: bool # Weather 链路已执行
    detection_ran: bool # 检测到未命中，对方链路未跑 → 补路


# ── 路由环：未命中标记检测 ─────────────────────────────────────────────────────
# 监督者是纯 LLM 决策，无法事后感知"选错了"；真正的纠错信号只能来自
# 执行结果：检索返回空 / 弱命中（rag_agent.EMPTY_RESULT / web_agent 失败文案）。
# 检测到未命中且对方链路未跑 → 补路，形成"监督者决策 → 执行 → 结果校验 →
# 纠错补路"的路由环（与生成后的评审环互补，一个是纠信息源、一个是纠答案质量）。

EMPTY_MARKERS = ("知识库中未找到相关文档。", "未找到网络搜索结果。", "网络搜索失败")

# 直答失败标记：监督者误判 direct_answer 时，答案常以"无法/抱歉/没有提供"开头。
# 路由环检测到直答失败 → 补路双路检索，弥补 GLM-4-Flash function calling 的误判。
ANSWER_FAIL_MARKERS = ("无法", "抱歉", "没有提供", "未提供", "不清楚", "未找到")


def _is_empty_result(text: str) -> bool:
    """判断检索结果是否为"未命中"标记文案。"""
    text = (text or "").strip()
    if not text:
        return True
    return any(text.startswith(m) for m in EMPTY_MARKERS)


def _is_answer_failed(text: str) -> bool:
    """判断直答结果是否"没答出来"（触发补路检索）。"""
    text = (text or "").strip()
    return any(m in text for m in ANSWER_FAIL_MARKERS)


# ── 对话历史工具 ──────────────────────────────────────────────────────────────
# LLM 工厂统一在 app/llm.py 的 get_llm()，本文件不再各自构造 ChatZhipuAI。

def format_history(messages: List[BaseMessage], window: int | None = None) -> str:
    """取当前问题之前的最近若干轮对话，格式化为文本。

    当前问题本身是 messages 的最后一条，因此被排除在历史之外。
    """
    window = window or settings.history_window
    prior = messages[:-1] if messages else []
    prior = prior[-window:]
    lines = []
    for m in prior:
        role = "用户" if isinstance(m, HumanMessage) else "助手"
        content = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}: {content[:500]}")
    return "\n".join(lines)


# ── 监督者节点（Function Calling 路由）───────────────────────────────────────
# 旧方案：关键词预路由 + LLM 文本路由（输出纯文本 rag_agent/web_agent/both/synthesis）
# 新方案：LLM bind_tools(ROUTING_TOOLS) → GLM 自主选择工具调用 → 解析 tool_calls
# 优势：路由判断完全交给 LLM 的 function calling，不再维护关键词列表；
#       工具 schema 与 MCP server 共享（app.tools.registry），一处定义两处使用。

SUPERVISOR_SYSTEM_PROMPT = """你是监督者，负责为研究查询选择最合适的检索工具（通过 function calling 调用其一）。

背景：本系统有内部知识库，只包含用户上传的文档（项目报告、论文、笔记、README 等）；
另有 web_search 可获取实时外部信息。

决策规则：
1. 先判断是否需要检索：
   - 仅问候、寒暄、简单算术等不依赖任何外部信息的问题 → direct_answer（不检索）
   - 其余问题必须选择检索工具，不能只回复文本而不调用工具。
   - 绝对绝对绝对禁止用自身知识直接回答需要事实依据的问题——天气、新闻、股价等实时信息!
     以及文档内容、具体数据，一律必须检索，绝不允许 direct_answer 后编造。
2. 依据查询语义选择工具：
   -    - 问题内容可能来自用户上传的文档 → rag_search。这包括：
     * 报告、论文、笔记、README 的内容
     * 【公司内部信息】公司制度、员工福利、假期政策、绩效考核、招聘信息
     * 【产品细节】产品功能、定价、技术架构、性能指标
     * 【具体数据】文档中出现的数字、日期、名称
     只要问题涉及上述任意一类，必须优先选择 rag_search，禁止走 web_search。
   - 问题涉及外部世界的最新信息、产品或模型 → web_search
   - 需要同时结合内部文档和外部信息 → parallel_search
   - 拿不准时优先 parallel_search，而不是不调用工具。
   - 如果问题涉及天气，必须使用 get_weather 工具（不走 RAG 或 Web）。另外要从问题中提取城市名（例如用户说"北京天气"，你就提取 city 参数为"北京"）。要处理用户没说城市的情况，例如“今天天气怎么样”，可以先从对话历史里找最近提到的城市；如果仍然没有，就请用户补充城市，而不要猜。
   - 如果问题涉及图像检测或者图片物体识别内容，务必使用 object_detection 工具（不走 RAG 或 Web）。
3. 结合对话历史判断意图：追问（如"它呢"）应沿用上一轮的信息源。
4. 只能选择一个工具调用。"""


async def supervisor_node(state: AgentState) -> AgentState:
    """通过 Function Calling 路由到相应的 Agent。

    LLM 绑定 ROUTING_TOOLS 后自主选择工具，解析 tool_calls[0].name
    映射到 LangGraph route 值（见 TOOL_TO_ROUTE）。
    """
    _t = time.perf_counter()
    # 注意：supervisor 只在图首轮执行一次（评审循环回 synthesis 不经过这里），
    # 因此 rag_context / web_context 此刻恒为空，这里不做"上下文就绪"判断。
    rag_ctx = state.get("rag_context", "")
    web_ctx = state.get("web_context", "")
    weather_ctx = state.get("weather_context", "")
    detection_ctx = state.get("detection_context", "")

    # Function Calling 路由：LLM 绑定 ROUTING_TOOLS 后自主选择工具，
    # 由 tool_calls[0].name 经 TOOL_TO_ROUTE 映射到 LangGraph route 值。
    llm = get_llm().bind_tools(ROUTING_TOOLS)
    chain = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_SYSTEM_PROMPT),
        ("human", "对话历史:\n{history}\n\n查询: {query}\n已有的 RAG 上下文: {rag_context}\n已有的 Web 上下文: {web_context}\n已有的天气上下文: {weather_context}\n已有的检测上下文: {detection_context}"),
    ]) | llm
    response = await asyncio.to_thread(
        chain.invoke,
        {
            "query": state["query"],
            "history": format_history(state.get("messages", [])),
            "rag_context": rag_ctx,
            "web_context": web_ctx,
            "weather_context": weather_ctx,
            "detection_context": detection_ctx,
        },
    )

    # 解析 tool_calls
    tool_calls = getattr(response, "tool_calls", None) or []
    weather_city = state.get("weather_city", "")

    if tool_calls:
        tool_name = tool_calls[0]["name"]
        tool_args = tool_calls[0]["args"] or {}
        route = TOOL_TO_ROUTE.get(tool_name, "both")
        direct = tool_name in DIRECT_ANSWER_TOOLS
        logger.info(
            "监督者 function calling: %s → route=%s（%.1fs）",
            tool_name, route, time.perf_counter() - _t,
        )

        if tool_name == "get_weather":
            weather_city = tool_args.get("city", "").strip()
            logger.info("提取到城市: %s", weather_city)
    else:
        # 兜底：LLM 未调用工具，默认走双路并行检索
        route = "both"
        direct = False
        logger.warning(
            "监督者未返回 tool_calls，兜底 route=both（%.1fs）",
            time.perf_counter() - _t,
        )
    return {**state,
            "route": route,
            "direct_answer": direct,
            "weather_city": weather_city}


# ── Agent 节点（全部异步化，同步阻塞调用放入线程池）──────────────────────────

async def rag_node(state: AgentState) -> AgentState:
    _t = time.perf_counter()
    agent = create_rag_agent()
    context = await asyncio.to_thread(agent.run, state["query"])
    logger.info("RAG 节点耗时 %.1fs", time.perf_counter() - _t)
    empty = _is_empty_result(context)
    return {
        **state,
        "rag_context": "" if empty else context,
        "rag_empty": empty,
        "rag_ran": True,
    }


async def web_node(state: AgentState) -> AgentState:
    _t = time.perf_counter()
    agent = create_web_agent()
    context = await asyncio.to_thread(agent.run, state["query"])
    logger.info("Web 节点耗时 %.1fs", time.perf_counter() - _t)
    empty = _is_empty_result(context)
    return {
        **state,
        "web_context": "" if empty else context,
        "web_empty": empty,
        "web_ran": True,
    }

async def weather_node(state: AgentState) -> AgentState:
    from app.tools.registry import get_weather
    import json

    _t = time.perf_counter()
    city = state.get("weather_city", "").strip()

    if not city:
        # 没有城市，返回提示而不是硬查
        context = "请提供要查询天气的城市。"
        empty = True
    else:
        # ★ 用 StructuredTool 的异步接口 .ainvoke
        result = await get_weather.ainvoke({"city": city})
        # 结果是 dict，转成字符串存进 context
        context = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
        empty = False
    return {
        **state,
        "weather_context": context,
        "weather_empty": empty,
        "weather_ran": True,
    }

async def detection_node(state: AgentState) -> AgentState:
    from app.tools.registry import object_detection

    _t = time.perf_counter()

    # 从 state 里取图片路径（需要前端或用户提供）
    image_path = state.get("image_path", "").strip()
    logger.info(
        "目标检测节点开始: query=%r image_path=%r",
        state.get("query", ""),
        image_path,
    )
    if not image_path:
        raise ValueError("目标检测请求缺少 image_path")

    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"测试图片不存在: {path}")

    resolved_path = str(path.resolve())
    logger.info("目标检测节点调用工具: image_path=%s", resolved_path)
    context = await object_detection.ainvoke({
        "image_path": resolved_path,
    })
    logger.info("目标检测工具返回: %s", context)
    logger.info("目标检测完成，图片路径: %s", path)
    logger.info("检测节点耗时 %.1fs", time.perf_counter() - _t)
    return {
        **state,
        "detection_context": context,
        "detection_empty": context == "没有检测到目标。",
        "detection_ran": True,
        "draft_answer": context,
        "final_answer": context,
    }

async def both_node(state: AgentState) -> AgentState:
    """RAG 与 Web 检索真正并行执行（asyncio.gather）。

    双路已并行执行，天然覆盖"两条信息源"，无需路由环补路，
    因此直接标记两条链路均已执行。

    同时重置 direct_answer 与旧草稿：
    - 本节点执行了真实检索，之后必须走正常综合 + 评审（不能再被当直答跳过评审）
    - 清空 draft_answer，避免直答失败的旧答案污染第二次综合
    """
    rag_fut, web_fut = await asyncio.gather(
        rag_node(state),
        web_node(state),
    )
    return {
        **state,
        "rag_context": rag_fut["rag_context"],
        "web_context": web_fut["web_context"],
        "rag_empty": rag_fut["rag_empty"],
        "web_empty": web_fut["web_empty"],
        "rag_ran": True,
        "web_ran": True,
        "direct_answer": False,
        "draft_answer": "",
    }


async def synthesis_node(state: AgentState) -> AgentState:
    """综合生成；把上一版答案与评审反馈一起传入，实现真正的修订闭环。

    direct_answer（简单问题直接回答）时答案即为最终答案：
    由 route_after_synthesis 跳过评审直接结束，省一次 LLM 往返。

    token 级流式：节点内用 get_stream_writer() 把 LLM 逐 token 内容透传
    给 stream_query（stream_mode="custom"），前端约 1s 内看到首字，
    而不是等完整生成（感知延迟优化）。非流式调用（graph.ainvoke）时
    writer 为 no-op，行为与原来完全一致。
    """
    _t = time.perf_counter()
    agent = create_synthesis_agent()
    current_date = datetime.now().strftime("%Y年%m月%d日")

    # 把时间拼接到用户问题里传给写手
    enriched_query = f"【系统当前日期：{current_date}】\n\n用户问题：{state['query']}"

    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
    except Exception:
        writer = None

    parts: list[str] = []
    async for token in agent.arun_stream(
        enriched_query,
        state.get("rag_context", ""),
        state.get("web_context", ""),
        state.get("weather_context", ""),
        state.get("critique", ""),
        state.get("draft_answer", ""),
        format_history(state.get("messages", [])),
        state.get("direct_answer", False),
    ):
        parts.append(token)
        if writer is not None:
            try:
                await writer({"type": "token", "text": token})
            except Exception:
                pass  # 非 custom 流式模式时 writer 丢弃数据，不影响答案

    answer = "".join(parts)
    logger.info("综合节点耗时 %.1fs", time.perf_counter() - _t)
    direct = state.get("direct_answer", False)
    return {
        **state,
        "draft_answer": answer,
        "final_answer": answer if direct else state.get("final_answer", ""),
    }


async def critique_node(
    state: AgentState,
    max_iterations: Optional[int] = None,
) -> AgentState:
    """评审答案；通过则结束，否则带反馈回到综合节点。

    max_iterations 通过 build_graph 注入（见 build_graph），
    不修改全局 settings，避免并发请求互相影响。
    """
    agent = create_critique_agent()
    _t = time.perf_counter()
    critique = await asyncio.to_thread(
        agent.evaluate,
        state["query"],
        state["draft_answer"],
        state.get("rag_context", "") + "\n" + state.get("web_context", "") + "\n" + state.get("weather_context", ""),
    )
    logger.info("评审节点耗时 %.1fs", time.perf_counter() - _t)
    iterations = state.get("iterations", 0) + 1
    limit = max_iterations if max_iterations is not None else settings.max_iterations
    passed = _critique_passed(critique)
    if passed or iterations >= limit:
        final = state["draft_answer"]
    else:
        final = ""
    logger.info(
        "评审结果: %s（第 %d/%d 轮）",
        "PASS" if passed else "REVISE",
        iterations,
        limit,
    )
    result = {
        **state,
        "critique": critique,
        "final_answer": final,
        "iterations": iterations,
    }
    # 评审通过时，把 AI 最终答案追加到 messages 列表，
    # 这样 checkpoint 保存后，下一轮查询能从历史中看到 AI 的回答。
    if final:
        result["messages"] = [AIMessage(content=final)]
    return result


def _critique_passed(critique: str) -> bool:
    """判断评审结果是否通过。

    评审 Agent 可能返回两种格式：
    - "GOOD"（直接通过）
    - "评估：… 评分：GOOD"（结论在末尾）

    同时避免 "NOT GOOD" 被误判为通过。
    """
    text = critique.upper().strip()
    if text.startswith("GOOD"):
        return True
    if "NOT GOOD" in text or "不通过" in text:
        return False
    last_line = text.splitlines()[-1] if text.splitlines() else ""
    return "GOOD" in last_line


# ── 路由函数 ───────────────────────────────────────────────────────────────────

def route_supervisor(state: AgentState) -> Literal["rag_node", "web_node", "both_node", "synthesis_node", "weather_node", "detection_node"]:
    route = state.get("route", "both")
    direct = state.get("direct_answer", False)
    # 代码层保险：如果上下文都为空且不是直接回答，绝不允许直接 synthesis
    has_rag = bool(state.get("rag_context", "").strip())
    has_web = bool(state.get("web_context", "").strip())
    if route == "synthesis" and not has_rag and not has_web and not direct:
        logger.warning("监督者路由至 synthesis 但上下文为空，强制改为 both")
        route = "both"
    if route == "rag_agent":
        return "rag_node"
    elif route == "web_agent":
        return "web_node"
    elif route == "both":
        return "both_node"
    elif route == "weather_node":
        return "weather_node"
    elif route in {"detection_node", "vision_node"}:
        return "detection_node"  # vision_node 用于兼容旧路由值
    else:
        return "synthesis_node"


def route_critique(state: AgentState) -> Literal["synthesis_node", "end"]:
    """如果评审标记了问题，则携带反馈循环回综合，否则结束。"""
    if state.get("final_answer"):
        return "end"
    return "synthesis_node"


def route_after_synthesis(state: AgentState) -> Literal["both_node", "critique_node", "end"]:
    """综合节点之后的分流。

    - direct_answer 且直答成功 → 直接结束（省一次 LLM 往返，简单问答无需质量门禁）
    - direct_answer 但直答失败（答案含"无法/抱歉/没有提供"等）→ 补路双路检索
      （监督者误判 direct_answer 的路由环兜底；both_node 会重置 direct_answer，
      第二次综合后走正常评审，最多补路一次不会死循环）
    - 其余情况 → 进入评审节点做事实核查
    """
    if state.get("direct_answer"):
        if _is_answer_failed(state.get("draft_answer", "")):
            logger.info("路由环: direct_answer 直答失败，补路双路检索: %s", state["query"][:40])
            return "both_node"
        return "end"
    return "critique_node"


def route_after_rag(state: AgentState) -> Literal["web_node", "synthesis_node"]:
    """路由环：RAG 未命中且 Web 没跑过 → 补路 Web；否则进入综合。

    依赖 rag_ran/web_ran 标记防止互补死循环：
    即使补路后的 Web 也未命中，route_after_web 看到 rag_ran=True 也不会跳回 RAG。
    """
    if state.get("rag_empty") and not state.get("web_ran"):
        logger.info("路由环: RAG 未命中，补路 Web: %s", state["query"][:40])
        return "web_node"
    return "synthesis_node"


def route_after_web(state: AgentState) -> Literal["rag_node", "synthesis_node"]:
    """路由环：Web 未命中且 RAG 没跑过 → 补路 RAG；否则进入综合。"""
    if state.get("web_empty") and not state.get("rag_ran"):
        logger.info("路由环: Web 未命中，补路 RAG: %s", state["query"][:40])
        return "rag_node"
    return "synthesis_node"


# ── 图组装 ─────────────────────────────────────────────────────────────────────

def build_graph(checkpointer=None, max_iterations: Optional[int] = None):
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("web_node", web_node)
    graph.add_node("both_node", both_node)
    graph.add_node("synthesis_node", synthesis_node)
    graph.add_node("critique_node", partial(critique_node, max_iterations=max_iterations))
    graph.add_node("weather_node", weather_node)
    graph.add_node("detection_node", detection_node)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "rag_node": "rag_node",
            "web_node": "web_node",
            "both_node": "both_node",
            "synthesis_node": "synthesis_node",
            "weather_node": "weather_node",
            "detection_node": "detection_node",
        },
    )

    # 检索后路由环：单路未命中 → 补路另一条信息源；双路已并行则直通综合
    graph.add_conditional_edges(
        "rag_node",
        route_after_rag,
        {"web_node": "web_node", "synthesis_node": "synthesis_node"},
    )
    graph.add_conditional_edges(
        "web_node",
        route_after_web,
        {"rag_node": "rag_node", "synthesis_node": "synthesis_node"},
    )
    graph.add_edge("both_node", "synthesis_node")

    graph.add_edge("weather_node", "synthesis_node")
    graph.add_edge("detection_node", END)
    # 综合后分流：direct_answer 成功直接结束（跳过评审），失败补路双路检索，
    # 其余进入评审闭环
    graph.add_conditional_edges(
        "synthesis_node",
        route_after_synthesis,
        {
            "critique_node": "critique_node",
            "both_node": "both_node",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "critique_node",
        route_critique,
        {"synthesis_node": "synthesis_node", "end": END},
    )

    return graph.compile(checkpointer=checkpointer)


# ── 公共接口 ───────────────────────────────────────────────────────────────────

def _initial_state(query: str) -> AgentState:
    return {
        "messages": [HumanMessage(content=query)],
        "query": query,

        "image_path": "",  # 用于 object_detection 节点，需前端或用户提供

        "rag_context": "",
        "web_context": "",

        "weather_city": "",  # 用于 weather_node 节点，需前端或用户提供
        "weather_context":"",
        "detection_context":"",

        "draft_answer": "",
        "final_answer": "",
        "critique": "",
        "iterations": 0,
        "route": "",
        "rag_empty": False,
        "web_empty": False,

        "weather_empty": False,
        "detection_empty": False,

        "rag_ran": False,
        "web_ran": False,

        "weather_ran": False,
        "detection_ran": False,
    }



def _extract_result(result) -> dict:
    return {
        "final_answer": result.get("final_answer") or result.get("draft_answer"),
        "detection_context": result.get("detection_context", ""),
        "detection_ran": result.get("detection_ran", False),
        "rag_context": result.get("rag_context", ""),
        "web_context": result.get("web_context", ""),
        "critique": result.get("critique", ""),
        "iterations": result.get("iterations", 0),
        "route": result.get("route", ""),
    }


async def arun_query(
    query: str,
    thread_id: str = "default",
    max_iterations: Optional[int] = None,
    db_path: Optional[str] = None,
) -> dict:
    """异步执行完整的多 Agent 管道。

    max_iterations 用于覆盖评审闭环轮数（评估时传 1 可关闭循环），
    不修改全局 settings。async 调用方请使用本函数而非 run_query。

    db_path 可指定独立的 SQLite checkpoint 路径，避免评估与用户对话
    共用同一数据库导致 SQLite 写锁冲突（对话被强制中断）。
    """
    db_path = db_path or os.path.join(settings.data_dir, "checkpoints.db")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_graph(checkpointer, max_iterations=max_iterations)
        config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
        result = await graph.ainvoke(_initial_state(query), config=config)
        return _extract_result(result)


def run_query(
    query: str,
    thread_id: str = "default",
    max_iterations: Optional[int] = None,
    db_path: Optional[str] = None,
) -> dict:
    """同步包装：通过 asyncio.run 执行完整管道。

    注意：只能在没有运行中事件循环的上下文调用（如 FastAPI 同步端点、
    脚本）。若在 async 环境中调用会抛 RuntimeError，请改用 arun_query。
    """
    return asyncio.run(arun_query(query, thread_id, max_iterations=max_iterations, db_path=db_path))


def _node_event(node: str, frag: dict) -> dict | None:
    """把每个节点输出转换为 SSE 友好的轻量事件。"""
    if node == "supervisor":
        return {"type": "status", "step": "route", "detail": f"路由至 {frag.get('route', 'both')}"}
    if node == "rag_node":
        if frag.get("rag_empty"):
            return {
                "type": "status",
                "step": "rag",
                "detail": "知识库未命中，准备补路 Web",
            }
        return {
            "type": "status",
            "step": "rag",
            "detail": f"知识库检索完成（{len(frag.get('rag_context', ''))} 字符）",
        }
    if node == "web_node":
        if frag.get("web_empty"):
            return {
                "type": "status",
                "step": "web",
                "detail": "网络搜索未命中，准备补路 RAG",
            }
        return {
            "type": "status",
            "step": "web",
            "detail": f"网络搜索完成（{len(frag.get('web_context', ''))} 字符）",
        }
    if node == "both_node":
        return {
            "type": "status",
            "step": "both",
            "detail": "双路并行检索完成",
        }
    if node == "synthesis_node":
        return {
            "type": "status",
            "step": "synthesis",
            "detail": f"综合生成完成（第 {frag.get('iterations', 0) + 1} 轮）",
        }
    if node == "critique_node":
        passed = bool(frag.get("final_answer"))
        return {
            "type": "critique",
            "passed": passed,
            "detail": (frag.get("critique") or "")[:300],
        }
    return None


async def stream_query(
    query: str,
    thread_id: str = "default",
    max_iterations: Optional[int] = None,
    image_path: str = "",
):
    """流式执行管道，逐步产出事件字典（供 SSE 使用）。

    事件类型：
    - start:    会话开始，携带 thread_id
    - status:   Agent 节点进度（route / rag / web / both / synthesis）
    - critique: 评审结果（passed + detail）
    - final:    最终结果（完整 answer + 上下文）
    """
    thread_id = thread_id or str(uuid.uuid4())
    db_path = os.path.join(settings.data_dir, "checkpoints.db")
    os.makedirs(settings.data_dir, exist_ok=True)

    yield {"type": "start", "thread_id": thread_id}

    last_state = None
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_graph(checkpointer, max_iterations=max_iterations)
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = _initial_state(query)
        if image_path:
            initial_state["image_path"] = image_path  # ★ 注入

        # updates：节点级状态事件；custom：synthesis_node 透传的逐 token 内容
        async for mode, data in graph.astream(
            initial_state, config=config, stream_mode=["updates", "custom"]
        ):
            if mode == "updates":
                for node, frag in data.items():
                    event = _node_event(node, frag)
                    if event:
                        yield event
                    # synthesis_node（direct_answer 直答）或 critique_node（常规闭环）
                    # 都可能是最后产出答案的节点，记录供最终结果提取
                    if node in ("synthesis_node", "critique_node","detection_node"):
                        last_state = frag
            elif mode == "custom":
                # token 级事件：前端逐字渲染，改善感知延迟
                if isinstance(data, dict) and data.get("type") == "token":
                    yield {"type": "token", "text": data.get("text", "")}

    if last_state:
        yield {"type": "final", "result": _extract_result(last_state)}
    else:
        yield {"type": "error", "detail": "工作流未返回结果"}
