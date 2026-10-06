"""
OmniRAG — 测试套件
测试核心组件，无需实时 API 调用（在需要时使用 mock）。
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import Runnable


class FakeLLM(Runnable):
    """可被 ChatPromptTemplate | llm 组合的假 LLM。

    - content：invoke 返回的文本（兼容非 FC 场景）
    - tool_calls：invoke 返回的 tool_calls 列表（模拟 function calling 路由）
    """

    def __init__(self, content: str = "", tool_calls: list | None = None):
        self._content = content
        self._tool_calls = tool_calls or []
        self.captured_input = None

    def bind_tools(self, *args, **kwargs):
        """模拟 llm.bind_tools(...)，返回自身即可（FC 测试用）。"""
        return self

    def invoke(self, *args, **kwargs):
        self.captured_input = args[0] if args else kwargs.get("input")
        return SimpleNamespace(content=self._content, tool_calls=self._tool_calls)

# ── 重排序器测试 ──────────────────────────────────────────────────────────────

class TestCrossEncoderReranker:
    def test_rerank_returns_top_n(self):
        from app.rag.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(top_n=3)

        docs = [
            Document(page_content=f"Document {i} content", metadata={"id": i})
            for i in range(10)
        ]

        with patch.object(reranker, '_get_model') as mock_model:
            model = MagicMock()
            model.predict.return_value = list(range(10, 0, -1))  # 分数为 10..1
            mock_model.return_value = model

            result = reranker.rerank("test query", docs)

        assert len(result) == 3
        # 第一个结果应该有最高分数
        assert result[0].metadata["rerank_score"] == 10.0

    def test_rerank_handles_empty_docs(self):
        from app.rag.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(top_n=5)
        with patch.object(reranker, '_get_model', return_value=MagicMock()):
            result = reranker.rerank("query", [])
        assert result == []

    def test_rerank_handles_model_failure(self):
        from app.rag.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(top_n=3)
        docs = [Document(page_content=f"doc {i}") for i in range(5)]

        with patch.object(reranker, '_get_model', return_value=None):
            result = reranker.rerank("query", docs)
        assert len(result) == 3  # 回退到截断


# ── 导入去重测试 ─────────────────────────────────────────────────────────────

class TestIngestion:
    def test_deduplication(self):
        from app.rag.ingestion import ingest_documents
        docs = [
            Document(page_content="相同内容", metadata={"source": "test"}),
            Document(page_content="相同内容", metadata={"source": "test"}),  # 重复
            Document(page_content="不同内容", metadata={"source": "test"}),
        ]

        with patch("app.rag.ingestion.get_vector_store") as mock_vs:
            vs = MagicMock()
            vs.add_documents.return_value = ["id1", "id2"]
            mock_vs.return_value = vs

            ingest_documents(docs)
            # 应该去重为 2 个唯一文档
            call_args = vs.add_documents.call_args[0][0]
            assert len(call_args) == 2

    def test_text_extraction(self):
        from app.rag.ingestion import extract_from_text
        text = "这是测试内容。 " * 100  # 足够长以进行分块
        docs = extract_from_text(text, source="test")
        assert len(docs) > 0
        assert all(d.metadata["source"] == "test" for d in docs)
        assert all(d.metadata["content_type"] == "text" for d in docs)


# ── Agent 路由测试 ───────────────────────────────────────────────────────────

class TestAgentRouting:
    @pytest.mark.asyncio
    async def test_supervisor_routes_to_rag(self):
        """监督者 function calling：LLM 选 rag_search 工具 → 路由至 RAG。"""
        from langchain_core.messages import HumanMessage

        from app.graph.workflow import AgentState, route_supervisor, supervisor_node

        state = AgentState(
            messages=[HumanMessage(content="What does the uploaded document say?")],
            query="What does the uploaded document say?",
            rag_context="",
            web_context="",
            draft_answer="",
            final_answer="",
            critique="",
            iterations=0,
            route="",
        )

        fc = [{"name": "rag_search", "args": {"query": state["query"]}, "id": "1", "type": "function"}]
        with patch("app.graph.workflow.get_llm") as mock_get_llm:
            mock_get_llm.return_value = FakeLLM(tool_calls=fc)

            result = await supervisor_node(state)

        assert result["route"] == "rag_agent"
        assert route_supervisor(result) == "rag_node"

    @pytest.mark.asyncio
    async def test_supervisor_fallback_to_both(self):
        """LLM 未返回 tool_calls 时，监督者应兜底到 both。"""
        from langchain_core.messages import HumanMessage

        from app.graph.workflow import AgentState, supervisor_node

        state = AgentState(
            messages=[HumanMessage(content="anything")],
            query="anything",
            rag_context="",
            web_context="",
            draft_answer="",
            final_answer="",
            critique="",
            iterations=0,
            route="",
        )

        with patch("app.graph.workflow.get_llm") as mock_get_llm:
            mock_get_llm.return_value = FakeLLM()  # 无 tool_calls → 兜底

            result = await supervisor_node(state)

        assert result["route"] == "both"

    @pytest.mark.asyncio
    async def test_supervisor_direct_answer(self):
        """监督者 function calling：LLM 选 direct_answer 工具 → 标记直接回答。"""
        from langchain_core.messages import HumanMessage

        from app.graph.workflow import AgentState, supervisor_node

        state = AgentState(
            messages=[HumanMessage(content="你好")],
            query="你好",
            rag_context="",
            web_context="",
            draft_answer="",
            final_answer="",
            critique="",
            iterations=0,
            route="",
        )

        fc = [{"name": "direct_answer", "args": {"query": state["query"]}, "id": "1", "type": "function"}]
        with patch("app.graph.workflow.get_llm") as mock_get_llm:
            mock_get_llm.return_value = FakeLLM(tool_calls=fc)

            result = await supervisor_node(state)

        assert result["route"] == "synthesis"
        assert result["direct_answer"] is True

    def test_routing_web(self):

        from app.graph.workflow import AgentState, route_supervisor

        state = AgentState(
            messages=[], query="", rag_context="", web_context="",
            draft_answer="", final_answer="", critique="",
            iterations=0, route="web_agent",
        )
        assert route_supervisor(state) == "web_node"

    def test_routing_both(self):
        from app.graph.workflow import AgentState, route_supervisor

        state = AgentState(
            messages=[], query="", rag_context="", web_context="",
            draft_answer="", final_answer="", critique="",
            iterations=0, route="both",
        )
        assert route_supervisor(state) == "both_node"

    def test_critique_ends_on_good(self):
        from app.graph.workflow import AgentState, route_critique

        state = AgentState(
            messages=[], query="", rag_context="", web_context="",
            draft_answer="test answer", final_answer="GOOD answer here",
            critique="GOOD", iterations=1, route="",
        )
        assert route_critique(state) == "end"

    def test_critique_loops_on_revise(self):
        from app.graph.workflow import AgentState, route_critique

        state = AgentState(
            messages=[], query="", rag_context="", web_context="",
            draft_answer="bad answer", final_answer="",
            critique="REVISE: missing details", iterations=1, route="",
        )
        assert route_critique(state) == "synthesis_node"

    def test_critique_requires_good_prefix(self):
        """"NOT GOOD" 不应被误判为通过。"""
        from app.graph.workflow import AgentState, route_critique

        state = AgentState(
            messages=[], query="", rag_context="", web_context="",
            draft_answer="answer", final_answer="",
            critique="NOT GOOD: missing sources", iterations=1, route="",
        )
        assert route_critique(state) == "synthesis_node"

    def test_critique_accepts_trailing_verdict(self):
        """兼容"评分：GOOD"结尾的评审格式。"""
        from app.graph.workflow import _critique_passed

        assert _critique_passed("评估：\n1. 忠实性通过\n评分：GOOD")
        assert _critique_passed("GOOD")
        assert not _critique_passed("NOT GOOD: 缺少来源")
        assert not _critique_passed("评估：\n1. 缺少引用\n评分：REVISE")

    def test_format_history_excludes_current_query(self):
        """对话历史应排除当前问题本身。"""
        from langchain_core.messages import AIMessage, HumanMessage

        from app.graph.workflow import format_history

        messages = [
            HumanMessage(content="第一问"),
            AIMessage(content="第一答"),
            HumanMessage(content="当前问题"),
        ]
        history = format_history(messages)
        assert "第一问" in history
        assert "第一答" in history
        assert "当前问题" not in history


# ── RAG Agent 上下文格式化（回归测试）────────────────────────────────────────

class TestRAGAgentFormatting:
    def test_missing_rerank_score_does_not_crash(self):
        """检索结果没有 rerank_score 时（未触发重排），上下文格式化不应崩溃。"""
        from app.agents.rag_agent import RAGAgent

        retriever = MagicMock()
        retriever.invoke.return_value = [
            Document(page_content="内容", metadata={"source": "s1", "page": 1})
        ]

        fake = FakeLLM("ok")
        with patch("app.agents.rag_agent.get_llm", return_value=fake):
            agent = RAGAgent(retriever=retriever)
            out = agent.run("query")

        assert out == "ok"
        # 验证传给 LLM 的上下文里评分显示为 N/A 而不是崩溃
        prompt_text = str(fake.captured_input)
        assert "Score: N/A" in prompt_text


# ── 黄金集加载（评估模块）────────────────────────────────────────────────────────

class TestGoldenSet:
    def test_load_list_format(self, tmp_path):
        from app.rag.evaluation import load_golden_set

        p = tmp_path / "golden.json"
        p.write_text(json.dumps([
            {"query": "q1", "ground_truth": "a1", "source": "s1"},
            {"query": "q2", "ground_truth": "a2"},
            {"query": "q3"},  # 缺 ground_truth 应被过滤
        ]), encoding="utf-8")

        samples = load_golden_set(str(p))
        assert len(samples) == 2
        assert samples[0] == {"query": "q1", "ground_truth": "a1", "source": "s1"}
        assert samples[1]["source"] == ""

    def test_load_dict_format_with_comment(self, tmp_path):
        from app.rag.evaluation import load_golden_set

        p = tmp_path / "golden.json"
        p.write_text(json.dumps({
            "_comment": "模板说明字段，应被忽略",
            "samples": [{"query": "q", "ground_truth": "a"}],
        }), encoding="utf-8")

        samples = load_golden_set(str(p))
        assert len(samples) == 1
        assert samples[0]["query"] == "q"

    def test_load_missing_file(self):
        from app.rag.evaluation import load_golden_set

        assert load_golden_set("/nonexistent/golden.json") == []


# ── 多查询检索路径 ─────────────────────────────────────────────────────────────

class TestMultiQuery:
    def test_multi_query_uses_expansion(self):
        from app.rag.retriever import HybridRetriever

        retriever = HybridRetriever(use_multi_query=True)
        mq = MagicMock()
        mq.retrieve.return_value = [Document(page_content="x")]

        with patch("app.rag.retriever.MultiQueryRetriever", return_value=mq):
            docs = retriever.invoke("query")

        mq.retrieve.assert_called_once_with("query")
        assert docs[0].page_content == "x"


# ── MCP 混合检索工具（回归测试）────────────────────────────────────────────────

class TestMCPRagSearch:
    @pytest.mark.asyncio
    async def test_rag_search_uses_invoke(self):
        """MCP rag_search 应复用共享 create_retriever 并调新版 invoke API。"""
        from app.mcp.server import _rag_search

        retriever = MagicMock()
        retriever.invoke.return_value = [
            Document(page_content="测试内容", metadata={"source": "t"})
        ]

        with patch("app.agents.rag_agent.create_retriever", return_value=retriever):
            result = await _rag_search("q")

        retriever.invoke.assert_called_once_with("q")
        assert "测试内容" in result[0].text


class TestMCPWeather:
    @pytest.mark.asyncio
    async def test_weather_returns_current_conditions(self):
        from app.mcp.server import _weather

        class FakeResponse:
            def __init__(self, data):
                self.data = data

            def raise_for_status(self):
                pass

            def json(self):
                return self.data

        class FakeClient:
            def __init__(self, **kwargs):
                self.responses = [
                    FakeResponse({"results": [{
                        "name": "北京", "admin1": "北京市", "country": "中国",
                        "latitude": 39.9, "longitude": 116.4,
                    }]}),
                    FakeResponse({"current": {
                        "time": "2026-10-01T11:00", "temperature_2m": 22,
                        "apparent_temperature": 21, "relative_humidity_2m": 45,
                        "wind_speed_10m": 8, "precipitation": 0, "weather_code": 0,
                    }}),
                ]

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, url, params):
                return self.responses.pop(0)

        with patch("httpx.AsyncClient", FakeClient):
            result = await _weather("北京")

        assert "北京" in result[0].text
        assert "天气：晴" in result[0].text
        assert "气温：22°C" in result[0].text

    def test_weather_tool_is_exposed_to_mcp(self):
        from app.tools.registry import get_mcp_tools

        weather_tool = next(tool for tool in get_mcp_tools() if tool.name == "weather")
        assert weather_tool.inputSchema["required"] == ["location"]


# ── critique_node max_iterations 参数化 ───────────────────────────────────────

class TestCritiqueMaxIterations:
    @pytest.mark.asyncio
    async def test_max_iterations_forces_final(self):
        """max_iterations=1 时，即使评审未通过也应输出 final_answer，且不改全局 settings。"""
        from langchain_core.messages import HumanMessage

        from app.graph.workflow import AgentState, critique_node

        state = AgentState(
            messages=[HumanMessage(content="q")], query="q",
            rag_context="ctx", web_context="", draft_answer="draft",
            final_answer="", critique="", iterations=0, route="rag_agent",
        )
        agent = MagicMock()
        agent.evaluate.return_value = "REVISE: 不完整"
        with patch("app.graph.workflow.create_critique_agent", return_value=agent):
            result = await critique_node(state, max_iterations=1)

        assert result["final_answer"] == "draft"
        assert result["iterations"] == 1

    @pytest.mark.asyncio
    async def test_default_uses_settings_limit(self):
        """不传 max_iterations 时，达到 settings.max_iterations 也应输出 final_answer。"""
        from langchain_core.messages import HumanMessage

        from app.graph.workflow import AgentState, critique_node, settings

        state = AgentState(
            messages=[HumanMessage(content="q")], query="q",
            rag_context="ctx", web_context="", draft_answer="draft",
            final_answer="", critique="", iterations=settings.max_iterations,
            route="rag_agent",
        )
        agent = MagicMock()
        agent.evaluate.return_value = "REVISE: 不完整"
        with patch("app.graph.workflow.create_critique_agent", return_value=agent):
            result = await critique_node(state)

        assert result["final_answer"] == "draft"
