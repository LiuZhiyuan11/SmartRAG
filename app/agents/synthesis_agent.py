"""
OmniRAG — 综合 Agent
──────────────────────────
将 RAG 上下文 + 网络上下文合并为最终的、有依据的答案。
"""

import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_llm

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位资深研究分析师，负责生成最终的、结构良好的答案。
【当前时间】今天是 {current_date}。所有涉及"今天""最新""当前"的内容，都必须以此日期为准。

你有三个信息来源：
1. 知识库（RAG）：经过筛选的内部文档
2. 网络搜索：实时互联网结果
3. 天气信息：实时天气数据

【绝对禁止】
- 禁止使用你自身的训练知识回答问题——你只能从 RAG 上下文和 Web 上下文和天气上下文中提取信息
- 禁止编造上下文中不存在的内容、URL、日期、数字
- 如果两个来源都没有覆盖某个方面，直接说"当前信息中未提及"

指令：
- 将三个来源的信息综合为一个连贯、全面的答案
- 以事实准确性为先；在行内标注来源 [RAG: 来源/页码] 或 [Web: URL] 或 [Weather: 位置/时间]
- 标出来源之间的任何矛盾之处
- 使用清晰的 Markdown 格式，善用标题、列表和表格
- 如有信息缺失，明确指出哪些是未知的
- 如果提供了"评审反馈"，必须逐条解决反馈中提到的问题；
  如果反馈为 GOOD，说明上一版答案已达标，保持其质量和事实，不要无意义改写。

输出格式要求（严格遵守）：
- 开头：如果问题涉及"最新""当前"等时效性内容，简要说明信息来源的时间范围
  （如"基于当前网络搜索结果"或"基于知识库文档"），让用户知道信息时效
- 主体：用带编号的小节组织答案，每个小节有 **加粗标题**，标题下方展开具体内容
  - 每个小节聚焦一个主题，给出具体事实、数据、日期、代表性事件
  - 不要空泛概括，要有可验证的具体信息（模型名、数字、机构名、日期）
  - 小节数量根据问题复杂度决定，通常 5~8 个
- 结尾：附"来源"章节，列出引用的 [RAG: ...] 和 [Web: ...] 和 [Weather: ...]
- 最后可加一句引导，询问用户是否想深入了解某个具体方向

质量要求：
- 答案控制在 600~1200 字，宁可详细不要空泛
- 每个论断都要有上下文支撑，不允许编造
- 用条目和短段落，不要写大段流水账"""),
    ("human", """对话历史（供上下文参考，不要重复用户已提过的内容）:
{history}

问题: {query}

--- RAG 上下文 ---
{rag_context}

--- Web 上下文 ---
{web_context}

--- 天气上下文 ---
{weather_context}

--- 上一版答案 ---
{previous_answer}

--- 评审反馈 ---
{critique}

请给出你的综合答案:"""),
])

# 简单问题直接回答模式（不走检索，用 LLM 自身知识）
DIRECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位智能助手。用户提出了一个简单问题，不需要检索知识库或网络搜索。
请用你自身的知识直接回答，简洁明了。
【当前时间】今天是 {current_date}。

如果提供了"对话历史"，请结合上文理解用户意图（例如追问"它呢"应沿用上文主题）。

输出要求：
- 直接回答问题，不要开头说"根据搜索结果"之类的话
- 答案控制在 100~300 字，简洁为上
- 如果问题非常简单（如打招呼、算术），一两句话即可
- 用 Markdown 格式，适当用列表和加粗"""),
    ("human", """对话历史（供上下文参考）:
{history}

问题: {query}"""),
])


@dataclass
class SynthesisAgent:
    def run(
        self,
        query: str,
        rag_context: str,
        web_context: str,
        weather_context: str = "",
        critique: str = "",
        previous_answer: str = "",
        history: str = "",
        direct: bool = False,
    ) -> str:
        _t = time.perf_counter()
        current_date = datetime.now().strftime("%Y年%m月%d日 %A")

        if direct:
            # 简单问题直接回答模式（走 app.llm 工厂）
            llm = get_llm(temperature=0.1, max_tokens=500)
            chain = DIRECT_PROMPT | llm
            response = chain.invoke({"query": query, "history": history or "（无）", "current_date": current_date})
            logger.info("Synthesis agent (直接) LLM 耗时 %.1fs", time.perf_counter() - _t)
            return response.content

        llm = get_llm(temperature=0.1, max_tokens=1200)
        chain = SYNTHESIS_PROMPT | llm
        response = chain.invoke({
            "query": query,
            "history": history or "（无）",
            "rag_context": rag_context or "未检索到知识库上下文。",
            "web_context": web_context or "未执行网络搜索。",
            "weather_context": weather_context or "未获取到天气信息。",
            "previous_answer": previous_answer or "（无）",
            "critique": critique or "（无）",
            "current_date": current_date,
        })
        logger.info("Synthesis agent LLM 耗时 %.1fs", time.perf_counter() - _t)
        logger.info("Synthesis agent 完成（含评审反馈: %s）", bool(critique))
        return response.content

    async def arun_stream(
        self,
        query: str,
        rag_context: str,
        web_context: str,
        weather_context: str = "",
        critique: str = "",
        previous_answer: str = "",
        history: str = "",
        direct: bool = False,
    ):
        """流式生成：逐 token yield 内容片段。

        供 workflow 的 synthesis_node 做 token 级 SSE（感知延迟优化——
        用户约 1s 内看到首字，而不是干等完整生成）。与非流式 run()
        共用同一组 prompt / 参数，行为一致，只是传输方式不同。
        """
        current_date = datetime.now().strftime("%Y年%m月%d日 %A")

        if direct:
            chain = DIRECT_PROMPT | get_llm(temperature=0.1, max_tokens=500)
            inputs = {"query": query, "history": history or "（无）", "current_date": current_date}
        else:
            chain = SYNTHESIS_PROMPT | get_llm(temperature=0.1, max_tokens=1200)
            inputs = {
                "query": query,
                "history": history or "（无）",
                "rag_context": rag_context or "未检索到知识库上下文。",
                "web_context": web_context or "未执行网络搜索。",
                "previous_answer": previous_answer or "（无）",
                "critique": critique or "（无）",
                "weather_context": weather_context or "未获取到天气信息。",
                "current_date": current_date,
            }
        async for chunk in chain.astream(inputs):
            content = getattr(chunk, "content", "")
            if content:
                yield content


@lru_cache
def create_synthesis_agent() -> SynthesisAgent:
    return SynthesisAgent()
