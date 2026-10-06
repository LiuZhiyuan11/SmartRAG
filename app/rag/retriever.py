"""
OmniRAG — 混合检索器
──────────────────────────
双路混合召回：Dense（智谱 embedding）+ Sparse（BM25）→ 倒数秩融合
+ Cross-Encoder 重排序。支持通过 LLM 提取结构化过滤器进行元数据过滤。
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, ConfigDict, Field

from app.config import settings
from app.llm import get_llm
from app.rag.ingestion import get_vector_store
from app.rag.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)
from concurrent.futures import ThreadPoolExecutor
import hashlib

# ── 过滤器提取模式 ────────────────────────────────────────────────────────────

class RAGFilters(BaseModel):
    """从自然语言查询中提取的结构化过滤器。"""
    source: Optional[str] = Field(None, description="文档/文件名过滤器")
    content_type: Optional[str] = Field(
        None, description="可选值：text, table, image"
    )
    page: Optional[int] = Field(None, description="特定页码")


FILTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个过滤器提取助手。
给定用户查询，仅当其中【明确包含结构化元数据过滤器】时才提取，否则全部返回 null。

判定标准（同时满足才提取）：
- 出现了 "source: <名称>"、"来自 <文档名>"、"<文件名>" 这种明确的文档定位意图；
- 或 "第 N 页"（page）；
- 或 "表格/图片"（content_type）这类显式限定词。
普通的问题、问句、泛指名词一律不构成过滤器，全部返回 null。

模式字段：
- source：文档名称/标识符（精确匹配）
- content_type：仅可选 'text'、'table' 或 'image'
- page：整数页码

务必极度保守——99% 的查询都不该有过滤器，只有用户明确指定文件/页码时才设置。"""),
    ("human", "查询: {query}"),
])


# ── 混合检索器 ────────────────────────────────────────────────────────────────

class HybridRetriever(BaseRetriever):
    """
    双路混合检索器：
    1. LLM 从查询中提取元数据过滤器
    2. 在 Qdrant 上进行混合搜索（dense + sparse RRF）
    3. 对候选结果进行 Cross-encoder 重排序
    """

    top_k: int = Field(default_factory=lambda: settings.retrieval_top_k)
    reranker_top_n: int = Field(default_factory=lambda: settings.reranker_top_n)
    use_reranking: bool = True
    use_filter_extraction: bool = Field(default_factory=lambda: settings.use_filter_extraction)
    use_multi_query: bool = Field(default_factory=lambda: settings.use_multi_query)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:

        # ── 步骤 0：多查询扩展（可选）──────────────────────
        if self.use_multi_query:
            base = HybridRetriever(
                top_k=self.top_k,
                reranker_top_n=self.reranker_top_n,
                # use_reranking=self.use_reranking,
                use_reranking=False,    # 关掉子检索精排
                use_filter_extraction=self.use_filter_extraction,
                use_multi_query=False,  # 防止递归
            )
            mq = MultiQueryRetriever(base, num_queries=settings.multi_query_count)
            return mq.retrieve(query)

        # ── 步骤 1：提取结构化过滤器 ────────────────────
        filters = {}
        if self.use_filter_extraction:
            filters = self._extract_filters(query)
            if filters:
                logger.info("提取的过滤器: %s", filters)

        # ── 步骤 2：混合向量搜索 ──────────────────────────
        vs = get_vector_store()
        search_kwargs: Dict[str, Any] = {"k": self.top_k}
        if filters:
            search_kwargs["filter"] = self._build_qdrant_filter(filters)

        try:
            candidates = vs.similarity_search(query, **search_kwargs)
        except Exception as e:
            # 混合检索（prefetch dense+sparse）在南美云/代理下偶发失败，
            # 降级为纯 dense 检索保证查询仍可用（稳定性兜底）。
            logger.warning("混合检索失败 (%s)，降级为纯 dense 检索。", e)
            from app.rag.ingestion import get_dense_vector_store

            candidates = get_dense_vector_store().similarity_search(query, **search_kwargs)
        logger.info("通过混合检索检索到 %d 个候选结果。", len(candidates))

        if not candidates:
            return []

        # ── 步骤 3：Cross-encoder 重排序 ───────────────────────
        if self.use_reranking and len(candidates) > self.reranker_top_n:
            reranker = CrossEncoderReranker(top_n=self.reranker_top_n)
            candidates = reranker.rerank(query, candidates)
            logger.info("Reranked to top %d results.", len(candidates))

        return candidates

    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """使用 LLM 从查询中提取元数据过滤器。"""
        try:
            llm = get_llm(temperature=0)
            structured_llm = llm.with_structured_output(RAGFilters)
            chain = FILTER_PROMPT | structured_llm
            result: RAGFilters = chain.invoke({"query": query})
            return {k: v for k, v in result.model_dump().items() if v is not None}
        except Exception as e:
            logger.warning("过滤器提取失败: %s", e)
            return {}

    def _build_qdrant_filter(self, filters: Dict[str, Any]) -> Dict:
        """将提取的过滤器转换为 Qdrant 过滤器格式。

        注意：langchain_qdrant 把 metadata 存在 payload.metadata 嵌套对象下，
        所以过滤字段路径必须是 'metadata.<field>'，直接用 '<field>' 匹配不到。
        """
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue
        conditions = []
        for key, value in filters.items():
            # 嵌套字段路径：metadata.source / metadata.content_type / metadata.page
            conditions.append(
                FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
            )
        return Filter(must=conditions) if conditions else None


# ── 多查询扩展器 ──────────────────────────────────────────────────────────────

class MultiQueryRetriever:
    """
    生成多个查询变体以扩大检索覆盖范围，
    然后对结果进行去重。
    """

    def __init__(self, base_retriever: HybridRetriever, num_queries: int = 3):
        self.base_retriever = base_retriever
        self.num_queries = num_queries

    def retrieve(self, query: str) -> List[Document]:
        queries = self._generate_queries(query)
        logger.info("多查询变体: %s", queries)
        #  使用线程池进行多变体并发查询
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            results = list(executor.map(self.base_retriever.invoke, queries))

        all_docs: Dict[str, Document] = {}
        for i, docs in enumerate(results):
            # docs = self.base_retriever.invoke(q)
            logger.info("第 %d 路 [%s] 返回 %d 条", i + 1, queries[i], len(docs))
            for doc in docs:
                key = hashlib.md5(doc.page_content.encode()).hexdigest()
                if key not in all_docs:  # 不覆盖
                    all_docs[key] = doc

        merged = list(all_docs.values())
        logger.info("合并去重后共 %d 条，二次精排中...", len(merged))

        # ★ 关键新增：用**原始 query** 二次精排，统一分数尺度
        if len(merged) > self.base_retriever.reranker_top_n:
            reranker = CrossEncoderReranker(top_n=self.base_retriever.reranker_top_n)
            merged = reranker.rerank(query, merged)
            logger.info("二次精排后保留 %d 条，最高分 %.3f",
                        len(merged), merged[0].metadata.get("rerank_score", 0))
        else:
            logger.info("合并后条数不足，直接返回 %d 条", len(merged))

        return merged[:self.base_retriever.reranker_top_n]
        # logger.info("合并去重后共 %d 条，截取前 %d 条", len(all_docs), self.base_retriever.reranker_top_n)
        # return list(all_docs.values())[:self.base_retriever.reranker_top_n]

    def _generate_queries(self, query: str) -> List[str]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""生成 {self.num_queries} 个不同的搜索查询，
            用于检索与用户问题相关的信息。
            仅返回查询语句，每行一个，不要编号。"""),
            ("human", "{query}"),
        ])
        # 多查询扩展需要一点随机性来生成不同变体，temperature=0.3
        llm = get_llm(temperature=0.3)
        response = (prompt | llm).invoke({"query": query})
        queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        return [query] + [q for q in queries if q != query][: self.num_queries - 1]
