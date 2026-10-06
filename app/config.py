"""
OmniRAG 配置 — 通过 pydantic-settings 集中管理设置。
所有值均来自环境变量（参见 .env.example）。
"""
import os

# ★ 把 HuggingFace 缓存目录迁移到 D 盘
os.environ.setdefault("HF_HOME", r"D:\hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── 大语言模型 (LLM) — 智谱 GLM ────────────────────────────────
    zhipuai_api_key: str = Field(..., validation_alias="ZHIPUAI_API_KEY")
    zhipu_model: str = Field("glm-4-flash", validation_alias="ZHIPU_MODEL")
    zhipu_embedding_model: str = Field(
        "embedding-3", validation_alias="ZHIPU_EMBEDDING_MODEL"
    )

    # ── 评估 judge LLM（可选，DeepSeek 官方 API）───────────────────
    # 配了 DEEPSEEK_API_KEY 时，RAGAS 的 judge 用 DeepSeek 而非智谱 GLM，
    # 消除「judge 与生成同模型」的自引用偏差；未配置则回退智谱 GLM。
    deepseek_api_key: str = Field("", validation_alias="DEEPSEEK_API_KEY")
    eval_judge_model: str = Field("deepseek-chat", validation_alias="EVAL_JUDGE_MODEL")

    # ── LangSmith 追踪配置 ────────────────────────────────────────
    langchain_tracing_v2: str = Field("true", validation_alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(..., validation_alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(
        "omnirag-production", validation_alias="LANGCHAIN_PROJECT"
    )
    langchain_endpoint: str = Field(
        "https://api.smith.langchain.com", validation_alias="LANGCHAIN_ENDPOINT"
    )

    # ── Qdrant 向量数据库（免费云服务）───────────────────────
    qdrant_url: str = Field(..., validation_alias="QDRANT_URL")
    qdrant_api_key: str = Field(..., validation_alias="QDRANT_API_KEY")
    qdrant_collection: str = Field("omnirag_hybrid", validation_alias="QDRANT_COLLECTION")

    # ── Tavily 网页搜索（免费版）────────────────────────────
    tavily_api_key: str = Field(..., validation_alias="TAVILY_API_KEY")

    # ── RAG 检索调优 ─────────────────────────────────────────────
    retrieval_top_k: int = Field(10, validation_alias="RETRIEVAL_TOP_K")
    reranker_top_n: int = Field(5, validation_alias="RERANKER_TOP_N")
    # 小 chunk（400/40）+ 按 Markdown 标题优先切分（separators）保证主题聚焦，
    # 避免多主题大块稀释检索余弦相似度（实测 recall@k 从 0.50 → 1.00）。
    # 注意：默认值与 .env.example 保持一致，避免配置漂移。
    chunk_size: int = Field(400, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(40, validation_alias="CHUNK_OVERLAP")

    # ── 多查询扩展（Multi-Query Retrieval）────────────────────
    use_multi_query: bool = Field(False, validation_alias="USE_MULTI_QUERY")
    multi_query_count: int = Field(3, validation_alias="MULTI_QUERY_COUNT")
    # 元数据过滤器提取（默认关：GLM 易误判 query 为 source 导致 0 命中，且每查询多一次 LLM 调用）
    use_filter_extraction: bool = Field(False, validation_alias="USE_FILTER_EXTRACTION")

    # ── 对话记忆 ──────────────────────────────────────────────
    history_window: int = Field(6, validation_alias="HISTORY_WINDOW")

    # ── Agent 设置 ───────────────────────────────────────────
    # 评审闭环最大轮数。各 Agent 的温度统一在 app/llm.py 的 get_llm() 里按角色指定，
    # 不再提供全局 AGENT_TEMPERATURE（避免死配置）。
    max_iterations: int = Field(5, validation_alias="MAX_ITERATIONS")

    # ── 评估（黄金集 + LLM-as-judge）──────────────────────────
    eval_golden_path: str = Field(
        "data/golden_set.json", validation_alias="EVAL_GOLDEN_PATH"
    )
    eval_report_dir: str = Field(
        "data/eval_reports", validation_alias="EVAL_REPORT_DIR"
    )

    # ── 应用配置 ──────────────────────────────────────────────
    app_title: str = "OmniRAG — 多智能体混合研究平台"
    data_dir: str = "data"
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")


settings = Settings()
