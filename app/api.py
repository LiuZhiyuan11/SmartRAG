"""
OmniRAG — FastAPI 接口层
────────────────────────────────────
React 前端的后端 API，同时提供 REST 与 SSE 流式接口。

启动方式：
    uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

接口一览：
    GET  /api/health          健康检查
    POST /api/query           查询主接口（监督者路由 → RAG/Web → 综合 → 评估）
    POST /api/ingest/file     上传文件导入知识库（multipart/form-data）
    POST /api/ingest/text     直接导入文本
"""

import json
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from tkinter import image_names
from typing import Optional
from pathlib import Path
from fastapi import UploadFile,File
import uuid

# 图片文件上传文件夹
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 确保项目根目录在 sys.path 中（python -m uvicorn 时 cwd 可能不在项目根）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# .libs 目录装了所有第三方依赖，必须显式加入 sys.path
_LIBS_DIR = os.path.join(_PROJECT_ROOT, ".libs")
if os.path.isdir(_LIBS_DIR) and _LIBS_DIR not in sys.path:
    sys.path.insert(0, _LIBS_DIR)

# 加载 .env 到 os.environ
from dotenv import load_dotenv

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# 从 Windows 注册表读取系统代理（Clash/V2Ray 等），Python 默认不读注册表代理
if not os.environ.get("HTTPS_PROXY") and not os.environ.get("HTTP_PROXY"):
    try:
        import winreg
        reg_path = r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path) as key:
            proxy_enable = winreg.QueryValueEx(key, "ProxyEnable")[0]
            if proxy_enable:
                proxy_server = winreg.QueryValueEx(key, "ProxyServer")[0]
                proxy_url = f"http://{proxy_server}"
                os.environ["HTTP_PROXY"] = proxy_url
                os.environ["HTTPS_PROXY"] = proxy_url
    except Exception:
        pass

# HuggingFace 模型下载走国内镜像，避免直连 huggingface.co 超时
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from app.config import settings

# 复用工作流与导入模块的同一套核心函数
from app.graph.workflow import run_query, stream_query
from app.rag.evaluation import run_evaluation
from app.rag.ingestion import extract_from_text, ingest_documents, ingest_file

logger = logging.getLogger(__name__)

# 日志同时输出到终端和 logs/omnirag.log
_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(_LOG_DIR, "omnirag.log"), encoding="utf-8"),
    ],
)


# ── 启动预热 ──────────────────────────────────────────────────────────────────

def _warmup_worker():
    """后台预热：初始化向量库单例（Qdrant 客户端 + 集合校验 + dense/sparse 嵌入器）。

    向量库首次初始化约 25-30s（含 Qdrant 云多次往返），若放到首条查询才触发，
    用户会明显感到"卡顿"。启动时在后台线程提前完成，首条查询直接复用。
    同时预检重排序器：未缓存则提前标记失败，避免首条查询额外等待 ~8s 的模型检查。
    """
    try:
        from app.rag.ingestion import get_vector_store
        t = time.perf_counter()
        get_vector_store()
        logger.info("启动预热完成：向量库初始化 %.1fs", time.perf_counter() - t)
    except Exception as e:
        logger.warning("启动预热失败（不影响服务，查询时懒加载兜底）: %s", e)
    try:
        from app.rag.reranker import CrossEncoderReranker
        CrossEncoderReranker()._get_model()
    except Exception:
        pass


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warmup_worker, daemon=True).start()
    yield


# ── 应用实例 ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OmniRAG API",
    description="多智能体混合 RAG 研究平台 REST 接口",
    version="1.0.0",
    lifespan=lifespan,
)

# 允许前端跨域调用，方便 React/Vue 本地开发
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求耗时记录中间件 ─────────────────────────────────────────────────────

@app.middleware("http")
async def log_request_duration(request, call_next):
    """记录每个 HTTP 请求的方法、路径和耗时。"""
    start = time.perf_counter()
    method = request.method
    path = request.url.path
    response = await call_next(request)
    duration = time.perf_counter() - start
    logger.info("API %s %s → %d (%.1fs)", method, path, response.status_code, duration)
    return response


# ── 请求/响应模型 ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    thread_id: Optional[str] = Field(
        None, description="会话 ID（不传则自动生成新会话）"
    )
    image_path: str = ""


class QueryResponse(BaseModel):
    query: str
    thread_id: str
    final_answer: str
    rag_context: str = ""
    web_context: str = ""
    critique: str = ""
    iterations: int = 1
    route: str = "both"


class TextIngestRequest(BaseModel):
    text: str = Field(..., description="待导入的文本内容")
    source: str = Field("manual", description="来源标签")


class IngestResponse(BaseModel):
    chunks: int
    message: str = ""


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


class EvalRequest(BaseModel):
    k: Optional[int] = Field(
        None, description="检索 top-k（默认用 RETRIEVAL_TOP_K）"
    )


class EvalResponse(BaseModel):
    metrics: dict
    sample_count: int
    summary: str
    report_path: Optional[str] = None


class AppConfigResponse(BaseModel):
    app_title: str
    model: str
    embedding_model: str
    use_multi_query: bool
    max_iterations: int
    retrieval_top_k: int
    reranker_top_n: int
    history_window: int


# ── 路由 ─────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """健康检查，可用作 Docker/K8s 存活探针。"""
    return HealthResponse()


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """
    主查询接口：
        监督者路由 → RAG/Web Agent → 综合 → 评估循环
    等价于直接调用 run_query(query, thread_id)。
    """
    thread_id = req.thread_id or str(uuid.uuid4())
    try:
        result = run_query(req.query, thread_id=thread_id)
        return QueryResponse(
            query=req.query,
            thread_id=thread_id,
            final_answer=result.get("final_answer") or result.get("draft_answer", ""),
            rag_context=result.get("rag_context", ""),
            web_context=result.get("web_context", ""),
            critique=result.get("critique", ""),
            iterations=result.get("iterations", 1),
            route=result.get("route", "both"),
        )
    except Exception as e:
        logger.exception("查询失败: %s", req.query)
        raise HTTPException(status_code=500, detail=f"查询失败: {e}")


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """
    流式查询接口（SSE）：逐步推送 Agent 执行事件，
    前端可实时展示 路由 → 检索 → 综合 → 评审 的过程。

    事件格式：data: {"type": "status|critique|final|error", ...}
    """
    thread_id = req.thread_id or str(uuid.uuid4())

    async def event_gen():
        try:
            async for event in stream_query(req.query,
                                            thread_id,
                                            image_path=req.image_path):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("流式查询失败")
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/config", response_model=AppConfigResponse)
def app_config() -> AppConfigResponse:
    """向前端暴露应用配置（不含任何密钥）。"""
    return AppConfigResponse(
        app_title=settings.app_title,
        model=settings.zhipu_model,
        embedding_model=settings.zhipu_embedding_model,
        use_multi_query=settings.use_multi_query,
        max_iterations=settings.max_iterations,
        retrieval_top_k=settings.retrieval_top_k,
        reranker_top_n=settings.reranker_top_n,
        history_window=settings.history_window,
    )


@app.get("/api/stats")
def collection_stats() -> dict:
    """知识库统计：向量数量、来源分布、内容类型分布。"""
    from app.rag.ingestion import get_qdrant_client

    try:
        client = get_qdrant_client()
        info = client.get_collection(settings.qdrant_collection)
        records, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=10000,
            with_payload=["metadata.source", "metadata.content_type"],
            with_vectors=False,
        )
        sources: dict = {}
        content_types: dict = {}
        for r in records:
            meta = (r.payload or {}).get("metadata") or {}
            src = meta.get("source") or "unknown"
            ct = meta.get("content_type") or "text"
            sources[src] = sources.get(src, 0) + 1
            content_types[ct] = content_types.get(ct, 0) + 1
        return {
            "points": getattr(info, "points_count", None),
            "sources": sources,
            "content_types": content_types,
            "collection": settings.qdrant_collection,
        }
    except Exception as e:
        logger.warning("读取集合统计失败: %s", e)
        # 始终返回完整结构，避免前端拿到缺字段的 200 响应
        return {
            "points": None,
            "sources": {},
            "content_types": {},
            "collection": settings.qdrant_collection,
            "error": str(e),
        }


@app.post("/api/ingest/file", response_model=IngestResponse)
async def ingest_file_endpoint(file: UploadFile = File(...)) -> IngestResponse:
    """
    上传文件导入知识库（支持 .pdf/.docx/.txt/.md）。
    使用临时文件落地后复用 app.rag.ingestion.ingest_file。
    """
    suffix = os.path.splitext(file.filename or "")[1]
    if suffix.lower() not in {".pdf", ".docx", ".txt", ".md"}:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {suffix}（仅支持 pdf/docx/txt/md）",
        )

    tmp_path = None
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        chunks = ingest_file(tmp_path)
        return IngestResponse(
            chunks=chunks,
            message=f"已从 {file.filename} 导入 {chunks} 个分块",
        )
    except Exception as e:
        logger.exception("文件导入失败: %s", file.filename)
        raise HTTPException(status_code=500, detail=f"导入失败: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """接收前端上传的图片，保存到本地，返回路径"""
    original_name = file.filename or ""
    logger.info(
        "图片上传开始: filename=%s content_type=%s",
        original_name,
        file.content_type,
    )

    # 1、生成唯一文件名，防止冲突
    ext = Path(original_name).suffix.lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        logger.warning("图片上传拒绝: filename=%s extension=%s", original_name, ext)
        raise HTTPException(status_code=400, detail="不支持的图片格式")
    # 生成随即名，减少重名冲突
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / filename

    # 2、写入磁盘
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    logger.info(
        "图片已保存: filename=%s bytes=%d path=%s",
        original_name,
        len(content),
        save_path.resolve(),
    )

    # 3、返回绝对路径(因为YOLO模型需要绝对路径)
    return {
        "image_path": str(save_path.resolve()),
        "file_name": original_name,
    }



@app.post("/api/ingest/text", response_model=IngestResponse)
def ingest_text_endpoint(req: TextIngestRequest) -> IngestResponse:
    """直接导入文本（先切分再向量化）。"""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="文本内容不能为空")

    try:
        docs = extract_from_text(req.text, source=req.source)
        chunks = ingest_documents(docs)
        return IngestResponse(
            chunks=chunks,
            message=f"已导入 {chunks} 个分块（来源: {req.source}）",
        )
    except Exception as e:
        logger.exception("文本导入失败")
        raise HTTPException(status_code=500, detail=f"导入失败: {e}")


# ── RAGAS 评估（黄金集 + 4 维 LLM-judge）────────────────────────────────────

@app.post("/api/evaluate", response_model=EvalResponse)
def evaluate_endpoint(req: EvalRequest) -> EvalResponse:
    """
    跑一次 RAGAS 评估：
        1. 从 data/golden_set.json 加载人工校验样本
        2. 对每条样本调 run_query 获取答案 + 检索 top-k 原始 chunk
        3. 构造成 RAGAS SingleTurnSample，输出 4 个指标：
           - faithfulness        答案是否忠于检索上下文（无幻觉）
           - answer_relevancy    答案是否切题
           - context_precision   检索上下文是否包含相关信息
           - context_recall      标准答案信息是否被检索到
    """
    try:
        result = run_evaluation(k=req.k)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return EvalResponse(
            metrics=result["metrics"],
            sample_count=result["sample_count"],
            summary=result["summary"],
            report_path=result.get("report_path"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("评估失败")
        raise HTTPException(status_code=500, detail=f"评估失败: {e}")


# ── 对话管理 ──────────────────────────────────────────────────────────────────

@app.delete("/api/conversation/{thread_id}")
def delete_conversation(thread_id: str):
    """删除指定对话的 SQLite checkpoint 数据。"""
    import sqlite3
    db_path = os.path.join(settings.data_dir, "checkpoints.db")
    if not os.path.exists(db_path):
        return {"message": "数据库不存在", "deleted": 0}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # langgraph 的 SQLite saver 表名通常是 checkpoints / writes
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        logger.info("删除对话 %s: %d 条 checkpoint 记录", thread_id, deleted)
        return {"message": "对话已删除", "thread_id": thread_id}
    except Exception as e:
        logger.error("删除对话失败: %s", e)
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")


# ── 前端静态资源（frontend/dist，构建后由 FastAPI 同源托管）────────────────
_FRONTEND_DIST = os.path.join(_PROJECT_ROOT, "frontend", "dist")

if os.path.isdir(os.path.join(_FRONTEND_DIST, "assets")):
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(_FRONTEND_DIST, "assets")),
        name="assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        index_file = os.path.join(_FRONTEND_DIST, "index.html")
        if full_path.startswith("api") or not os.path.exists(index_file):
            raise HTTPException(status_code=404)
        return FileResponse(index_file)


# ── 直接运行入口 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
