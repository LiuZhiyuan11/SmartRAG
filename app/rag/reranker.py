"""
OmniRAG — Cross-Encoder 重排序器
──────────────────────────────────
使用 BAAI/bge-reranker-base 对查询-文档对进行评分，
并返回排名前 N 的最相关结果。
"""

import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    使用 sentence-transformers 的轻量级 cross-encoder 重排序器。
    模型：BAAI/bge-reranker-base（免费，本地运行）。
    """

    _model = None  # 懒加载单例
    _model_failed = False  # 加载失败/未安装后不再重复尝试（避免每查询重下模型拖慢）

    def __init__(self, top_n: int = 5, model_name: str = "BAAI/bge-reranker-base"):
        self.top_n = top_n
        self.model_name = model_name

    @staticmethod
    def _model_is_cached(model_name: str) -> bool:
        """模型是否已在本地 HF 缓存（直接检查权重文件，避免网络下载拖慢查询）。

        注意：不能用 snapshot_download(local_files_only=True) 探测——
        部分缓存/残留 .lock 时它会阻塞等待文件锁，导致查询卡死（实测坑）。
        直接遍历 snapshots 目录检查权重文件存在性，纯本地 I/O，毫秒级返回。
        """
        try:
            import os

            # ★ 优先读 HF_HOME，回退到默认路径
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                cache_root = os.path.join(hf_home, "hub")
            else:
                cache_root = os.path.join(
                    os.path.expanduser("~"), ".cache", "huggingface", "hub"
                )

            repo_dir = os.path.join(
                cache_root, "models--" + model_name.replace("/", "--")
            )
            snapshots = os.path.join(repo_dir, "snapshots")
            if not os.path.isdir(snapshots):
                return False
            for entry in os.scandir(snapshots):
                if not entry.is_dir():
                    continue
                for weight_file in ("model.safetensors", "pytorch_model.bin"):
                    if os.path.exists(os.path.join(entry.path, weight_file)):
                        return True
            return False
        except Exception:
            return False

    def _get_model(self):
        if CrossEncoderReranker._model_failed:
            return None
        if CrossEncoderReranker._model is None:
            if not self._model_is_cached(self.model_name):
                logger.warning(
                    "重排序模型 %s 未本地缓存，跳过重排。"
                    "可用 HF_ENDPOINT=https://hf-mirror.com 预下载后启用。",
                    self.model_name,
                )
                CrossEncoderReranker._model_failed = True
                return None
            try:
                from sentence_transformers import CrossEncoder
                CrossEncoderReranker._model = CrossEncoder(self.model_name)
                logger.info("已加载重排序器: %s", self.model_name)
            except Exception as e:
                logger.warning("重排序器加载失败（%s），跳过重排序。", e)
                CrossEncoderReranker._model = None
                CrossEncoderReranker._model_failed = True
        return CrossEncoderReranker._model

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """根据与查询的相关性对文档进行评分和排序。"""
        model = self._get_model()
        if model is None or not documents:
            return documents[: self.top_n]

        pairs = [(query, doc.page_content) for doc in documents]
        scores = model.predict(pairs)

        scored = sorted(
            zip(scores, documents), key=lambda x: x[0], reverse=True
        )

        reranked = []
        for score, doc in scored[: self.top_n]:
            doc.metadata["rerank_score"] = float(score)
            reranked.append(doc)

        return reranked
