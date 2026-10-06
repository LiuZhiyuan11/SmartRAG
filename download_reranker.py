"""
手动下载 BAAI/bge-reranker-base 模型，带实时进度显示。
用法：python download_reranker.py
"""

import os
import sys
import time

# ★ 强制设置 HF_HOME 到 D 盘（如果你想改路径，改这里）
os.environ.setdefault("HF_HOME", r"D:\hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

print(f"HF_HOME = {os.environ['HF_HOME']}")
print(f"HF_ENDPOINT = {os.environ['HF_ENDPOINT']}")
print("=" * 60)

# ★ 让 huggingface_hub 打开进度条
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

MODEL_NAME = "BAAI/bge-reranker-base"

print(f"\n开始下载模型：{MODEL_NAME}")
print(f"缓存目录：{os.path.join(os.environ['HF_HOME'], 'hub')}\n")

t0 = time.time()

try:
    # 用 snapshot_download 单独下载，能看到文件级别的进度
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=MODEL_NAME,
        repo_type="model",
        resume_download=True,          # 断点续传
        local_files_only=False,        # 允许联网下载
        max_workers=4,                 # 并发下载
    )
    print(f"\n✅ 模型已下载到：{path}")

except Exception as e:
    print(f"\n❌ 下载失败：{e}")
    sys.exit(1)

# ★ 再验证一次能不能加载
print("\n验证模型能否加载...")
try:
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(MODEL_NAME)
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"⚠️ 模型加载失败：{e}")
    sys.exit(1)

elapsed = time.time() - t0
print(f"\n总耗时：{elapsed:.1f} 秒")
print(f"缓存位置：{os.path.join(os.environ['HF_HOME'], 'hub', 'models--BAAI--bge-reranker-base')}")