# -*- coding: utf-8 -*-
"""全局配置。

统一从环境变量（或项目根目录的 .env）读取，API Key 不写进代码。
三套实现（手写版 / LangChain 版 / Qwen-Agent 版）共用这里的参数，
保证切分粒度、召回条数、相似度阈值、模型选择完全一致，便于横向对比。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_DOC_PATH = DATA_DIR / "yingke.txt"


def load_env_file(path: Optional[Union[str, Path]] = None, override: bool = False) -> None:
    """加载 .env。装了 python-dotenv 就用它，没装则用内置极简解析兜底。"""
    env_path = Path(path) if path else PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if override or key not in os.environ:
                os.environ[key] = value
    else:
        load_dotenv(env_path, override=override)


load_env_file()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ========== 智谱 GLM 接口 ==========
API_KEY = (os.getenv("ZHIPU_API_KEY") or os.getenv("ZHIPUAI_API_KEY") or "").strip()
API_BASE = os.getenv("ZHIPU_API_BASE", "https://open.bigmodel.cn/api/paas/v4").rstrip("/")
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "glm-4-flash")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "embedding-3")
REQUEST_TIMEOUT = _env_int("RAG_REQUEST_TIMEOUT", 60)
MAX_RETRIES = _env_int("RAG_MAX_RETRIES", 2)
EMBED_BATCH_SIZE = _env_int("RAG_EMBED_BATCH_SIZE", 32)

# ========== 文档与切分 ==========
DOC_PATH = Path(os.getenv("RAG_DOC_PATH", str(DEFAULT_DOC_PATH)))
CHUNK_SIZE = _env_int("RAG_CHUNK_SIZE", 200)
CHUNK_OVERLAP = _env_int("RAG_CHUNK_OVERLAP", 30)

# ========== 检索 ==========
TOP_K = _env_int("RAG_TOP_K", 3)
# 余弦相似度阈值：低于它的召回结果直接丢弃，是"拒答"的第一道闸门。
# embedding-3 的中文向量基线相似度偏高，这个值必须用真实分数校准：
#   python -m handcrafted_rag.main --diagnose "英科医疗2025年上半年营收是多少"
#   python -m handcrafted_rag.main --diagnose "帮我写一个快速排序"
MIN_SCORE = _env_float("RAG_MIN_SCORE", 0.35)

REJECT_MSG = "根据现有资料无法回答"

# langchain 的 ZhipuAIEmbeddings 默认读 ZHIPUAI_API_KEY，这里顺手对齐，避免两套变量名混用
if API_KEY:
    os.environ.setdefault("ZHIPUAI_API_KEY", API_KEY)


def require_api_key() -> str:
    """取 API Key；未配置时直接给出可操作的报错，而不是等到 401 才发现。"""
    if not API_KEY:
        raise RuntimeError(
            "未配置 API Key：请复制 .env.example 为 .env 并填入 ZHIPU_API_KEY=你的key，"
            "或直接设置同名环境变量。"
        )
    return API_KEY
