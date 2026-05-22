"""Parent 文档存储：JSON 文件持久化 + 内存 dict。

职责：
    存储 parent chunks（LLM 消费的上下文块），以 JSON 文件持久化。
    启动时加载到内存，检索时 O(1) 查询。

设计：
    1. 最小接口：put / get / mget / persist / load / clear
    2. 可迁移：后续切换为 Redis/SQLite 只需重新实现这六个方法
    3. 线程安全注意：当前未加锁，仅适用于单线程入库操作

使用方式：
    store = DocStore("db/doc_store.json")
    store.put("0", parent_doc)
    store.persist()
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


def _json_safe(value: Any) -> Any:
    """将非 JSON 可序列化类型转换为安全类型。

    常见来源：YAML frontmatter 中的 lastmod（datetime）等。
    """
    if isinstance(value, datetime):
        return value.isoformat()
    # 如有其他不可序列化类型（date, Decimal 等），在此扩展
    # 但不做静默 str() 转换——捕捉未知类型以避免数据静默丢失
    return value


def _serialize_doc(doc: Document) -> dict:
    """将 Document 序列化为可 JSON 化的 dict。

    注意：
        - datetime 等非原生类型会被转换为 ISO 格式字符串
        - Chroma metadata 中只支持 str/int/float/bool，
          但 DocStore 的 JSON 存储宽松得多，不过仍需 JSON 兼容。
    """
    return {
        "page_content": doc.page_content,
        "metadata": {
            k: _json_safe(v) for k, v in doc.metadata.items()
        },
    }


def _deserialize_doc(data: dict) -> Document:
    """从 dict 反序列化为 Document。"""
    return Document(page_content=data["page_content"], metadata=data["metadata"])


class DocStore:
    """Parent chunks 存储：内存 dict + JSON 文件持久化。

    Args:
        path: JSON 持久化文件路径。不存在的文件会在首次 persist 时创建。
    """

    def __init__(self, path: str):
        self._path = path
        self._store: Dict[str, Document] = {}

    # -----------------------------------------------------------
    # 写操作
    # -----------------------------------------------------------

    def put(self, doc_id: str, document: Document) -> None:
        """存储一篇 parent chunk。

        Args:
            doc_id: 唯一标识（通常对应 chunk_id）。
            document: parent Document（page_content + metadata）。
        """
        self._store[doc_id] = document

    # -----------------------------------------------------------
    # 读操作
    # -----------------------------------------------------------

    def get(self, doc_id: str) -> Optional[Document]:
        """按 doc_id 获取 parent chunk。"""
        return self._store.get(doc_id)

    def mget(self, doc_ids: List[str]) -> Dict[str, Document]:
        """批量获取，返回 {doc_id: Document}，不存在的 ID 不出现。

        返回 dict 而非 list，方便 caller 做 doc_id→Document 查找。
        保留 doc_ids 顺序：调用方通过 `ordered_ids` 列表保证顺序。
        """
        return {
            pid: self._store[pid]
            for pid in doc_ids
            if pid in self._store
        }

    # -----------------------------------------------------------
    # 持久化
    # -----------------------------------------------------------

    def persist(self) -> None:
        """将内存中的 store 写入 JSON 文件。

        为什么写 JSON 而非 pickle：
            JSON 可读、可 diff、可版本控制，方便调试。
        """
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)

        serialized = {
            doc_id: _serialize_doc(doc)
            for doc_id, doc in self._store.items()
        }
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        """从 JSON 文件加载到内存。文件不存在时静默跳过。

        在入库脚本开始时调用，用于增量入库场景（追加而非重建）。
        """
        if not Path(self._path).exists():
            return
        with open(self._path, "r", encoding="utf-8") as f:
            serialized = json.load(f)
        self._store = {
            doc_id: _deserialize_doc(data)
            for doc_id, data in serialized.items()
        }

    def clear(self) -> None:
        """清空内存 store（不影响已持久化的文件）。

        重新入库时调用：先 clear()，再 put()，最后 persist() 覆盖。
        """
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._store

    def __repr__(self) -> str:
        return f"DocStore(path={self._path!r}, count={len(self._store)})"


__all__ = [
    "DocStore",
]
