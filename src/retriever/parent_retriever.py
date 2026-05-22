"""搜 child 向量库，返回保序、元数据合并后的 parent DocStore 内容。

满足 RetrieverProtocol（结构子类型），调用方无需感知内部差异。
"""

from typing import Any, Dict, List

from langchain_core.documents import Document


class ParentRetriever:
    """搜索 child → 保序去重 → parent 元数据合并。

    设计要点：
        - 保序：按相关性排序的 child 中 parent 首次出现的顺序为准
        - 去重：同一 parent 被多个 child 命中时只返回一份
        - 元数据合并：child 的检索信号字段（has_code, code_language 等）合入 parent

    元数据合并规则（容易出错）：
        child 的 chunk_id 和 parent_chunk_id 不继承到输出，
        因为输出是 parent 级别的，parent 用自己的 chunk_id 标识自己。
        child 的检索信号字段（has_code, code_language）保留。

    使用方式：
        from src.core.factories import create_parent_retriever
        retriever = create_parent_retriever(settings)
        docs = retriever.invoke("what is LangGraph")
    """

    # child 的身份字段（不合并到 parent 输出，避免覆盖 parent 自身标识）
    # parent_chunk_id 保留（它是 child 的检索链路信息，parent 自身没有该字段）
    _CHILD_IDENTITY_KEYS: frozenset = frozenset({"chunk_id"})

    def __init__(self, child_retriever: Any, doc_store: Any):
        self._child = child_retriever
        self._store = doc_store

    def invoke(self, query: str) -> List[Document]:
        """搜 child → 保序去重 → 批量读 parent → 合并元数据。"""
        children = self._child.invoke(query)
        if not children:
            return []

        # 保序去重：记录 parent_id 首次出现顺序
        parent_to_child_meta: Dict[str, Dict[str, Any]] = {}
        ordered_ids: List[str] = []

        for c in children:
            p_id = c.metadata.get("parent_chunk_id")
            if p_id is None:
                continue
            p_id_str = str(p_id)
            if p_id_str not in parent_to_child_meta:
                ordered_ids.append(p_id_str)
                # 暂存最相关（首次命中）子块的元数据用于后续合并
                parent_to_child_meta[p_id_str] = c.metadata

        # 批量读取 parent（返回 {doc_id: Document}）
        parent_dict = self._store.mget(ordered_ids)

        # 按序组装，合并 child 检索信号（但不覆盖 parent 自身 ID）
        result = []
        for p_id in ordered_ids:
            parent = parent_dict.get(p_id)
            if parent is None:
                continue

            # 从 child 筛选出非身份字段的信号（跳过 chunk_id / parent_chunk_id）
            child_signals = {
                k: v
                for k, v in parent_to_child_meta[p_id].items()
                if k not in self._CHILD_IDENTITY_KEYS
            }
            merged = {**parent.metadata, **child_signals}
            result.append(Document(
                page_content=parent.page_content,
                metadata=merged,
            ))
        return result

    async def ainvoke(self, query: str) -> List[Document]:
        """异步版本：线上高并发服务必备。"""
        children = await self._child.ainvoke(query)
        if not children:
            return []

        parent_to_child_meta: Dict[str, Dict[str, Any]] = {}
        ordered_ids: List[str] = []

        for c in children:
            p_id = c.metadata.get("parent_chunk_id")
            if p_id is None:
                continue
            p_id_str = str(p_id)
            if p_id_str not in parent_to_child_meta:
                ordered_ids.append(p_id_str)
                parent_to_child_meta[p_id_str] = c.metadata

        parent_dict = self._store.mget(ordered_ids)

        return [
            Document(
                page_content=parent_dict[p_id].page_content,
                metadata={**parent_dict[p_id].metadata, **parent_to_child_meta[p_id]},
            )
            for p_id in ordered_ids
            if p_id in parent_dict
        ]


__all__ = [
    "ParentRetriever",
]
