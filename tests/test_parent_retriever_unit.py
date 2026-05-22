"""ParentRetriever 单元测试：Mock 依赖，验证核心逻辑。

覆盖场景：
    - 正常检索：多个 child → 保序去重 → parent 返回
    - 空结果：child 返回空列表
    - 无 parent_chunk_id 的 child 被跳过
    - doc_store 中不存在的 parent 被跳过
    - 元数据合并：child meta 覆盖 parent meta
    - 去重：同 parent 的多 child 只返回一份
"""
from typing import Any, Dict, List

import pytest
from langchain_core.documents import Document

from src.retriever.parent_retriever import ParentRetriever


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def child_docs() -> List[Document]:
    """模拟从 Chroma 返回的 child chunks。

    设计意图：
        child_0/child_1 指向同一个 parent（parent_0）
        child_2 指向 parent_1
        用于验证去重逻辑：3 个 child → 2 个 unique parents
    """
    return [
        Document(
            page_content="child zero content",
            metadata={"chunk_id": 0, "parent_chunk_id": 0, "has_code": False, "source": "child_a"},
        ),
        Document(
            page_content="child one content",
            metadata={"chunk_id": 1, "parent_chunk_id": 0, "has_code": True, "source": "child_b"},
        ),
        Document(
            page_content="child two content",
            metadata={"chunk_id": 2, "parent_chunk_id": 1, "has_code": False, "source": "child_c"},
        ),
    ]


@pytest.fixture
def child_docs_without_pid() -> List[Document]:
    """部分 child 缺少 parent_chunk_id，应被跳过。"""
    return [
        Document(
            page_content="has parent",
            metadata={"chunk_id": 0, "parent_chunk_id": 0},
        ),
        Document(
            page_content="no parent",
            metadata={"chunk_id": 1},  # 缺少 parent_chunk_id
        ),
    ]


@pytest.fixture
def parent_dict() -> Dict[str, Document]:
    """模拟 DocStore 中的 parent chunks。"""
    return {
        "0": Document(
            page_content="PARENT ZERO: long context content",
            metadata={"chunk_id": 0, "title": "Parent Zero", "source": "parent_original"},
        ),
        "1": Document(
            page_content="PARENT ONE: another long context",
            metadata={"chunk_id": 1, "title": "Parent One"},
        ),
    }


@pytest.fixture
def mock_child_retriever(child_docs: List[Document]):
    """返回固定结果的 child retriever mock。"""
    class _Mock:
        def invoke(self, query: str) -> List[Document]:
            return child_docs
    return _Mock()


@pytest.fixture
def mock_doc_store(parent_dict: Dict[str, Document]):
    """基于 dict 的 DocStore mock。"""
    class _Mock:
        def mget(self, doc_ids: List[str]) -> Dict[str, Document]:
            return {
                pid: parent_dict[pid]
                for pid in doc_ids
                if pid in parent_dict
            }
    return _Mock()


@pytest.fixture
def retriever(mock_child_retriever, mock_doc_store) -> ParentRetriever:
    return ParentRetriever(child_retriever=mock_child_retriever, doc_store=mock_doc_store)


# ============================================================
# Tests
# ============================================================


class TestParentRetrieverInvoke:

    def test_normal_dedup(self, retriever: ParentRetriever):
        """3 个 child（2 个同 parent）→ 返回 2 个 parent docs。"""
        docs = retriever.invoke("test query")
        assert len(docs) == 2
        # 保序：parent_0（child_0 最先出现）应在 parent_1 之前
        assert docs[0].metadata["chunk_id"] == 0
        assert docs[1].metadata["chunk_id"] == 1

    def test_parent_has_child_content(self, retriever: ParentRetriever):
        """parent 返回的是 parent 的 page_content（长文本），不是 child 的。"""
        docs = retriever.invoke("test query")
        assert docs[0].page_content == "PARENT ZERO: long context content"
        assert "long context" in docs[0].page_content

    def test_metadata_merge(self, retriever: ParentRetriever):
        """child metadata 覆盖 parent metadata（child's source > parent's source）。"""
        docs = retriever.invoke("test query")
        # child_0 的 source="child_a" 应覆盖 parent_0 的 source="parent_original"
        assert docs[0].metadata["source"] == "child_a"
        # parent 特有的字段应保留
        assert docs[0].metadata["title"] == "Parent Zero"

    def test_empty_child(self, mock_child_retriever, mock_doc_store):
        """child 返回空列表 → 返回空列表。"""
        class _Empty:
            def invoke(self, query):
                return []
        r = ParentRetriever(child_retriever=_Empty(), doc_store=mock_doc_store)
        assert r.invoke("anything") == []

    def test_skip_missing_parent_chunk_id(self, mock_doc_store):
        """缺少 parent_chunk_id 的 child 被跳过。"""
        class _SomeMissing:
            def invoke(self, query):
                return [
                    Document(page_content="a", metadata={"parent_chunk_id": 0}),
                    Document(page_content="b", metadata={}),  # 无 parent_chunk_id
                    Document(page_content="c", metadata={"parent_chunk_id": 1}),
                ]
        r = ParentRetriever(child_retriever=_SomeMissing(), doc_store=mock_doc_store)
        docs = r.invoke("q")
        # 2 个有效的 child → 2 个 unique parents
        assert len(docs) == 2

    def test_skip_missing_parent_in_store(self, mock_child_retriever):
        """child 指向的 parent 在 doc_store 中不存在 → 跳过。"""
        class _PartialStore:
            def mget(self, doc_ids):
                # 只返回 "0"，不返回 "1"
                return {"0": Document(page_content="only zero", metadata={})}
        r = ParentRetriever(
            child_retriever=mock_child_retriever,
            doc_store=_PartialStore(),
        )
        docs = r.invoke("q")
        assert len(docs) == 1
        assert docs[0].page_content == "only zero"

    def test_all_children_same_parent(self, mock_doc_store):
        """3 个 child 同 parent → 1 个 parent。"""
        class _SameParent:
            def invoke(self, query):
                return [
                    Document(page_content="a", metadata={"parent_chunk_id": 0}),
                    Document(page_content="b", metadata={"parent_chunk_id": 0}),
                    Document(page_content="c", metadata={"parent_chunk_id": 0}),
                ]
        r = ParentRetriever(child_retriever=_SameParent(), doc_store=mock_doc_store)
        docs = r.invoke("q")
        assert len(docs) == 1


class TestParentRetrieverPreservesOrder:

    def test_relevance_order(self):
        """child 按相关性降序排列 → parent 保持此顺序。"""
        class _Ordered:
            def invoke(self, query):
                return [
                    Document(page_content="rel_high", metadata={"parent_chunk_id": 5}),
                    Document(page_content="rel_mid", metadata={"parent_chunk_id": 2}),
                    Document(page_content="rel_low", metadata={"parent_chunk_id": 7}),
                ]
        class _Store:
            def mget(self, doc_ids):
                return {
                    pid: Document(
                        page_content=f"parent_{pid}",
                        metadata={"chunk_id": int(pid), "title": f"Title {pid}"},
                    )
                    for pid in doc_ids
                }
        r = ParentRetriever(child_retriever=_Ordered(), doc_store=_Store())
        docs = r.invoke("q")
        assert [d.metadata["chunk_id"] for d in docs] == [5, 2, 7]


# ============================================================
# Metadata merge 边界测试
# ============================================================


class TestParentRetrieverMetadata:

    def test_numeric_parent_chunk_id(self):
        """parent_chunk_id 为 int 时 str() 转换不应丢失精度。"""
        children = [Document(
            page_content="test",
            metadata={"parent_chunk_id": 12345678901234567890},
        )]
        class _MockChild:
            def invoke(self, query):
                return children
        class _MockStore:
            def mget(self, doc_ids):
                # doc_ids 中的 id 应为 str 且不丢失精度
                assert doc_ids == ["12345678901234567890"]
                return {}
        r = ParentRetriever(child_retriever=_MockChild(), doc_store=_MockStore())
        assert r.invoke("q") == []
