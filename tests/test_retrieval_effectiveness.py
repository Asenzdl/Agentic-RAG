"""父子分块检索效果验证：使用真实 Chroma + DocStore 数据。

这些测试依赖已入库的数据（db/langchain_docs_db1 + db/doc_store.json），
在数据未准备时会被 pytest skip。

覆盖维度：
    - 上下文完整性：parent 的文本长度应显著大于 child
    - 来源追溯：每个 doc 都有 source/doc_id 等元数据
    - 去重有效性：关键词搜索返回不重复的 parent docs
    - 保序性：检索结果按相关性降序排列（首个 docs 应包含 query 关键词）
"""
from pathlib import Path
from typing import List

import pytest
from langchain_core.documents import Document

from src.core.config import settings

# ============================================================
# 环境检查：仅在数据就绪时运行
# ============================================================

HAS_CHROMA = Path("db/langchain_docs_db1").exists()
HAS_DOCSTORE = Path(settings.doc_store_path).exists()
SKIP_REASON = (
    "跳过：需要预入库数据。\n"
    f"  检查 Chroma: {'✓' if HAS_CHROMA else '✗'} db/langchain_docs_db1\n"
    f"  检查 DocStore: {'✓' if HAS_DOCSTORE else '✗'} {settings.doc_store_path}\n"
    "  运行 python -m src.ingestion.load_data 准备数据"
)

pytestmark = pytest.mark.skipif(not (HAS_CHROMA and HAS_DOCSTORE), reason=SKIP_REASON)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(scope="session")
def parent_retriever():
    """共享：真实 ParentRetriever 实例。"""
    from src.core.factories import create_parent_retriever
    return create_parent_retriever(settings)


@pytest.fixture(scope="session")
def child_retriever():
    """共享：原始 child 级检索器（用于对比）。"""
    from src.core.factories import create_retriever
    return create_retriever(settings, search_kwargs={"k": 5})


@pytest.fixture(scope="session")
def doc_store():
    """共享：真实 DocStore 实例。"""
    from src.ingestion.doc_store import DocStore
    store = DocStore(settings.doc_store_path)
    store.load()
    return store


@pytest.fixture(scope="session")
def known_queries() -> List[str]:
    """根据已入库的 LangChain 文档内容设计的问题。

    注意：如果数据重新入库（文档范围变化），以下问题可能需要调整。
    """
    return [
        "what is LangGraph",
        "how to define a tool in LangChain",
        "what is streaming in LangChain",
        "how to create a vector store retriever",
        "what is the ToolNode",
        "structured output in LangChain",
    ]


# ============================================================
# 检索效果测试
# ============================================================


class TestParentContextCompleteness:
    """parent chunk 是否真正提供了"上下文"而非碎片？"""

    def test_parent_is_larger_than_child(self, parent_retriever, child_retriever):
        """同 query 下 parent 的文本应显著大于 child。"""
        q = "what is LangGraph"

        parents = parent_retriever.invoke(q)
        children = child_retriever.invoke(q)

        assert len(parents) > 0
        assert len(children) > 0

        avg_parent_len = sum(len(d.page_content) for d in parents) / len(parents)
        avg_child_len = sum(len(d.page_content) for d in children) / len(children)

        print(f"\n  Parent avg length: {avg_parent_len:.0f} chars")
        print(f"  Child avg length:  {avg_child_len:.0f} chars")
        print(f"  Ratio:             {avg_parent_len / avg_child_len:.1f}x")

        # parent 应至少是 child 的 1.5 倍
        assert avg_parent_len > avg_child_len * 1.5, (
            f"Parent chunks ({avg_parent_len:.0f}) 不比 Child ({avg_child_len:.0f}) "
            f"大足够多，可能是父子切块参数不合理"
        )

    def test_parent_has_readable_length(self, parent_retriever):
        """parent chunk 通常应在合理范围内。

        注意：极少数文档自身内容很短（如一句话简介），
        此时 parent chunk 确实很短——这是文档特性，不是检索器 bug。
        验证"大部分" chunk 达标即可。
        """
        for query in ["what is LangGraph", "how to define a tool"]:
            docs = parent_retriever.invoke(query)
            assert len(docs) > 0

            short_count = sum(1 for d in docs if len(d.page_content) < 200)
            # 允许最多 1 个过短的 chunk（文档特性）
            assert short_count <= 1, (
                f"Query '{query}': {short_count}/{len(docs)} 个 parent 长度 < 200 chars，"
                f"可能是父子切分参数异常"
            )


class TestSourceAttribution:
    """检索结果是否可追溯到来源？"""

    def test_every_doc_has_source(self, parent_retriever, known_queries):
        """每个 parent doc 都应携带 source URL。"""
        for query in known_queries:
            docs = parent_retriever.invoke(query)
            for d in docs:
                assert d.metadata.get("source"), (
                    f"Query '{query[:30]}...' 返回了无 source 的 parent\n"
                    f"  content preview: {d.page_content[:80]}"
                )

    def test_every_doc_has_doc_id(self, parent_retriever, known_queries):
        """每个 parent doc 都应携带 doc_id（用于溯源）。"""
        for query in known_queries:
            docs = parent_retriever.invoke(query)
            for d in docs:
                assert d.metadata.get("doc_id"), (
                    f"Query '{query[:30]}...' 返回了无 doc_id 的 parent"
                )

    def test_every_doc_has_parent_chunk_id(self, parent_retriever):
        """返回的 parent 自身的 chunk_id 应存在且可追溯到 DocStore。"""
        from src.ingestion.doc_store import DocStore
        store = DocStore(settings.doc_store_path)
        store.load()

        docs = parent_retriever.invoke("streaming")
        for d in docs:
            pid = d.metadata.get("parent_chunk_id")
            assert pid is not None, "返回的 doc 缺少 parent_chunk_id"
            assert str(pid) in store, (
                f"parent_chunk_id={pid} 在 DocStore 中不存在，"
                f"可能是入库时 DocStore 索引没有正确更新"
            )


class TestDedupEffectiveness:
    """去重逻辑是否有效？"""

    def test_dedup_reduces_count(self, parent_retriever, child_retriever):
        """child 检索结果应多于 parent 检索结果（去重生效）。"""
        q = "LangGraph"

        parents = parent_retriever.invoke(q)
        children = child_retriever.invoke(q)

        print(f"\n  Children: {len(children)}, Parents: {len(parents)}")
        assert len(parents) <= len(children), (
            f"Parent 数量 ({len(parents)}) 不应超过 Child 数量 ({len(children)})"
        )

    def test_no_duplicate_parents(self, parent_retriever):
        """parent 检索结果不应包含重复的 chunk_id。"""
        docs = parent_retriever.invoke("tool")
        seen = set()
        for d in docs:
            pid = d.metadata.get("parent_chunk_id")
            assert pid is not None
            assert pid not in seen, f"重复的 parent_chunk_id={pid}"
            seen.add(pid)


class TestRelevanceOrder:
    """相关性排序是否得到保留？"""

    def test_first_result_contains_query_term(self, parent_retriever, known_queries):
        """首个返回结果应包含查询关键词（相关性降序的直观验证）。"""
        for query in known_queries:
            docs = parent_retriever.invoke(query)
            assert len(docs) > 0

            first = docs[0]
            # 取 query 中的核心词（拆分为小写 token）
            tokens = query.lower().split()
            content_lower = first.page_content.lower()
            match_count = sum(1 for t in tokens if t in content_lower)

            assert match_count > 0, (
                f"Query: '{query}'\n"
                f"Top-1 parent ({first.metadata.get('source', '?')}) "
                f"未包含任何 query 关键词\n"
                f"  Content starts: {first.page_content[:120]}"
            )


class TestRetrievalRobustness:
    """检索容错性。"""

    def test_empty_query_does_not_crash(self, parent_retriever):
        """空字符串查询不应抛异常。"""
        docs = parent_retriever.invoke("")
        # 空查询返回 0 个结果或全部结果都可接受
        assert isinstance(docs, list)

    def test_nonsense_query_returns_something(self, parent_retriever):
        """乱码查询不应抛异常。"""
        docs = parent_retriever.invoke("zzzzzxxxxxccccvvvvvbbbbnnnn")
        assert isinstance(docs, list)

    def test_known_term_returns_valid_results(self, parent_retriever):
        """已知文档中的概念应返回非空、有来源、内容合理的结果。"""
        docs = parent_retriever.invoke("ToolNode")
        assert len(docs) > 0, "已知概念应返回非空结果"

        for d in docs:
            # 每个结果必须有 source
            assert d.metadata.get("source"), f"结果缺少 source URL"
            # 每个结果必须有父块 ID
            assert d.metadata.get("parent_chunk_id") is not None, f"结果缺少 parent_chunk_id"
            # 每个结果必须有 doc_id（溯源用）
            assert d.metadata.get("doc_id"), f"结果缺少 doc_id"
            # 内容不应为空
            assert len(d.page_content) > 0, f"结果内容为空"
