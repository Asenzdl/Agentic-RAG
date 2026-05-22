"""检索效果查看器：同时跑 child 和 parent 检索，打印详细结果对比。

用法：
    python scripts/inspect_retrieval.py
    python scripts/inspect_retrieval.py "what is LangGraph"
    python scripts/inspect_retrieval.py "ToolNode" --top-k 10
"""

import argparse
import textwrap

from src.core.config import settings
from src.core.factories import create_retriever, create_parent_retriever
from src.ingestion.doc_store import DocStore


def fmt(doc, label: str, show_content: bool = True):
    """格式化打印一个 Document。"""
    meta = doc.metadata
    print(f"{'='*60}")
    print(f"  [{label}]")
    print(f"  source:         {meta.get('source', '-')}")
    print(f"  title:          {meta.get('title', '-')}")
    print(f"  chunk_id:       {meta.get('chunk_id', '-')}")
    print(f"  parent_chunk_id:{meta.get('parent_chunk_id', '-')}")
    print(f"  has_code:       {meta.get('has_code', '-')}")
    print(f"  doc_id:         {meta.get('doc_id', '-')}")
    print(f"  doc_category:   {meta.get('doc_category', '-')}")
    if show_content:
        content = doc.page_content
        # 截断过长的内容
        if len(content) > 400:
            content = content[:400] + f"\n  ...(truncated, total {len(doc.page_content)} chars)"
        print(f"  content ({len(doc.page_content)} chars):")
        # 缩进内容
        for line in content.split("\n"):
            print(f"    {line}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="检索效果对比工具")
    parser.add_argument("query", nargs="?", default="what is LangGraph",
                        help="检索查询（默认: 'what is LangGraph'）")
    parser.add_argument("--top-k", type=int, default=5,
                        help="检索 Top-K（默认: 5）")
    parser.add_argument("--content", action="store_true",
                        help="显示完整内容（默认只显示前 400 字符）")
    args = parser.parse_args()

    query = args.query
    top_k = args.top_k
    show_full = args.content

    print(f"\n{'#'*60}")
    print(f"  检索引擎：Chroma (child) + DocStore (parent)")
    print(f"  查询：      \"{query}\"")
    print(f"  Top-K：     {top_k}")
    print(f"{'#'*60}\n")

    # ============================================================
    # 1. Child 检索（原始向量检索）
    # ============================================================
    print("▶ CHILD 检索（嵌入粒度，直接从 Chroma 返回）")
    print("─" * 60)
    child_r = create_retriever(settings, search_kwargs={"k": top_k})
    children = child_r.invoke(query)
    print(f"  → 返回 {len(children)} 个 child chunks\n")
    for i, doc in enumerate(children):
        fmt(doc, f"CHILD #{i+1}", show_content=show_full)

    # ============================================================
    # 2. Parent 检索（去重 + 完整上下文）
    # ============================================================
    print("▶ PARENT 检索（搜 child → 去重 → 返回 parent 上下文）")
    print("─" * 60)
    parent_r = create_parent_retriever(settings, search_kwargs={"k": top_k})
    parents = parent_r.invoke(query)
    print(f"  → 返回 {len(parents)} 个 parent chunks")
    print(f"  → 去重比: {len(children)} child → {len(parents)} parent\n")
    for i, doc in enumerate(parents):
        fmt(doc, f"PARENT #{i+1}", show_content=show_full)

    # ============================================================
    # 3. 汇总对比
    # ============================================================
    print("▶ 对比摘要")
    print("─" * 60)
    if parents and children:
        avg_child = sum(len(c.page_content) for c in children) / len(children)
        avg_parent = sum(len(p.page_content) for p in parents) / len(parents)
        print(f"  平均 child length:  {avg_child:.0f} chars")
        print(f"  平均 parent length: {avg_parent:.0f} chars")
        print(f"  上下文增益:         {avg_parent/avg_child:.1f}x")

        # 显示每个 parent 由几个 child 聚合而来
        child_parents = {}
        for c in children:
            pid = c.metadata.get("parent_chunk_id")
            if pid is not None:
                child_parents.setdefault(pid, 0)
                child_parents[pid] += 1
        if child_parents:
            print(f"  child→parent 映射分布:")
            for pid, cnt in sorted(child_parents.items(), key=lambda x: -x[1])[:5]:
                print(f"    parent_id={pid}: {cnt} 个 child 命中")

    print(f"\n{'#'*60}\n")


if __name__ == "__main__":
    main()
