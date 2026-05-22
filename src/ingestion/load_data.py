"""数据入库编排脚本 — 加载 → 合并 metadata → 父子切分 → 入库。

调用 ingestion 包中的各模块完成完整 pipeline。

流水线步骤：
    1. 加载 Markdown 文档（解析 YAML frontmatter）
    2. 整合 metadata_index.json 补充字段
    3. 父子双层切分：parent（LLM 上下文） + child（嵌入检索）
    4. Parent chunks → DocStore（JSON 持久化）
    5. Child chunks → Chroma 向量库

使用方式：
    # 默认配置（父子切分）
    from src.ingestion import run_pipeline
    run_pipeline()

    # 恢复旧版单层切分（兼容模式）
    run_pipeline(use_parent_child=False)
"""

from pathlib import Path
from typing import List, Optional

# 注意：使用相对导入避免循环导入
# from src.ingestion import ... 会导致循环，因为 __init__.py 会导入本模块
from .doc_store import DocStore
from .loader import (
    load_directory,
    load_metadata_index,
    enrich_docs_with_index,
)
from .splitter import SmartDocumentSplitter
from .vectorstore import ingest_to_chroma


# ============================================================
# 默认配置
# ============================================================
DATA_DIR = [
    "data/langchain_docs_separated/oss/python/langchain",
    "data/langchain_docs_separated/oss/python/langgraph",
]
EXCLUDE_DIRS = ["frontend"]


def run_pipeline(
    data_dir: str | List[str] = DATA_DIR,
    exclude_dirs: List[str] = EXCLUDE_DIRS,
    metadata_json: str = "data/langchain_docs_separated/metadata_index.json",
    persist_dir: str = "db/langchain_docs_db1",
    collection_name: str = "langchain_docs1",
    doc_store_path: str = "db/doc_store.json",
    use_parent_child: bool = True,
    parent_chunk_size: int = 2000,
    child_chunk_size: int = 500,
) -> Optional[DocStore]:
    """完整流水线：加载 → 合并 metadata → 切分 → 入库。

    Args:
        data_dir: Markdown 文档目录（单个路径或路径列表）。
        exclude_dirs: 需要排除的子目录名列表。
        metadata_json: metadata_index.json 路径。
        persist_dir: Chroma 持久化目录。
        collection_name: Chroma 集合名称。
        doc_store_path: DocStore JSON 持久化路径（仅父子模式）。
        use_parent_child: True=父子双层切分，False=旧版单层切分。
        parent_chunk_size: parent 层 chunk 上限（仅父子模式）。
        child_chunk_size: child 层目标 chunk 尺寸（仅父子模式）。

    Returns:
        仅父子模式返回 DocStore 实例（内含 parent chunks）；
        单层切分模式返回 None（旧版行为）。
    """
    # ============================================================
    # 1. 加载文档（解析 frontmatter）
    # ============================================================
    print("[1/4] 加载 Markdown 文档...")
    docs = load_directory(data_dir, exclude_dirs=exclude_dirs)
    print(f"  共加载 {len(docs)} 篇文档")

    # ============================================================
    # 2. 整合 metadata_index.json
    # ============================================================
    print("[2/4] 整合 metadata_index.json...")
    index = load_metadata_index(metadata_json)
    # 用 metadata_json 所在目录作为基准路径（多目录时 data_dir 是列表）
    index_base_dir = str(Path(metadata_json).parent)
    docs = enrich_docs_with_index(docs, index, index_base_dir)

    # ============================================================
    # 3. 切分
    # ============================================================
    splitter = SmartDocumentSplitter(chunk_size=parent_chunk_size, chunk_overlap=200)

    if use_parent_child:
        print(f"[3/4] 父子双层切分（parent={parent_chunk_size}, child={child_chunk_size}）...")
        parent_docs, child_docs = splitter.parent_child_split(
            docs,
            parent_chunk_size=parent_chunk_size,
            child_chunk_size=child_chunk_size,
        )
        print(f"  父层: {len(parent_docs)} chunks")
        print(f"  子层: {len(child_docs)} chunks")

        # ============================================================
        # 4. Parent chunks → DocStore
        # ============================================================
        print("[4/5] Parent chunks → DocStore...")
        store = DocStore(doc_store_path)
        for parent in parent_docs:
            # 用 chunk_id（str(pid)）作为 DocStore 的 key
            pid = str(parent.metadata["chunk_id"])
            store.put(pid, parent)
        store.persist()
        print(f"  DocStore 已持久化: {doc_store_path} ({len(store)} parent chunks)")

        # ============================================================
        # 5. Child chunks → Chroma
        # ============================================================
        print("[5/5] Child chunks → Chroma...")
        ingest_to_chroma(
            child_docs,
            persist_directory=persist_dir,
            collection_name=collection_name,
        )
        return store

    # ============================================================
    # 旧版模式：单层切分 + Chroma
    # ============================================================
    print("[3/4] 单层智能切分...")
    chunks = splitter.smart_split(docs)
    print(f"  共产出 {len(chunks)} 个 chunks")

    print("[4/4] 存入 Chroma...")
    ingest_to_chroma(
        chunks,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    return None


if __name__ == "__main__":
    run_pipeline()
