"""智能文档切分模块。

职责：
- 按 Markdown 标题（h1/h2/h3）进行第一阶段切分
- 代码块边界保护，确保 ``` 块不被截断
- 超大段递归字符切分
- 传播文档级 + 标题级 metadata
- 父子双层切分（parent→LLM, child→embedding）
"""

import re
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


class SmartDocumentSplitter:
    """智能文档切分器：标题切分 + 代码块保护 + 递归字符切分 + 父子双层切分。"""

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n```\n",  # 代码块结束标记
                "\n```",
                "\n\n",    # 段落
                "\n",      # 行
                ".", "!", "?", ";",
                " ",
                ""
            ],
            length_function=len,
        )

    @staticmethod
    def _protect_code_blocks(text: str, chunk_size: int) -> List[str]:
        """按代码块边界将文本切分为安全段，确保代码块不被截断。

        策略：
        1. 用正则找出所有 ```...``` 代码块的位置
        2. 在代码块边界之间的「文本段」处切分（而非在代码块内部）
        3. 如果「说明文字 + 紧跟的代码块」总长 < chunk_size，合并为一段
        4. 如果单个代码块本身超过 chunk_size，保持完整（后续递归切分会处理）

        Returns:
            字符串列表，每个元素是一个"安全段"（代码块完整不被截断）
        """
        if not text.strip():
            return []

        # 找出所有代码块的起止位置
        code_blocks = list(re.finditer(r'```[\w]*\n[\s\S]*?```', text))

        if not code_blocks:
            # 没有代码块，直接返回原文本
            return [text]

        segments = []
        last_end = 0

        for match in code_blocks:
            code_start = match.start()
            code_end = match.end()

            # 代码块之前的文本
            text_before = text[last_end:code_start]

            if text_before.strip():
                # 判断「前置文本 + 代码块」是否可以合并
                combined = text_before + text[code_start:code_end]
                if len(combined) <= chunk_size:
                    # 合并为一段
                    segments.append(combined)
                else:
                    # 分别添加：先加文本，再加代码块
                    segments.append(text_before)
                    segments.append(text[code_start:code_end])
            else:
                # 没有前置文本，直接添加代码块
                segments.append(text[code_start:code_end])

            last_end = code_end

        # 处理最后一个代码块之后的文本
        if last_end < len(text):
            remaining = text[last_end:]
            if remaining.strip():
                segments.append(remaining)

        return segments

    def smart_split(self, documents: List[Document]) -> List[Document]:
        """切分文档：
        第一阶段 → 标题切分（仅 h1/h2）
        第二阶段 → 代码块保护 + 递归切分
        """
        final_chunks: List[Document] = []

        for doc in documents:
            # 保存文档级 metadata（source, title, doc_id 等）
            doc_meta = dict(doc.metadata)

            # 第一阶段：按标题切分
            header_chunks = self.markdown_splitter.split_text(doc.page_content)

            chunk_id = 0
            for chunk in header_chunks:
                # 第二阶段：代码块保护 + 递归切分
                # 先用 _protect_code_blocks 得到安全段
                safe_segments = self._protect_code_blocks(
                    chunk.page_content, self.chunk_size
                )

                for segment in safe_segments:
                    # 对每个安全段调用递归切分
                    # RecursiveCharacterTextSplitter 会自动判断是否需要切分
                    sub_chunks = self.recursive_splitter.split_text(segment)

                    for sub_chunk in sub_chunks:
                        # 合并：文档级 meta + 标题层级 meta（h1-h2）
                        merged = {**doc_meta, **chunk.metadata}
                        merged["chunk_id"] = chunk_id

                        # 检测代码块信息
                        has_code, code_language = _extract_code_info(sub_chunk)
                        merged["has_code"] = has_code
                        merged["code_language"] = code_language

                        final_chunks.append(Document(
                            page_content=sub_chunk,
                            metadata=merged
                        ))
                        chunk_id += 1

        return final_chunks

    def parent_child_split(
        self,
        documents: List[Document],
        parent_chunk_size: int = 2000,
        parent_overlap: int = 200,
        child_chunk_size: int = 500,
        child_overlap: int = 100,
    ) -> Tuple[List[Document], List[Document]]:
        """父子双层切分：返回 (parent_docs, child_docs)。

        第 1 遍 — parent 层：
            复用 smart_split 逻辑（header 边界 + 代码块保护），
            但用 parent_chunk_size 控制最大块尺寸。
            parent 是最终给 LLM 的上下文单位。

        第 2 遍 — child 层：
            对每个 parent 内部用 RecursiveCharacterTextSplitter
            以 child_chunk_size 做进一步切分。
            不保护代码块（被截断不影响 embedding 精度）。
            child 只用于嵌入和检索，metadata 中记录 parent_chunk_id。

        Args:
            documents: 原始 Document 列表（含完整 metadata）。
            parent_chunk_size: parent 层 chunk 上限。
            parent_overlap: parent 层重叠字符数。
            child_chunk_size: child 层目标 chunk 尺寸（嵌入精度最优）。
            child_overlap: child 层重叠字符数。

        Returns:
            (parent_docs, child_docs):
                parent_docs — 给 LLM 的大块（存 DocStore），
                child_docs  — 给 embedding 的小块（存 Chroma）。
        """
        # ============================================================
        # Phase 1：构建 parent chunks（代码块保护 + header 语义边界）
        # ============================================================
        # 创建独立实例避免修改 self 的默认参数
        # 分隔符列表与 __init__ 保持一致（复制而非引用，避免意外耦合）
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
            separators=[
                "\n```\n",
                "\n```",
                "\n\n",
                "\n",
                ".", "!", "?", ";",
                " ",
                "",
            ],
            length_function=len,
        )

        parent_docs: List[Document] = []
        p_id = 0  # parent chunk_id（每篇文档独立编号）

        for doc in documents:
            doc_meta = dict(doc.metadata)

            # 第 1 阶段：按标题切分（h1/h2/h3）
            header_chunks = self.markdown_splitter.split_text(doc.page_content)

            for chunk in header_chunks:
                # 第 2 阶段：代码块保护
                safe_segments = self._protect_code_blocks(
                    chunk.page_content, parent_chunk_size
                )

                for segment in safe_segments:
                    # 第 3 阶段：递归切分（受 parent_chunk_size 上限保护）
                    sub_parts = parent_splitter.split_text(segment)

                    for part in sub_parts:
                        merged = {**doc_meta, **chunk.metadata}
                        merged["chunk_id"] = p_id

                        has_code, code_language = _extract_code_info(part)
                        merged["has_code"] = has_code
                        merged["code_language"] = code_language

                        parent_docs.append(Document(
                            page_content=part, metadata=merged
                        ))
                        p_id += 1

        # ============================================================
        # Phase 2：对每个 parent 切 child（无代码块保护）
        # ============================================================
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
            separators=[
                "\n```\n",  # 作为高优先级分隔符（在代码块边界处切）
                "\n```",
                "\n\n",
                "\n",
                ".", "!", "?", ";",
                " ",
                "",
            ],
            length_function=len,
        )

        child_docs: List[Document] = []
        c_id = 0  # child chunk_id（每篇文档独立编号）

        for parent in parent_docs:
            # 从 parent 继承所有 metadata，覆盖/新增特有字段
            raw_children = child_splitter.split_text(parent.page_content)

            for text in raw_children:
                merged = {
                    **parent.metadata,
                    "chunk_id": c_id,
                    "parent_chunk_id": parent.metadata["chunk_id"],
                }
                has_code, code_language = _extract_code_info(text)
                merged["has_code"] = has_code
                merged["code_language"] = code_language

                child_docs.append(Document(page_content=text, metadata=merged))
                c_id += 1

        return parent_docs, child_docs


def _extract_code_info(content: str) -> Tuple[bool, str]:
    """检测 chunk 中的代码块，返回 (has_code, code_language)。
    code_language 为逗号分隔的去重排序语言列表。
    """
    code_blocks = re.findall(r"```(\w*)", content)
    has_code = len(code_blocks) > 0
    languages = sorted(set(lang for lang in code_blocks if lang))
    return has_code, ",".join(languages) if languages else ""
