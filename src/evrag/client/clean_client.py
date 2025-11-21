"""
文档清理客户端

使用LLM批量清理和整理文档。
"""

import concurrent.futures
from typing import List, Optional
from tqdm import tqdm
from more_itertools import divide
from langchain_core.documents import Document

from .base import BaseLLMClient


# 文档清理的prompt模板
LLM_CLEAN_PROMPT = """
你是一个专业的文档整理助手，负责对汽车用户手册中的内容进行整理和总结。请根据以下要求对文档进行处理：

1. **让句子变得更加通顺**：重新整合句子、段落，去除一些不必要的符号，例如换行符等。
2. **按标题归类整理**：按照文档的语义关系，把属于同一个标题下的文档做归类合并, 记住标题要用markdown的形式加粗，例如###。

请根据以下文档内容进行整理：
{}
整理后的输出：
"""


class CleanClient:
    """
    文档清理客户端

    使用LLM批量清理和整理文档。
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_template: Optional[str] = None,
        max_workers: int = 20,
    ):
        """
        初始化文档清理客户端

        Args:
            llm_client: LLM客户端实例
            prompt_template: 自定义prompt模板（可选）
            max_workers: 最大并发工作线程数
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template or LLM_CLEAN_PROMPT
        self.max_workers = max_workers

    def clean_document(self, doc_content: str, **kwargs) -> Optional[str]:
        """
        清理单个文档

        Args:
            doc_content: 文档内容
            **kwargs: 其他LLM参数

        Returns:
            清理后的文档内容，失败时返回None
        """
        prompt = self.prompt_template.format(doc_content)

        messages = [{"role": "user", "content": prompt}]

        # 设置默认参数
        default_kwargs = {
            "temperature": 0.001,
            "top_p": 0,
        }
        default_kwargs.update(kwargs)

        try:
            result = self.llm_client.chat(
                messages=messages, stream=False, **default_kwargs
            )
            return result
        except Exception as e:
            print(f"Error cleaning document: {e}")
            return None

    def clean_documents(self, documents: List[Document], **kwargs) -> List[Document]:
        """
        批量清理文档

        Args:
            documents: 文档列表
            **kwargs: 其他LLM参数

        Returns:
            清理后的文档列表
        """
        clean_docs = []
        docs_mapping = {
            doc.metadata.get("unique_id", i): doc for i, doc in enumerate(documents)
        }

        # 将文档分组以控制并发
        docs_groups = [list(group) for group in divide(self.max_workers, documents)]

        for groups in docs_groups:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                futures = {
                    doc.metadata.get("unique_id", i): executor.submit(
                        self.clean_document, doc.page_content, **kwargs
                    )
                    for i, doc in enumerate(groups)
                }

                for unique_id in tqdm(futures, desc="Cleaning documents"):
                    future = futures[unique_id]
                    result = future.result()

                    if result is None:
                        continue

                    # 保留原始metadata
                    original_doc = docs_mapping[unique_id]
                    clean_docs.append(
                        Document(page_content=result, metadata=original_doc.metadata)
                    )

        return clean_docs
