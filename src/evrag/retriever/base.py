from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document


# 继承ABC，成为抽象基类，不能直接实例化
class BaseRetriever(ABC):
    def __init__(
        self,
        docs: Optional[List[Document]] = None,
        retrieve: bool = True,
    ) -> None:
        """
        Initialize the retriever.

        Args:
            docs: The documents to retrieve from.
            retrieve: 是否从已有索引检索（True）还是构建新索引（False）
        """
        self.documents = docs  # 修改：统一使用 documents
        self.retrieve = retrieve

    @abstractmethod
    def retrieve_topk(self, query: str, topk: int = 10) -> List[Document]:
        """
        检索Top-K相关文档

        这是一个抽象方法，子类必须实现。

        Args:
            query: 查询字符串
            topk: 返回的文档数量

        Returns:
            相关文档列表，按相关性排序（最相关的在前）

        Raises:
            NotImplementedError: 如果子类没有实现此方法
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement retrieve_topk method"
        )
