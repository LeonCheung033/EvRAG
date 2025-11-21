"""
重排序器基类模块

定义所有重排序器必须实现的接口。
"""

from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class BaseReranker(ABC):
    @abstractmethod
    def rank(
        self,
        query: str,
        candidates: List[Document],
        topk: int = 10,
    ) -> List[Document]:
        """
        Rank the candidates based on the query.

        Args:
            query: The query string.
            candidates: The candidates to rank.
            topk: The number of candidates to return.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement rank method"
        )
