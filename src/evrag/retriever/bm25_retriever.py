"""
BM25检索器模块

基于BM25算法的稀疏检索器，适合关键词匹配。
"""

import pickle
from pathlib import Path
from typing import List, Optional, Set
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever as LangchainBM25Retriever
import jieba

from src.evrag.retriever.base import BaseRetriever
from src.evrag.config import get_settings


class BM25Retriever(BaseRetriever):
    """
    BM25检索器

    使用BM25算法进行文档检索，适合关键词匹配场景。
    支持索引持久化，避免重复构建。
    """

    def __init__(
        self,
        docs: Optional[List[Document]] = None,
        retrieve: bool = False,
        stopwords_path: Optional[Path] = None,
    ) -> None:
        """
        初始化BM25检索器

        Args:
            docs: 文档列表，用于构建索引
            retrieve: 是否从已有索引检索
            stopwords_path: 停用词文件路径（可选）
        """

        super().__init__(docs, retrieve)
        # 获取配置
        settings = get_settings()
        self.stopwords_path = stopwords_path or settings.stopwords_path
        self.stopwords = self._load_stopwords()
        self.retriever = self._get_retriever()  # 会被执行，将传入的doc切分

    def _load_stopwords(self) -> Set[str]:
        """
        load stopwords from file

        Returns:
            stopwords: 停用词集合
        """
        stopwords = set()
        if self.stopwords_path and self.stopwords_path.exists():
            try:
                with open(self.stopwords_path, "r", encoding="utf-8") as f:
                    # 只处理非空行, 因为空行的.strip()是空字符串，在Python中空字符串是 falsy
                    stopwords = {line.strip() for line in f if line.strip()}

            except Exception as e:
                print(
                    f"Warning: Failed to load stopwords from {self.stopwords_path}: {e}"
                )

        return stopwords

    def _get_retriever(self) -> LangchainBM25Retriever:
        """
        获取BM25检索器

        如果已有索引且retrieve=True，则加载；否则创建新索引。

        Returns:
            BM25检索器实例
        """
        settings = get_settings()
        index_path = settings.bm25_pickle_path
        # 如果已有索引且需要检索，直接加载
        if self.retrieve and index_path.exists():
            try:
                with open(index_path, "rb") as f:
                    retriever = pickle.load(f)
                print(f"Loaded BM25 index from {index_path}")
                return retriever
            except Exception as e:
                print(f"Warning: Failed to load BM25 index: {e}")
                # 如果加载失败，继续创建新索引
        # 创建新索引
        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "Cannot create BM25 index: documents is None or empty. "
                "Provide documents or set retrieve=True to load existing index."
            )

        # 使用langchain的BM25Retriever创建索引
        # LangChain 1.0 API保持不变
        retriever = LangchainBM25Retriever.from_documents(
            self.documents, preprocess_func=self.tokenize
        )

        # 持久化索引
        try:
            # 确保目录存在
            index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(index_path, "wb") as f:
                pickle.dump(retriever, f)
            print(f"Saved BM25 index to {index_path}")
        except Exception as e:
            print(f"Warning: Failed to save BM25 index: {e}")

        return retriever

    def tokenize(self, text: str) -> List[str]:
        """
        文本分词和停用词过滤

        使用jieba进行中文分词，并过滤停用词。

        Args:
            text: 输入文本

        Returns:
            分词后的词列表（已过滤停用词）
        """
        # 使用jieba进行分词
        tokens = jieba.lcut(text)

        # 过滤停用词和空字符串
        filtered_tokens = [
            token.strip()
            for token in tokens
            if token.strip() and token.strip() not in self.stopwords
        ]

        return filtered_tokens

    def retrieve_topk(self, query: str, topk: int = 10) -> List[Document]:
        """
        检索Top-K相关文档

        Args:
            query: 查询字符串
            topk: 返回的文档数量

        Returns:
            相关文档列表，按BM25分数排序
        """
        if self.retriever is None:
            raise ValueError("BM25 retriever not initialized")

        # 设置检索数量
        self.retriever.k = topk

        # 执行检索
        # LangChain 1.0: 使用 invoke 方法（官方推荐）
        # 根据官方文档，invoke 直接返回文档列表
        relevant_docs = self.retriever.invoke(query)

        return relevant_docs
