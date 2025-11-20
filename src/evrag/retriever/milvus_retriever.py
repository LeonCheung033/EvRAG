"""
Milvus检索器模块

基于Milvus的混合检索器，支持dense和sparse向量检索。
使用BGEM3模型进行embedding。
"""

from typing import List, Optional
from langchain_core.documents import Document
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    RRFRanker,
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from src.evrag.retriever.base import BaseRetriever
from src.evrag.config import get_settings

# 常量定义
EMB_BATCH = 50
MAX_TEXT_LENGTH = 512
ID_MAX_LENGTH = 100
COL_NAME = "hybrid_bge_m3"


class MilvusRetriever(BaseRetriever):
    """
    Milvus混合检索器

    支持dense和sparse向量的混合检索，使用BGEM3模型进行embedding。
    """

    def __init__(
        self,
        docs: Optional[List[Document]] = None,
        retrieve: bool = False,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the Milvus retriever.

        Args:
            docs: The documents to retrieve from.
            retrieve: Whether to retrieve from existing index or build new index.
            collection_name: The name of the collection to retrieve from.
        """
        super().__init__(docs, retrieve)

        # 获取配置
        settings = get_settings()

        self.collection_name = collection_name or COL_NAME
        self.milvus_db_path = settings.milvus_db_path
        self.bge_m3_model_path = settings.bge_m3_model_path

        # connect to milvus
        self._connect_milvus()

        # initialize embedding function
        self.embedding_handler = BGEM3EmbeddingFunction(
            model_name=str(self.bge_m3_model_path),
            device="cuda" if settings.device == "cuda" else "cpu",
        )
        self.col = self._init_collection()

        if not self.retrieve and self.documents:
            self._save_vectorstore(self.documents)

    def _connect_milvus(self) -> None:
        """connect to milvus"""
        try:
            connections.connect(
                uri=str(self.milvus_db_path),
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {e}")

    def _init_collection(self) -> Collection:
        """
        initialize the collection

        Returns:
            The collection object.
        """
        # declare fields
        fields = [
            FieldSchema(
                name="unique_id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=ID_MAX_LENGTH,
            ),
            FieldSchema(
                name="text", dtype=DataType.VARCHAR, max_length=MAX_TEXT_LENGTH
            ),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(
                name="dense_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedding_handler.dim["dense"],
            ),
        ]

        schema = CollectionSchema(
            fields=fields, description="Hybrid BGE-M3 Collection Schema"
        )

        # if not retrieve mode and collection already exists, drop it
        if not self.retrieve and utility.has_collection(self.collection_name):
            Collection(self.collection_name).drop()

        # create collection
        col = Collection(self.collection_name, schema, consistency_level="Strong")

        # create index
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
        col.create_index("sparse_vector", sparse_index)
        col.create_index("dense_vector", dense_index)
        col.load()

        return col

    def _save_vectorstore(self, docs: List[Document]) -> None:
        """
        save the vectorstore to milvus

        Args:
            documents: The documents to save.
        """
        raw_texts = [doc.page_content for doc in docs]
        unique_ids = [
            doc.metadata.get("unique_id", str(i)) for i, doc in enumerate(docs)
        ]

        # 计算embedding
        texts_embeddings = self.embedding_handler(raw_texts)

        # 批量插入
        for i in range(0, len(docs), EMB_BATCH):
            batched_entities = [
                unique_ids[i : i + EMB_BATCH],
                raw_texts[i : i + EMB_BATCH],
                texts_embeddings["sparse"][i : i + EMB_BATCH],
                texts_embeddings["dense"][i : i + EMB_BATCH],
            ]
            self.col.insert(batched_entities)

        print(f"索引构建完成，插入了{self.col.num_entities}条数据")

    def _hybrid_search(
        self,
        query_dense_embedding,
        query_sparse_embedding,
        limit: int = 10,
    ) -> List[dict]:
        """
        混合检索

        Args:
            query_dense_embedding: 查询的dense向量
            query_sparse_embedding: 查询的sparse向量
            limit: 返回结果数量

        Returns:
            检索结果列表
        """
        dense_search_params = {"metric_type": "IP", "params": {}}
        dense_req = AnnSearchRequest(
            [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
        )

        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_req = AnnSearchRequest(
            [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
        )

        # 使用RRF进行重排序
        rerank = RRFRanker()
        res = self.col.hybrid_search(
            [sparse_req, dense_req],
            rerank=rerank,
            limit=limit,
            output_fields=["unique_id", "text"],
        )

        return res[0]

    def retrieve_topk(self, query: str, topk: int = 10) -> List[Document]:
        """
        检索Top-K相关文档

        Args:
            query: 查询字符串
            topk: 返回的文档数量

        Returns:
            相关文档列表
        """
        if self.col is None:
            raise ValueError("Milvus collection not initialized")

        # 计算查询的embedding
        query_embeddings = self.embedding_handler.encode_queries([query])

        # 混合检索
        hybrid_results = self._hybrid_search(
            query_embeddings["dense"][0], query_embeddings["sparse"][0], limit=topk
        )

        # 转换为Document对象
        related_docs = []
        for result in hybrid_results:
            # 从结果中获取文本和元数据
            text = result.get("text", "")
            unique_id = result.get("id", "")

            doc = Document(page_content=text, metadata={"unique_id": unique_id})
            related_docs.append(doc)

        return related_docs
