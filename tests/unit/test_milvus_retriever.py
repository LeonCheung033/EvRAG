"""
Milvus检索器模块的单元测试

测试Milvus混合检索器的功能。
"""

import pytest
from langchain_core.documents import Document
from unittest.mock import Mock, patch

from src.evrag.retriever import MilvusRetriever
from src.evrag.config import reload_settings


class TestMilvusRetriever:
    """MilvusRetriever的测试"""

    @pytest.fixture
    def sample_documents(self):
        """
        测试夹具：创建示例文档

        Returns:
            示例文档列表
        """
        texts = [
            "打开车窗的方法",
            "空调加热功能",
            "座椅加热设置",
            "如何开启车窗",
        ]
        docs = []
        for i, text in enumerate(texts):
            doc = Document(
                page_content=text, metadata={"unique_id": f"doc_{i}", "index": i}
            )
            docs.append(doc)
        return docs

    @pytest.fixture
    def temp_milvus_db_path(self, tmp_path, monkeypatch):
        """
        测试夹具：设置临时Milvus数据库路径

        Args:
            tmp_path: pytest提供的临时目录
            monkeypatch: pytest的monkeypatch fixture

        Returns:
            临时Milvus数据库路径
        """
        db_path = tmp_path / "milvus_test.db"
        # 使用monkeypatch设置环境变量
        monkeypatch.setenv("MILVUS_DB_PATH", str(db_path))

        # 重新加载配置以应用环境变量
        reload_settings()

        return db_path

    @pytest.fixture
    def mock_bge_m3_model_path(self, tmp_path, monkeypatch):
        """
        测试夹具：设置模拟的BGE-M3模型路径

        Args:
            tmp_path: pytest提供的临时目录
            monkeypatch: pytest的monkeypatch fixture

        Returns:
            模拟模型路径
        """
        model_path = tmp_path / "bge_m3_model"
        model_path.mkdir(exist_ok=True)

        # 设置环境变量
        monkeypatch.setenv("BGE_M3_MODEL_PATH", str(model_path))

        # 重新加载配置
        reload_settings()

        return model_path

    @pytest.fixture
    def mock_embedding_handler(self):
        """
        测试夹具：模拟BGEM3EmbeddingFunction

        Returns:
            模拟的embedding handler
        """
        mock_handler = Mock()

        # 模拟dim属性
        mock_handler.dim = {"dense": 1024, "sparse": 1000}

        # 模拟embedding方法（用于文档）
        def mock_embed(texts):
            # 返回模拟的sparse和dense向量
            num_texts = len(texts)
            return {
                "sparse": [
                    {i: 0.1 * (i + 1) for i in range(10)} for _ in range(num_texts)
                ],
                "dense": [[0.1] * 1024 for _ in range(num_texts)],
            }

        # 模拟encode_queries方法（用于查询）
        def mock_encode_queries(queries):
            num_queries = len(queries)
            return {
                "dense": [[0.2] * 1024 for _ in range(num_queries)],
                "sparse": [
                    {i: 0.2 * (i + 1) for i in range(10)} for _ in range(num_queries)
                ],
            }

        mock_handler.side_effect = mock_embed
        mock_handler.encode_queries = mock_encode_queries

        return mock_handler

    @pytest.fixture
    def mock_milvus_collection(self):
        """
        测试夹具：模拟Milvus Collection对象

        Returns:
            模拟的Collection对象
        """
        mock_col = Mock()
        mock_col.num_entities = 0

        # 模拟insert方法 - 使用Mock对象
        mock_insert = Mock()

        def insert_side_effect(entities):
            mock_col.num_entities += len(entities[0])

        mock_insert.side_effect = insert_side_effect
        mock_col.insert = mock_insert

        # 模拟search方法
        def mock_search(*args, **kwargs):
            limit = kwargs.get("limit", 10)
            results = []
            for i in range(min(limit, 4)):
                results.append(
                    {
                        "id": f"doc_{i}",
                        "text": f"文档{i}的内容",
                        "distance": 0.9 - i * 0.1,
                    }
                )
            return [results]

        mock_col.search = mock_search

        # 模拟hybrid_search方法 - 使用Mock对象
        mock_hybrid_search = Mock()

        def hybrid_search_side_effect(*args, **kwargs):
            limit = kwargs.get("limit", 10)
            results = []
            for i in range(min(limit, 4)):
                results.append(
                    {
                        "id": f"doc_{i}",
                        "text": f"文档{i}的内容",
                        "distance": 0.9 - i * 0.1,
                    }
                )
            return [results]

        mock_hybrid_search.side_effect = hybrid_search_side_effect
        mock_col.hybrid_search = mock_hybrid_search

        # 模拟create_index方法
        mock_col.create_index = Mock()

        # 模拟load方法
        mock_col.load = Mock()

        return mock_col

    def test_init_with_documents(
        self,
        sample_documents,
        temp_milvus_db_path,
        mock_bge_m3_model_path,
        mock_embedding_handler,
        mock_milvus_collection,
    ):
        """
        测试：使用文档初始化Milvus检索器

        验证：可以成功创建检索器并构建索引
        """
        with (
            patch(
                "src.evrag.retriever.milvus_retriever.connections"
            ) as mock_connections,
            patch(
                "src.evrag.retriever.milvus_retriever.BGEM3EmbeddingFunction",
                return_value=mock_embedding_handler,
            ),
            patch(
                "src.evrag.retriever.milvus_retriever.Collection",
                return_value=mock_milvus_collection,
            ),
            patch("src.evrag.retriever.milvus_retriever.utility") as mock_utility,
        ):
            # 模拟utility.has_collection返回False（集合不存在）
            mock_utility.has_collection.return_value = False

            retriever = MilvusRetriever(docs=sample_documents, retrieve=False)

            # 验证检索器已初始化
            assert retriever.col is not None
            assert len(retriever.documents) == 4
            assert retriever.collection_name == "hybrid_bge_m3"

            # 验证连接被调用
            mock_connections.connect.assert_called_once()

            # 验证索引构建（insert被调用）
            assert mock_milvus_collection.insert.called

    def test_init_with_custom_collection_name(
        self,
        sample_documents,
        temp_milvus_db_path,
        mock_bge_m3_model_path,
        mock_embedding_handler,
        mock_milvus_collection,
    ):
        """
        测试：使用自定义集合名称初始化

        验证：可以指定自定义的集合名称
        """
        with (
            patch("src.evrag.retriever.milvus_retriever.connections"),
            patch(
                "src.evrag.retriever.milvus_retriever.BGEM3EmbeddingFunction",
                return_value=mock_embedding_handler,
            ),
            patch(
                "src.evrag.retriever.milvus_retriever.Collection",
                return_value=mock_milvus_collection,
            ),
            patch("src.evrag.retriever.milvus_retriever.utility") as mock_utility,
        ):
            mock_utility.has_collection.return_value = False

            custom_name = "custom_collection"
            retriever = MilvusRetriever(
                docs=sample_documents, retrieve=False, collection_name=custom_name
            )

            # 验证使用了自定义集合名称
            assert retriever.collection_name == custom_name

    def test_retrieve_topk(
        self,
        sample_documents,
        temp_milvus_db_path,
        mock_bge_m3_model_path,
        mock_embedding_handler,
        mock_milvus_collection,
    ):
        """
        测试：检索Top-K文档

        验证：可以检索到相关文档
        """
        with (
            patch("src.evrag.retriever.milvus_retriever.connections"),
            patch(
                "src.evrag.retriever.milvus_retriever.BGEM3EmbeddingFunction",
                return_value=mock_embedding_handler,
            ),
            patch(
                "src.evrag.retriever.milvus_retriever.Collection",
                return_value=mock_milvus_collection,
            ),
            patch("src.evrag.retriever.milvus_retriever.utility") as mock_utility,
        ):
            mock_utility.has_collection.return_value = False

            # 设置hybrid_search返回包含实际文档内容的结果
            def mock_hybrid_search(*args, **kwargs):
                limit = kwargs.get("limit", 10)
                results = []
                for i, doc in enumerate(
                    sample_documents[: min(limit, len(sample_documents))]
                ):
                    results.append(
                        {
                            "id": doc.metadata["unique_id"],
                            "text": doc.page_content,
                            "distance": 0.9 - i * 0.1,
                        }
                    )
                return [results]

            mock_milvus_collection.hybrid_search = mock_hybrid_search

            retriever = MilvusRetriever(docs=sample_documents, retrieve=False)

            # 执行检索
            query = "车窗"
            results = retriever.retrieve_topk(query, topk=2)

            # 验证结果
            assert isinstance(results, list)
            assert len(results) <= 2
            assert all(isinstance(doc, Document) for doc in results)

            # 验证结果包含文档内容
            if len(results) > 0:
                assert results[0].page_content in [
                    doc.page_content for doc in sample_documents
                ]

    def test_retrieve_topk_empty_query(
        self,
        sample_documents,
        temp_milvus_db_path,
        mock_bge_m3_model_path,
        mock_embedding_handler,
        mock_milvus_collection,
    ):
        """
        测试：空查询的处理

        验证：空查询不会导致错误
        """
        with (
            patch("src.evrag.retriever.milvus_retriever.connections"),
            patch(
                "src.evrag.retriever.milvus_retriever.BGEM3EmbeddingFunction",
                return_value=mock_embedding_handler,
            ),
            patch(
                "src.evrag.retriever.milvus_retriever.Collection",
                return_value=mock_milvus_collection,
            ),
            patch("src.evrag.retriever.milvus_retriever.utility") as mock_utility,
        ):
            mock_utility.has_collection.return_value = False

            retriever = MilvusRetriever(docs=sample_documents, retrieve=False)

            # 空查询应该也能处理
            results = retriever.retrieve_topk("", topk=5)
            assert isinstance(results, list)

    def test_retrieve_without_collection(
        self,
        temp_milvus_db_path,
        mock_bge_m3_model_path,
        mock_embedding_handler,
        mock_milvus_collection,
    ):
        """
        测试：集合未初始化时的错误处理

        验证：如果集合未初始化，应该抛出ValueError
        """
        with (
            patch("src.evrag.retriever.milvus_retriever.connections"),
            patch(
                "src.evrag.retriever.milvus_retriever.BGEM3EmbeddingFunction",
                return_value=mock_embedding_handler,
            ),
            patch(
                "src.evrag.retriever.milvus_retriever.Collection",
                return_value=mock_milvus_collection,
            ),
            patch("src.evrag.retriever.milvus_retriever.utility") as mock_utility,
        ):
            mock_utility.has_collection.return_value = False

            retriever = MilvusRetriever(docs=None, retrieve=True)

            # 手动设置col为None来模拟未初始化的情况
            retriever.col = None

            # 应该抛出ValueError
            with pytest.raises(ValueError, match="Milvus collection not initialized"):
                retriever.retrieve_topk("test query", topk=5)

    def test_connection_error(
        self, sample_documents, temp_milvus_db_path, mock_bge_m3_model_path
    ):
        """
        测试：Milvus连接失败的错误处理

        验证：连接失败时应该抛出ConnectionError
        """
        with patch(
            "src.evrag.retriever.milvus_retriever.connections"
        ) as mock_connections:
            # 模拟连接失败
            mock_connections.connect.side_effect = Exception("Connection failed")

            with pytest.raises(ConnectionError, match="Failed to connect to Milvus"):
                MilvusRetriever(docs=sample_documents, retrieve=False)

    def test_drop_existing_collection(
        self,
        sample_documents,
        temp_milvus_db_path,
        mock_bge_m3_model_path,
        mock_embedding_handler,
        mock_milvus_collection,
    ):
        """
        测试：删除已存在的集合

        验证：当retrieve=False且集合已存在时，应该删除旧集合
        """
        with (
            patch("src.evrag.retriever.milvus_retriever.connections"),
            patch(
                "src.evrag.retriever.milvus_retriever.BGEM3EmbeddingFunction",
                return_value=mock_embedding_handler,
            ),
            patch(
                "src.evrag.retriever.milvus_retriever.Collection"
            ) as mock_collection_class,
            patch("src.evrag.retriever.milvus_retriever.utility") as mock_utility,
        ):
            # 模拟集合已存在
            mock_utility.has_collection.return_value = True

            # 模拟旧的Collection对象（用于drop）
            mock_old_collection = Mock()
            mock_collection_class.side_effect = [
                mock_old_collection,  # 第一次调用：用于drop旧集合
                mock_milvus_collection,  # 第二次调用：创建新集合
            ]

            # 创建检索器实例，这会触发drop操作
            retriever = MilvusRetriever(docs=sample_documents, retrieve=False)

            # 验证旧集合的drop方法被调用
            mock_old_collection.drop.assert_called_once()
            # 验证新集合被创建
            assert retriever.col is not None

    def test_hybrid_search_integration(
        self,
        sample_documents,
        temp_milvus_db_path,
        mock_bge_m3_model_path,
        mock_embedding_handler,
        mock_milvus_collection,
    ):
        """
        测试：混合检索的集成测试

        验证：hybrid_search方法被正确调用
        """
        with (
            patch("src.evrag.retriever.milvus_retriever.connections"),
            patch(
                "src.evrag.retriever.milvus_retriever.BGEM3EmbeddingFunction",
                return_value=mock_embedding_handler,
            ),
            patch(
                "src.evrag.retriever.milvus_retriever.Collection",
                return_value=mock_milvus_collection,
            ),
            patch("src.evrag.retriever.milvus_retriever.utility") as mock_utility,
        ):
            mock_utility.has_collection.return_value = False

            # 设置hybrid_search返回结果 - 使用Mock对象
            mock_hybrid_search_func = Mock()

            def hybrid_search_side_effect(*args, **kwargs):
                return [[{"id": "doc_0", "text": "打开车窗的方法", "distance": 0.9}]]

            mock_hybrid_search_func.side_effect = hybrid_search_side_effect
            mock_milvus_collection.hybrid_search = mock_hybrid_search_func

            retriever = MilvusRetriever(docs=sample_documents, retrieve=False)

            # 执行检索
            results = retriever.retrieve_topk("车窗", topk=1)

            # 验证hybrid_search被调用
            assert mock_milvus_collection.hybrid_search.called

            # 验证结果
            assert len(results) == 1
            assert results[0].page_content == "打开车窗的方法"
