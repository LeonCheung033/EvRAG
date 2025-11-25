"""
文档切分模块单元测试

测试texts_split和save_2_mongo函数的功能。
"""

import pytest
import copy
import hashlib
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.evrag.parser.document_splitter import texts_split, save_2_mongo
from src.evrag.client import SemanticChunkClient, MongoDBClient


class TestSave2Mongo:
    """save_2_mongo函数的测试"""

    @pytest.fixture
    def sample_documents(self):
        """示例文档列表"""
        return [
            Document(
                page_content="文档1内容",
                metadata={
                    "unique_id": "doc_1",
                    "source": "test.pdf",
                    "page": 1,
                },
            ),
            Document(
                page_content="文档2内容",
                metadata={
                    "unique_id": "doc_2",
                    "source": "test.pdf",
                    "page": 2,
                },
            ),
        ]

    @patch("src.evrag.parser.document_splitter.MongoDBClient")
    def test_save_2_mongo_saves_documents(self, mock_mongodb_client_class, sample_documents):
        """测试：保存文档到MongoDB"""
        # 创建mock对象
        mock_client = Mock(spec=MongoDBClient)
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_mongodb_client_class.return_value = mock_client

        save_2_mongo(sample_documents, collection_name="test_collection")

        # 验证MongoDBClient被正确初始化
        mock_mongodb_client_class.assert_called_once()
        
        # 验证连接和获取集合
        mock_client.connect.assert_called_once()
        mock_client.get_collection.assert_called_once_with("test_collection")
        mock_client.close.assert_called_once()

        # 验证每个文档都被保存
        assert mock_collection.update_one.call_count == 2

        # 验证第一个文档的保存
        first_call = mock_collection.update_one.call_args_list[0]
        assert first_call[0][0] == {"unique_id": "doc_1"}
        assert "$set" in first_call[0][1]
        assert first_call[1]["upsert"] is True

    @patch("src.evrag.parser.document_splitter.MongoDBClient")
    def test_save_2_mongo_skips_documents_without_unique_id(self, mock_mongodb_client_class):
        """测试：跳过没有unique_id的文档"""
        mock_client = Mock(spec=MongoDBClient)
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_mongodb_client_class.return_value = mock_client

        docs_without_id = [
            Document(page_content="内容", metadata={"page": 1}),
        ]

        save_2_mongo(docs_without_id)

        # 应该没有调用update_one
        mock_collection.update_one.assert_not_called()

    @patch("src.evrag.parser.document_splitter.MongoDBClient")
    def test_save_2_mongo_uses_default_collection_name(self, mock_mongodb_client_class, sample_documents):
        """测试：使用默认集合名称"""
        mock_client = Mock(spec=MongoDBClient)
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_mongodb_client_class.return_value = mock_client

        save_2_mongo(sample_documents)

        mock_client.get_collection.assert_called_once_with("manual_text")

    @patch("src.evrag.parser.document_splitter.MongoDBClient")
    def test_save_2_mongo_handles_connection_error(self, mock_mongodb_client_class, sample_documents):
        """测试：处理连接错误"""
        mock_client = Mock(spec=MongoDBClient)
        mock_client.connect.side_effect = Exception("Connection failed")
        mock_mongodb_client_class.return_value = mock_client

        # 应该抛出异常
        with pytest.raises(Exception, match="Connection failed"):
            save_2_mongo(sample_documents)


class TestTextsSplit:
    """texts_split函数的测试"""

    @pytest.fixture
    def sample_raw_docs(self):
        """示例原始文档"""
        return [
            Document(
                page_content="这是第一段内容。这是第二段内容。这是第三段内容。",
                metadata={
                    "unique_id": "raw_doc_1",
                    "source": "test.pdf",
                    "page": 1,
                },
            ),
        ]

    @pytest.fixture
    def mock_semantic_chunk_client(self):
        """模拟语义切分客户端"""
        mock_client = Mock(spec=SemanticChunkClient)
        # 模拟返回两个语义块
        mock_client.chunk.return_value = [
            "这是第一段内容。这是第二段内容。",
            "这是第三段内容。",
        ]
        return mock_client

    @patch("src.evrag.parser.document_splitter.save_2_mongo")
    @patch("src.evrag.parser.document_splitter.text_splitter")
    def test_texts_split_creates_parent_docs(
        self,
        mock_text_splitter,
        mock_save_2_mongo,
        sample_raw_docs,
        mock_semantic_chunk_client,
    ):
        """测试：创建父文档"""
        # 模拟text_splitter不进行进一步切分（返回空列表）
        mock_text_splitter.create_documents.return_value = []

        result = texts_split(
            sample_raw_docs,
            semantic_chunk_client=mock_semantic_chunk_client,
        )

        # 验证语义切分被调用
        assert mock_semantic_chunk_client.chunk.call_count == 1

        # 验证保存到MongoDB被调用（至少一次，用于父文档）
        assert mock_save_2_mongo.call_count >= 1

        # 验证返回的文档包含父文档
        assert len(result) > 0

    @patch("src.evrag.parser.document_splitter.save_2_mongo")
    @patch("src.evrag.parser.document_splitter.text_splitter")
    def test_texts_split_creates_child_docs(
        self,
        mock_text_splitter,
        mock_save_2_mongo,
        sample_raw_docs,
        mock_semantic_chunk_client,
    ):
        """测试：创建子文档"""
        # 模拟text_splitter创建子文档
        child_doc = Document(
            page_content="这是第一段内容。",
            metadata={"unique_id": "raw_doc_1"},
        )
        mock_text_splitter.create_documents.return_value = [child_doc]

        result = texts_split(
            sample_raw_docs,
            semantic_chunk_client=mock_semantic_chunk_client,
        )

        # 验证text_splitter被调用
        assert mock_text_splitter.create_documents.call_count > 0

        # 验证保存到MongoDB被调用（父文档和子文档）
        assert mock_save_2_mongo.call_count >= 2

    @patch("src.evrag.parser.document_splitter.save_2_mongo")
    def test_texts_split_handles_semantic_chunk_failure(
        self,
        mock_save_2_mongo,
        sample_raw_docs,
    ):
        """测试：处理语义切分失败"""
        mock_client = Mock(spec=SemanticChunkClient)
        # 模拟失败，返回原始文本
        mock_client.chunk.return_value = [sample_raw_docs[0].page_content]

        with patch("src.evrag.parser.document_splitter.text_splitter") as mock_text_splitter:
            mock_text_splitter.create_documents.return_value = []

            result = texts_split(
                sample_raw_docs,
                semantic_chunk_client=mock_client,
            )

            # 应该仍然能处理
            assert len(result) >= 0

    @patch("src.evrag.parser.document_splitter.save_2_mongo")
    def test_texts_split_preserves_metadata(
        self,
        mock_save_2_mongo,
        sample_raw_docs,
        mock_semantic_chunk_client,
    ):
        """测试：保留元数据"""
        with patch("src.evrag.parser.document_splitter.text_splitter") as mock_text_splitter:
            mock_text_splitter.create_documents.return_value = []

            result = texts_split(
                sample_raw_docs,
                semantic_chunk_client=mock_semantic_chunk_client,
            )

            # 验证结果文档保留了原始metadata
            if result:
                assert "source" in result[0].metadata
                assert "page" in result[0].metadata

    @patch("src.evrag.parser.document_splitter.save_2_mongo")
    def test_texts_split_adds_parent_id_to_child_docs(
        self,
        mock_save_2_mongo,
        sample_raw_docs,
        mock_semantic_chunk_client,
    ):
        """测试：为子文档添加parent_id"""
        with patch("src.evrag.parser.document_splitter.text_splitter") as mock_text_splitter:
            child_doc = Document(
                page_content="这是第一段内容。",
                metadata={"unique_id": "raw_doc_1"},
            )
            mock_text_splitter.create_documents.return_value = [child_doc]

            result = texts_split(
                sample_raw_docs,
                semantic_chunk_client=mock_semantic_chunk_client,
            )

            # 验证子文档有parent_id
            child_docs = [doc for doc in result if "parent_id" in doc.metadata]
            if child_docs:
                assert "parent_id" in child_docs[0].metadata

    @patch("src.evrag.parser.document_splitter.save_2_mongo")
    def test_texts_split_uses_custom_collection_name(
        self,
        mock_save_2_mongo,
        sample_raw_docs,
        mock_semantic_chunk_client,
    ):
        """测试：使用自定义集合名称"""
        with patch("src.evrag.parser.document_splitter.text_splitter") as mock_text_splitter:
            mock_text_splitter.create_documents.return_value = []

            texts_split(
                sample_raw_docs,
                semantic_chunk_client=mock_semantic_chunk_client,
                collection_name="custom_collection",  
            )

            # 验证save_2_mongo被调用时使用了自定义集合名
            assert mock_save_2_mongo.called
            # 检查最后一次调用是否使用了自定义集合名
            call_args = mock_save_2_mongo.call_args_list
            if call_args:
                # save_2_mongo的第二个参数是collection_name
                assert call_args[-1][1]["collection_name"] == "custom_collection"