"""
客户端模块单元测试

测试LLM客户端和MongoDB客户端的功能。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.evrag.client import (
    BaseLLMClient,
    LocalLLMClient,
    OpenAIClient,
    MongoDBClient,
    ChatClient,
    CleanClient,
    HydeClient,
    SemanticChunkClient,
)


class TestBaseLLMClient:
    """BaseLLMClient的测试"""

    def test_cannot_instantiate_base_class(self):
        """测试：不能直接实例化基类"""
        with pytest.raises(TypeError):
            BaseLLMClient()

    def test_abstract_method(self):
        """测试：未实现抽象方法的子类不能实例化"""

        class IncompleteClient(BaseLLMClient):
            pass

        with pytest.raises(TypeError):
            IncompleteClient()


class TestLocalLLMClient:
    """LocalLLMClient的测试"""

    @pytest.fixture
    def mock_openai_client(self):
        """模拟OpenAI客户端"""
        mock_client = Mock()
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = Mock()
        mock_completion.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_completion
        return mock_client

    @pytest.fixture
    def sample_messages(self):
        """示例消息列表"""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]

    def test_init_with_default_settings(self):
        """测试：使用默认配置初始化"""
        with patch("src.evrag.client.local_client.OpenAI") as mock_openai_class:
            mock_openai_class.return_value = Mock()

            with patch(
                "src.evrag.client.local_client.get_settings"
            ) as mock_get_settings:
                mock_settings = Mock()
                mock_settings.llm_api_key = "EMPTY"
                mock_settings.llm_base_url = "http://localhost:8000/v1"
                mock_settings.llm_model_name = "test-model"
                mock_get_settings.return_value = mock_settings

                client = LocalLLMClient()

                assert client.api_key == "EMPTY"
                assert client.base_url == "http://localhost:8000/v1"
                assert client.model == "test-model"
                mock_openai_class.assert_called_once_with(
                    api_key="EMPTY", base_url="http://localhost:8000/v1"
                )

    def test_init_with_custom_params(self):
        """测试：使用自定义参数初始化"""
        with patch("src.evrag.client.local_client.OpenAI") as mock_openai_class:
            mock_openai_class.return_value = Mock()

            client = LocalLLMClient(
                api_key="custom_key",
                base_url="http://custom:9000/v1",
                model="custom-model",
            )

            assert client.api_key == "custom_key"
            assert client.base_url == "http://custom:9000/v1"
            assert client.model == "custom-model"

    def test_chat_non_stream(self, mock_openai_client, sample_messages):
        """测试：非流式聊天请求"""
        with patch("src.evrag.client.local_client.OpenAI") as mock_openai_class:
            mock_openai_class.return_value = mock_openai_client

            client = LocalLLMClient(
                api_key="EMPTY", base_url="http://localhost:8000/v1", model="test-model"
            )

            response = client.chat(
                messages=sample_messages, temperature=0.7, max_tokens=100, stream=False
            )

            assert response == "Test response"
            mock_openai_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
            assert call_kwargs["model"] == "test-model"
            assert call_kwargs["messages"] == sample_messages
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["max_tokens"] == 100
            assert call_kwargs["stream"] is False

    def test_chat_with_custom_model(self, mock_openai_client, sample_messages):
        """测试：使用自定义模型名称"""
        with patch("src.evrag.client.local_client.OpenAI") as mock_openai_class:
            mock_openai_class.return_value = mock_openai_client

            client = LocalLLMClient(model="default-model")

            client.chat(messages=sample_messages, model="custom-model")

            call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
            assert call_kwargs["model"] == "custom-model"

    def test_chat_stream(self, mock_openai_client, sample_messages):
        """测试：流式聊天请求"""
        # 模拟流式响应
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock()]
        mock_chunk1.choices[0].delta = Mock()
        mock_chunk1.choices[0].delta.content = "Hello"

        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock()]
        mock_chunk2.choices[0].delta = Mock()
        mock_chunk2.choices[0].delta.content = " World"

        mock_openai_client.chat.completions.create.return_value = [
            mock_chunk1,
            mock_chunk2,
        ]

        with patch("src.evrag.client.local_client.OpenAI") as mock_openai_class:
            mock_openai_class.return_value = mock_openai_client

            client = LocalLLMClient()

            response = client.chat(messages=sample_messages, stream=True)

            # 流式返回应该是生成器
            assert hasattr(response, "__iter__")
            chunks = list(response)
            assert chunks == ["Hello", " World"]

    def test_chat_stream_method(self, mock_openai_client, sample_messages):
        """测试：chat_stream方法"""
        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta = Mock()
        mock_chunk.choices[0].delta.content = "Test"

        mock_openai_client.chat.completions.create.return_value = [mock_chunk]

        with patch("src.evrag.client.local_client.OpenAI") as mock_openai_class:
            mock_openai_class.return_value = mock_openai_client

            client = LocalLLMClient()

            response = client.chat_stream(messages=sample_messages)

            chunks = list(response)
            assert chunks == ["Test"]
            call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
            assert call_kwargs["stream"] is True


class TestOpenAIClient:
    """OpenAIClient的测试"""

    @pytest.fixture
    def mock_openai_client(self):
        """模拟OpenAI客户端"""
        mock_client = Mock()
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = Mock()
        mock_completion.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_completion
        return mock_client

    @pytest.fixture
    def sample_messages(self):
        """示例消息列表"""
        return [
            {"role": "user", "content": "Hello"},
        ]

    def test_init_with_valid_api_key(self):
        """测试：使用有效API密钥初始化"""
        with patch("src.evrag.client.openai_client.OpenAI") as mock_openai_class:
            mock_openai_class.return_value = Mock()

            client = OpenAIClient(
                api_key="valid_key", base_url="https://api.openai.com/v1", model="gpt-4"
            )

            assert client.api_key == "valid_key"
            assert client.base_url == "https://api.openai.com/v1"
            assert client.model == "gpt-4"

    def test_init_with_empty_api_key_raises_error(self):
        """测试：API密钥为空时抛出错误"""
        with patch("src.evrag.client.openai_client.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.llm_api_key = "EMPTY"
            mock_get_settings.return_value = mock_settings

            with pytest.raises(ValueError, match="OpenAI API key is required"):
                OpenAIClient()

    def test_init_with_none_api_key_raises_error(self):
        """测试：API密钥为None时抛出错误"""
        with patch("src.evrag.client.openai_client.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.llm_api_key = None
            mock_get_settings.return_value = mock_settings

            with pytest.raises(ValueError, match="OpenAI API key is required"):
                OpenAIClient()

    def test_chat_non_stream(self, mock_openai_client, sample_messages):
        """测试：非流式聊天请求"""
        with patch("src.evrag.client.openai_client.OpenAI") as mock_openai_class:
            mock_openai_class.return_value = mock_openai_client

            client = OpenAIClient(
                api_key="valid_key", base_url="https://api.openai.com/v1", model="gpt-4"
            )

            response = client.chat(
                messages=sample_messages, temperature=0.5, max_tokens=200
            )

            assert response == "Test response"
            mock_openai_client.chat.completions.create.assert_called_once()

    def test_chat_stream(self, mock_openai_client, sample_messages):
        """测试：流式聊天请求"""
        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta = Mock()
        mock_chunk.choices[0].delta.content = "Stream"

        mock_openai_client.chat.completions.create.return_value = [mock_chunk]

        with patch("src.evrag.client.openai_client.OpenAI") as mock_openai_class:
            mock_openai_class.return_value = mock_openai_client

            client = OpenAIClient(api_key="valid_key")

            response = client.chat(messages=sample_messages, stream=True)
            chunks = list(response)
            assert chunks == ["Stream"]


class TestMongoDBClient:
    """MongoDBClient的测试"""

    @pytest.fixture
    def mock_mongo_client(self):
        """模拟MongoDB客户端"""
        # 使用MagicMock以支持__getitem__操作
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()

        # 设置数据库的__getitem__返回集合
        mock_db.__getitem__.return_value = mock_collection

        # 设置客户端的__getitem__返回数据库
        mock_client.__getitem__.return_value = mock_db

        # 设置admin.command用于ping
        mock_client.admin = MagicMock()
        mock_client.admin.command.return_value = {"ok": 1}

        return mock_client

    def test_init_with_default_settings(self):
        """测试：使用默认配置初始化"""
        with patch("src.evrag.client.mongodb_client.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.mongodb_host = "localhost"
            mock_settings.mongodb_port = 27017
            mock_settings.mongodb_database = "evrag"
            mock_get_settings.return_value = mock_settings

            client = MongoDBClient()

            assert client.host == "localhost"
            assert client.port == 27017
            assert client.database == "evrag"

    def test_init_with_custom_params(self):
        """测试：使用自定义参数初始化"""
        client = MongoDBClient(
            host="custom_host",
            port=27018,
            database="custom_db",
            username="user",
            password="pass",
        )

        assert client.host == "custom_host"
        assert client.port == 27018
        assert client.database == "custom_db"
        assert client.username == "user"
        assert client.password == "pass"

    def test_build_connection_uri_without_auth(self):
        """测试：构建无认证的连接URI"""
        client = MongoDBClient(host="localhost", port=27017)
        uri = client._build_connection_uri()
        assert uri == "mongodb://localhost:27017"

    def test_build_connection_uri_with_auth(self):
        """测试：构建带认证的连接URI"""
        client = MongoDBClient(
            host="localhost",
            port=27017,
            username="user",
            password="pass",
            auth_source="admin",
        )
        uri = client._build_connection_uri()
        assert uri == "mongodb://user:pass@localhost:27017/?authSource=admin"

    def test_connect_success(self, mock_mongo_client):
        """测试：成功连接MongoDB"""
        with patch("src.evrag.client.mongodb_client.MongoClient") as mock_mongo_class:
            mock_mongo_class.return_value = mock_mongo_client

            client = MongoDBClient(host="localhost", port=27017, database="test_db")
            client.connect()

            assert client._client is not None
            assert client._db is not None
            # 验证ping被调用
            mock_mongo_client.admin.command.assert_called_once_with("ping")
            # 验证数据库被访问
            mock_mongo_client.__getitem__.assert_called_with("test_db")

    def test_connect_configuration_error(self):
        """测试：配置错误时抛出异常"""
        with patch("src.evrag.client.mongodb_client.MongoClient") as mock_mongo_class:
            from pymongo.errors import ConfigurationError

            mock_mongo_class.side_effect = ConfigurationError("Invalid config")

            client = MongoDBClient()

            with pytest.raises(RuntimeError, match="MongoDB configuration error"):
                client.connect()

    def test_connect_connection_failure(self):
        """测试：连接失败时抛出异常"""
        with patch("src.evrag.client.mongodb_client.MongoClient") as mock_mongo_class:
            from pymongo.errors import ConnectionFailure

            mock_mongo_class.side_effect = ConnectionFailure("Connection failed")

            client = MongoDBClient()

            with pytest.raises(RuntimeError, match="Failed to connect to MongoDB"):
                client.connect()

    def test_get_db(self, mock_mongo_client):
        """测试：获取数据库实例"""
        with patch("src.evrag.client.mongodb_client.MongoClient") as mock_mongo_class:
            mock_mongo_class.return_value = mock_mongo_client

            client = MongoDBClient(database="test_db")
            db = client.get_db()

            assert db is not None
            # 验证数据库被访问
            mock_mongo_client.__getitem__.assert_called_with("test_db")

    def test_get_collection(self, mock_mongo_client):
        """测试：获取集合实例"""
        with patch("src.evrag.client.mongodb_client.MongoClient") as mock_mongo_class:
            mock_mongo_class.return_value = mock_mongo_client

            client = MongoDBClient(database="test_db")
            collection = client.get_collection("test_collection")

            assert collection is not None
            # 验证数据库和集合都被访问
            # 首先访问数据库
            mock_mongo_client.__getitem__.assert_called_with("test_db")
            # 然后通过数据库访问集合
            mock_db = mock_mongo_client.__getitem__.return_value
            mock_db.__getitem__.assert_called_with("test_collection")

    def test_close(self, mock_mongo_client):
        """测试：关闭连接"""
        with patch("src.evrag.client.mongodb_client.MongoClient") as mock_mongo_class:
            mock_mongo_class.return_value = mock_mongo_client

            client = MongoDBClient()
            client.connect()
            client.close()

            assert client._client is None
            assert client._db is None
            mock_mongo_client.close.assert_called_once()

    def test_context_manager(self, mock_mongo_client):
        """测试：上下文管理器"""
        with patch("src.evrag.client.mongodb_client.MongoClient") as mock_mongo_class:
            mock_mongo_class.return_value = mock_mongo_client

            with MongoDBClient() as client:
                assert client._client is not None
                assert client._db is not None

            # 退出上下文后应该关闭连接
            assert client._client is None
            assert client._db is None

    def test_get_db_without_connection_raises_error(self, mock_mongo_client):
        """测试：未连接时获取数据库抛出错误"""
        with patch("src.evrag.client.mongodb_client.MongoClient") as mock_mongo_class:
            # 模拟连接失败
            from pymongo.errors import ConnectionFailure

            mock_mongo_class.side_effect = ConnectionFailure("Connection refused")

            client = MongoDBClient()
            # 不调用connect()，直接调用get_db()
            # get_db()会尝试连接，但连接失败
            with pytest.raises(RuntimeError, match="Failed to connect to MongoDB"):
                client.get_db()


# ... existing code ...


class TestChatClient:
    """ChatClient的测试"""

    @pytest.fixture
    def mock_llm_client(self):
        """模拟LLM客户端"""
        mock_client = Mock(spec=BaseLLMClient)
        mock_client.chat.return_value = "Test answer【1, 2】"
        mock_client.chat_stream.return_value = iter(["Test ", "answer"])
        return mock_client

    def test_init_with_default_template(self, mock_llm_client):
        """测试：使用默认prompt模板初始化"""
        client = ChatClient(mock_llm_client)

        assert client.llm_client == mock_llm_client
        assert "信息" in client.prompt_template
        assert "任务" in client.prompt_template

    def test_init_with_custom_template(self, mock_llm_client):
        """测试：使用自定义prompt模板初始化"""
        custom_template = "Custom template: {query} with {context}"
        client = ChatClient(mock_llm_client, prompt_template=custom_template)

        assert client.prompt_template == custom_template

    def test_chat_non_stream(self, mock_llm_client):
        """测试：非流式RAG问答"""
        client = ChatClient(mock_llm_client)

        result = client.chat(
            query="如何打开车窗？", context="【1】打开车窗的方法\n【2】车窗控制"
        )

        assert result == "Test answer【1, 2】"
        mock_llm_client.chat.assert_called_once()
        call_kwargs = mock_llm_client.chat.call_args[1]
        assert call_kwargs["stream"] is False
        assert "max_tokens" in call_kwargs
        assert call_kwargs["max_tokens"] == 4096

    def test_chat_stream(self, mock_llm_client):
        """测试：流式RAG问答"""
        client = ChatClient(mock_llm_client)

        result = client.chat(
            query="如何打开车窗？", context="【1】打开车窗的方法", stream=True
        )

        # 流式返回应该是生成器
        assert hasattr(result, "__iter__")
        chunks = list(result)
        assert chunks == ["Test ", "answer"]

    def test_chat_with_custom_kwargs(self, mock_llm_client):
        """测试：使用自定义LLM参数"""
        client = ChatClient(mock_llm_client)

        client.chat(query="测试", context="上下文", temperature=0.5, max_tokens=2048)

        call_kwargs = mock_llm_client.chat.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 2048


class TestCleanClient:
    """CleanClient的测试"""

    @pytest.fixture
    def mock_llm_client(self):
        """模拟LLM客户端"""
        mock_client = Mock(spec=BaseLLMClient)
        mock_client.chat.return_value = "Cleaned document content"
        return mock_client

    @pytest.fixture
    def sample_documents(self):
        """示例文档列表"""
        from langchain_core.documents import Document

        return [
            Document(
                page_content="原始文档内容1",
                metadata={"unique_id": "doc_1", "index": 0},
            ),
            Document(
                page_content="原始文档内容2",
                metadata={"unique_id": "doc_2", "index": 1},
            ),
        ]

    def test_init_with_default_template(self, mock_llm_client):
        """测试：使用默认prompt模板初始化"""
        client = CleanClient(mock_llm_client)

        assert client.llm_client == mock_llm_client
        assert "整理" in client.prompt_template
        assert client.max_workers == 20

    def test_init_with_custom_params(self, mock_llm_client):
        """测试：使用自定义参数初始化"""
        custom_template = "Custom clean template: {}"
        client = CleanClient(
            mock_llm_client, prompt_template=custom_template, max_workers=10
        )

        assert client.prompt_template == custom_template
        assert client.max_workers == 10

    def test_clean_document_success(self, mock_llm_client):
        """测试：成功清理单个文档"""
        client = CleanClient(mock_llm_client)

        result = client.clean_document("原始文档内容")

        assert result == "Cleaned document content"
        mock_llm_client.chat.assert_called_once()
        call_kwargs = mock_llm_client.chat.call_args[1]
        assert call_kwargs["stream"] is False
        assert call_kwargs["temperature"] == 0.001

    def test_clean_document_failure(self, mock_llm_client):
        """测试：清理文档失败时返回None"""
        mock_llm_client.chat.side_effect = Exception("API error")
        client = CleanClient(mock_llm_client)

        result = client.clean_document("原始文档内容")

        assert result is None

    def test_clean_documents(self, mock_llm_client, sample_documents):
        """测试：批量清理文档"""
        client = CleanClient(mock_llm_client, max_workers=2)

        result = client.clean_documents(sample_documents)

        assert len(result) == 2
        assert all(doc.page_content == "Cleaned document content" for doc in result)
        # 验证metadata被保留
        assert result[0].metadata["unique_id"] == "doc_1"
        assert result[1].metadata["unique_id"] == "doc_2"

    def test_clean_documents_with_failed_items(self, mock_llm_client, sample_documents):
        """测试：部分文档清理失败"""
        # 第一个成功，第二个失败
        mock_llm_client.chat.side_effect = [
            "Cleaned document content",
            Exception("API error"),
        ]
        client = CleanClient(mock_llm_client, max_workers=2)

        result = client.clean_documents(sample_documents)

        # 只有成功的文档被返回
        assert len(result) == 1
        assert result[0].metadata["unique_id"] == "doc_1"

    def test_clean_documents_with_custom_kwargs(
        self, mock_llm_client, sample_documents
    ):
        """测试：使用自定义LLM参数"""
        client = CleanClient(mock_llm_client)

        client.clean_documents(sample_documents, temperature=0.5)

        # 验证自定义参数被传递
        call_kwargs = mock_llm_client.chat.call_args[1]
        assert call_kwargs["temperature"] == 0.5


class TestHydeClient:
    """HydeClient的测试"""

    @pytest.fixture
    def mock_llm_client(self):
        """模拟LLM客户端"""
        mock_client = Mock(spec=BaseLLMClient)
        mock_client.chat.return_value = "假设文档内容：关于车窗的使用方法"
        return mock_client

    def test_init_with_default_template(self, mock_llm_client):
        """测试：使用默认prompt模板初始化"""
        client = HydeClient(mock_llm_client)

        assert client.llm_client == mock_llm_client
        assert "Tesla" in client.prompt_template
        assert "专家" in client.prompt_template

    def test_init_with_custom_template(self, mock_llm_client):
        """测试：使用自定义prompt模板初始化"""
        custom_template = "Custom HYDE template: {query}"
        custom_system = "Custom system message"
        client = HydeClient(
            mock_llm_client,
            prompt_template=custom_template,
            system_message=custom_system,
        )

        assert client.prompt_template == custom_template
        assert client.system_message == custom_system

    def test_generate_hypothetical_document(self, mock_llm_client):
        """测试：生成假设文档"""
        client = HydeClient(mock_llm_client)

        result = client.generate_hypothetical_document("如何打开车窗？")

        assert result == "假设文档内容：关于车窗的使用方法"
        mock_llm_client.chat.assert_called_once()
        call_kwargs = mock_llm_client.chat.call_args[1]
        assert call_kwargs["stream"] is False
        assert call_kwargs["temperature"] == 0.001
        assert call_kwargs["top_p"] == 0

    def test_generate_hypothetical_document_with_custom_kwargs(self, mock_llm_client):
        """测试：使用自定义LLM参数"""
        client = HydeClient(mock_llm_client)

        client.generate_hypothetical_document(
            "测试查询", temperature=0.5, max_tokens=100
        )

        call_kwargs = mock_llm_client.chat.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    def test_prompt_formatting(self, mock_llm_client):
        """测试：prompt格式化"""
        client = HydeClient(mock_llm_client)

        client.generate_hypothetical_document("测试查询")

        # 验证prompt被正确格式化
        call_args = mock_llm_client.chat.call_args[1]
        messages = call_args["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "测试查询" in messages[1]["content"]


class TestSemanticChunkClient:
    """SemanticChunkClient的测试"""

    @pytest.fixture
    def mock_response(self):
        """模拟HTTP响应"""
        mock_response = Mock()
        mock_response.json.return_value = {"chunks": ["chunk1", "chunk2", "chunk3"]}
        mock_response.raise_for_status = Mock()
        return mock_response

    def test_init_with_default_url(self):
        """测试：使用默认URL初始化"""
        with patch(
            "src.evrag.client.semantic_chunk_client.get_settings"
        ) as mock_get_settings:
            mock_settings = Mock()
            mock_settings.semantic_chunk_url = "http://default:6000/v1/semantic-chunks"
            mock_get_settings.return_value = mock_settings

            client = SemanticChunkClient()

            assert client.url == "http://default:6000/v1/semantic-chunks"
            assert client.timeout == 30

    def test_init_with_custom_url(self):
        """测试：使用自定义URL初始化"""
        client = SemanticChunkClient(
            url="http://custom:7000/v1/semantic-chunks", timeout=60
        )

        assert client.url == "http://custom:7000/v1/semantic-chunks"
        assert client.timeout == 60

    def test_chunk_success(self, mock_response):
        """测试：成功进行语义分块"""
        with patch("src.evrag.client.semantic_chunk_client.requests.post") as mock_post:
            mock_post.return_value = mock_response

            client = SemanticChunkClient(url="http://test:6000/v1/semantic-chunks")
            result = client.chunk("测试文本内容", group_size=10)

            assert result == ["chunk1", "chunk2", "chunk3"]
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["timeout"] == 30
            assert "Content-Type" in call_kwargs["headers"]
            assert call_kwargs["headers"]["Content-Type"] == "application/json"

    def test_chunk_request_payload(self, mock_response):
        """测试：请求payload格式"""
        with patch("src.evrag.client.semantic_chunk_client.requests.post") as mock_post:
            mock_post.return_value = mock_response

            client = SemanticChunkClient()
            client.chunk("测试文本", group_size=5)

            call_args = mock_post.call_args
            payload = call_args[1]["data"]
            import json

            payload_dict = json.loads(payload)
            assert payload_dict["sentences"] == "测试文本"
            assert payload_dict["group_size"] == 5

    def test_chunk_http_error(self, mock_response):
        """测试：HTTP错误处理"""
        with patch("src.evrag.client.semantic_chunk_client.requests.post") as mock_post:
            import requests

            mock_response.raise_for_status.side_effect = requests.HTTPError(
                "404 Not Found"
            )
            mock_post.return_value = mock_response

            client = SemanticChunkClient()
            result = client.chunk("测试文本", group_size=10)

            # 失败时返回原始文本作为单个元素
            assert result == ["测试文本"]

    def test_chunk_connection_error(self):
        """测试：连接错误处理"""
        with patch("src.evrag.client.semantic_chunk_client.requests.post") as mock_post:
            import requests

            mock_post.side_effect = requests.ConnectionError("Connection failed")

            client = SemanticChunkClient()
            result = client.chunk("测试文本", group_size=10)

            # 失败时返回原始文本作为单个元素
            assert result == ["测试文本"]

    def test_chunk_timeout_error(self):
        """测试：超时错误处理"""
        with patch("src.evrag.client.semantic_chunk_client.requests.post") as mock_post:
            import requests

            mock_post.side_effect = requests.Timeout("Request timeout")

            client = SemanticChunkClient(timeout=10)
            result = client.chunk("测试文本", group_size=10)

            # 失败时返回原始文本作为单个元素
            assert result == ["测试文本"]

    def test_chunk_json_decode_error(self):
        """测试：JSON解析错误处理"""
        with patch("src.evrag.client.semantic_chunk_client.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_post.return_value = mock_response

            client = SemanticChunkClient()
            result = client.chunk("测试文本", group_size=10)

            # 失败时返回原始文本作为单个元素
            assert result == ["测试文本"]
