"""
重排序器单元测试模块

测试BGE-M3重排序器模型的功能和性能。
"""

import torch
import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from src.evrag.reranker import BGEReranker
from src.evrag.reranker.base import BaseReranker


class TestBaseReranker:
    """BGEReranker的测试"""

    def test_cannot_instantiate_base_class(self):
        with pytest.raises(TypeError):
            BaseReranker()

    def test_abstract_method(self):
        class IncompleteReranker(BaseReranker):
            pass

        with pytest.raises(TypeError):
            IncompleteReranker()


class TestBGEReranker:
    """BGEReranker的测试"""

    @pytest.fixture
    def sample_documents(self):
        return [
            Document(
                page_content="打开车窗的方法",
                metadata={"unique_id": "doc_0", "index": 0},
            ),
            Document(
                page_content="空调加热功能",
                metadata={"unique_id": "doc_1", "index": 1},
            ),
            Document(
                page_content="座椅加热设置",
                metadata={"unique_id": "doc_2", "index": 2},
            ),
            Document(
                page_content="如何开启车窗",
                metadata={"unique_id": "doc_3", "index": 3},
            ),
        ]

    @pytest.fixture
    def mock_model_path(self, tmp_path):
        model_dir = tmp_path / "bge_reranker"
        model_dir.mkdir()
        # 创建一些假文件以模拟模型目录
        (model_dir / "config.json").touch()
        (model_dir / "pytorch_model.bin").touch()
        return model_dir

    @pytest.fixture
    def mock_tokenizer(self):
        """模拟分词器"""
        tokenizer = Mock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        """模拟模型"""
        model = Mock()
        # 模拟模型输出
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.9], [0.7], [0.8], [0.6]])
        model.return_value = mock_output
        model.eval = Mock(return_value=model)
        model.half = Mock(return_value=model)
        model.cuda = Mock(return_value=model)
        model.float = Mock(return_value=model)
        model.cpu = Mock(return_value=model)
        return model

    def test_init_with_model_path(self, mock_model_path, mock_tokenizer, mock_model):
        """测试：使用指定模型路径初始化"""
        with (
            patch(
                "src.evrag.reranker.bge_reranker.AutoTokenizer"
            ) as mock_tokenizer_class,
            patch(
                "src.evrag.reranker.bge_reranker.AutoModelForSequenceClassification"
            ) as mock_model_class,
        ):
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model

            reranker = BGEReranker(model_path=mock_model_path, device="cpu")

            assert reranker.model_path == mock_model_path
            assert reranker.device == "cpu"
            assert reranker.max_length == 4096

    def test_init_from_config(self, mock_model_path, mock_tokenizer, mock_model):
        """测试：从配置中读取模型路径"""
        with (
            patch(
                "src.evrag.reranker.bge_reranker.AutoTokenizer"
            ) as mock_tokenizer_class,
            patch(
                "src.evrag.reranker.bge_reranker.AutoModelForSequenceClassification"
            ) as mock_model_class,
            patch("src.evrag.reranker.bge_reranker.get_settings") as mock_get_settings,
        ):
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model

            mock_settings = Mock()
            mock_settings.bge_reranker_tuned_model_path = None
            mock_settings.bge_reranker_model_path = mock_model_path
            mock_settings.device = "cpu"
            mock_get_settings.return_value = mock_settings

            reranker = BGEReranker(device="cpu")

            assert reranker.model_path == mock_model_path

    def test_init_without_model_path_raises_error(self):
        """测试：没有模型路径时抛出错误"""
        with patch("src.evrag.reranker.bge_reranker.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.bge_reranker_tuned_model_path = None
            mock_settings.bge_reranker_model_path = None
            mock_get_settings.return_value = mock_settings

            with pytest.raises(
                ValueError, match="BGE reranker model path not specified"
            ):
                BGEReranker()

    def test_init_with_nonexistent_model_path_raises_error(self, tmp_path):
        """测试：模型路径不存在时抛出错误"""
        nonexistent_path = tmp_path / "nonexistent_model"

        with pytest.raises(FileNotFoundError, match="BGE reranker model not found"):
            BGEReranker(model_path=nonexistent_path)

    def test_rank(self, sample_documents, mock_model_path, mock_tokenizer, mock_model):
        """测试：重排序功能"""
        with (
            patch(
                "src.evrag.reranker.bge_reranker.AutoTokenizer"
            ) as mock_tokenizer_class,
            patch(
                "src.evrag.reranker.bge_reranker.AutoModelForSequenceClassification"
            ) as mock_model_class,
        ):
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            # 设置模型输出：第一个文档分数最高
            mock_output = Mock()
            mock_output.logits = torch.tensor([[0.9], [0.6], [0.8], [0.5]])
            mock_model.return_value = mock_output
            mock_model.eval = Mock(return_value=mock_model)
            mock_model.float = Mock(return_value=mock_model)
            mock_model.cpu = Mock(return_value=mock_model)
            mock_model_class.from_pretrained.return_value = mock_model

            # 设置分词器返回值
            def tokenize_side_effect(*args, **kwargs):
                num_pairs = len(args[0]) if args else 4
                return {
                    "input_ids": torch.tensor([[1, 2, 3]] * num_pairs),
                    "attention_mask": torch.tensor([[1, 1, 1]] * num_pairs),
                }

            mock_tokenizer.side_effect = tokenize_side_effect

            reranker = BGEReranker(model_path=mock_model_path, device="cpu")

            query = "车窗"
            results = reranker.rank(query, sample_documents, topk=2)

            assert isinstance(results, list)
            assert len(results) == 2
            assert all(isinstance(doc, Document) for doc in results)
            # 第一个文档应该是最相关的（分数0.9）
            assert results[0].page_content == "打开车窗的方法"

    def test_rank_empty_docs(self, mock_model_path, mock_tokenizer, mock_model):
        """测试：空文档列表"""
        with (
            patch(
                "src.evrag.reranker.bge_reranker.AutoTokenizer"
            ) as mock_tokenizer_class,
            patch(
                "src.evrag.reranker.bge_reranker.AutoModelForSequenceClassification"
            ) as mock_model_class,
        ):
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model

            reranker = BGEReranker(model_path=mock_model_path, device="cpu")
            results = reranker.rank("query", [], topk=10)

            assert results == []

    def test_rank_empty_query(
        self, sample_documents, mock_model_path, mock_tokenizer, mock_model
    ):
        """测试：空查询"""
        with (
            patch(
                "src.evrag.reranker.bge_reranker.AutoTokenizer"
            ) as mock_tokenizer_class,
            patch(
                "src.evrag.reranker.bge_reranker.AutoModelForSequenceClassification"
            ) as mock_model_class,
        ):
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model

            reranker = BGEReranker(model_path=mock_model_path, device="cpu")
            results = reranker.rank("", sample_documents, topk=2)

            # 空查询应该返回原始文档列表的前topk个
            assert len(results) == 2
            assert results == sample_documents[:2]

    def test_rank_single_document(self, mock_model_path, mock_tokenizer, mock_model):
        """测试：单个文档重排序"""
        doc = Document(page_content="测试文档", metadata={"id": "1"})

        with (
            patch(
                "src.evrag.reranker.bge_reranker.AutoTokenizer"
            ) as mock_tokenizer_class,
            patch(
                "src.evrag.reranker.bge_reranker.AutoModelForSequenceClassification"
            ) as mock_model_class,
        ):
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            # 单个文档时，logits可能是标量
            mock_output = Mock()
            mock_output.logits = torch.tensor([0.8])  # 标量
            mock_model.return_value = mock_output
            mock_model.eval = Mock(return_value=mock_model)
            mock_model.float = Mock(return_value=mock_model)
            mock_model.cpu = Mock(return_value=mock_model)
            mock_model_class.from_pretrained.return_value = mock_model

            def tokenize_side_effect(*args, **kwargs):
                return {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                }

            mock_tokenizer.side_effect = tokenize_side_effect

            reranker = BGEReranker(model_path=mock_model_path, device="cpu")
            results = reranker.rank("查询", [doc], topk=1)

            assert len(results) == 1
            assert results[0] == doc

    def test_rank_topk_larger_than_docs(
        self, sample_documents, mock_model_path, mock_tokenizer, mock_model
    ):
        """测试：topk大于文档数量"""
        with (
            patch(
                "src.evrag.reranker.bge_reranker.AutoTokenizer"
            ) as mock_tokenizer_class,
            patch(
                "src.evrag.reranker.bge_reranker.AutoModelForSequenceClassification"
            ) as mock_model_class,
        ):
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            mock_output = Mock()
            mock_output.logits = torch.tensor([[0.9], [0.8], [0.7], [0.6]])
            mock_model.return_value = mock_output
            mock_model.eval = Mock(return_value=mock_model)
            mock_model.float = Mock(return_value=mock_model)
            mock_model.cpu = Mock(return_value=mock_model)
            mock_model_class.from_pretrained.return_value = mock_model

            def tokenize_side_effect(*args, **kwargs):
                num_pairs = len(args[0]) if args else 4
                return {
                    "input_ids": torch.tensor([[1, 2, 3]] * num_pairs),
                    "attention_mask": torch.tensor([[1, 1, 1]] * num_pairs),
                }

            mock_tokenizer.side_effect = tokenize_side_effect

            reranker = BGEReranker(model_path=mock_model_path, device="cpu")
            results = reranker.rank("查询", sample_documents, topk=10)

            # 应该返回所有文档（因为只有4个）
            assert len(results) == 4

    def test_cuda_device(self, mock_model_path, mock_tokenizer, mock_model):
        """测试：CUDA设备支持"""
        with (
            patch(
                "src.evrag.reranker.bge_reranker.AutoTokenizer"
            ) as mock_tokenizer_class,
            patch(
                "src.evrag.reranker.bge_reranker.AutoModelForSequenceClassification"
            ) as mock_model_class,
            patch("torch.cuda.is_available", return_value=True),
        ):
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model

            reranker = BGEReranker(model_path=mock_model_path, device="cuda")

            assert reranker.device == "cuda"
            # 验证模型被移动到CUDA
            mock_model.half.assert_called_once()
            mock_model.cuda.assert_called_once()

    def test_cuda_unavailable_falls_back_to_cpu(
        self, mock_model_path, mock_tokenizer, mock_model
    ):
        """测试：CUDA不可用时回退到CPU"""
        with (
            patch(
                "src.evrag.reranker.bge_reranker.AutoTokenizer"
            ) as mock_tokenizer_class,
            patch(
                "src.evrag.reranker.bge_reranker.AutoModelForSequenceClassification"
            ) as mock_model_class,
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model

            reranker = BGEReranker(model_path=mock_model_path, device="cuda")

            # 应该回退到CPU
            assert reranker.device == "cpu"
            mock_model.float.assert_called_once()
            mock_model.cpu.assert_called_once()
