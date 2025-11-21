"""
QA生成模块单元测试

测试QA生成器的功能和性能。
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.evrag.gen_qa import QAGenerator
from src.evrag.client import BaseLLMClient


class TestQAGenerator:
    """QAGenerator的测试"""

    @pytest.fixture
    def mock_llm_client(self):
        """模拟LLM客户端"""
        mock_client = Mock(spec=BaseLLMClient)
        mock_client.chat.return_value = '{"test": "response"}'
        return mock_client

    @pytest.fixture
    def sample_documents(self):
        """示例文档列表"""
        return [
            Document(
                page_content="这是一段关于如何打开车窗的详细说明。首先，找到车窗控制按钮。然后，按下按钮即可打开车窗。",
                metadata={"unique_id": "doc_1", "index": 0}
            ),
            Document(
                page_content="空调系统使用说明。",
                metadata={"unique_id": "doc_2", "index": 1}
            ),
        ]

    def test_init_with_default_params(self, mock_llm_client):
        """测试：使用默认参数初始化"""
        generator = QAGenerator(mock_llm_client)

        assert generator.llm_client == mock_llm_client
        assert generator.min_chunk_size == 100
        assert generator.max_workers == 20
        assert generator.max_retry == 3

    def test_init_with_custom_params(self, mock_llm_client):
        """测试：使用自定义参数初始化"""
        generator = QAGenerator(
            mock_llm_client,
            min_chunk_size=50,
            max_workers=10,
            max_retry=5,
            seed=123
        )

        assert generator.min_chunk_size == 50
        assert generator.max_workers == 10
        assert generator.max_retry == 5

    def test_build_prompt(self, mock_llm_client):
        """测试：构建prompt"""
        generator = QAGenerator(mock_llm_client)
        template = "Template with {{document}} placeholder"
        text = "test text"

        result = generator._build_prompt(template, text)

        assert "test text" in result
        assert "{{document}}" not in result

    def test_build_prompt_with_question_placeholder(self, mock_llm_client):
        """测试：构建包含question占位符的prompt"""
        generator = QAGenerator(mock_llm_client)
        template = "Template with {{question}} placeholder"
        text = "test question"

        result = generator._build_prompt(template, text)

        assert "test question" in result
        assert "{{question}}" not in result

    def test_call_llm_success(self, mock_llm_client):
        """测试：成功调用LLM"""
        mock_llm_client.chat.return_value = "Success response"
        generator = QAGenerator(mock_llm_client)

        result = generator._call_llm("test prompt")

        assert result == "Success response"
        mock_llm_client.chat.assert_called_once()

    def test_call_llm_with_retry(self, mock_llm_client):
        """测试：LLM调用失败后重试"""
        # 第一次失败，第二次成功
        mock_llm_client.chat.side_effect = [
            Exception("First failure"),
            "Success response"
        ]
        generator = QAGenerator(mock_llm_client, max_retry=3)

        with patch('time.sleep'):  # 跳过sleep
            result = generator._call_llm("test prompt")

        assert result == "Success response"
        assert mock_llm_client.chat.call_count == 2

    def test_call_llm_max_retry_exceeded(self, mock_llm_client):
        """测试：超过最大重试次数后返回None"""
        mock_llm_client.chat.side_effect = Exception("Always fails")
        generator = QAGenerator(mock_llm_client, max_retry=2)

        with patch('time.sleep'):  # 跳过sleep
            result = generator._call_llm("test prompt")

        assert result is None
        assert mock_llm_client.chat.call_count == 2

    def test_generate_qa_from_documents_success(self, mock_llm_client, sample_documents):
        """测试：成功从文档生成QA对"""
        mock_llm_client.chat.return_value = json.dumps([
            {"question": "如何打开车窗？", "answer": "找到车窗控制按钮，按下即可。"},
            {"question": "车窗控制按钮在哪里？", "answer": "在车门上。"}
        ])
        generator = QAGenerator(mock_llm_client, min_chunk_size=10)

        result = generator.generate_qa_from_documents(sample_documents)

        assert len(result) > 0
        assert "doc_1" in result
        assert "raw_resp" in result["doc_1"]

    def test_generate_qa_from_documents_filter_short_docs(self, mock_llm_client, sample_documents):
        """测试：过滤太短的文档"""
        generator = QAGenerator(mock_llm_client, min_chunk_size=100)

        result = generator.generate_qa_from_documents(sample_documents)

        # doc_2太短，应该被过滤
        assert "doc_2" not in result

    def test_generate_qa_from_documents_with_checkpoint(self, mock_llm_client, sample_documents):
        """测试：使用checkpoint跳过已处理的文档"""
        mock_llm_client.chat.return_value = json.dumps([{"question": "Q", "answer": "A"}])
        generator = QAGenerator(mock_llm_client, min_chunk_size=10)
        checkpoint = {"doc_1": {"unique_id": "doc_1", "raw_resp": "existing"}}

        result = generator.generate_qa_from_documents(sample_documents, checkpoint=checkpoint)

        # doc_1应该被跳过
        assert result["doc_1"]["raw_resp"] == "existing"
        # 不应该为doc_1调用LLM
        assert mock_llm_client.chat.call_count == 0

    def test_generate_qa_from_documents_with_output_file(self, mock_llm_client, sample_documents):
        """测试：生成QA对并保存到文件"""
        mock_llm_client.chat.return_value = json.dumps([{"question": "Q", "answer": "A"}])
        generator = QAGenerator(mock_llm_client, min_chunk_size=10)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            output_file = Path(f.name)

        try:
            result = generator.generate_qa_from_documents(sample_documents, output_file=output_file)

            assert output_file.exists()
            # 验证文件内容
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) > 0
        finally:
            output_file.unlink()

    def test_generalize_questions_success(self, mock_llm_client):
        """测试：成功泛化问题"""
        mock_llm_client.chat.return_value = "1. 怎么打开车窗\n2. 如何开启车窗\n3. 车窗怎么开"
        generator = QAGenerator(mock_llm_client)
        questions = ["如何打开车窗？"]

        result = generator.generalize_questions(questions)

        assert len(result) > 0
        assert "如何打开车窗？" in result

    def test_generalize_questions_with_output_file(self, mock_llm_client):
        """测试：泛化问题并保存到文件"""
        mock_llm_client.chat.return_value = "1. 问题1\n2. 问题2"
        generator = QAGenerator(mock_llm_client)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            output_file = Path(f.name)

        try:
            result = generator.generalize_questions(["测试问题"], output_file=output_file)

            assert output_file.exists()
        finally:
            output_file.unlink()

    def test_extract_keywords_success(self, mock_llm_client):
        """测试：成功提取关键词"""
        mock_llm_client.chat.return_value = "行车记录仪,探测功能,辅助驾驶"
        generator = QAGenerator(mock_llm_client)
        texts = ["这是一段关于行车记录仪和辅助驾驶的文本"]

        result = generator.extract_keywords(texts)

        assert len(result) > 0

    def test_score_qa_quality_success(self, mock_llm_client):
        """测试：成功进行QA质量评分"""
        mock_llm_client.chat.return_value = '{"score": 4, "reason": "Good QA pair"}'
        generator = QAGenerator(mock_llm_client)

        result = generator.score_qa_quality("如何打开车窗？", "找到按钮按下即可")

        assert result is not None
        assert result["score"] == 4
        assert "reason" in result

    def test_score_qa_quality_with_result_tags(self, mock_llm_client):
        """测试：从带标签的结果中提取JSON"""
        mock_llm_client.chat.return_value = '<result>{"score": 5, "reason": "Excellent"}</result>'
        generator = QAGenerator(mock_llm_client)

        result = generator.score_qa_quality("Q", "A")

        assert result is not None
        assert result["score"] == 5

    def test_score_qa_quality_parse_failure(self, mock_llm_client):
        """测试：JSON解析失败时返回None"""
        mock_llm_client.chat.return_value = "Invalid response"
        generator = QAGenerator(mock_llm_client)

        result = generator.score_qa_quality("Q", "A")

        assert result is None

    def test_parse_qa_response_valid_json(self):
        """测试：解析有效的QA响应JSON"""
        raw_resp = json.dumps([
            {"question": "如何打开车窗？", "answer": "按下按钮"},
            {"question": "如何关闭车窗？", "answer": "再次按下按钮"}
        ])

        result = QAGenerator.parse_qa_response(raw_resp)

        assert len(result) == 2
        assert result[0]["question"] == "如何打开车窗？"
        assert result[0]["answer"] == "按下按钮"

    def test_parse_qa_response_with_text_wrapper(self):
        """测试：解析带文本包装的JSON响应"""
        raw_resp = "Here is the result:\n" + json.dumps([
            {"question": "Q", "answer": "A"}
        ]) + "\nEnd of result"

        result = QAGenerator.parse_qa_response(raw_resp)

        assert len(result) == 1
        assert result[0]["question"] == "Q"

    def test_parse_qa_response_invalid_json(self):
        """测试：解析无效JSON时返回空列表"""
        raw_resp = "This is not valid JSON"

        result = QAGenerator.parse_qa_response(raw_resp)

        assert result == []

    def test_parse_qa_response_empty_list(self):
        """测试：解析空列表"""
        raw_resp = json.dumps([])

        result = QAGenerator.parse_qa_response(raw_resp)

        assert result == []

    def test_parse_generalized_questions_success(self):
        """测试：成功解析泛化问题"""
        raw_resp = "1. 怎么打开车窗\n2. 如何开启车窗\n3. 车窗怎么开\n4. 开启车窗的方法\n5. 如何操作车窗"

        result = QAGenerator.parse_generalized_questions(raw_resp)

        assert len(result) == 5
        assert "怎么打开车窗" in result
        assert "如何开启车窗" in result
        # 验证序号被移除
        assert not any(q.startswith("1.") or q.startswith("2.") for q in result)

    def test_parse_generalized_questions_with_different_formats(self):
        """测试：解析不同格式的序号"""
        raw_resp = "1. 问题1\n2. 问题2\n3) 问题3\n4 问题4"

        result = QAGenerator.parse_generalized_questions(raw_resp)

        assert len(result) == 4
        assert all(not q[0].isdigit() for q in result)  # 所有问题都不应该以数字开头

    def test_parse_generalized_questions_empty_lines(self):
        """测试：过滤空行"""
        raw_resp = "1. 问题1\n\n2. 问题2\n   \n3. 问题3"

        result = QAGenerator.parse_generalized_questions(raw_resp)

        assert len(result) == 3
        assert "" not in result

    def test_parse_keywords_success(self):
        """测试：成功解析关键词"""
        raw_resp = "行车记录仪,探测功能,辅助驾驶,车辆功率"

        result = QAGenerator.parse_keywords(raw_resp)

        assert len(result) == 4
        assert "行车记录仪" in result
        assert "探测功能" in result

    def test_parse_keywords_filter_invalid(self):
        """测试：过滤无效关键词"""
        raw_resp = "行车记录仪,无,Model 3,辅助驾驶"

        result = QAGenerator.parse_keywords(raw_resp)

        assert "无" not in result
        assert "Model 3" not in result
        assert "行车记录仪" in result
        assert "辅助驾驶" in result

    def test_parse_keywords_with_spaces(self):
        """测试：处理带空格的关键词"""
        raw_resp = "行车记录仪, 探测功能 , 辅助驾驶"

        result = QAGenerator.parse_keywords(raw_resp)

        assert all(not k.startswith(" ") and not k.endswith(" ") for k in result)

    def test_generate_qa_llm_failure(self, mock_llm_client, sample_documents):
        """测试：LLM调用失败时跳过文档"""
        mock_llm_client.chat.side_effect = Exception("LLM error")
        generator = QAGenerator(mock_llm_client, min_chunk_size=10, max_retry=1)

        with patch('time.sleep'):  # 跳过sleep
            result = generator.generate_qa_from_documents(sample_documents)

        # 所有文档都失败，结果应该为空或只包含checkpoint
        assert "doc_1" not in result or result.get("doc_1", {}).get("raw_resp") is None

    def test_generate_qa_concurrent_processing(self, mock_llm_client):
        """测试：并发处理多个文档"""
        mock_llm_client.chat.return_value = json.dumps([{"question": "Q", "answer": "A"}])
        generator = QAGenerator(mock_llm_client, min_chunk_size=10, max_workers=2)
        
        documents = [
            Document(
                page_content="文档内容" * 20,  # 确保长度足够
                metadata={"unique_id": f"doc_{i}"}
            )
            for i in range(5)
        ]

        result = generator.generate_qa_from_documents(documents)

        # 验证所有文档都被处理
        assert len(result) == 5
        assert all(f"doc_{i}" in result for i in range(5))