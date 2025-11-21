# 测试指南

本文档介绍EvRAG项目的测试框架和测试规范。

## 测试框架

- **测试框架**: pytest
- **覆盖率工具**: pytest-cov
- **Mock工具**: unittest.mock

## 测试结构

```
tests/
├── conftest.py          # pytest配置和共享fixtures
├── unit/                # 单元测试
│   ├── test_config.py
│   ├── test_retriever.py
│   ├── test_reranker.py
│   ├── test_parser.py
│   ├── test_client.py
│   ├── test_gen_qa.py
│   └── test_utils.py
└── integration/         # 集成测试
    └── test_end_to_end.py
```

## 运行测试

### 运行所有测试

```bash
pytest tests/ -v
```

### 运行单元测试

```bash
pytest tests/unit/ -v
```

### 运行特定测试文件

```bash
pytest tests/unit/test_retriever.py -v
```

### 运行特定测试类

```bash
pytest tests/unit/test_retriever.py::TestBM25Retriever -v
```

### 运行特定测试方法

```bash
pytest tests/unit/test_retriever.py::TestBM25Retriever::test_retrieve_topk -v
```

### 运行并显示输出

```bash
pytest tests/ -v -s
```

## 测试覆盖率

### 生成覆盖率报告

```bash
# 生成HTML报告
pytest tests/ --cov=src/evrag --cov-report=html

# 查看报告
open htmlcov/index.html
# 或
xdg-open htmlcov/index.html  # Linux
```

### 覆盖率要求

- **目标覆盖率**: > 80%
- **最低要求**: > 75%
- **关键模块**: > 90%

### 查看覆盖率

```bash
# 终端输出
pytest tests/ --cov=src/evrag --cov-report=term

# XML报告（用于CI）
pytest tests/ --cov=src/evrag --cov-report=xml

# 同时生成多种格式
pytest tests/ --cov=src/evrag --cov-report=term --cov-report=html --cov-report=xml
```

## 编写测试

### 测试文件命名

- 测试文件：`test_<module>.py`
- 测试类：`Test<ClassName>`
- 测试方法：`test_<functionality>`

### 测试类结构

```python
import pytest
from unittest.mock import Mock, patch
from src.evrag.retriever import BM25Retriever

class TestBM25Retriever:
    """BM25Retriever的测试"""
    
    @pytest.fixture
    def sample_documents(self):
        """示例文档fixture"""
        return [
            Document(
                page_content="测试文档",
                metadata={"unique_id": "doc_1"}
            )
        ]
    
    def test_init(self, sample_documents):
        """测试初始化"""
        retriever = BM25Retriever(docs=sample_documents)
        assert retriever is not None
    
    def test_retrieve_topk(self, sample_documents):
        """测试检索功能"""
        retriever = BM25Retriever(docs=sample_documents)
        results = retriever.retrieve_topk("query", topk=5)
        assert len(results) <= 5
```

### 使用Fixtures

```python
# conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def project_root():
    """返回项目根目录"""
    return Path(__file__).parent.parent

# test_*.py
def test_something(project_root):
    config_file = project_root / "config" / "config.yaml"
    assert config_file.exists()
```

### Mock外部依赖

```python
from unittest.mock import Mock, patch, MagicMock

def test_with_mock():
    # Mock外部服务
    with patch("src.evrag.client.openai_client.OpenAI") as mock_openai:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content="mocked response"))
        ]
        mock_openai.return_value = mock_client
        
        # 测试代码
        from src.evrag.client import OpenAIClient
        client = OpenAIClient(api_key="test_key")
        result = client.chat([{"role": "user", "content": "test"}])
        
        assert result == "mocked response"
```

## 测试类型

### 单元测试

- **目标**: 测试单个函数或类的功能
- **特点**: 快速、独立、可重复
- **位置**: `tests/unit/`
- **要求**: 使用mock隔离外部依赖

### 集成测试

- **目标**: 测试多个模块的协作
- **特点**: 验证完整流程
- **位置**: `tests/integration/`
- **要求**: 可以依赖外部服务（如MongoDB、Milvus）

### 测试标记

```python
import pytest

@pytest.mark.slow
def test_slow_operation():
    """标记为慢速测试"""
    pass

@pytest.mark.integration
def test_integration():
    """标记为集成测试"""
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_skipped():
    """跳过测试"""
    pass
```

运行标记的测试：

```bash
# 运行慢速测试
pytest -m slow

# 运行集成测试
pytest -m integration

# 跳过慢速测试
pytest -m "not slow"

# 运行多个标记
pytest -m "slow or integration"
```

## 测试最佳实践

### 1. 测试独立性

- 每个测试应该独立运行
- 不依赖其他测试的执行顺序
- 使用fixtures设置和清理

### 2. 测试命名

- 使用描述性的测试名称
- 遵循 `test_<what>_<condition>_<expected_result>` 格式

示例：
```python
def test_retrieve_topk_with_empty_query_returns_empty_list():
    """测试：空查询时返回空列表"""
    pass

def test_retrieve_topk_with_invalid_topk_raises_error():
    """测试：无效topk值时抛出错误"""
    pass
```

### 3. 测试覆盖

- 测试正常流程
- 测试边界情况
- 测试异常情况

### 4. 断言清晰

```python
# 好的断言
assert len(results) == 5, "应该返回5个文档"
assert "expected" in result, f"结果中应包含'expected'，实际为: {result}"

# 避免模糊的断言
assert result  # 不够清晰
```

### 5. 使用参数化测试

```python
@pytest.mark.parametrize("input,expected", [
    ("test1", "result1"),
    ("test2", "result2"),
    ("", ""),  # 边界情况
])
def test_multiple_cases(input, expected):
    assert process(input) == expected
```

### 6. 测试异常

```python
def test_raises_exception():
    """测试：抛出异常"""
    with pytest.raises(ValueError, match="Invalid input"):
        function_that_raises("invalid")
```

## 持续集成

### GitHub Actions

项目使用GitHub Actions进行CI/CD，每次push和PR都会：

1. 运行代码质量检查（ruff, mypy）
2. 运行所有测试
3. 生成覆盖率报告
4. 上传覆盖率到Codecov

### 本地验证CI

```bash
# 运行CI脚本
bash scripts/check_code_quality.sh

# 手动运行所有检查
ruff format --check src/
ruff check src/
mypy src/
pytest tests/ --cov=src/evrag --cov-fail-under=80
```

## 常见问题

### 问题1：测试失败但本地通过

**解决方案**：
- 检查环境变量
- 检查依赖版本
- 清理缓存：`pytest --cache-clear`
- 检查是否有未提交的更改

### 问题2：覆盖率不达标

**解决方案**：
- 检查未覆盖的代码行：`pytest --cov=src/evrag --cov-report=term-missing`
- 添加边界测试
- 添加异常处理测试
- 检查是否有不必要的代码

### 问题3：测试运行缓慢

**解决方案**：
- 使用mock替代真实服务调用
- 使用fixtures缓存
- 标记慢速测试并选择性运行
- 并行运行测试：`pytest -n auto`

### 问题4：Mock不生效

**解决方案**：
- 确保patch路径正确（使用完整模块路径）
- 检查是否在正确的位置使用patch
- 使用`patch.object`而不是`patch`（如果需要）

### 问题5：导入错误

**解决方案**：
```bash
# 确保在项目根目录
cd /remote-home/share/liangZhang/EvRAG

# 检查Python路径
python -c "import sys; print('\n'.join(sys.path))"

# 使用PYTHONPATH
PYTHONPATH=/remote-home/share/liangZhang/EvRAG pytest tests/
```

## 测试示例

### 完整测试示例

```python
"""
检索器模块单元测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.evrag.retriever import BM25Retriever, BaseRetriever


class TestBaseRetriever:
    """BaseRetriever的测试"""

    def test_cannot_instantiate_base_class(self):
        """测试：不能直接实例化基类"""
        with pytest.raises(TypeError):
            BaseRetriever()


class TestBM25Retriever:
    """BM25Retriever的测试"""

    @pytest.fixture
    def sample_documents(self):
        """示例文档列表"""
        return [
            Document(
                page_content="打开车窗的方法",
                metadata={"unique_id": "doc_0", "index": 0}
            ),
            Document(
                page_content="空调加热功能",
                metadata={"unique_id": "doc_1", "index": 1}
            ),
        ]

    def test_init_with_documents(self, sample_documents):
        """测试：使用文档初始化"""
        retriever = BM25Retriever(docs=sample_documents, retrieve=False)
        assert retriever is not None
        assert retriever.documents == sample_documents

    def test_retrieve_topk(self, sample_documents):
        """测试：检索Top-K文档"""
        retriever = BM25Retriever(docs=sample_documents, retrieve=False)
        results = retriever.retrieve_topk("车窗", topk=2)
        
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)
```

## 下一步

- [开发指南](DEVELOPMENT.md) - 了解开发流程
- [环境设置指南](ENVIRONMENT.md) - 了解环境配置
- [README.md](../README.md) - 了解项目概览

