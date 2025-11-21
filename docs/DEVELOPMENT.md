# 开发指南

本文档介绍EvRAG项目的开发流程和规范。

## 开发环境设置

参考 [环境设置指南](ENVIRONMENT.md) 完成开发环境的配置。

## GitFlow工作流

本项目采用GitFlow工作流进行版本管理。

### 分支结构

```
main (生产分支，稳定版本)
  └── develop (开发主分支)
      ├── feature/* (功能开发分支)
      ├── release/* (发布准备分支)
      └── hotfix/* (热修复分支)
```

### 分支命名规范

- **feature/**: `feature/模块名-功能描述`，如 `feature/retriever-bm25`
- **release/**: `release/版本号`，如 `release/v1.0.0`
- **hotfix/**: `hotfix/问题描述`，如 `hotfix/fix-milvus-connection`

### 工作流程

#### 1. 开始新功能开发

```bash
# 切换到develop分支
git checkout develop
git pull origin develop

# 创建功能分支（使用git flow）
git flow feature start <feature-name>

# 或使用git命令
git checkout -b feature/<feature-name> develop
```

#### 2. 开发过程中提交

```bash
# 添加文件
git add <files>

# 提交（使用Conventional Commits格式）
git commit -m "feat: add new feature"
```

#### 3. 完成功能开发

```bash
# 完成功能分支（合并到develop）
git flow feature finish <feature-name>

# 或使用git命令
git checkout develop
git merge --no-ff feature/<feature-name>
git branch -d feature/<feature-name>

# 推送到远程
git push origin develop
```

#### 4. 发布版本

```bash
# 创建发布分支
git flow release start v1.0.0

# 完成发布（合并到main和develop）
git flow release finish v1.0.0

# 推送标签
git push origin --tags
```

#### 5. 热修复

```bash
# 从main创建热修复分支
git flow hotfix start <hotfix-name>

# 完成热修复
git flow hotfix finish <hotfix-name>
```

## 提交信息规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

### 格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type类型

- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整（不影响代码运行）
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

### 示例

```bash
# 新功能
git commit -m "feat: add BM25 retriever implementation"

# 修复bug
git commit -m "fix: resolve MongoDB connection timeout issue"

# 文档更新
git commit -m "docs: update README with installation instructions"

# 测试
git commit -m "test: add unit tests for retriever module"

# 重构
git commit -m "refactor: simplify config loading logic"
```

## 代码规范

### Python代码风格

- 遵循 [PEP 8](https://pep8.org/) 规范
- 使用 `ruff` 进行代码格式化和检查
- 使用 `mypy` 进行类型检查

### 代码格式化

```bash
# 自动格式化
ruff format src/

# 检查格式
ruff format --check src/
```

### 代码检查

```bash
# 运行ruff检查
ruff check src/

# 运行mypy类型检查
mypy src/
```

### 代码质量检查脚本

```bash
# 运行完整的代码质量检查
bash scripts/check_code_quality.sh
```

## 模块开发规范

### 1. 模块结构

每个模块应包含：
- `__init__.py`: 导出公共接口
- 实现文件：如 `bm25_retriever.py`
- 单元测试：`tests/unit/test_<module>.py`

### 2. 文档字符串

所有公共函数和类都应包含文档字符串：

```python
def retrieve_topk(self, query: str, topk: int = 5) -> List[Document]:
    """
    检索Top-K文档
    
    Args:
        query: 查询文本
        topk: 返回的文档数量
        
    Returns:
        检索到的文档列表
    """
    pass
```

### 3. 类型提示

所有函数参数和返回值都应包含类型提示：

```python
from typing import List, Optional

def process_documents(
    docs: List[Document],
    max_size: Optional[int] = None
) -> List[Document]:
    pass
```

### 4. 异常处理

```python
try:
    # 可能失败的操作
    result = risky_operation()
except SpecificException as e:
    # 处理特定异常
    logger.error(f"Operation failed: {e}")
    raise
except Exception as e:
    # 处理其他异常
    logger.error(f"Unexpected error: {e}")
    raise RuntimeError(f"Operation failed: {e}") from e
```

## 依赖管理

### 添加新依赖

1. 安装依赖：`pip install <package>`
2. 更新requirements.txt：`pipreqs . --force`
3. 提交更改：`git commit -m "chore: add <package> dependency"`

### 依赖分类

- **生产依赖**: `requirements.txt`
- **开发依赖**: `requirements-dev.txt`
- **Conda依赖**: `environment.yml`（仅Python和基础工具）

## 测试要求

- 每个模块必须有对应的单元测试
- 测试覆盖率目标：> 80%
- 所有测试必须通过才能合并到develop分支

参考 [测试指南](TESTING.md) 了解详细的测试规范。

## 代码审查

### Pull Request流程

1. 创建PR从feature分支到develop分支
2. 确保所有测试通过
3. 确保代码质量检查通过
4. 等待代码审查
5. 根据反馈修改代码
6. 审查通过后合并

### 审查检查清单

- [ ] 代码符合规范
- [ ] 所有测试通过
- [ ] 测试覆盖率达标
- [ ] 文档已更新
- [ ] 提交信息符合规范
- [ ] 没有引入新的警告或错误

## 开发最佳实践

### 1. 小步提交

- 频繁提交，每次提交包含一个完整的功能或修复
- 提交信息清晰描述变更内容

### 2. 测试驱动开发

- 先写测试，再写实现
- 确保测试覆盖正常流程、边界情况和异常情况

### 3. 代码复用

- 提取公共功能到工具函数
- 使用抽象基类定义接口
- 避免重复代码

### 4. 文档同步

- 代码变更时同步更新文档
- 保持README和代码示例的一致性

## 调试技巧

### 使用日志

```python
import logging

logger = logging.getLogger(__name__)

logger.debug("Debug information")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### 使用调试器

```bash
# 使用pdb
python -m pdb main.py

# 使用ipdb（需要安装）
pip install ipdb
python -m ipdb main.py
```

### 运行单个测试

```bash
# 运行特定测试并进入调试器
pytest tests/unit/test_retriever.py::TestBM25Retriever::test_retrieve_topk -v -s --pdb
```

## 性能优化

### 性能分析

```bash
# 使用cProfile
python -m cProfile -o profile.stats main.py

# 使用py-spy（需要安装）
pip install py-spy
py-spy record -o profile.svg -- python main.py
```

### 内存分析

```bash
# 使用memory_profiler
pip install memory-profiler
python -m memory_profiler main.py
```

## 下一步

- [测试指南](TESTING.md) - 了解测试规范
- [环境设置指南](ENVIRONMENT.md) - 了解环境配置
- [README.md](../README.md) - 了解项目概览

