# 开发指南

本文档总结了EvRAG项目的开发规范、项目结构和测试指南。

## 📋 目录

- [项目结构](#项目结构)
- [开发流程](#开发流程)
- [测试指南](#测试指南)
- [实现检查清单](#实现检查清单)

---

## 📁 项目结构

### 核心目录

```
EvRAG/
├── main.py                    # ⭐ 主入口文件，所有CLI命令的入口
├── config/                    # 配置文件目录
├── src/evrag/                 # ⭐ 核心源代码目录
│   ├── config.py              # 配置管理
│   ├── retriever/             # 检索器模块
│   ├── reranker/              # 重排序模块
│   ├── parser/                # 文档解析模块
│   ├── client/                # 客户端模块
│   ├── server/                # 服务模块
│   ├── evaluation/            # 评估模块
│   └── utils/                 # 工具函数
├── scripts/                   # 脚本目录
├── data/                      # 数据目录
├── models/                    # 模型文件目录
└── tests/                     # 测试目录
```

### 核心文件说明

#### `main.py` - 主入口文件

**作用**: 项目的CLI入口，提供所有命令行功能

**主要命令**:
- `prepare-data`: PDF解析、文档清洗、文档切分
- `build-index`: 构建BM25和Milvus检索索引
- `gen-qa`: 生成QA对
- `process-qa`: QA数据处理
- `generate-sft-data`: 生成SFT训练数据
- `evaluate-rag`: RAG系统评估

#### 核心模块

- **检索器** (`retriever/`): BM25和Milvus检索器
- **重排序器** (`reranker/`): BGE Reranker
- **解析器** (`parser/`): PDF解析和文档切分
- **客户端** (`client/`): LLM客户端和MongoDB客户端
- **服务** (`server/`): FastAPI RAG服务和Gradio前端

---

## 🔄 开发流程

### GitFlow工作流

本项目采用GitFlow工作流进行版本管理。

#### 分支结构

```
main (生产分支，稳定版本)
  └── develop (开发主分支)
      ├── feature/* (功能开发分支)
      ├── release/* (发布准备分支)
      └── hotfix/* (热修复分支)
```

#### 分支命名规范

- **feature/**: `feature/模块名-功能描述`，如 `feature/retriever-bm25`
- **release/**: `release/版本号`，如 `release/v1.0.0`
- **hotfix/**: `hotfix/问题描述`，如 `hotfix/fix-milvus-connection`

#### 工作流程

##### 1. 开始新功能开发

```bash
# 切换到develop分支
git checkout develop
git pull origin develop

# 创建功能分支（使用git flow）
git flow feature start <feature-name>

# 或使用git命令
git checkout -b feature/<feature-name> develop
```

##### 2. 开发过程中提交

```bash
# 添加文件
git add <files>

# 提交（遵循约定式提交规范）
git commit -m "feat: Add new feature"

# 推送到远程
git push origin feature/<feature-name>
```

##### 3. 完成功能开发

```bash
# 合并到develop分支
git flow feature finish <feature-name>

# 或手动合并
git checkout develop
git merge feature/<feature-name>
git push origin develop
```

### 代码规范

- **Python风格**: 遵循PEP 8
- **类型提示**: 使用类型注解
- **文档字符串**: 所有公共函数和类都需要文档字符串
- **测试覆盖率**: 要求 > 80%，关键模块 > 90%

### 提交规范

遵循约定式提交规范（Conventional Commits）：

- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建/工具相关

示例：
```bash
git commit -m "feat: Add BM25 retriever"
git commit -m "fix: Fix MongoDB connection issue"
git commit -m "docs: Update README.md"
```

---

## 🧪 测试指南

### 测试框架

- **测试框架**: pytest
- **覆盖率工具**: pytest-cov
- **Mock工具**: unittest.mock

### 测试结构

```
tests/
├── conftest.py          # pytest配置和共享fixtures
├── unit/                # 单元测试
│   ├── test_config.py
│   ├── test_retriever.py
│   ├── test_reranker.py
│   ├── test_parser.py
│   ├── test_client.py
│   └── test_utils.py
└── integration/         # 集成测试
    └── test_end_to_end.py
```

### 运行测试

#### 运行所有测试

```bash
pytest tests/ -v
```

#### 运行单元测试

```bash
pytest tests/unit/ -v
```

#### 运行特定测试文件

```bash
pytest tests/unit/test_retriever.py -v
```

#### 运行特定测试类

```bash
pytest tests/unit/test_retriever.py::TestBM25Retriever -v
```

#### 运行特定测试方法

```bash
pytest tests/unit/test_retriever.py::TestBM25Retriever::test_retrieve_topk -v
```

#### 生成覆盖率报告

```bash
# HTML报告
pytest tests/ --cov=src/evrag --cov-report=html

# 控制台报告
pytest tests/ --cov=src/evrag --cov-report=term-missing
```

### 测试最佳实践

1. **单元测试**: 测试单个函数或类的功能
2. **集成测试**: 测试多个模块的协作
3. **Mock外部依赖**: 使用mock避免依赖外部服务
4. **测试覆盖率**: 确保关键代码有足够的测试覆盖
5. **测试命名**: 使用描述性的测试名称

---

## ✅ 实现检查清单

在实现新功能时，请确保完成以下检查：

### 代码质量

- [ ] 代码遵循PEP 8规范
- [ ] 添加了类型提示
- [ ] 添加了文档字符串
- [ ] 代码通过linting检查
- [ ] 代码通过格式化检查

### 测试

- [ ] 编写了单元测试
- [ ] 编写了集成测试（如需要）
- [ ] 测试覆盖率 > 80%
- [ ] 所有测试通过

### 文档

- [ ] 更新了README.md（如需要）
- [ ] 更新了API文档（如需要）
- [ ] 添加了使用示例

### 配置

- [ ] 添加了配置项到`config.py`
- [ ] 更新了`config.example.yaml`
- [ ] 更新了`.env.example`（如需要）

### Git

- [ ] 提交信息遵循约定式提交规范
- [ ] 代码已推送到远程仓库
- [ ] 创建了Pull Request（如需要）

---

## 📚 相关文档

- **详细项目结构**: `dev_docs/development/PROJECT_STRUCTURE.md`
- **开发流程详情**: `dev_docs/development/DEVELOPMENT.md`
- **测试指南详情**: `dev_docs/development/TESTING.md`
- **实现检查清单**: `dev_docs/development/implementation_details_checklist.md`
- **主README**: `README.md`

