# EvRAG - Enhanced RAG System

一个增强的检索增强生成（RAG）系统，支持多种检索器、重排序器和LLM客户端。

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 功能特性

- 🔍 **多种检索器**: BM25（稀疏检索）、Milvus（混合检索）
- 📊 **重排序**: BGE Reranker支持
- 📄 **文档解析**: PDF解析、图片提取
- 🤖 **LLM集成**: 支持本地和远程LLM客户端（OpenAI兼容API）
- 💬 **RAG问答**: 完整的RAG问答流程
- 📝 **QA生成**: 自动生成问答对
- 🛠️ **CLI工具**: 命令行接口支持
- ⚙️ **配置管理**: 灵活的配置系统（环境变量、YAML文件）

## 快速开始

### 环境要求

- Python 3.12+
- Conda (推荐使用 Miniconda 或 Anaconda)
- CUDA (可选，用于GPU加速)

### 安装

1. **克隆仓库**

```bash
git clone https://github.com/LeonCheung033/EvRAG.git
cd EvRAG
```

2. **创建Conda环境**

```bash
conda env create -f environment.yml
conda activate evrag
```

3. **安装依赖**

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. **配置环境变量**

```bash
cp .env.example .env
# 编辑.env文件，填入你的配置
```

详细的环境设置请参考 [环境设置指南](docs/ENVIRONMENT.md)。

## 使用示例

### 构建索引

从PDF文件构建检索索引：

```bash
python main.py build-index --pdf-path data/Tesla_Manual.pdf
```

### 问答

使用RAG系统回答问题：

```bash
python main.py infer "如何打开车窗？" --topk 5
```

### 流式问答

支持流式输出：

```bash
python main.py infer "如何打开车窗？" --stream
```

### 生成QA对

从文档生成问答对：

```bash
python main.py gen-qa data/processed_docs/clean_docs.pkl --output data/qa_pairs/qa_pair.json
```

### 查看帮助

```bash
# 查看所有命令
python main.py --help

# 查看特定命令的帮助
python main.py build-index --help
python main.py infer --help
python main.py gen-qa --help
```

## 项目结构

```
EvRAG/
├── src/evrag/           # 源代码
│   ├── __init__.py      # 包初始化
│   ├── config.py        # 配置管理
│   ├── retriever/       # 检索器模块
│   │   ├── base.py      # 检索器基类
│   │   ├── bm25_retriever.py
│   │   └── milvus_retriever.py
│   ├── reranker/        # 重排序模块
│   │   ├── base.py      # 重排序器基类
│   │   └── bge_reranker.py
│   ├── parser/          # 文档解析模块
│   │   ├── pdf_parser.py
│   │   └── image_handler.py
│   ├── client/          # 客户端模块
│   │   ├── base.py      # LLM客户端基类
│   │   ├── local_client.py
│   │   ├── openai_client.py
│   │   ├── mongodb_client.py
│   │   ├── chat_client.py
│   │   ├── clean_client.py
│   │   ├── hyde_client.py
│   │   └── semantic_chunk_client.py
│   ├── gen_qa/          # QA生成模块
│   │   └── generator.py
│   └── utils.py         # 工具函数
├── tests/               # 测试
│   ├── conftest.py      # pytest配置
│   ├── unit/            # 单元测试
│   └── integration/     # 集成测试
├── config/              # 配置文件
│   ├── config.example.yaml
│   └── config.yaml
├── data/                # 数据目录
│   ├── saved_images/    # 保存的图片
│   └── processed_docs/  # 处理后的文档
├── docs/                # 文档
│   ├── ENVIRONMENT.md   # 环境设置指南
│   ├── DEVELOPMENT.md   # 开发指南
│   └── TESTING.md       # 测试指南
├── scripts/             # 脚本
│   ├── check_code_quality.sh
│   └── manage_dependencies.sh
├── main.py              # 主入口
├── environment.yml      # Conda环境配置
├── requirements.txt     # 生产依赖
├── requirements-dev.txt # 开发依赖
└── README.md            # 项目说明
```

## 核心模块

### 检索器 (Retriever)

- **BM25Retriever**: 基于BM25算法的稀疏检索器
- **MilvusRetriever**: 基于Milvus的混合检索器（支持稠密+稀疏向量）

### 重排序器 (Reranker)

- **BGEReranker**: 基于BGE模型的重排序器

### 解析器 (Parser)

- **PDFParser**: PDF文档解析，支持文本和图片提取
- **ImageHandler**: 图片处理和关联

### 客户端 (Client)

- **LocalLLMClient**: 本地LLM客户端
- **OpenAIClient**: OpenAI兼容API客户端
- **MongoDBClient**: MongoDB数据库客户端
- **ChatClient**: RAG问答客户端
- **CleanClient**: 文档清理客户端
- **HydeClient**: HYDE（假设文档生成）客户端
- **SemanticChunkClient**: 语义分块客户端

## 开发

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行单元测试
pytest tests/unit/ -v

# 生成覆盖率报告
pytest tests/ --cov=src/evrag --cov-report=html
```

### 代码质量检查

```bash
bash scripts/check_code_quality.sh
```

### GitFlow工作流

本项目采用GitFlow工作流，详细说明请参考 [开发指南](docs/DEVELOPMENT.md)。

## 文档

- [环境设置指南](docs/ENVIRONMENT.md) - 详细的环境配置说明
- [开发指南](docs/DEVELOPMENT.md) - 开发流程和代码规范
- [测试指南](docs/TESTING.md) - 测试框架和测试规范

## 配置

项目支持多种配置方式，优先级从高到低：

1. 环境变量
2. 命令行参数
3. YAML配置文件
4. 默认值

配置文件示例：`config/config.example.yaml`

## 贡献

欢迎贡献！请遵循以下步骤：

1. Fork本项目
2. 创建功能分支 (`git flow feature start feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'feat: Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

详细贡献指南请参考 [开发指南](docs/DEVELOPMENT.md)。

## 测试覆盖率

项目要求测试覆盖率 > 80%，关键模块 > 90%。

查看覆盖率报告：

```bash
pytest tests/ --cov=src/evrag --cov-report=html
open htmlcov/index.html
```

## 许可证

[添加许可证信息]

## 联系方式

- 项目地址: https://github.com/LeonCheung033/EvRAG
- 问题反馈: [GitHub Issues](https://github.com/LeonCheung033/EvRAG/issues)

## 致谢

感谢所有为这个项目做出贡献的开发者！
