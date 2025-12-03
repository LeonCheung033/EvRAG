# EvRAG - Enhanced RAG System

一个增强的检索增强生成（RAG）系统，支持多种检索器、重排序器和LLM客户端，提供完整的端到端RAG解决方案。

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [使用指南](#使用指南)
- [项目结构](#项目结构)
- [核心模块](#核心模块)
- [开发指南](#开发指南)
- [配置说明](#配置说明)
- [常见问题](#常见问题)

---

## 🎯 项目概述

### 问题定义

EvRAG是一个面向文档问答的检索增强生成（RAG）系统，旨在解决以下问题：

1. **文档检索挑战**：从大量文档中快速准确地检索相关信息
2. **答案生成质量**：基于检索到的上下文生成准确、相关的答案
3. **多模态支持**：支持文本和图片的联合检索与展示
4. **系统集成**：提供完整的端到端解决方案，包括CLI和Web界面

### 系统范围

- **输入**：用户问题（自然语言查询）
- **处理**：混合检索（BM25 + Milvus）→ 重排序（BGE Reranker）→ LLM生成
- **输出**：答案文本、引用页码、相关图片、性能指标
- **应用场景**：文档问答、知识库查询、用户手册查询等

---

## 🏗️ 系统架构

### 整体架构

```
┌─────────────────┐
│   用户界面层     │
│  (CLI / Gradio) │
└────────┬────────┘
         │
┌────────▼────────┐
│   RAG服务层     │
│  (FastAPI)      │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────┐
│检索层 │ │生成层   │
│BM25   │ │LLM     │
│Milvus │ │(vLLM)  │
└───┬───┘ └────────┘
    │
┌───▼──────┐
│重排序层  │
│Reranker │
└─────────┘
```

### 核心流程

1. **检索阶段**：使用BM25（稀疏检索）和Milvus（向量检索）并行检索相关文档
2. **合并去重**：合并两种检索结果，去除重复文档
3. **重排序**：使用BGE Reranker对文档进行精排
4. **上下文构建**：将top-k文档组织成上下文
5. **答案生成**：使用微调的LLM生成答案
6. **后处理**：提取答案、引用页码、相关图片

### 技术栈

- **检索**：BM25（rank-bm25）、Milvus（向量数据库）
- **重排序**：BGE-Reranker-v2-m3（支持微调）
- **LLM**：Qwen3（支持LoRA微调）
- **框架**：FastAPI（后端服务）、Gradio（前端界面）
- **数据库**：MongoDB（文档存储）

---

## ✨ 功能特性

- 🔍 **混合检索**: BM25（稀疏检索）+ Milvus（向量检索）
- 📊 **智能重排序**: BGE Reranker支持，可微调优化
- 📄 **文档解析**: PDF解析、文本提取、图片提取
- 🤖 **LLM集成**: 支持本地vLLM和远程OpenAI兼容API
- 💬 **RAG问答**: 完整的端到端RAG流程
- 🌐 **Web界面**: Gradio前端，支持流式输出和多轮对话
- 📝 **QA生成**: 自动生成问答对用于训练和评估
- 🛠️ **CLI工具**: 完整的命令行接口
- ⚙️ **配置管理**: 灵活的配置系统（环境变量、YAML文件）
- 📈 **性能监控**: 实时性能指标追踪

---

## 🚀 快速开始

### 环境要求

- **Python**: 3.12+
- **Conda**: Miniconda 或 Anaconda（推荐）
- **CUDA**: 11.8+（可选，用于GPU加速）
- **内存**: 至少16GB RAM（推荐32GB+）
- **GPU**: 推荐使用GPU（用于LLM推理和重排序）

### 安装步骤

#### 1. 克隆仓库

```bash
git clone https://github.com/LeonCheung033/EvRAG.git
cd EvRAG
```

#### 2. 创建Conda环境

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate evrag
```

#### 3. 安装依赖

```bash
# 安装生产依赖
pip install -r requirements.txt

# 安装开发依赖（可选）
pip install -r requirements-dev.txt
```

#### 4. 配置环境

```bash
# 复制配置文件
cp config/config.example.yaml config/config.yaml

# 编辑配置文件，设置模型路径等
vim config/config.yaml
```

#### 5. 下载模型（可选）

```bash
# 使用脚本下载所需模型
./scripts/setup/download_models.sh
```

**注意**：模型文件较大，建议提前下载或使用HuggingFace镜像。

---

## 📖 使用指南

### 方式一：命令行接口（CLI）

#### 1. 数据准备

```bash
# 解析PDF文件，提取文本和图片
python main.py prepare-data --pdf-path data/Tesla_Manual.pdf
```

#### 2. 构建索引

```bash
# 构建BM25和Milvus检索索引
python main.py build-index --pdf-path data/Tesla_Manual.pdf
```

#### 3. 问答

```bash
# 基础问答
python main.py infer "如何打开车窗？" --topk 5

# 流式输出
python main.py infer "如何打开车窗？" --stream

# 启用思考模式（复杂推理任务）
python main.py infer "如何打开车窗？" --enable-thinking
```

#### 4. 生成QA对

```bash
# 从文档生成问答对
python main.py gen-qa data/processed_docs/clean_docs.pkl \
    --output data/qa_pairs/qa_pair.json
```

### 方式二：Web界面（Gradio）

#### 1. 启动服务（按顺序）

```bash
# 1. 启动vLLM服务（端口8001）
./scripts/deployment/start_vllm_finetuned.sh

# 2. 启动RAG服务（端口8002）
python -m src.evrag.server.rag_server
# 或使用uvicorn:
# uvicorn src.evrag.server.rag_server:app --host 0.0.0.0 --port 8002

# 3. 启动Gradio前端（端口8080）
./scripts/deployment/start_gradio.sh
# 或直接运行:
# python -m src.evrag.server.gradio_app
```

#### 2. 访问界面

打开浏览器访问：`http://localhost:8080`

#### 3. 使用界面

- **输入问题**：在输入框中输入问题
- **调整参数**：可以调整BM25 TopK、Milvus TopK、Reranker TopK等参数
- **流式输出**：开启流式输出开关，实时查看生成过程
- **查看结果**：查看答案、相关图片和性能指标

#### 4. 停止服务

```bash
./scripts/deployment/stop_gradio.sh
./scripts/deployment/stop_vllm_finetuned.sh
```

### 完整复现流程

#### 步骤1：环境准备

```bash
# 1. 克隆仓库
git clone https://github.com/LeonCheung033/EvRAG.git
cd EvRAG

# 2. 创建环境
conda env create -f environment.yml
conda activate evrag

# 3. 安装依赖
pip install -r requirements.txt
```

#### 步骤2：数据准备

```bash
# 准备PDF文件（示例：Tesla手册）
# 将PDF文件放到 data/ 目录下

# 解析PDF
python main.py prepare-data --pdf-path data/Tesla_Manual.pdf

# 构建索引
python main.py build-index --pdf-path data/Tesla_Manual.pdf
```

#### 步骤3：启动服务

```bash
# 启动vLLM服务
./scripts/deployment/start_vllm_finetuned.sh

# 启动RAG服务（新终端）
python -m src.evrag.server.rag_server

# 启动Gradio前端（新终端）
./scripts/deployment/start_gradio.sh
```

#### 步骤4：测试

访问 `http://localhost:8080`，输入问题进行测试。

---

## 📁 项目结构

```
EvRAG/
├── src/evrag/                    # 核心源代码
│   ├── config.py                 # 配置管理
│   ├── retriever/                # 检索器模块
│   │   ├── base.py               # 检索器基类
│   │   ├── bm25_retriever.py     # BM25检索器
│   │   └── milvus_retriever.py   # Milvus检索器
│   ├── reranker/                 # 重排序模块
│   │   ├── base.py               # 重排序器基类
│   │   └── bge_reranker.py       # BGE重排序器
│   ├── parser/                   # 文档解析模块
│   │   ├── pdf_parser.py         # PDF解析器
│   │   └── image_handler.py      # 图片处理
│   ├── client/                   # 客户端模块
│   │   ├── local_client.py       # 本地LLM客户端
│   │   ├── chat_client.py        # RAG问答客户端
│   │   └── ...                   # 其他客户端
│   ├── server/                   # 服务模块
│   │   ├── rag_server.py         # RAG服务（FastAPI）
│   │   └── gradio_app.py         # Gradio前端
│   ├── evaluation/               # 评估模块
│   ├── finetune/                # 微调模块
│   └── utils/                   # 工具函数
├── scripts/                      # 脚本目录
│   ├── deployment/              # 部署脚本
│   │   ├── start_gradio.sh
│   │   ├── start_vllm_finetuned.sh
│   │   └── ...
│   ├── evaluation/              # 评估脚本
│   │   ├── run_finetuned_evaluation.sh
│   │   └── ...
│   ├── training/                # 训练脚本
│   │   ├── train_llm.sh
│   │   └── train_reranker.sh
│   ├── utils/                   # 工具脚本
│   │   ├── check_code_quality.sh
│   │   └── monitor_gpu.sh
│   └── setup/                    # 设置脚本
│       └── download_models.sh
├── config/                       # 配置文件
│   ├── config.example.yaml       # 配置示例
│   └── config.yaml               # 实际配置（需创建）
├── data/                         # 数据目录
│   ├── saved_images/            # 保存的图片
│   └── processed_docs/          # 处理后的文档
├── tests/                        # 测试
│   ├── unit/                    # 单元测试
│   └── integration/             # 集成测试
├── main.py                       # 主入口（CLI）
├── environment.yml               # Conda环境配置
├── requirements.txt              # 生产依赖
├── requirements-dev.txt          # 开发依赖
└── README.md                     # 项目说明
```

**详细脚本说明**：请参考 [scripts/README.md](scripts/README.md)

---

## 🔧 核心模块

### 检索器 (Retriever)

#### BM25Retriever
- **功能**：基于BM25算法的稀疏检索器
- **特点**：关键词匹配，检索速度快
- **使用场景**：精确关键词匹配的查询

#### MilvusRetriever
- **功能**：基于Milvus的混合检索器
- **特点**：支持稠密向量和稀疏向量，语义相似度检索
- **使用场景**：语义相似查询

### 重排序器 (Reranker)

#### BGEReranker
- **功能**：基于BGE-Reranker-v2-m3模型的重排序器
- **特点**：支持微调，提升领域适应性
- **使用场景**：对检索结果进行精排

### 解析器 (Parser)

#### PDFParser
- **功能**：PDF文档解析
- **特点**：提取文本和图片，支持OCR

#### ImageHandler
- **功能**：图片处理和关联
- **特点**：图片路径管理，图片检索

### 客户端 (Client)

#### LocalLLMClient
- **功能**：本地LLM客户端（vLLM服务）
- **特点**：支持流式输出，高性能推理

#### ChatClient
- **功能**：RAG问答客户端
- **特点**：整合检索、重排序、生成流程

### 服务 (Server)

#### RAGServer (FastAPI)
- **功能**：RAG服务API
- **接口**：
  - `POST /chat` - 非流式聊天
  - `POST /chat/stream` - 流式聊天
  - `GET /health` - 健康检查

#### GradioApp
- **功能**：Web前端界面
- **特点**：支持流式输出、多轮对话、图片展示

---

## 💻 开发指南

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行单元测试
pytest tests/unit/ -v

# 运行集成测试
pytest tests/integration/ -v

# 生成覆盖率报告
pytest tests/ --cov=src/evrag --cov-report=html
```

### 代码质量检查

```bash
# 运行代码质量检查
./scripts/utils/check_code_quality.sh

# 格式化代码
ruff format src/

# 检查代码风格
ruff check src/
```

### 代码规范

- **Python风格**：遵循PEP 8
- **类型提示**：使用类型注解
- **文档字符串**：所有公共函数和类都需要文档字符串
- **测试覆盖率**：要求 > 80%，关键模块 > 90%

### 添加新功能

1. 创建功能分支：`git flow feature start feature/NewFeature`
2. 实现功能并添加测试
3. 运行测试确保通过
4. 提交更改：`git commit -m 'feat: Add NewFeature'`
5. 合并到develop分支

---

## ⚙️ 配置说明

### 配置优先级

配置按以下优先级加载（高到低）：

1. **环境变量**
2. **命令行参数**
3. **YAML配置文件** (`config/config.yaml`)
4. **默认值**

### 配置文件

配置文件示例：`config/config.example.yaml`

主要配置项：

- **数据路径**：PDF路径、索引路径、图片保存目录
- **模型路径**：检索模型、重排序模型、LLM模型
- **服务配置**：MongoDB配置、LLM服务地址
- **检索参数**：BM25 TopK、Milvus TopK、Reranker TopK

### 环境变量

可以通过环境变量覆盖配置：

```bash
export LLM_BASE_URL="http://localhost:8001/v1"
export LLM_MODEL_NAME="qwen3_lora_sft"
export MONGODB_HOST="localhost"
export MONGODB_PORT=27017
```

---

## ❓ 常见问题

### Q1: 如何修改LLM服务地址？

**A**: 修改 `config/config.yaml` 中的 `llm_base_url`，或设置环境变量 `LLM_BASE_URL`。

### Q2: 如何调整检索参数？

**A**: 
- CLI方式：使用 `--topk` 参数
- Web界面：在参数配置面板中调整
- 代码方式：修改 `config/config.yaml` 中的默认值

### Q3: 如何添加新的检索器？

**A**: 
1. 继承 `BaseRetriever` 类
2. 实现 `retrieve_topk` 方法
3. 在 `main.py` 中注册使用

### Q4: 如何微调模型？

**A**: 参考 `scripts/training/` 目录下的训练脚本，详细说明请参考 `dev_docs/` 目录下的相关文档。

### Q5: 性能优化建议？

**A**:
- 使用GPU加速LLM推理
- 调整检索TopK参数，平衡准确率和速度
- 使用微调模型提升领域适应性
- 启用流式输出提升用户体验

---

## 📚 相关文档

- **脚本说明**：[scripts/README.md](scripts/README.md)
- **环境设置**：`dev_docs/ENVIRONMENT.md`
- **项目结构**：`dev_docs/PROJECT_STRUCTURE.md`
- **Gradio实现**：`dev_docs/gradio_implementation_summary.md`

---

## 📄 许可证

[添加许可证信息]

---

## 👥 贡献

欢迎贡献！请遵循以下步骤：

1. Fork本项目
2. 创建功能分支 (`git flow feature start feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'feat: Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

---

## 📮 联系方式

- **项目地址**: https://github.com/LeonCheung033/EvRAG
- **问题反馈**: [GitHub Issues](https://github.com/LeonCheung033/EvRAG/issues)

---

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者！

特别感谢以下开源项目：
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) - BM25检索
- [Milvus](https://milvus.io/) - 向量数据库
- [BGE](https://github.com/FlagOpen/FlagEmbedding) - 嵌入和重排序模型
- [Gradio](https://gradio.app/) - Web界面框架
- [FastAPI](https://fastapi.tiangolo.com/) - Web框架
