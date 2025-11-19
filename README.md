# EvRAG

Enhanced Vector Retrieval-Augmented Generation

一个增强的检索增强生成系统，支持多种检索器和重排序方法。

## 功能特性

- 多种检索器支持（BM25、Faiss、Milvus等）
- 重排序功能（BGE-M3等）
- PDF文档解析
- LLM客户端集成
- QA生成功能

## 快速开始

### 环境要求

- Python 3.12+
- Conda

### 安装步骤

1. 克隆仓库
git clone <repository-url>
cd EvRAG2. 创建conda环境
conda env create -f environment.yml
conda activate evrag3. 安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt## 项目结构
