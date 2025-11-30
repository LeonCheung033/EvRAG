# EvRAG 项目结构详解

本文档详细说明EvRAG项目的目录结构和各文件的作用，重点关注真正发挥作用的文件。

## 📁 项目根目录结构

```
EvRAG/
├── main.py                    # ⭐ 主入口文件，所有CLI命令的入口
├── config/                    # 配置文件目录
├── src/                       # ⭐ 核心源代码目录
├── data/                      # 数据目录
├── scripts/                   # 脚本目录
├── docs/                      # 文档目录
├── models/                    # 模型文件目录
├── tests/                     # 测试目录
├── logs/                      # 日志目录
├── rag_test_reports/          # RAG评估报告
├── reports/                   # 其他报告
└── requirements.txt           # 依赖文件
```

---

## ⭐ 核心文件详解

### 1. 主入口文件

#### `main.py` - **最重要的文件**
**作用**: 项目的CLI入口，提供所有命令行功能

**主要命令**:
- `prepare-data`: PDF解析、文档清洗、文档切分
- `build-index`: 构建BM25和Milvus检索索引
- `infer`: RAG问答推理（核心功能）
- `gen-qa`: 生成QA对
- `evaluate-rag`: RAG系统评估
- `evaluate-qwen-baseline`: Qwen基线模型评估
- `finetune-llm`: LLM微调
- `finetune-reranker`: Reranker微调
- `evaluate-model`: 模型评估，评估微调后的问答llm的效果
- `compare-models`: 模型对比，将微调后的问答llm与基线问答llm对比

**为什么重要**: 这是用户与系统交互的唯一入口，所有功能都通过这里调用

---

## 📂 核心源代码目录 (`src/evrag/`)

### 2. 配置管理

#### `config.py`
**作用**: 统一管理所有配置项（模型路径、API地址、超参数等）
**重要性**: ⭐⭐⭐⭐⭐ 所有模块都依赖这个文件

---

### 3. 检索器模块 (`retriever/`)

#### `bm25_retriever.py` ⭐⭐⭐⭐⭐
**作用**: BM25稀疏检索器，基于关键词匹配
**核心功能**: 
- 构建BM25索引
- 检索top-k文档
- 支持索引持久化

#### `milvus_retriever.py` ⭐⭐⭐⭐⭐
**作用**: Milvus向量检索器，基于语义相似度
**核心功能**:
- 向量化查询和文档
- 使用Milvus进行向量检索
- 支持混合检索（稠密+稀疏）

**为什么重要**: 这两个检索器是RAG系统的核心，负责从大量文档中检索相关信息

---

### 4. 重排序器模块 (`reranker/`)

#### `bge_reranker.py` ⭐⭐⭐⭐⭐
**作用**: BGE重排序模型，对检索结果进行精排
**核心功能**:
- 加载BGE reranker模型（baseline或finetuned）
- 对检索到的文档进行相关性排序
- 返回top-k最相关的文档

**为什么重要**: 重排序直接影响最终答案的质量，是性能提升的关键

---

### 5. 客户端模块 (`client/`)

#### `local_client.py` ⭐⭐⭐⭐⭐
**作用**: 本地LLM客户端（vLLM服务）
**核心功能**: 调用本地部署的LLM进行文本生成

#### `chat_client.py` ⭐⭐⭐⭐⭐
**作用**: RAG问答客户端，整合检索+重排序+LLM生成
**核心功能**:
- 接收用户问题
- 调用检索器获取相关文档
- 调用重排序器精排
- 构建prompt并调用LLM生成答案
- 后处理答案（提取引用等）

**为什么重要**: 这是RAG流程的核心编排者，连接所有组件

#### `clean_client.py` ⭐⭐⭐
**作用**: 文档清洗客户端
**功能**: 使用LLM清洗和格式化文档

#### `hyde_client.py` ⭐⭐
**作用**: HYDE（假设文档生成）客户端
**功能**: 生成假设文档用于检索增强

#### `mongodb_client.py` ⭐⭐⭐
**作用**: MongoDB数据库客户端
**功能**: 存储和查询文档数据

---

### 6. 评估模块 (`evaluation/`)

#### `rag_evaluator.py` ⭐⭐⭐⭐⭐
**作用**: RAG系统端到端评估器
**核心功能**:
- 执行完整的RAG流程评估
- 计算语义相似度、关键词匹配、BLEU、ROUGE等指标
- 支持RAGas框架评估（ContextRecall、ContextPrecision）
- 生成评估报告

**为什么重要**: 用于评估系统性能，是优化和调试的关键工具

#### `ragas_evaluator.py` ⭐⭐⭐⭐
**作用**: RAGas框架评估器
**功能**: 使用RAGas框架评估检索上下文质量

#### `metrics.py` ⭐⭐⭐
**作用**: 评估指标计算
**功能**: 实现各种评估指标的计算逻辑

---

### 7. 微调模块 (`finetune/`)

#### `llm_finetuner.py` ⭐⭐⭐⭐
**作用**: LLM微调器
**功能**: 使用LLaMA-Factory进行LLM微调

#### `reranker_finetuner.py` ⭐⭐⭐⭐
**作用**: Reranker微调器
**功能**: 使用RAG-Retrieval框架进行reranker微调

#### `model_evaluator.py` ⭐⭐⭐⭐
**作用**: 模型评估器
**功能**: 评估微调后的模型性能（NDCG、MRR等）

#### `performance_comparison.py` ⭐⭐⭐
**作用**: 性能对比工具
**功能**: 对比baseline和finetuned模型的性能

#### 训练数据准备流程 ⭐⭐⭐⭐⭐

**核心文件**: `src/evrag/gen_qa/sft_data_generator.py`

**完整流程**:

```
1. 输入数据: train_qa_pair.json (QA对数据)
   ↓
2. 生成 train_data.json (RAG检索数据)
   - 对每个问题执行RAG检索:
     * BM25检索 (topk=5)
     * Milvus检索 (topk=10)
     * 合并文档 (merge_docs)
     * Reranker重排序 (topk=5)
   - 使用LLM生成响应（带引用标记格式：答案【引用编号1,引用编号2】）
   - 格式: JSONL，每行 {"query": "...", "context": [...], "response": "...", "merged_docs": [...]}
   ↓
3. 生成 summary_data (SFT训练数据)
   - 从 train_data.json 解析LLM响应
   - 提取答案和引用标记
   - 格式化instruction和output
   - 切分训练集和测试集 (test_rate=0.08)
   - 输出: summary_data/train.json, summary_data/test.json
   - 格式: JSON数组，{"query": "...", "context": "...", "instruction": "...", "input": "", "output": "..."}
   ↓
4. 生成 rerank_data (Reranker训练数据)
   - 从 train_data.json 生成正/中/负样本
   - Label生成策略:
     * Label 2 (正样本): context[0] - 最相关的文档
     * Label 1 (中等样本): context[-2:] 随机选择
     * Label 0 (负样本): neg_docs 或 merged_docs 随机选择
   - 切分训练集/开发集/测试集
   - 输出: rerank_data/train.json, dev.json, test.json
   - 格式: JSONL，每行 {"query": "...", "content": "...", "label": 0/1/2}
```

**数据统计**:
- **train_data.json**: RAG检索数据，包含完整的检索上下文和LLM响应
- **summary_data/train.json**: SFT训练数据，用于LLM微调
- **rerank_data/train.json**: Reranker训练数据（约30000条）
  - Label 0 (负样本): ~35%
  - Label 1 (中等样本): ~32.5%
  - Label 2 (正样本): ~32.5%

**使用方法**:
```bash
# 生成所有微调数据
python main.py generate-sft-data --train-qa-path data/qa_pairs/train_qa_pair.json

# 分步骤生成
python main.py generate-sft-data --step train_data    # 只生成train_data.json
python main.py generate-sft-data --step summary       # 只生成summary_data
python main.py generate-sft-data --step rerank       # 只生成rerank_data
```

**为什么重要**: 训练数据质量直接影响微调效果，这个流程确保数据格式正确且分布合理

---

### 8. QA生成模块 (`gen_qa/`)

#### `sft_data_generator.py` ⭐⭐⭐⭐
**作用**: SFT数据生成器
**功能**: 从文档生成用于微调的QA对

#### `generator.py` ⭐⭐⭐
**作用**: QA对生成器
**功能**: 使用LLM生成问答对

---

### 9. 解析器模块 (`parser/`)

#### `pdf_parser.py` ⭐⭐⭐
**作用**: PDF解析器
**功能**: 解析PDF文件，提取文本和图片

#### `document_splitter.py` ⭐⭐⭐
**作用**: 文档切分器
**功能**: 将长文档切分成适合检索的chunks

---

### 10. 工具函数 (`utils/`)

#### `tool_func.py` ⭐⭐⭐⭐
**作用**: 核心工具函数
**核心函数**:
- `merge_docs()`: 合并BM25和Milvus检索结果
- `post_processing()`: 答案后处理（提取引用等）

**为什么重要**: 这些函数在RAG流程中被频繁调用

---

## 📁 配置文件 (`config/`)

### `config.yaml` ⭐⭐⭐⭐⭐
**作用**: 主配置文件
**内容**:
- 模型路径（baseline和finetuned）
- API地址和端口
- 检索参数（topk等）
- 数据路径

**为什么重要**: 所有模块都从这里读取配置

### `finetune/qwen3_lora_sft.yaml` ⭐⭐⭐⭐
**作用**: LLM微调配置
**内容**: 学习率、batch size、训练数据路径等

### `finetune/reranker_training.yaml` ⭐⭐⭐⭐
**作用**: Reranker微调配置
**内容**: 训练参数、数据路径等

---

## 📁 脚本目录 (`scripts/`)

### 重要脚本

#### `run_baseline_finetuned_comparison.sh` ⭐⭐⭐⭐⭐
**作用**: 运行baseline和finetuned模型对比测试
**功能**: 自动化执行评估和对比

#### `start_vllm.sh` / `start_vllm_for_evaluation.sh` ⭐⭐⭐⭐
**作用**: 启动vLLM服务
**功能**: 启动本地LLM服务供推理使用

#### `train_llm.sh` / `train_reranker.sh` ⭐⭐⭐⭐
**作用**: 训练脚本
**功能**: 自动化执行模型微调

---

## 📁 数据目录 (`data/`)

### 重要子目录

#### `qa_pairs/` ⭐⭐⭐⭐⭐
**作用**: 存储QA对数据
**重要文件**:
- `test_qa_pair.json`: 测试数据（718条）
- `test_qa_pair_handmade.json`: 手工标注测试数据（676条）
- `train_data.json`: 训练数据

#### `rerank_data/` ⭐⭐⭐⭐
**作用**: Reranker训练数据
**文件**:
- `train.json`: 训练数据（17,131条）
- `dev.json`: 验证数据（1,000条）

#### `saved_index/` ⭐⭐⭐⭐
**作用**: 保存的检索索引
**文件**:
- `bm25retriever.pkl`: BM25索引
- Milvus索引存储在Milvus数据库中

#### `processed_docs/` ⭐⭐⭐
**作用**: 处理后的文档
**文件**: 清洗和切分后的文档

---

## 📁 模型目录 (`models/`)

### 重要子目录

#### `finetuned/bge_reranker/` ⭐⭐⭐⭐⭐
**作用**: 微调后的reranker模型
**内容**: 微调后的模型权重和配置

#### `finetuned/qwen3_lora_sft/` ⭐⭐⭐⭐⭐
**作用**: 微调后的LLM模型（LoRA权重）
**内容**: LoRA适配器权重

#### `bge-reranker-v2-m3/` ⭐⭐⭐⭐
**作用**: Baseline reranker模型
**内容**: 预训练的BGE reranker模型

---

## 📁 评估报告目录

### `rag_test_reports/` ⭐⭐⭐⭐⭐
**作用**: RAG评估报告
**内容**:
- `comparison_*/`: 对比评估结果
- `baseline/`: Baseline模型评估结果
- `finetuned/`: Finetuned模型评估结果

### `reports/evaluation/` ⭐⭐⭐⭐
**作用**: 模型评估报告
**内容**: 详细的评估指标和分析

---

## 🔍 核心工作流程

### 1. 数据准备流程
```
PDF文件 → pdf_parser.py → 清洗(clean_client.py) → 切分(document_splitter.py) → MongoDB
```

### 2. 索引构建流程
```
文档 → bm25_retriever.py (构建BM25索引) → saved_index/bm25retriever.pkl
文档 → milvus_retriever.py (向量化) → Milvus数据库
```

### 3. RAG推理流程 ⭐⭐⭐⭐⭐
```
用户问题 
  → bm25_retriever.py (BM25检索)
  → milvus_retriever.py (向量检索)
  → merge_docs() (合并结果)
  → bge_reranker.py (重排序)
  → chat_client.py (构建prompt)
  → local_client.py (LLM生成)
  → post_processing() (后处理)
  → 返回答案
```

### 4. 评估流程
```
测试数据 → rag_evaluator.py → 计算指标 → 生成报告
```

### 5. 微调流程
```
训练数据 → llm_finetuner.py / reranker_finetuner.py → 微调模型 → model_evaluator.py → 评估
```

---

## 🎯 真正发挥作用的文件总结

### ⭐⭐⭐⭐⭐ 核心文件（必须理解）

1. **main.py** - CLI入口
2. **src/evrag/config.py** - 配置管理
3. **src/evrag/retriever/bm25_retriever.py** - BM25检索
4. **src/evrag/retriever/milvus_retriever.py** - 向量检索
5. **src/evrag/reranker/bge_reranker.py** - 重排序
6. **src/evrag/client/chat_client.py** - RAG流程编排
7. **src/evrag/client/local_client.py** - LLM调用
8. **src/evrag/evaluation/rag_evaluator.py** - 评估器
9. **src/evrag/utils/tool_func.py** - 工具函数
10. **config/config.yaml** - 配置文件

### ⭐⭐⭐⭐ 重要文件（需要了解）

1. **src/evrag/finetune/llm_finetuner.py** - LLM微调
2. **src/evrag/finetune/reranker_finetuner.py** - Reranker微调
3. **src/evrag/finetune/model_evaluator.py** - 模型评估
4. **src/evrag/gen_qa/sft_data_generator.py** - 数据生成
5. **scripts/run_baseline_finetuned_comparison.sh** - 对比脚本

### ⭐⭐⭐ 辅助文件（可选了解）

1. **src/evrag/parser/** - 文档解析
2. **src/evrag/client/clean_client.py** - 文档清洗
3. **src/evrag/evaluation/ragas_evaluator.py** - RAGas评估

---

## 📊 文件依赖关系

```
main.py
  ├── config.py (配置)
  ├── retriever/ (检索)
  │   ├── bm25_retriever.py
  │   └── milvus_retriever.py
  ├── reranker/ (重排序)
  │   └── bge_reranker.py
  ├── client/ (客户端)
  │   ├── chat_client.py (核心)
  │   └── local_client.py
  ├── evaluation/ (评估)
  │   └── rag_evaluator.py
  └── utils/ (工具)
      └── tool_func.py
```



---

## 📝 总结

**核心原则**: 
- `main.py` 是入口
- `chat_client.py` 是RAG流程的核心编排者
- `retriever/` 和 `reranker/` 是检索和排序的核心
- `config.yaml` 是配置中心
- `evaluation/` 是性能评估的关键

**建议学习路径**:
1. 先理解 `main.py` 的命令结构
2. 理解 `chat_client.py` 的RAG流程
3. 深入 `retriever/` 和 `reranker/` 的实现
4. 学习 `evaluation/` 的评估方法
5. 了解 `finetune/` 的微调流程

