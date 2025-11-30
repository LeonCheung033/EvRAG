# EvRAG系统实现细节文档清单

本文档列出了EvRAG项目中关于系统实现细节的相关文档位置，用于编写后续章节。

---

## 一、Document Processing Pipeline 的真实实现信息

### 1.1 文档切分方法

**主要文档**：
- **`docs/PHASE2_DATA_PREPARATION.md`** (第97-180行)
  - 详细说明语义切分和句子级切分的实现
  - 语义切分：使用M3E-small模型进行语义聚类（`group_size=10`）
  - 句子级切分：使用`RecursiveCharacterTextSplitter`（`chunk_size=256 tokens`, `chunk_overlap=50 tokens`）
  - 分隔符优先级：`["\n\n", "\n"]`
  - 使用tiktoken编码计算长度

**代码位置**：
- `src/evrag/parser/document_splitter.py` - 文档切分实现
- `src/evrag/parser/pdf_parser.py` - PDF解析实现

**关键信息**：
- **切分策略**：两阶段切分（语义切分 → 句子级切分）
- **语义切分**：基于M3E-small模型的语义聚类，`group_size=10`（每组目标最大句子数）
- **句子级切分**：基于长度（256 tokens），使用`RecursiveCharacterTextSplitter`
- **重叠策略**：`chunk_overlap=50 tokens`
- **父子关系**：父文档（语义块）和子文档（句子级块）通过`parent_id`关联

### 1.2 正则化处理

**主要文档**：
- **`docs/PHASE2_DATA_PREPARATION.md`** (第58-96行)
  - 文档清洗使用LLM进行清理和整理
  - Prompt要求：让句子更通顺，按标题归类整理
  - 使用豆包API（temperature=0.001, top_p=0）

**代码位置**：
- `src/evrag/client/clean_client.py` - 文档清洗客户端

**关键信息**：
- **标点修复**：通过LLM清理，去除不必要的符号（如换行符）
- **空格处理**：LLM自动处理
- **繁简转换**：未明确提及（可能需要查看代码）
- **标题整理**：使用markdown格式（###）按标题归类合并

### 1.3 Metadata提取

**主要文档**：
- **`docs/PHASE2_DATA_PREPARATION.md`** (第147-160行)
  - MongoDB文档结构说明
  - 包含`source`（PDF路径）、`page`（页码）、`images_info`（图片信息）、`parent_id`（父文档ID）

**代码位置**：
- `src/evrag/parser/pdf_parser.py` - PDF解析时提取metadata
- `src/evrag/parser/document_splitter.py` - 切分时保留和添加metadata

**关键信息**：
- **提取的metadata**：
  - `unique_id`: MD5哈希值（基于内容）
  - `source`: PDF文件路径
  - `page`: 页码
  - `images_info`: 图片信息列表
  - `parent_id`: 父文档ID（仅子文档有此字段）

---

## 二、Multi-Stage Retrieval 的具体实现

### 2.1 BM25实现

**主要文档**：
- **`docs/PHASE2_DATA_PREPARATION.md`** (第184-198行)
  - BM25索引构建说明
  - 使用LangChain的`BM25Retriever`
  - 使用jieba进行中文分词
  - 过滤停用词

**代码位置**：
- `src/evrag/retriever/bm25_retriever.py` - BM25检索器实现

**关键信息**：
- **使用的库**：`langchain_community.retrievers.BM25Retriever`
- **分词工具**：`jieba`（中文分词）
- **停用词过滤**：从配置文件读取停用词列表
- **索引保存**：pickle文件（`data/saved_index/bm25retriever.pkl`）

### 2.2 Dense模型实现

**主要文档**：
- **`docs/PHASE2_DATA_PREPARATION.md`** (第199-230行)
  - Milvus索引构建说明
  - 使用BGE-M3模型进行embedding
  - 生成dense vector和sparse vector

**代码位置**：
- `src/evrag/retriever/milvus_retriever.py` - Milvus检索器实现

**关键信息**：
- **Dense模型**：BGE-M3（BAAI/bge-m3）
- **Embedding函数**：`pymilvus.model.hybrid.BGEM3EmbeddingFunction`
- **向量类型**：同时生成dense vector和sparse vector（混合检索）
- **批量处理**：`EMB_BATCH=50`
- **文本长度限制**：超过512字符自动截断

### 2.3 Reranker模型实现

**主要文档**：
- **`docs/PROJECT_STRUCTURE.md`** (第80-85行)
  - BGE重排序模型说明
- **`docs/RERANKER_TRAINING_SUMMARY.md`** (全文)
  - Reranker训练详细说明

**代码位置**：
- `src/evrag/reranker/bge_reranker.py` - Reranker实现

**关键信息**：
- **Reranker模型**：BGE-Reranker-v2-m3（BAAI）
- **模型类型**：Cross-encoder（BERT-based encoder）
- **基线模型**：`models/bge-reranker-v2-m3/`
- **微调模型**：`models/finetuned/bge_reranker/`

### 2.4 Top-K设置

**主要文档**：
- **`docs/rag_test_plan.md`** (第236-246行)
  - 检索参数配置说明
- **`docs/PROJECT_STRUCTURE.md`** (第174-180行)
  - 检索流程说明

**关键信息**：
- **BM25 topk**：默认5（可配置）
- **Milvus topk**：默认10（可配置）
- **Reranker topk**：默认5（可配置）
- **混合检索**：BM25和Milvus结果合并后，使用Reranker重排序

---

## 三、Fine-Tuning 细节

### 3.1 Reranker微调

#### 3.1.1 数据规模

**主要文档**：
- **`docs/RERANKER_TRAINING_SUMMARY.md`** (第85-142行)
  - 数据合并和统计说明

**关键信息**：
- **训练集**：~30,600条（合并新旧项目数据后）
- **验证集**：~1,700条（5%）
- **测试集**：~1,700条（5%）
- **数据格式**：`{"query": str, "content": str, "label": int}`（pointwise格式）
- **Label分布**：
  - Label 0（负样本）：~35%
  - Label 1（中等样本）：~32.5%
  - Label 2（正样本）：~32.5%

#### 3.1.2 Hard Negatives来源

**主要文档**：
- **`docs/PHASE3_SFT_DATA_GENERATION.md`** (第100-120行)
  - Reranker数据生成说明
- **`docs/sft_data_generation_plan.md`** (第130-150行)
  - 负样本生成策略

**关键信息**：
- **正样本（label=2）**：来自Reranker排序后的第一个文档（`context[0]`）
- **中等样本（label=1）**：从`context[-2:]`随机选择（倒数两个文档中随机选）
- **负样本（label=0）**：从`neg_docs`随机选择（不在context中的文档）
- **Hard negatives**：中等样本（label=1）可以视为hard negatives，因为它们来自检索结果但相关性较低

#### 3.1.3 Loss Function

**主要文档**：
- **`docs/RERANKER_TRAINING_SUMMARY.md`** (第195-207行)
  - 最终训练配置
- **`config/finetune/reranker_training.yaml`** (第7行)

**关键信息**：
- **Loss类型**：`pointwise_bce`（Binary Cross-Entropy）
- **数据格式**：pointwise（每个样本独立标注）
- **Label范围**：0-2（自动缩放到0-1）

#### 3.1.4 超参数

**主要文档**：
- **`docs/RERANKER_TRAINING_SUMMARY.md`** (第195-207行)
  - 最终训练配置
- **`config/finetune/reranker_training.yaml`** (完整文件)

**关键信息**：
- **Epochs**：2
- **Batch size**：8
- **Learning rate**：2e-5
- **Gradient accumulation steps**：2
- **Warmup proportion**：0.1
- **Mixed precision**：fp16
- **Max length**：4096
- **Loss type**：pointwise_bce

### 3.2 LLM微调

#### 3.2.1 采用的LLM

**主要文档**：
- **`docs/FINETUNE_TRAINING_SUMMARY.md`** (第8-30行)
  - LLM训练概述
- **`docs/sft_plan.md`** (第90-117行)
  - LLM微调配置

**关键信息**：
- **模型**：Qwen3-8B
- **微调方法**：LoRA (Low-Rank Adaptation)
- **训练类型**：SFT (Supervised Fine-Tuning)
- **基础模型路径**：`models/Qwen3-8B/`

#### 3.2.2 SFT数据格式

**主要文档**：
- **`docs/FINETUNE_TRAINING_SUMMARY.md`** (第140-183行)
  - 训练配置说明
- **`docs/PHASE3_SFT_DATA_GENERATION.md`** (第95-100行)
  - SFT数据生成说明
- **`docs/sft_plan.md`** (第72-89行)
  - 数据格式转换说明

**关键信息**：
- **数据格式**：`{"instruction": str, "input": str, "output": str}`
- **Template**：`qwen3_nothink`（Qwen3模板，无思考过程）
- **数据来源**：从`train_data.json`解析LLM响应，提取答案和引用标记
- **训练集**：6,406个样本
- **测试集**：516个样本

#### 3.2.3 超参数

**主要文档**：
- **`docs/FINETUNE_TRAINING_SUMMARY.md`** (第140-183行)
  - 最终优化配置
- **`config/finetune/qwen3_lora_sft.yaml`** (完整文件)

**关键信息**：
- **LoRA rank**：8
- **LoRA target**：all
- **Cutoff length**：2048 tokens
- **Batch size**：4 (per device)
- **Gradient accumulation steps**：4
- **Effective batch size**：6卡 × 4 × 4 = 96
- **Learning rate**：2.0e-5
- **Epochs**：3
- **LR scheduler**：cosine
- **Warmup ratio**：0.1
- **Mixed precision**：bf16
- **Flash attention**：sdpa (PyTorch SDPA)

---

## 四、相关文档完整列表

### 4.1 Document Processing Pipeline

1. **`docs/PHASE2_DATA_PREPARATION.md`** ⭐⭐⭐⭐⭐
   - 完整的文档处理流程说明
   - PDF解析、清洗、切分、入库
   - 配置参数和代码示例

2. **`docs/PHASE2_DATA_FIXES_SUMMARY.md`**
   - 文档处理过程中的问题修复
   - Milvus文本长度限制修复

3. **`src/evrag/parser/pdf_parser.py`**
   - PDF解析实现代码

4. **`src/evrag/parser/document_splitter.py`**
   - 文档切分实现代码

5. **`src/evrag/client/clean_client.py`**
   - 文档清洗实现代码

### 4.2 Multi-Stage Retrieval

1. **`docs/PHASE2_DATA_PREPARATION.md`** (第184-230行) ⭐⭐⭐⭐⭐
   - BM25和Milvus索引构建说明

2. **`docs/PROJECT_STRUCTURE.md`** (第67-85行) ⭐⭐⭐⭐⭐
   - 检索器架构说明

3. **`docs/RERANKER_TRAINING_SUMMARY.md`** ⭐⭐⭐⭐⭐
   - Reranker模型详细说明

4. **`src/evrag/retriever/bm25_retriever.py`**
   - BM25检索器实现代码

5. **`src/evrag/retriever/milvus_retriever.py`**
   - Milvus检索器实现代码

6. **`src/evrag/reranker/bge_reranker.py`**
   - Reranker实现代码

### 4.3 Fine-Tuning

#### Reranker微调

1. **`docs/RERANKER_TRAINING_SUMMARY.md`** ⭐⭐⭐⭐⭐
   - Reranker训练完整总结
   - 数据规模、超参数、训练过程

2. **`docs/FINETUNE_TRAINING_SUMMARY.md`** (第764-894行)
   - Reranker训练部分

3. **`config/finetune/reranker_training.yaml`** ⭐⭐⭐⭐⭐
   - Reranker训练配置文件

4. **`docs/PHASE3_SFT_DATA_GENERATION.md`** (第100-120行)
   - Reranker数据生成说明

5. **`docs/sft_data_generation_plan.md`** (第130-150行)
   - 负样本生成策略

#### LLM微调

1. **`docs/FINETUNE_TRAINING_SUMMARY.md`** ⭐⭐⭐⭐⭐
   - LLM训练完整总结
   - 训练配置、过程、结果

2. **`docs/sft_plan.md`** ⭐⭐⭐⭐⭐
   - LLM微调实现计划
   - 数据格式、配置参数

3. **`config/finetune/qwen3_lora_sft.yaml`** ⭐⭐⭐⭐⭐
   - LLM训练配置文件

4. **`docs/PHASE3_SFT_DATA_GENERATION.md`** (第95-100行)
   - SFT数据生成说明

---

## 五、快速查找指南

### 5.1 按问题查找

| 问题 | 主要文档 | 关键章节 |
|------|---------|---------|
| **文档切分方法** | `PHASE2_DATA_PREPARATION.md` | 第97-180行 |
| **正则化处理** | `PHASE2_DATA_PREPARATION.md` | 第58-96行 |
| **Metadata提取** | `PHASE2_DATA_PREPARATION.md` | 第147-160行 |
| **BM25实现** | `PHASE2_DATA_PREPARATION.md` | 第184-198行 |
| **Dense模型** | `PHASE2_DATA_PREPARATION.md` | 第199-230行 |
| **Reranker模型** | `RERANKER_TRAINING_SUMMARY.md` | 全文 |
| **Reranker数据规模** | `RERANKER_TRAINING_SUMMARY.md` | 第85-142行 |
| **Hard negatives** | `PHASE3_SFT_DATA_GENERATION.md` | 第100-120行 |
| **Reranker超参数** | `RERANKER_TRAINING_SUMMARY.md` | 第195-207行 |
| **LLM模型** | `FINETUNE_TRAINING_SUMMARY.md` | 第8-30行 |
| **SFT数据格式** | `FINETUNE_TRAINING_SUMMARY.md` | 第140-183行 |
| **LLM超参数** | `FINETUNE_TRAINING_SUMMARY.md` | 第140-183行 |

### 5.2 配置文件位置

| 配置类型 | 文件路径 |
|---------|---------|
| **Reranker训练** | `config/finetune/reranker_training.yaml` |
| **LLM训练** | `config/finetune/qwen3_lora_sft.yaml` |
| **数据集配置** | `config/datasets/faq_summary.yaml` |

### 5.3 代码实现位置

| 功能 | 代码文件 |
|------|---------|
| **PDF解析** | `src/evrag/parser/pdf_parser.py` |
| **文档切分** | `src/evrag/parser/document_splitter.py` |
| **文档清洗** | `src/evrag/client/clean_client.py` |
| **BM25检索** | `src/evrag/retriever/bm25_retriever.py` |
| **Milvus检索** | `src/evrag/retriever/milvus_retriever.py` |
| **Reranker** | `src/evrag/reranker/bge_reranker.py` |
| **LLM微调** | `src/evrag/finetune/llm_finetuner.py` |
| **Reranker微调** | `src/evrag/finetune/reranker_finetuner.py` |

---

## 六、关键数据统计

### 6.1 文档处理统计

- **原始文档**：PDF页面数（从PDF解析）
- **清洗后文档**：与原始文档数量相同
- **切分后文档**：约为清洗后的2-5倍（包含父文档和子文档）
- **分块参数**：
  - 语义分组：`group_size=10`
  - 句子级切分：`chunk_size=256 tokens`, `chunk_overlap=50 tokens`

### 6.2 检索统计

- **BM25 topk**：5（默认）
- **Milvus topk**：10（默认）
- **Reranker topk**：5（默认）
- **混合检索**：BM25和Milvus结果合并后重排序

### 6.3 训练数据统计

**Reranker**：
- 训练集：~30,600条
- 验证集：~1,700条
- 测试集：~1,700条
- Label分布：0(35%), 1(32.5%), 2(32.5%)

**LLM**：
- 训练集：6,406条
- 测试集：516条
- 数据格式：instruction-input-output

---

**文档版本**：v1.0  
**最后更新**：2024-11-30  
**维护者**：EvRAG团队

