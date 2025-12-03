# 数据处理指南

本文档总结了EvRAG项目的数据处理流程，包括PDF解析、文档清洗、文档切分、QA生成和数据处理。

## 📋 目录

- [数据准备流程](#数据准备流程)
- [QA数据处理](#qa数据处理)
- [数据修复和优化](#数据修复和优化)
- [数据格式说明](#数据格式说明)

---

## 📊 数据准备流程

### 1. PDF解析

**模块**: `src/evrag/parser/pdf_parser.py`

**功能**: 从PDF文件中提取文本和图片信息

**处理步骤**:
1. 打开PDF文件（使用PyMuPDF）
2. 遍历每一页PDF：
   - 过滤指定页面范围（跳过封面、目录等）
   - 裁剪页面底部（去除页眉页脚）
   - 提取文本内容
   - 提取图片信息（使用ImageHandler）
3. 为每个页面创建Document对象

**配置参数**（`config.yaml`）:
```yaml
pdf_min_filter_pages: 0      # 最小页码（从0开始）
pdf_max_filter_pages: null   # 最大页码（null表示不限制）
pdf_page_clip: 50            # 页面底部裁剪像素数
```

**输出**:
- 文件: `data/processed_docs/raw_docs.pkl`
- 数据结构: `List[Document]`，每个Document代表一页PDF

### 2. 文档清洗

**模块**: `src/evrag/client/clean_client.py`

**功能**: 使用LLM批量清理和整理文档内容

**处理步骤**:
1. 初始化CleanClient（需要LLM客户端，默认使用豆包API）
2. 对每个原始文档：
   - 构建清理prompt（要求LLM让句子更通顺，按标题归类整理）
   - 调用LLM进行文档清理（temperature=0.001, top_p=0）
   - 保留原始metadata
3. 批量处理（默认20个并发线程）

**输出**:
- 文件: `data/processed_docs/clean_docs.pkl`
- 数据结构: `List[Document]`，每个Document包含清理后的内容

### 3. 文档切分

**模块**: `src/evrag/parser/document_splitter.py`

**功能**: 将清洗后的文档进行语义切分和句子级切分

**处理步骤**:
1. **语义切分（父文档）**：
   - 使用M3E-small模型进行语义切分
   - 生成父文档（包含完整的语义段落）
   - 保存到MongoDB和pickle文件

2. **句子级切分（子文档）**：
   - 使用RecursiveCharacterTextSplitter进行句子级切分
   - 参数：chunk_size=256 tokens, overlap=50 tokens
   - 生成子文档（包含parent_id指向父文档）
   - 保存到MongoDB和pickle文件

**输出**:
- 文件: `data/processed_docs/split_docs.pkl`
- MongoDB集合: `manual_text`
- 数据结构: `List[Document]`，包含父文档和子文档

### 4. 索引构建

**模块**: `src/evrag/retriever/`

**功能**: 构建BM25和Milvus检索索引

**处理步骤**:
1. **BM25索引**：
   - 使用rank-bm25库构建BM25索引
   - 保存到: `data/saved_index/bm25retriever.pkl`

2. **Milvus索引**：
   - 使用BGE-M3模型生成文档向量
   - 使用Milvus存储向量索引
   - 保存到: `data/saved_index/milvus.db`

**输出**:
- BM25索引: `data/saved_index/bm25retriever.pkl`
- Milvus索引: `data/saved_index/milvus.db`

---

## 🔄 QA数据处理

### 流程概览

```
qa_pair.json (原始QA对)
    ↓
[Step 1] 质量打分和过滤
    ↓
filtered_qa_pair.json (过滤后的QA对)
    ↓
[Step 2] 问题改写（生成同义问题）
    ↓
expand_qa_pair.json (问题改写结果)
    ↓
[Step 3] 答案匹配和扩充 + 训练/测试集切分
    ↓
train_qa_pair.json, test_qa_pair.json (切分后的数据集)
    ↓
[Step 4] 测试集关键词提取
    ↓
test_keywords_pair.json (关键词映射)
test_qa_pair.json (更新，添加keywords字段)
    ↓
[Step 5] 添加负样本
    ↓
train_qa_pair.json, test_qa_pair.json (最终数据集)
```

### Step 1: QA质量打分和过滤

**功能**：对QA对进行质量评分，过滤低质量数据

**实现**：
- 使用Deepseek API对每个QA对进行质量评分（1-5分）
- 过滤包含"无法准确"或"未提及"的答案
- 保留评分≥阈值的QA对（默认阈值：3）

**输入**：`data/qa_pairs/qa_pair.json`

**输出**：`data/qa_pairs/filtered_qa_pair.json`

**运行命令**：
```bash
python main.py process-qa \
    --step filter \
    --qa-pair-path data/qa_pairs/qa_pair.json \
    --output-dir data/qa_pairs \
    --quality-threshold 3 \
    --workers 5
```

### Step 2: 问题改写

**功能**：为每个问题生成5个同义问题，扩充数据集

**实现**：
- 从 `filtered_qa_pair.json` 加载过滤后的QA对
- 提取所有问题
- 使用Deepseek API为每个问题生成5个同义问题
- 保存到 `expand_qa_pair.json`（JSONL格式）

**输入**：`data/qa_pairs/filtered_qa_pair.json`

**输出**：`data/qa_pairs/expand_qa_pair.json`

### Step 3: 训练/测试集切分

**功能**：将扩充后的QA对切分为训练集和测试集

**实现**：
- 90% 训练集，10% 测试集
- 确保测试集中的答案唯一性
- 为测试集添加关键词字段

**输出**：
- `data/qa_pairs/train_qa_pair.json`
- `data/qa_pairs/test_qa_pair.json`

### Step 4: 关键词提取

**功能**：从测试集的唯一答案中提取关键词

**实现**：
- 使用LLM提取答案中的关键词
- 保存到 `test_keywords_pair.json`
- 更新 `test_qa_pair.json`，添加keywords字段

**输出**：
- `data/qa_pairs/test_keywords_pair.json`
- `data/qa_pairs/test_qa_pair.json`（更新）

### Step 5: 添加负样本

**功能**：为训练集和测试集添加负样本，用于训练Reranker

**实现**：
- 从检索结果中选择不相关的文档作为负样本
- 添加到训练集和测试集中

**输出**：
- `data/qa_pairs/train_qa_pair.json`（最终版本）
- `data/qa_pairs/test_qa_pair.json`（最终版本）

---

## 🔧 数据修复和优化

### 1. parent_id 问题修复

**问题描述**：
- 父文档被错误地设置了 `parent_id`，且 `parent_id` 等于自己的 `unique_id`
- 子文档的 `parent_id` 指向了父文档的 `parent_id` 而不是父文档的 `unique_id`

**修复方案**：
1. **父文档修复**：父文档只设置 `unique_id`，不设置 `parent_id`
2. **子文档修复**：子文档的 `parent_id` 指向父文档的 `unique_id`

**影响**：
- 修复了父子关系错误，确保子文档正确关联到父文档
- 修复后需要清空 MongoDB 并重新生成数据

### 2. PDF 解析页面过滤修复

**问题描述**：
- PDF 解析没有使用配置的页面过滤参数

**修复方案**：
- 在配置文件中添加 PDF 解析配置字段
- 修改 PDFParser 初始化，传递配置参数

### 3. 文档切分优化

**问题描述**：
- 文档切分时可能出现重复或遗漏

**优化方案**：
- 改进语义切分算法
- 优化句子级切分参数
- 添加去重逻辑

---

## 📝 数据格式说明

### QA对格式

#### 原始QA对 (qa_pair.json)

```json
{
  "unique_id": "文档唯一标识符",
  "raw_resp": "JSON字符串，包含该文档生成的所有QA对"
}
```

#### 过滤后的QA对 (filtered_qa_pair.json)

```json
{
  "question": "问题文本",
  "answer": "答案文本",
  "quality_score": 5,
  "quality_reason": "质量评分的原因"
}
```

#### 最终训练/测试集 (train_qa_pair.json / test_qa_pair.json)

```json
{
  "question": "问题文本",
  "answer": "答案文本",
  "keywords": ["关键词1", "关键词2"],
  "negative_samples": ["负样本1", "负样本2"]
}
```

### 文档格式

#### Document对象

```python
Document(
    page_content="文档内容",
    metadata={
        "unique_id": "文档唯一标识符",
        "page": 页码,
        "parent_id": "父文档ID（子文档才有）",
        "images_info": [图片信息列表]
    }
)
```

---

## 📚 相关文档

- **数据准备详情**: `dev_docs/data/PHASE2_DATA_PREPARATION.md`
- **QA数据处理详情**: `dev_docs/data/PHASE2_QA_DATA_PROCESSING.md`
- **数据修复总结**: `dev_docs/data/PHASE2_DATA_FIXES_SUMMARY.md`
- **QA处理计划**: `dev_docs/data/QA_DATA_PROCESSING_PLAN.md`
- **QA处理示例**: `dev_docs/data/QA_PROCESSING_EXAMPLE.md`
- **主README**: `README.md`

