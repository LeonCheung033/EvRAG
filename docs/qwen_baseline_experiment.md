# Qwen基线系统评估实验说明

## 1. 实验目的

本实验旨在构建并评估一个**Qwen基线RAG系统**，作为对比基准，用于评估我们优化后的RAG系统的性能提升。该基线系统模拟了真实场景中chatbot对PDF文档的**粗糙处理方式**，使用最基础的RAG流程，不包含任何优化技术。

### 1.1 实验意义

- **建立性能基准**：为我们的优化系统提供对比基线
- **模拟真实场景**：使用粗糙的PDF处理方式，更贴近实际应用
- **评估优化效果**：量化我们的系统相比基线系统的性能提升
- **验证技术价值**：证明BM25混合检索、重排序、微调等技术的价值

## 2. 实验设计

### 2.1 基线系统配置

| 组件 | 配置 | 说明 |
|------|------|------|
| **LLM模型** | `Qwen/Qwen3-32B` | 通过SiliconFlow API调用，未微调 |
| **Embedding模型** | `Qwen/Qwen3-Embedding-8B` | 通过SiliconFlow API调用 |
| **向量数据库** | FAISS | 使用Qwen Embedding构建的向量索引 |
| **检索方式** | 仅向量检索 | 无BM25混合，无重排序 |
| **文档处理** | 从原始PDF加载并分块 | 使用PyMuPDF加载，RecursiveCharacterTextSplitter分块 |
| **分块参数** | chunk_size=500, chunk_overlap=50 | 基础分块策略 |
| **生成方式** | 直接生成答案 | 无后处理，无引用提取 |
| **Reranker** | 不使用 | 无重排序步骤 |

### 2.2 与我们的系统对比

| 特性 | 基线系统 | 我们的系统 |
|------|---------|-----------|
| **LLM** | Qwen3-32B（未微调） | Qwen3-8B（微调后） |
| **Embedding** | Qwen3-Embedding-8B | BGE-M3 |
| **检索** | 仅向量检索 | BM25 + 向量混合检索 |
| **重排序** | 无 | 使用微调后的Reranker |
| **文档处理** | 粗糙分块（500字符） | 精心处理的分块 |
| **后处理** | 无 | 引用提取、格式化等 |

## 3. 技术架构

### 3.1 系统架构图

```
┌─────────────────────────────────────────────────────────┐
│              Qwen基线RAG系统架构                        │
└─────────────────────────────────────────────────────────┘

PDF文档
  │
  ├─→ PyMuPDF加载
  │     │
  │     └─→ 逐页提取文本
  │
  ├─→ RecursiveCharacterTextSplitter分块
  │     │
  │     └─→ chunk_size=500, chunk_overlap=50
  │
  ├─→ Qwen3-Embedding-8B向量化
  │     │
  │     └─→ 通过SiliconFlow API
  │
  └─→ FAISS向量索引构建
        │
        └─→ 保存到本地

用户问题
  │
  ├─→ Qwen3-Embedding-8B向量化
  │     │
  │     └─→ 通过SiliconFlow API
  │
  ├─→ FAISS向量检索（Top-K）
  │     │
  │     └─→ 无BM25，无重排序
  │
  ├─→ 构建上下文
  │     │
  │     └─→ 简单拼接检索到的文档
  │
  └─→ Qwen3-32B生成答案
        │
        └─→ 通过SiliconFlow API，无后处理
```

### 3.2 关键技术点

#### 3.2.1 PDF文档加载

使用**PyMuPDF（fitz）**库加载PDF，避免对cryptography的依赖：

```python
import fitz  # PyMuPDF

doc = fitz.open(pdf_path)
for page_num, page in enumerate(doc):
    text = page.get_text()
    # 转换为LangChain Document对象
```

**优势**：
- 不需要cryptography依赖
- 性能更好（C语言实现）
- 对加密PDF处理更稳定

#### 3.2.2 文本分块

使用LangChain的`RecursiveCharacterTextSplitter`进行基础分块：

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每块500字符
    chunk_overlap=50,    # 重叠50字符
    length_function=len,
)
chunks = text_splitter.split_documents(docs)
```

**特点**：
- 固定大小分块，无语义感知
- 简单重叠策略
- 模拟真实chatbot的粗糙处理

#### 3.2.3 向量检索

使用FAISS进行向量相似度搜索：

```python
results = vectorstore.similarity_search_with_score(query, k=topk)
```

**特点**：
- 仅使用向量相似度
- 无BM25混合检索
- 无重排序步骤

#### 3.2.4 答案生成

使用LangChain的RAG链生成答案：

```python
# 简单的Prompt模板
prompt = """你是一个专业助手，请严格根据以下提供的参考资料回答问题。

参考资料：
{context}

问题：{question}

请直接输出答案，不要输出任何思考过程或解释性文字。
"""

# 使用ChatOpenAI调用Qwen3-32B
llm = ChatOpenAI(
    model="Qwen/Qwen3-32B",
    openai_api_key=api_key,
    openai_api_base=base_url,
    temperature=0.1,
    max_tokens=512,
)
```

**特点**：
- 简单的Prompt模板
- 无后处理步骤
- 无引用提取

## 4. 数据处理流程

### 4.1 文档处理流程

```
原始PDF文件
    ↓
[步骤1] PyMuPDF加载
    ├─ 逐页打开PDF
    ├─ 提取每页文本内容
    └─ 转换为LangChain Document对象
    ↓
[步骤2] 文本分块
    ├─ RecursiveCharacterTextSplitter
    ├─ chunk_size=500字符
    ├─ chunk_overlap=50字符
    └─ 生成文档块列表
    ↓
[步骤3] 向量化
    ├─ 使用Qwen3-Embedding-8B
    ├─ 通过SiliconFlow API
    └─ 生成向量表示
    ↓
[步骤4] 构建FAISS索引
    ├─ 将向量和文档存储到FAISS
    └─ 保存索引到本地
```

### 4.2 问答流程

```
用户问题
    ↓
[步骤1] 问题向量化
    ├─ 使用Qwen3-Embedding-8B
    └─ 生成查询向量
    ↓
[步骤2] 向量检索
    ├─ FAISS相似度搜索
    ├─ 返回Top-K文档
    └─ 无BM25，无重排序
    ↓
[步骤3] 构建上下文
    ├─ 简单拼接检索到的文档
    └─ 格式化输出
    ↓
[步骤4] 生成答案
    ├─ 使用Qwen3-32B
    ├─ 通过SiliconFlow API
    └─ 直接输出答案（无后处理）
```

## 5. 评估指标

### 5.1 主要评估指标

#### 5.1.1 语义相似度+关键词加权得分

**计算公式**：
```
final_score = 0.8 × semantic_score + 0.2 × keyword_score
```

- **semantic_score**：使用text2vec计算生成答案与标准答案的语义相似度
- **keyword_score**：关键词匹配得分（关键词在答案中的覆盖率）

**目标值**：≥ 0.75

#### 5.1.2 RAGas框架指标

- **Context Recall**：检索到的上下文是否包含标准答案所需的信息
- **Context Precision**：检索到的上下文是否与问题相关

**计算公式**：
```
ragas_average = 0.7 × context_recall + 0.3 × context_precision
```

**目标值**：
- Context Recall ≥ 0.85
- Context Precision ≥ 0.80

#### 5.1.3 生成质量指标

- **BLEU**：n-gram重叠度
- **ROUGE-1**：单字重叠度
- **ROUGE-2**：双字重叠度
- **ROUGE-L**：最长公共子序列

**目标值**：
- BLEU ≥ 0.40
- ROUGE-L ≥ 0.60

#### 5.1.4 综合准确率

**计算公式**：
```
comprehensive_accuracy = 0.7 × semantic_keyword_score + 0.3 × ragas_average
```

**目标值**：≥ 0.80

### 5.2 性能指标

- **检索时间**：向量检索的平均耗时
- **生成时间**：LLM生成答案的平均耗时
- **总响应时间**：端到端的平均耗时

### 5.3 无答案样本统计

- **命中率**：标准答案为"无答案"时，系统正确识别为"无答案"的比例
- **误报率**：标准答案为"无答案"时，系统错误生成答案的比例

## 6. 运行方法

### 6.1 环境准备

#### 6.1.1 安装依赖

```bash
# 安装PyMuPDF（推荐）
pip install pymupdf

# 或安装PyPDFLoader的依赖（备选）
pip install cryptography>=3.1
```

#### 6.1.2 配置设置

确保`config/config.yaml`中配置了以下参数：

```yaml
# SiliconFlow API配置
siliconflow_api_key: "your_api_key_here"
siliconflow_base_url: "https://api.siliconflow.cn/v1"

# PDF文件路径
pdf_path: "data/Tesla_Manual.pdf"
```

### 6.2 运行评估

#### 6.2.1 快速测试（抽样30条）

```bash
cd /remote-home/share/liangZhang/EvRAG
python scripts/run_qwen_baseline_evaluation.py
```

#### 6.2.2 使用指定测试文件（全量测试）

```bash
python scripts/run_qwen_baseline_evaluation.py \
    --test-data data/qa_pairs/test_qa_pair_handmade_verify01.json \
    --full-test
```

#### 6.2.3 自定义参数

```bash
python scripts/run_qwen_baseline_evaluation.py \
    --test-data data/qa_pairs/test_qa_pair_handmade_verify01.json \
    --pdf-path data/Tesla_Manual.pdf \
    --topk 10 \
    --chunk-size 500 \
    --chunk-overlap 50 \
    --sample-size 10 \
    --max-workers 4
```

### 6.3 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--test-data` | str | `data/qa_pairs/test_qa_pair_handmade_verify01.json` | 测试数据文件路径 |
| `--output-dir` | str | None | 输出目录（None时自动生成） |
| `--pdf-path` | str | None | PDF文件路径（None时从配置读取） |
| `--topk` | int | 10 | 检索数量 |
| `--chunk-size` | int | 500 | 文本分块大小 |
| `--chunk-overlap` | int | 50 | 文本分块重叠大小 |
| `--sample-size` | int | None | 抽样测试样本数 |
| `--full-test` | flag | False | 是否执行全量测试 |
| `--no-ragas` | flag | False | 不使用RAGas评估 |
| `--max-workers` | int | None | 最大并发工作线程数 |
| `--llm-model` | str | None | LLM模型名称 |
| `--embedding-model` | str | None | Embedding模型名称 |
| `--temperature` | float | 0.1 | LLM温度参数 |
| `--max-tokens` | int | 512 | LLM最大生成token数 |

## 7. 输出结果

### 7.1 输出文件结构

```
rag_test_reports/qwen_baseline_YYYYMMDD_HHMMSS/
├── qwen_baseline_evaluation_results.json          # 详细评估结果
└── qwen_baseline_evaluation_results_summary.json  # 汇总指标
```

### 7.2 结果文件格式

#### 7.2.1 详细结果（results.json）

每个样本包含：

```json
{
  "unique_id": "样本唯一标识",
  "question": "问题",
  "ground_truth_answer": "标准答案",
  "generated_answer": "生成的答案",
  "keywords": ["关键词列表"],
  "context": "检索到的上下文",
  "retrieved_docs_count": 5,
  "metrics": {
    "semantic_keyword_score": {
      "semantic_score": 0.93,
      "keyword_score": 0.0,
      "final_score": 0.75,
      "matched_keywords": [],
      "total_keywords": 1
    },
    "generation_quality": {
      "bleu": 0.50,
      "rouge_1": 0.67,
      "rouge_2": 0.40,
      "rouge_l": 0.67
    },
    "response_time": {
      "retrieval_time": 0.38,
      "generation_time": 0.84,
      "total_time": 1.21
    }
  },
  "timings": {
    "retrieval_time": 0.38,
    "generation_time": 0.84,
    "total_time": 1.21
  }
}
```

#### 7.2.2 汇总指标（summary.json）

```json
{
  "total_samples": 30,
  "semantic_keyword_score": {
    "mean": 0.84,
    "std": 0.17
  },
  "generation_quality": {
    "bleu": 0.55,
    "rouge_1": 0.68,
    "rouge_2": 0.48,
    "rouge_l": 0.63
  },
  "response_time": {
    "mean_retrieval_time": 0.27,
    "mean_generation_time": 1.69,
    "mean_total_time": 1.97
  },
  "ragas_scores": {
    "context_recall": 0.93,
    "context_precision": 0.83,
    "average": 0.90
  },
  "comprehensive_accuracy": {
    "score": 0.86,
    "semantic_keyword_score": 0.84,
    "ragas_score": 0.90,
    "weights": {
      "semantic_weight": 0.7,
      "ragas_weight": 0.3
    },
    "use_ragas": true
  }
}
```

## 8. 结果解读

### 8.1 指标解读指南

#### 8.1.1 语义相似度+关键词得分

- **0.8-1.0**：优秀，答案与标准答案高度一致
- **0.6-0.8**：良好，答案基本正确但可能有细节差异
- **0.4-0.6**：一般，答案部分正确但存在明显差异
- **< 0.4**：较差，答案与标准答案差异较大

#### 8.1.2 RAGas指标

- **Context Recall ≥ 0.85**：检索到的上下文能够覆盖标准答案所需的大部分信息
- **Context Precision ≥ 0.80**：检索到的上下文与问题高度相关，噪声较少

#### 8.1.3 生成质量指标

- **BLEU ≥ 0.40**：生成答案与标准答案在n-gram层面有较好的重叠
- **ROUGE-L ≥ 0.60**：生成答案与标准答案在序列层面有较好的相似度

#### 8.1.4 综合准确率

- **≥ 0.80**：系统整体性能良好
- **0.70-0.80**：系统性能中等，有改进空间
- **< 0.70**：系统性能较差，需要优化

### 8.2 性能分析

#### 8.2.1 响应时间分析

- **检索时间**：通常 < 0.5秒（向量检索较快）
- **生成时间**：通常 1-3秒（取决于LLM响应速度）
- **总响应时间**：通常 1.5-4秒

#### 8.2.2 常见问题

1. **检索时间过长**：
   - 检查FAISS索引是否已构建
   - 检查网络连接（SiliconFlow API调用）

2. **生成时间过长**：
   - 检查SiliconFlow API响应速度
   - 考虑调整`max_tokens`参数

3. **准确率较低**：
   - 检查检索到的上下文是否相关
   - 考虑调整`topk`参数
   - 检查PDF分块质量

## 9. 与我们的系统对比

### 9.1 对比维度

| 维度 | 基线系统 | 我们的系统 | 预期提升 |
|------|---------|-----------|---------|
| **综合准确率** | ~0.80-0.85 | 目标≥0.95 | ≥18% |
| **Context Recall** | ~0.85-0.93 | 目标≥0.95 | ≥10% |
| **Context Precision** | ~0.80-0.83 | 目标≥0.90 | ≥8% |
| **BLEU** | ~0.50-0.55 | 目标≥0.60 | ≥10% |
| **ROUGE-L** | ~0.60-0.63 | 目标≥0.75 | ≥20% |

### 9.2 性能提升计算

```
相对提升 = (我们的系统得分 - 基线系统得分) / 基线系统得分 × 100%
```

**示例**：
- 基线系统综合准确率：0.85
- 我们的系统综合准确率：1.00
- 相对提升 = (1.00 - 0.85) / 0.85 × 100% = 17.6%

### 9.3 技术优势分析

通过对比基线系统，可以量化以下技术的价值：

1. **BM25混合检索**：提升检索精度
2. **重排序技术**：提升Top-K文档质量
3. **LLM微调**：提升生成质量
4. **精心文档处理**：提升检索效果
5. **后处理技术**：提升答案格式和质量

## 10. 实验注意事项

### 10.1 数据一致性

- **使用相同的测试集**：确保对比公平性
- **使用相同的PDF文档**：确保知识库一致
- **使用相同的评估指标**：确保结果可比较

### 10.2 实验可复现性

- **固定随机种子**：确保抽样结果可复现
- **保存配置参数**：记录所有实验参数
- **保存详细结果**：便于后续分析

### 10.3 常见问题处理

1. **PDF加载失败**：
   - 检查PDF文件路径是否正确
   - 检查PDF文件是否损坏
   - 尝试使用PyMuPDF替代PyPDFLoader

2. **API调用失败**：
   - 检查SiliconFlow API密钥是否正确
   - 检查网络连接
   - 检查API配额是否充足

3. **索引构建失败**：
   - 检查磁盘空间是否充足
   - 检查FAISS索引路径权限
   - 检查Embedding API是否正常

## 11. 实验总结

### 11.1 实验价值

1. **建立性能基准**：为优化系统提供对比基线
2. **验证技术价值**：量化各项优化技术的效果
3. **指导系统优化**：识别系统薄弱环节
4. **支持决策制定**：为技术选型提供数据支持

### 11.2 后续工作

1. **运行全量测试**：使用完整的测试集进行评估
2. **对比分析**：与我们的优化系统进行详细对比
3. **性能分析**：分析各项指标的提升原因
4. **报告生成**：生成详细的对比分析报告

## 12. 参考资料

- [Qwen基线评估脚本使用说明](../scripts/README_qwen_baseline.md)
- [RAG测试计划](./rag_test_plan.md)
- [LangChain文档](https://python.langchain.com/)
- [PyMuPDF文档](https://pymupdf.readthedocs.io/)
- [SiliconFlow API文档](https://siliconflow.cn/)

---

**文档版本**：v1.0  
**最后更新**：2024-11-30  
**维护者**：EvRAG团队

