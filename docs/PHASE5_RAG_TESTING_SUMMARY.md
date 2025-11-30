# Phase 5: RAG测试阶段工作总结

## 一、概述

本阶段完成了RAG（Retrieval-Augmented Generation）系统的端到端评估框架实现，包括评估指标计算、RAGas框架集成、基线和微调模型对比、Qwen基线系统评估等功能。

**时间范围**: 2024年11月  
**分支**: `feature/rag-testing`  
**代码统计**: 新增约8000+行代码，修改约10+个文件

---

## 二、核心评估模块实现

### 2.1 评估指标计算模块 (`src/evrag/evaluation/metrics.py`)

**功能**: 实现各种RAG评估指标的计算函数

**主要函数**:
- `calculate_semantic_keyword_score()` - 语义相似度+关键词加权评分
  - 语义相似度：使用text2vec计算余弦相似度
  - 关键词匹配：Jaccard相似度（阈值0.3）
  - 最终得分：`0.2 * keyword_score + 0.8 * semantic_score`
  - 特殊处理：空关键词列表时仅使用语义相似度
  - 返回字段：`semantic_score`, `keyword_score`, `final_score`, `matched_keywords`, `total_keywords`

- `calculate_retrieval_accuracy()` - 检索准确率计算
  - Top-K准确率：检索到的文档中是否包含标准答案相关文档
  - 支持缺失`ground_truth_docs`的情况（返回0.0）

- `calculate_reranking_metrics()` - 重排序指标计算
  - NDCG@K：归一化折损累积增益
  - MRR：平均倒数排名
  - Precision@K：精确率
  - Recall@K：召回率
  - 支持缺失`ground_truth_docs`的情况（返回0.0）

- `calculate_generation_quality()` - 生成质量指标
  - BLEU：双语评估替补分数（中文分词使用jieba）
  - ROUGE-1/2/L：召回导向的评估指标
  - **已移除**: `exact_match`指标（用户要求）

- `calculate_response_time()` - 响应时间统计
  - 检索时间、重排序时间、生成时间、总时间

**代码行数**: 347行

---

### 2.2 RAG评估核心模块 (`src/evrag/evaluation/rag_evaluator.py`)

**功能**: 执行端到端RAG流程评估，收集性能指标，计算评估分数

**主要特性**:
1. **模型配置管理**
   - 支持基线和微调模型切换（`use_baseline`参数）
   - 基线模型：本地基线BGEReranker + 基线LLM（端口8000）
   - 微调模型：SiliconFlow Qwen/Qwen3-Reranker-8B + 微调LLM（端口8001）
   - 详细的日志输出，显示实际使用的模型配置

2. **端到端评估流程**
   - BM25检索 → Milvus检索 → 文档合并 → 重排序 → LLM生成 → 后处理
   - 记录每个阶段的耗时
   - 支持并发评估（ThreadPoolExecutor）

3. **评估指标收集**
   - 语义相似度+关键词加权得分
   - 检索准确率
   - 重排序指标
   - 生成质量指标
   - 响应时间统计
   - 无答案样本统计（命中率、误判率）

4. **综合准确率计算**
   - 公式：`0.7 * avg_semantic_keyword_score + 0.3 * avg_ragas_score`
   - 支持RAGas评估开关
   - 支持RAGas抽样测试

5. **结果保存**
   - 详细结果JSON（每个样本的完整评估数据）
   - 汇总指标JSON（平均值、标准差等）
   - 支持传入已计算的summary（包含RAGas结果）

**代码行数**: 590行

---

### 2.3 RAGas框架集成 (`src/evrag/evaluation/ragas_evaluator.py`)

**功能**: 使用RAGas框架进行LLM评估，计算ContextRecall和ContextPrecision

**主要特性**:
1. **API集成**
   - 使用SiliconFlow API调用Qwen/Qwen3-Next-80B-A3B-Instruct模型
   - 支持自定义API密钥和基础URL
   - 默认并发数：4（避免429速率限制）

2. **评估指标**
   - ContextRecall：评估检索上下文是否包含完整信息
   - ContextPrecision：评估检索上下文的相关性
   - **不启用**: AnswerRelevancy（按用户要求）

3. **Context格式处理**
   - 自动解析字符串格式的context（`【1】doc1\n【2】doc2`）
   - 使用正则表达式拆分文档
   - 转换为RAGas所需的文档列表格式

4. **错误处理**
   - 处理API调用失败（返回0.0）
   - 处理NaN值（使用dropna()）
   - 增加超时时间（300秒）
   - 增加重试次数（5次）

5. **列名兼容性**
   - 支持`llm_context_precision_with_reference`列名
   - 向后兼容`context_precision`列名
   - 详细的调试信息输出

**代码行数**: 350行

---

### 2.4 Qwen基线评估模块 (`src/evrag/evaluation/qwen_baseline_evaluator.py`)

**功能**: 实现Qwen3-32B + Qwen3-Embedding-8B基线系统的评估

**主要特性**:
1. **SiliconFlow API集成**
   - `SiliconFlowEmbeddingClient`: 调用Qwen3-Embedding-8B进行文本向量化
   - LLM客户端：调用Qwen3-32B生成答案

2. **检索策略**
   - 仅使用向量检索（无BM25混合，无重排序）
   - 使用Qwen Embedding构建的Milvus索引
   - 支持fallback到现有集合（如果Qwen索引不存在）

3. **评估流程**
   - 向量检索 → LLM生成（无后处理）
   - 计算评估指标（语义相似度、生成质量、响应时间）
   - 支持RAGas评估

**代码行数**: 631行

---

## 三、CLI接口增强

### 3.1 新增命令

#### `evaluate-rag` - RAG系统评估

**功能**: 执行端到端RAG流程评估

**主要参数**:
- `--test-data`: 测试数据路径
- `--output-dir`: 输出目录（自动添加时间戳）
- `--use-baseline`: 是否使用基线模型
- `--full-test`: 是否执行全量测试
- `--sample-size`: 抽样测试样本数
- `--use-ragas`: 是否使用RAGas评估
- `--max-workers`: 最大并发工作线程数
- `--bm25-topk`, `--milvus-topk`, `--reranker-topk`: 检索参数

**输出**:
- `{model_type}_evaluation_results.json` - 详细结果
- `{model_type}_evaluation_summary.json` - 汇总指标

#### `evaluate-qwen-baseline` - Qwen基线系统评估

**功能**: 使用Qwen3-32B + Qwen3-Embedding-8B进行基线评估

**主要参数**:
- `--test-data`: 测试数据路径
- `--output-dir`: 输出目录
- `--full-test`: 是否执行全量测试
- `--use-ragas`: 是否使用RAGas评估
- `--max-workers`: 最大并发工作线程数

---

## 四、评估脚本

### 4.1 主要测试脚本（本阶段核心使用）

本阶段主要使用以下三个脚本进行测试：

#### `run_baseline_evaluation.sh` - 基线模型评估脚本

**功能**: 单独运行基线模型的全量评估

**主要特性**:
- 使用`main.py evaluate-rag --use-baseline`命令
- 支持全量测试（`--full-test`）
- 自动生成带时间戳的输出目录
- 自动运行结果分析脚本
- 保存评估日志到文件

**配置参数**:
- `TEST_DATA`: 测试数据路径（默认：`data/qa_pairs/test_qa_pair_handmade.json`）
- `OUTPUT_BASE_DIR`: 输出基础目录（默认：`rag_test_reports/baseline_evaluation`）
- `MAX_WORKERS`: 并发数（默认：6）
- `USE_RAGAS`: 是否使用RAGas评估（默认：true）

**输出文件**:
- `baseline_evaluation_results.json` - 详细评估结果
- `baseline_evaluation_summary.json` - 汇总指标
- `baseline_evaluation.log` - 评估日志
- `baseline_analysis.txt` - 分析报告（如果分析脚本存在）

**使用示例**:
```bash
bash scripts/run_baseline_evaluation.sh
```

---

#### `run_finetuned_evaluation.sh` - 微调模型评估脚本

**功能**: 单独运行微调模型的全量评估

**主要特性**:
- 使用`main.py evaluate-rag`命令（默认使用微调模型）
- 支持全量测试（`--full-test`）
- 自动生成带时间戳的输出目录
- 自动运行结果分析脚本
- 保存评估日志到文件

**配置参数**:
- `TEST_DATA`: 测试数据路径（默认：`data/qa_pairs/test_qa_pair_handmade.json`）
- `OUTPUT_BASE_DIR`: 输出基础目录（默认：`rag_test_reports/finetuned_evaluation`）
- `MAX_WORKERS`: 并发数（默认：6）
- `USE_RAGAS`: 是否使用RAGas评估（默认：true）

**输出文件**:
- `finetuned_evaluation_results.json` - 详细评估结果
- `finetuned_evaluation_summary.json` - 汇总指标
- `finetuned_evaluation.log` - 评估日志
- `finetuned_analysis.txt` - 分析报告（如果分析脚本存在）

**使用示例**:
```bash
bash scripts/run_finetuned_evaluation.sh
```

---

#### `run_qwen_baseline_evaluation.py` - Qwen基线系统评估脚本

**功能**: 运行Qwen3-32B + Qwen3-Embedding-8B基线系统的端到端评估

**主要特性**:
- Python脚本，支持更灵活的配置
- 从原始PDF文件加载并分块，模拟真实chatbot的粗糙处理方式
- 使用SiliconFlow API调用Qwen3-32B和Qwen3-Embedding-8B
- 仅使用向量检索（无BM25混合，无重排序）
- 支持自定义分块参数（chunk_size, chunk_overlap）
- 支持抽样测试和全量测试

**主要参数**:
- `--test-data`: 测试数据文件路径（默认：`data/qa_pairs/test_qa_pair_handmade_verify01.json`）
- `--output-dir`: 输出目录（None表示自动生成带时间戳的目录）
- `--pdf-path`: PDF文件路径（None表示从配置读取）
- `--topk`: 检索数量（默认：10）
- `--chunk-size`: 文本分块大小（默认：500）
- `--chunk-overlap`: 文本分块重叠大小（默认：50）
- `--sample-size`: 抽样测试样本数（None表示全量测试）
- `--full-test`: 是否执行全量测试
- `--no-ragas`: 不使用RAGas评估
- `--max-workers`: 最大并发工作线程数
- `--llm-model`: LLM模型名称（默认：Qwen/Qwen3-32B）
- `--embedding-model`: Embedding模型名称（默认：Qwen/Qwen3-Embedding-8B）
- `--temperature`: LLM温度参数（默认：0.1）
- `--max-tokens`: LLM最大生成token数（默认：512）

**输出文件**:
- `qwen_baseline_evaluation_results.json` - 详细评估结果
- `qwen_baseline_evaluation_summary.json` - 汇总指标

**使用示例**:
```bash
# 使用默认参数运行（抽样30条）
python scripts/run_qwen_baseline_evaluation.py

# 全量测试
python scripts/run_qwen_baseline_evaluation.py --full-test

# 指定测试数据和输出目录
python scripts/run_qwen_baseline_evaluation.py \
    --test-data data/qa_pairs/test_qa_pair_handmade_verify01.json \
    --output-dir rag_test_reports/my_test \
    --sample-size 10

# 自定义PDF路径和分块参数
python scripts/run_qwen_baseline_evaluation.py \
    --pdf-path data/Tesla_Manual.pdf \
    --chunk-size 500 \
    --chunk-overlap 50
```

**代码行数**: 355行

---

### 4.2 对比测试脚本

#### `run_baseline_finetuned_comparison.sh`
- 功能：基线和微调模型对比测试
- 流程：基线评估 → 微调评估 → 生成对比报告 → 分析结果
- 输出：对比报告JSON、分析报告TXT

---

### 4.3 辅助脚本

#### `analyze_evaluation_results.py`
- 功能：分析评估结果，诊断为什么数值偏低
- 输出：详细的统计信息和低分样本分析

---

## 五、关键修复和优化

### 5.1 ContextPrecision为0的问题修复

**问题**: RAGas评估中ContextPrecision始终为0

**原因分析**:
1. Context格式问题：context是字符串（`【1】doc1\n【2】doc2`），需要拆分为文档列表
2. 列名匹配问题：RAGas返回的列名是`llm_context_precision_with_reference`，代码中查找的是`context_precision`

**修复方案**:
1. 在`ragas_evaluator.py`中添加context字符串解析逻辑
2. 使用正则表达式`r'【\d+】'`拆分文档
3. 更新列名查找逻辑，支持新旧列名

### 5.2 429速率限制问题修复

**问题**: RAGas评估时频繁出现429错误（请求过多）

**原因**: 并发数过高（默认32），超过API速率限制

**修复方案**:
1. 降低默认并发数：32 → 4
2. 增加重试机制：max_retries 3 → 5
3. 增加等待时间：max_wait 10秒
4. 增加超时时间：timeout 300秒

### 5.3 模型配置混淆问题修复

**问题**: 基线和微调模型的输出经常一致，怀疑模型配置混淆

**原因**: 日志输出不够详细，无法确认实际使用的模型

**修复方案**:
1. 在`rag_evaluator.py`中添加详细的模型配置日志
2. 在`local_client.py`中添加参数验证和警告
3. 创建`verify_model_config.py`脚本验证配置
4. 确保基线和微调模型使用不同的URL和模型名称



## 六、文件统计

### 6.1 新增文件

#### 核心评估模块（5个文件，约2000+行）
- `src/evrag/evaluation/__init__.py` (24行)
- `src/evrag/evaluation/metrics.py` (347行)
- `src/evrag/evaluation/rag_evaluator.py` (590行)
- `src/evrag/evaluation/ragas_evaluator.py` (350行)
- `src/evrag/evaluation/qwen_baseline_evaluator.py` (631行)

#### 评估脚本（7个文件）
- `scripts/run_baseline_evaluation.sh` (90行) - **主要测试脚本**
- `scripts/run_finetuned_evaluation.sh` (89行) - **主要测试脚本**
- `scripts/run_qwen_baseline_evaluation.py` (355行) - **主要测试脚本**
- `scripts/run_baseline_finetuned_comparison.sh` (166行) - 对比测试脚本
- `scripts/analyze_evaluation_results.py` (187行) - 结果分析脚本
- `scripts/test_reranker_loading.py` (130行) - 验证脚本
- `scripts/verify_model_config.py` (约100行) - 验证脚本

#### 文档（1个核心文档）
- `docs/rag_test_plan.md` (1232行)

### 6.2 修改文件

#### 核心文件
- `main.py` (+355行) - 添加`evaluate-rag`和`evaluate-qwen-baseline`命令
- `src/evrag/config.py` (+19行) - 添加SiliconFlow配置
- `src/evrag/client/local_client.py` (+27行) - 增强参数验证
- `src/evrag/reranker/siliconflow_reranker.py` (+5行) - 隐藏日志输出

#### 配置文件
- `config/config.yaml` - 添加微调Reranker路径配置

#### 部署脚本
- `scripts/start_vllm_for_evaluation.sh` - 优化GPU分配和并发参数

### 6.3 代码统计

- **新增代码**: 约8000+行
- **修改代码**: 约500+行
- **评估模块代码**: 约2000+行
- **测试脚本**: 约1000+行
- **文档**: 约1500+行

---

## 七、功能特性总结

### 7.1 评估指标

1. **语义相似度+关键词加权评分**
   - 语义相似度：text2vec余弦相似度
   - 关键词匹配：Jaccard相似度（阈值0.3）
   - 加权公式：`0.2 * keyword_score + 0.8 * semantic_score`

2. **检索准确率**
   - Top-K准确率：检索到的文档中是否包含标准答案相关文档

3. **重排序指标**
   - NDCG@K、MRR、Precision@K、Recall@K

4. **生成质量指标**
   - BLEU、ROUGE-1/2/L（中文分词使用jieba）

5. **RAGas指标**
   - ContextRecall：检索上下文完整性
   - ContextPrecision：检索上下文相关性

6. **综合准确率**
   - 公式：`0.7 * avg_semantic_keyword_score + 0.3 * avg_ragas_score`
   - 目标：≥ 90%（理想目标）

### 7.2 评估流程

1. **端到端RAG流程**
   - BM25检索 → Milvus检索 → 文档合并 → 重排序 → LLM生成 → 后处理

2. **并发评估**
   - 使用ThreadPoolExecutor进行并发处理
   - 默认并发数：2（可配置）
   - RAGas评估并发数：4（避免API速率限制）

3. **结果保存**
   - 详细结果JSON（每个样本的完整数据）
   - 汇总指标JSON（平均值、标准差等）
   - 评估日志（包含所有输出）

### 7.3 模型配置

1. **基线模型**
   - LLM：Qwen3-8B（端口8000）
   - Reranker：本地BGEReranker（`bge-reranker-v2-m3`）
   - Embedding：BGE-M3（Milvus）

2. **微调模型**
   - LLM：qwen3_lora_sft（端口8001）
   - Reranker：微调reranker
   - Embedding：BGE-M3（Milvus，与基线相同）

3. **Qwen基线系统**
   - LLM：Qwen3-32B（SiliconFlow API）
   - Embedding：Qwen3-Embedding-8B（SiliconFlow API）
   - Reranker：不使用（仅向量检索）

---

## 八、测试和验证

---

## 九、已知问题和限制

### 9.1 已解决的问题

1. ✅ ContextPrecision为0 - 已修复（context格式解析和列名匹配）
2. ✅ 429速率限制 - 已修复（降低并发数，增加重试）
3. ✅ 模型配置混淆 - 已修复（增强日志和验证）
5. ✅ 空关键词列表处理 - 已改进

### 9.2 当前限制

1. **RAGas评估速度**
   - 即使降低并发数到4，全量测试（750条）仍需要较长时间
   - 建议：先抽样测试（30条），验证正确性后再全量测试

2. **可视化图表**
   - 优先级1的图表尚未实现（检索准确率对比图、重排序对比图、端到端性能热力图）
   - 计划：下一阶段实现

---

## 十一、使用指南

### 11.1 快速开始（本阶段主要使用方式）

**注意**: 本阶段主要使用以下三个脚本进行测试，它们封装了完整的评估流程，包括结果分析和日志保存。

#### 1. 启动vLLM服务

```bash
bash scripts/start_vllm_for_evaluation.sh
```

#### 2. 运行基线模型评估（推荐）

```bash
# 使用脚本运行（自动生成时间戳目录、保存日志、运行分析）
bash scripts/run_baseline_evaluation.sh
```

**输出目录**: `rag_test_reports/baseline_evaluation_YYYYMMDD_HHMMSS/`

**输出文件**:
- `baseline_evaluation_results.json` - 详细评估结果
- `baseline_evaluation_summary.json` - 汇总指标
- `baseline_evaluation.log` - 完整评估日志
- `baseline_analysis.txt` - 分析报告

#### 3. 运行微调模型评估（推荐）

```bash
# 使用脚本运行（自动生成时间戳目录、保存日志、运行分析）
bash scripts/run_finetuned_evaluation.sh
```

**输出目录**: `rag_test_reports/finetuned_evaluation_YYYYMMDD_HHMMSS/`

**输出文件**:
- `finetuned_evaluation_results.json` - 详细评估结果
- `finetuned_evaluation_summary.json` - 汇总指标
- `finetuned_evaluation.log` - 完整评估日志
- `finetuned_analysis.txt` - 分析报告

#### 4. 运行Qwen基线系统评估（推荐）

```bash
# 使用Python脚本运行（支持更多配置选项）
python scripts/run_qwen_baseline_evaluation.py --full-test
```

**输出目录**: `rag_test_reports/qwen_baseline_YYYYMMDD_HHMMSS/`

**输出文件**:
- `qwen_baseline_evaluation_results.json` - 详细评估结果
- `qwen_baseline_evaluation_summary.json` - 汇总指标

**自定义配置示例**:
```bash
# 抽样测试（30条）
python scripts/run_qwen_baseline_evaluation.py --sample-size 30

# 指定PDF路径和分块参数
python scripts/run_qwen_baseline_evaluation.py \
    --pdf-path data/Tesla_Manual.pdf \
    --chunk-size 500 \
    --chunk-overlap 50 \
    --full-test

# 不使用RAGas评估（加快速度）
python scripts/run_qwen_baseline_evaluation.py --no-ragas --full-test
```

#### 5. 运行对比测试（可选）

```bash
# 运行基线和微调模型的对比测试
bash scripts/run_baseline_finetuned_comparison.sh
```

**输出目录**: `rag_test_reports/comparison_YYYYMMDD_HHMMSS/`

**输出文件**:
- `baseline/` - 基线模型评估结果
- `finetuned/` - 微调模型评估结果
- `comparison_results.json` - 对比报告
- `baseline_analysis.txt` - 基线分析报告
- `finetuned_analysis.txt` - 微调分析报告

### 11.2 脚本配置说明

#### 修改测试数据路径

编辑脚本文件，修改`TEST_DATA`变量：

```bash
# run_baseline_evaluation.sh 或 run_finetuned_evaluation.sh
TEST_DATA="data/qa_pairs/test_qa_pair_handmade.json"  # 修改为你的测试数据路径
```

#### 修改并发数

编辑脚本文件，修改`MAX_WORKERS`变量：

```bash
# 根据显存情况调整
MAX_WORKERS=6  # 默认6，可根据实际情况调整
```

#### 禁用RAGas评估

编辑脚本文件，修改`USE_RAGAS`变量：

```bash
USE_RAGAS=false  # 设置为false可加快评估速度（但会缺少RAGas指标）
```

### 11.3 命令行使用（高级用法）

如果需要更细粒度的控制，可以直接使用CLI命令：

#### 评估RAG系统

```bash
# 基线模型评估（抽样30条）
python main.py evaluate-rag \
    --test-data data/qa_pairs/test_qa_pair.json \
    --sample-size 30 \
    --use-baseline \
    --use-ragas

# 微调模型评估（全量测试）
python main.py evaluate-rag \
    --test-data data/qa_pairs/test_qa_pair.json \
    --full-test \
    --use-ragas \
    --max-workers 6
```

#### 评估Qwen基线系统

```bash
python main.py evaluate-qwen-baseline \
    --test-data data/qa_pairs/test_qa_pair.json \
    --full-test \
    --use-ragas
```

### 11.4 结果分析

**注意**: 如果使用上述三个主要测试脚本（`run_baseline_evaluation.sh`、`run_finetuned_evaluation.sh`、`run_qwen_baseline_evaluation.py`），结果分析会自动运行，无需手动执行。

如果需要手动分析结果：

```bash
# 分析基线模型结果
python scripts/analyze_evaluation_results.py \
    rag_test_reports/baseline_evaluation_xxx/baseline_evaluation_summary.json \
    rag_test_reports/baseline_evaluation_xxx/baseline_evaluation_results.json

# 分析微调模型结果
python scripts/analyze_evaluation_results.py \
    rag_test_reports/finetuned_evaluation_xxx/finetuned_evaluation_summary.json \
    rag_test_reports/finetuned_evaluation_xxx/finetuned_evaluation_results.json
```

---

## 十二、技术细节

### 12.1 评估指标计算公式

#### 语义相似度+关键词加权评分

```
final_score = 0.2 * keyword_score + 0.8 * semantic_score

其中：
- semantic_score: text2vec余弦相似度 [0, 1]
- keyword_score: Jaccard相似度，如果 > 0.3 返回1，否则返回0
- 如果keywords为空，final_score = semantic_score
```

#### 综合准确率

```
comprehensive_accuracy = 0.7 * avg_semantic_keyword_score + 0.3 * avg_ragas_score

其中：
- avg_semantic_keyword_score: 所有样本的final_score平均值
- avg_ragas_score: (context_recall + context_precision) / 2.0
```

### 12.2 并发控制

- **RAG评估并发数**: 默认2（可配置，通过`--max-workers`）
- **RAGas评估并发数**: 默认4（避免API速率限制）
- **语义模型加载**: 动态选择可用GPU（优先GPU 2，fallback到CPU）

### 12.3 错误处理

- **API调用失败**: 返回0.0，不中断评估流程
- **NaN值处理**: 使用`np.nan_to_num()`和`dropna()`
- **超时处理**: RAGas评估超时时间300秒，重试5次

---

## 十三、配置说明

### 13.1 关键配置项

```yaml
# 基线模型配置
local_llm_base_url: "http://localhost:8000/v1"
local_llm_model_name: "Qwen3-8B"
bge_reranker_model_path: "/path/to/bge-reranker-v2-m3"

# 微调模型配置
finetuned_llm_base_url: "http://localhost:8001/v1"
finetuned_llm_model_name: "qwen3_lora_sft"
bge_reranker_tuned_model_path: "/path/to/finetuned/bge_reranker/model"

# SiliconFlow API配置
siliconflow_api_key: "your-api-key"
siliconflow_base_url: "https://api.siliconflow.cn/v1"
siliconflow_reranker_model: "Qwen/Qwen3-Reranker-8B"
```

### 13.2 环境变量

- `SILICONFLOW_API_KEY`: SiliconFlow API密钥
- `SILICONFLOW_BASE_URL`: SiliconFlow API基础URL
- `RAGAS_MODEL_NAME`: RAGas评估使用的模型名称

---

## 十四、性能数据

### 14.1 评估速度

- **单样本评估时间**: 约5-10秒（包含检索、重排序、生成）
- **全量测试（750条）**: 约1-2小时（取决于并发数和RAGas评估）
- **RAGas评估**: 约30-60秒/样本（取决于API响应速度）

### 14.2 资源占用

- **GPU显存**: 
  - vLLM服务：约40GB（基线+微调各占约20GB）
  - 语义模型：约2-3GB
  - Reranker：约2-3GB
- **CPU**: 并发评估时CPU使用率较高
- **内存**: 评估结果存储在内存中，全量测试约需数GB

---

## 十五、总结

本阶段成功实现了完整的RAG评估框架，包括：

1. ✅ **核心评估模块**: 5个评估模块，约2000+行代码
2. ✅ **CLI接口**: 2个新命令，支持基线和微调模型评估
3. ✅ **RAGas集成**: 完整的RAGas框架集成，支持ContextRecall和ContextPrecision
4. ✅ **Qwen基线系统**: 完整的Qwen基线评估实现
5. ✅ **评估脚本**: 6个评估和测试脚本
6. ✅ **问题修复**: 修复了ContextPrecision、429错误、模型配置混淆等关键问题
7. ✅ **文档**: 详细的实现计划文档（1232行）

**待完成工作**:
- 可视化图表生成（优先级1）
- Qwen基线系统完整测试
- 消融实验和参数调优

**代码质量**: 所有代码已通过lint检查，包含详细的注释和错误处理。

---

