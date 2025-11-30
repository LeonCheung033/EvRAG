<!-- c4822b3f-1ee8-44f5-8a16-f46f66d8c0fc ebb0163a-c40d-4832-bd50-4d44bbb0056f -->
# RAG系统测试阶段详细实现计划（Phase 5）

## 一、分支规划和提交计划

### 1.1 Git分支操作

```bash
# 从develop分支创建feature分支
git checkout develop
git pull origin develop
git flow feature start rag-testing

# 开发完成后合并
git flow feature finish rag-testing
git push origin develop
```

### 1.2 提交计划

- **提交1**: 实现RAG评估核心模块（rag_evaluator.py）
- **提交2**: 实现消融实验和参数调优功能
- **提交3**: 集成RAGas框架评估
- **提交4**: 实现可视化图表生成
- **提交5**: 添加命令行接口和文档

## 二、RAG评估核心实现

### 2.1 原项目代码参考路径

**核心评估脚本**: `/remote-home/share/liangZhang/EVRAG/final_score.py`

**关键功能**:

- 端到端RAG流程测试
- 语义相似度+关键词匹配评分
- RAGas框架评估（ContextRecall、ContextPrecision）

### 2.2 实现步骤

#### 步骤1: 创建RAG评估核心模块

**新项目文件**: `src/evrag/evaluation/rag_evaluator.py`（**全新创建**，evaluation目录下目前无文件）

**主要功能**:

- 执行端到端RAG流程（检索→重排序→生成）
- 收集性能指标（检索准确率、重排序效果、生成质量、响应时间）
- 计算评估分数（语义相似度+关键词匹配、RAGas指标）
- 支持消融实验（不同组件组合）
- 支持参数调优（不同topk、temperature等）
- **重要**: 拆出一套专门用于评估的pipeline，与线上infer流程独立（允许有差异）

**参考代码片段**:

```python
# 原项目 final_score.py:133-156
for item in test_qa_pairs:
    query = item["question"].strip()
    bm25_docs = bm25_retriever.retrieve_topk(query, topk=BM25_RETRIEVE_SIZE)
    milvus_docs = milvus_retriever.retrieve_topk(query, topk=MILVUS_RETRIEVE_SIZE)
    merged_docs = merge_docs(bm25_docs, milvus_docs)
    ranked_docs = bge_m3_reranker.rank(query, merged_docs, topk=RERANK_SIZE)
    context = "\n".join([str(idx+1) + "." + doc.page_content for idx, doc in enumerate(ranked_docs)])
    response = request_chat(query, context)
    answer = post_processing(response, ranked_docs)
```

**新项目实现**:

- 创建 `RAGEvaluator` 类
- 支持配置不同的检索器、重排序器、LLM
- 记录每个阶段的耗时和结果
- 支持批量评估和并发处理（使用多进程/多线程/asyncio）
- **并发度控制**: 需要探测当前vLLM服务的并发度上限，动态调整并发数
- **抽样测试**: 先抽样30条样本进行端到端测试，确保程序执行正确后再全量测试

#### 步骤2: 实现评估指标计算

**新项目文件**: `src/evrag/evaluation/metrics.py`（**全新创建**）

**需要实现的指标**:

1. **检索准确率** (`calculate_retrieval_accuracy`)

   - Top-K准确率：Top-K检索结果中包含正确答案的比例
   - 支持BM25、Milvus、混合检索的分别计算

2. **重排序效果** (`calculate_reranking_metrics`)

   - NDCG@K：归一化折损累积增益
   - MRR：平均倒数排名
   - Precision@K、Recall@K

3. **生成质量** (`calculate_generation_quality`)

   - BLEU分数（使用jieba分词）
   - ROUGE分数（使用jieba分词）
   - 语义相似度（使用sentence_transformers）
   - 引用准确率

4. **语义相似度+实体关键词加权评分** (`calculate_semantic_keyword_score`)

   - 参考原项目 `report_score` 函数（`final_score.py:78-130`）
   - **语义相似度计算**:
     - 使用 `text2vec` 的 `SentenceModel` 对标准答案和生成答案进行编码
     - 使用 `semantic_search` 计算余弦相似度
     - 得分范围：[0, 1]，值越大表示语义越相似
   - **实体关键词匹配计算**:
     - 提取生成答案中包含的关键词：`join_keywords = [word for word in keywords if word in pred]`
     - 使用Jaccard相似度：`keyword_score = len(join_keywords) / (len(keywords) + 1e-6)`
     - 阈值判断：如果 `keyword_score > 0.3`，返回1，否则返回0
   - **加权计算**:
     - 如果有关键词：`final_score = 0.2 * keyword_score + 0.8 * semantic_score`
     - 如果无关键词：`final_score = semantic_score`
   - **特殊处理**:
     - 如果标准答案为"无答案"且生成答案也为"无答案"：`final_score = 1.0`
     - 如果标准答案为"无答案"但生成答案不为"无答案"：`final_score = 0.0`
   - **无答案样本统计**:
     - 需要单独统计"无答案样本"的命中率/误判率
     - 命中率：标准答案为"无答案"且生成答案也为"无答案"的比例
     - 误判率（FPR）：标准答案为"无答案"但生成答案不为"无答案"的比例

5. **响应时间** (`calculate_response_time`)

   - 检索时间（BM25、Milvus）
   - 重排序时间
   - 生成时间
   - 总时间

6. **吞吐量** (`calculate_throughput`)

   - QPS（每秒查询数）

**参考代码片段**:

```python
# 原项目 final_score.py:78-130
def report_score(result):
    for item in result:
        semantic_score = semantic_search(simModel.encode([gold]), simModel.encode(pred), top_k=1)[0][0]['score']
        keyword_score = calc_jaccard(join_keywords, keywords)
        score = 0.2 * keyword_score + 0.8 * semantic_score
```

#### 步骤3: 集成RAGas框架评估

**新项目文件**: `src/evrag/evaluation/ragas_evaluator.py`（**全新创建**）

**主要功能**:

- 使用RAGas框架进行LLM评估
- 计算ContextRecall和ContextPrecision（**不启用AnswerRelevancy**）
- 支持自定义LLM（使用LangchainLLMWrapper）
- **评估LLM配置**: 使用单独的Doubao/OpenAI兼容接口（通过siliconflow调用）
- **抽样测试**: 先抽样30条样本进行RAGas评估，确保程序执行正确后再全量测试

**RAGas评估指标说明**:

1. **ContextRecall（上下文召回率）**:
   - 衡量检索到的上下文是否包含回答问题所需的所有相关信息
   - 使用LLM判断：给定问题和参考答案，检索到的上下文是否包含足够信息
   - 得分范围：[0, 1]，值越大表示上下文越完整

2. **ContextPrecision（上下文精确率）**:
   - 衡量检索到的上下文中与问题相关的比例
   - 使用LLM判断：给定问题、参考答案和生成答案，检索到的上下文的相关性
   - 得分范围：[0, 1]，值越大表示上下文越相关

**评估数据集格式**:
```python
dataset = [
    {
        "user_input": "问题文本",
        "retrieved_contexts": ["检索到的上下文"],
        "response": "生成的答案",
        "reference": "标准答案"
    },
    ...
]
```

**参考代码片段**:

```python
# 原项目 final_score.py:171-204
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper

evaluation_dataset = EvaluationDataset.from_list(dataset)
evaluator_llm = LangchainLLMWrapper(llm)
result = evaluate(dataset=evaluation_dataset, metrics=[LLMContextRecall(), LLMContextPrecisionWithReference()], llm=evaluator_llm)
```

**新项目实现**:

- 创建 `RAGasEvaluator` 类
- 支持使用Doubao/OpenAI兼容接口（通过siliconflow）
- 处理评估数据集格式转换
- 返回详细的评估结果
- **注意**: 评估LLM与RAG生成LLM分离，使用独立的API服务

## 三、消融实验实现

### 3.1 组件组合配置

**新项目文件**: `src/evrag/evaluation/ablation_study.py`（**全新创建**）

**支持的组件组合**:

1. **仅BM25**: BM25检索 → LLM生成（无重排序）
2. **仅Milvus**: Milvus检索 → LLM生成（无重排序）
3. **BM25 + Milvus**: 混合检索 → LLM生成（无重排序）
4. **BM25 + Milvus + Reranker**: 混合检索 → 重排序 → LLM生成（完整流程）
5. **仅Reranker**: 随机文档 → 重排序 → LLM生成（测试重排序效果）

**实现要点**:

- 创建 `AblationStudy` 类
- 支持配置不同的组件组合
- 对同一测试集执行所有组合的评估
- 生成对比报告

## 四、参数调优实现

### 4.1 参数配置

**新项目文件**: `src/evrag/evaluation/parameter_tuning.py`（**全新创建**）

**需要调优的参数**:

1. **检索参数**:

   - `bm25_topk`: [3, 5, 10, 15]
   - `milvus_topk`: [5, 10, 15, 20]
   - `reranker_topk`: [3, 5, 10]

2. **LLM参数**:

   - `temperature`: [0.0, 0.3, 0.7, 1.0]
   - `max_tokens`: [256, 512, 1024]

3. **混合检索参数**:

   - BM25和Milvus的权重比例

**实现要点**:

- 创建 `ParameterTuner` 类
- 支持网格搜索或随机搜索
- 记录所有参数组合的性能
- 生成参数性能热力图

## 五、微调前后对比

### 5.1 对比配置

**实现要点**:

- 支持配置基线模型和微调后模型
- 对比指标：
  - 检索准确率对比
  - 重排序效果对比
  - 生成质量对比
  - 端到端性能对比

**模型配置**:

- **基线Reranker**: `models/bge-reranker-v2-m3`
- **微调Reranker**: `models/finetuned/bge_reranker`
- **基线LLM**: `models/Qwen3-8B`（通过vLLM服务）
- **微调LLM**: `models/finetuned/qwen3_lora_sft`（通过vLLM服务，LoRA）

### 5.2 综合准确率计算

**综合准确率定义**:

综合准确率 = 语义相似度+关键词加权评分 + RAGas指标的综合得分

**原项目实现** (`final_score.py:165-168, 197-204`):
```python
# 1. 计算语义相似度+关键词加权得分
results = report_score(result)
semantic_keyword_score = np.mean([item["score"] for item in results])
print(f"预测问题数：{len(results)}, 语义相似度+关键词加权得分：{semantic_keyword_score}")

# 2. 计算RAGas指标
ragas_result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), LLMContextPrecisionWithReference()],
    llm=evaluator_llm
)
# ragas_result 是一个RAGas评估结果对象，包含多个指标的DataFrame
# 可以通过 to_pandas() 转换为DataFrame，然后提取各指标的平均值
print(f"预测问题数：{len(results)}, LLM+RAGas综合得分：{ragas_result}")
```

**注意**: 
- 原项目中的 `result` 变量（第197行）是RAGas评估结果对象，**不是**一个简单的综合得分
- 它包含 `context_recall` 和 `context_precision` 两个指标的详细结果（每个样本的得分）
- 打印时直接打印整个对象，所以看起来不太清晰
- 原项目只使用了 `ContextRecall` 和 `ContextPrecision` 两个指标，没有使用 `AnswerRelevancy`

**新项目实现（结合RAGas分数）**:

**方案1：使用ContextRecall和ContextPrecision（默认方案）**
```python
# 1. 对每个测试样本计算语义相似度+关键词加权得分
semantic_keyword_scores = []
for item in test_data:
    pred = item["generated_answer"]
    gold = item["ground_truth_answer"]
    keywords = item["keywords"]
    score_dict = calculate_semantic_keyword_score(pred, gold, keywords, semantic_model)
    semantic_keyword_scores.append(score_dict["final_score"])

# 计算语义相似度+关键词加权得分的平均值
avg_semantic_keyword_score = np.mean(semantic_keyword_scores)

# 2. 计算RAGas指标（ContextRecall和ContextPrecision的平均值，不启用AnswerRelevancy）
ragas_results = ragas_evaluator.evaluate_from_rag_results(test_data)
context_recall = ragas_results.get("context_recall", 0.0)
context_precision = ragas_results.get("context_precision", 0.0)
avg_ragas_score = (context_recall + context_precision) / 2.0

# 3. 结合两个分数计算综合准确率（默认权重0.7/0.3）
comprehensive_accuracy = 0.7 * avg_semantic_keyword_score + 0.3 * avg_ragas_score
```

**方案2：使用ContextRecall、ContextPrecision和AnswerRelevancy（更全面，但不使用）**
```python
# 注意：此方案已废弃，不使用AnswerRelevancy
# 仅保留方案1（2个RAGas指标）
```

**方案选择**:
- **使用方案1**（2个RAGas指标）：与原项目一致，评估检索上下文质量
- **不使用方案2**：不启用AnswerRelevancy，简化评估流程

**计算逻辑**:
1. **语义相似度+关键词加权得分**:
   - 对每个样本：`final_score = 0.2 * keyword_score + 0.8 * semantic_score`
   - 平均值：`avg_semantic_keyword_score = mean(all_final_scores)`

2. **RAGas指标**:
   - ContextRecall：检索上下文是否包含完整信息
   - ContextPrecision：检索上下文的相关性
   - 平均值：`avg_ragas_score = (context_recall + context_precision) / 2.0`
   - **注意**: 不启用AnswerRelevancy

3. **综合准确率**（默认权重）:
   - 加权结合：`comprehensive_accuracy = 0.7 * avg_semantic_keyword_score + 0.3 * avg_ragas_score`
   - 权重可配置（通过YAML配置文件），但默认使用0.7/0.3

**目标**: 综合准确率 ≥ 90%（**理想目标**，允许作为阶段性结果）

**说明**: 
- 语义相似度+关键词加权得分主要评估答案质量
- RAGas指标主要评估检索上下文的质量
- 两者结合可以更全面地评估RAG系统的性能
- 如果最终达不到目标值（如86%），需要在报告中给出解释，允许作为阶段性结果

**评估指标组合**:

1. **主要指标**: 语义相似度+实体关键词加权评分（综合准确率）
   - 权重：20%关键词 + 80%语义相似度
   - 目标：≥ 90%

2. **辅助指标**: RAGas框架评估
   - ContextRecall：≥ 0.85
   - ContextPrecision：≥ 0.80

3. **其他指标**:
   - BLEU：≥ 0.40
   - ROUGE-L：≥ 0.60
   - 引用准确率：≥ 0.75

### 5.3 与Qwen基线系统对比（外挂知识库方案）

**对比基线系统配置**:

- **LLM**: `Qwen/Qwen3-32B`（通过siliconflow调用，未微调）
- **Embedding模型**: `Qwen/Qwen3-Embedding-8B`（通过siliconflow调用）
- **向量数据库**: 使用Qwen Embedding构建的向量索引（Milvus）
- **检索方式**: 仅使用向量检索（无BM25混合，无重排序）
- **生成方式**: 直接使用Qwen3-32B生成答案（无后处理）
- **Reranker**: 不使用重排序器
- **底层文档**: 与我们的系统使用相同的底层文档集合（保证对比公平性）
- **注意**: Embedding模型和LLM都与我们的系统不同，这是我们要对比的场景

**我们的系统配置**:

- **LLM**: `models/finetuned/qwen3_lora_sft`（微调后，通过vLLM服务）
- **Embedding模型**: 使用现有的Milvus向量库（与基线使用不同的embedding模型）
- **检索方式**: BM25 + Milvus混合检索 + 重排序
- **Reranker**: `models/finetuned/bge_reranker`（微调后）
- **底层文档**: 与基线系统使用相同的底层文档集合（保证对比公平性）

**对比实验设置**:

1. **使用相同测试集**
2. **评估指标**: 
   - 语义相似度+实体关键词加权评分（综合准确率）
   - RAGas指标（ContextRecall、ContextPrecision）
   - BLEU、ROUGE等传统指标
3. **性能对比**:
   - 计算相对提升：`improvement = (our_score - baseline_score) / baseline_score * 100%`
   - 目标：相对提升 ≥ 18%

**对比实验实现**:

**新项目文件**: `src/evrag/evaluation/qwen_baseline_comparison.py`

**注意**: 文件名使用 `qwen_baseline_comparison.py` 以避免与现有的 `rag_comparison.py`（微调前后对比）冲突

**主要功能**:
- 实现Qwen3-32B+Qwen3-Embedding-8B基线系统
- 使用相同的测试集进行评估
- 计算性能提升百分比
- 生成对比报告

**基线系统实现要点**:
```python
# 基线系统评估流程
def evaluate_qwen_baseline_system(test_data):
    # 1. 初始化Qwen3-Embedding-8B模型（通过siliconflow API）
    # 2. 使用Qwen Embedding对知识库文档进行向量化（如果尚未构建）
    #    注意：使用与我们的系统相同的底层文档集合
    # 3. 构建Milvus向量索引
    # 4. 对每个问题：
    #    - 使用Qwen Embedding对问题进行编码（通过siliconflow）
    #    - 使用向量检索获取Top-K文档（仅Milvus，无BM25，无重排序）
    #    - 使用Qwen3-32B生成答案（通过siliconflow，无后处理）
    # 5. 计算评估指标
    # 6. 返回评估结果
```

**对比报告内容**:
- 各项指标的绝对值和相对提升
- 性能提升分析（哪些方面提升最大）
- 响应时间对比
- 组件贡献度分析（混合检索、重排序、微调LLM的贡献）

**输出路径**:

- **基线系统结果**: `rag_test_reports/qwen_baseline_comparison/baseline/`
- **我们的系统结果**: `rag_test_reports/qwen_baseline_comparison/our_system/`
- **对比报告**: `rag_test_reports/qwen_baseline_comparison/comparison_results.json`
- **对比图表**: `rag_test_reports/qwen_baseline_comparison/plots/`

**注意**: 统一使用 `rag_test_reports/` 作为顶层目录，避免与现有的 `reports/` 冲突

## 六、可视化图表生成

### 6.1 图表生成模块

**新项目文件**: `src/evrag/evaluation/visualization.py`（**全新创建**）

**需要生成的图表**（按优先级排序）:

**优先级1（必做）**:

1. **检索准确率对比图** (`plot_retrieval_accuracy`)
   - BM25 vs Milvus vs 混合检索（柱状图）
   - X轴：Top-K值
   - Y轴：准确率

2. **重排序效果对比图** (`plot_reranking_comparison`)
   - 微调前后reranker性能对比（折线图）
   - 指标：NDCG@K、MRR、Precision@K

3. **端到端性能对比图** (`plot_end_to_end_performance`)
   - 不同配置下的整体性能（热力图）
   - 行：不同组件组合
   - 列：不同评估指标

**优先级2（可选，有时间再做）**:

4. **响应时间分布图** (`plot_response_time_distribution`)
   - 各阶段耗时分析（堆叠柱状图）
   - 检索时间、重排序时间、生成时间

5. **准确率-召回率曲线** (`plot_pr_curve`)
   - PR曲线、ROC曲线（如果适用）

6. **失败案例分析图** (`plot_error_analysis`)
   - 错误类型分布（饼图）
   - 错误类型：检索失败、重排序失败、生成失败、引用错误等

7. **性能热力图** (`plot_performance_heatmap`)
   - 不同参数组合的性能热力图
   - 行：参数组合
   - 列：评估指标

8. **组件贡献度分析图** (`plot_component_contribution`)
   - 各组件对最终性能的贡献（雷达图）
   - 维度：检索、重排序、生成

9. **检索效果可视化** (`visualize_retrieval_results`)
   - Top-K检索结果示例展示
   - 展示检索到的文档和相关性分数

**实现要点**:

- 使用matplotlib和seaborn生成图表
- 所有图表标签使用英文
- 支持保存为PNG和PDF格式
- 图表样式统一、美观

## 七、命令行接口

### 7.1 在main.py中添加命令

**命令1**: `evaluate-rag`

**参数**:

- `--test-data`: 测试数据路径（默认：`data/qa_pairs/test_qa_pair_verify.json`）
- `--output-dir`: 评估结果输出目录（默认：`rag_test_reports/rag_evaluation/`）
- `--config`: 评估配置文件路径（默认：`config/evaluation/rag_evaluation.yaml`）
- `--sample-size`: 抽样测试样本数（默认：30，用于快速验证）
- `--full-test`: 是否执行全量测试（默认：False，需要显式指定）
- `--baseline-reranker`: 基线Reranker模型路径（覆盖配置文件）
- `--finetuned-reranker`: 微调后Reranker模型路径（覆盖配置文件）
- `--baseline-llm-url`: 基线LLM的vLLM服务地址（覆盖配置文件）
- `--baseline-llm-model`: 基线LLM模型名称（覆盖配置文件）
- `--finetuned-llm-url`: 微调后LLM的vLLM服务地址（覆盖配置文件）
- `--finetuned-llm-model`: 微调后LLM模型名称（覆盖配置文件）
- `--bm25-topk`: BM25检索数量（覆盖配置文件）
- `--milvus-topk`: Milvus检索数量（覆盖配置文件）
- `--reranker-topk`: 重排序数量（覆盖配置文件）
- `--use-ragas`: 是否使用RAGas框架评估（覆盖配置文件）
- `--use-semantic`: 是否使用语义相似度评估（覆盖配置文件）
- `--max-workers`: 最大并发工作线程数（默认：自动探测vLLM并发度上限）

**命令2**: `ablation-study`

**参数**:

- `--test-data`: 测试数据路径（默认：`data/qa_pairs/test_qa_pair_verify.json`）
- `--output-dir`: 结果输出目录（默认：`rag_test_reports/ablation/`）
- `--config`: 消融实验配置文件路径（默认：`config/evaluation/ablation_configs.yaml`）
- `--sample-size`: 抽样测试样本数（默认：30）
- `--full-test`: 是否执行全量测试（默认：False）

**命令3**: `tune-parameters`

**参数**:

- `--test-data`: 测试数据路径（默认：`data/qa_pairs/test_qa_pair_verify.json`）
- `--output-dir`: 结果输出目录（默认：`rag_test_reports/parameter_tuning/`）
- `--config`: 参数调优配置文件路径（默认：`config/evaluation/parameter_tuning.yaml`）
- `--sample-size`: 抽样测试样本数（默认：30）
- `--full-test`: 是否执行全量测试（默认：False）

**命令4**: `plot-rag-metrics`

**参数**:

- `--results-dir`: 评估结果目录（必需）
- `--output-dir`: 图表输出目录（默认：`rag_test_reports/plots/`）
- `--metrics`: 要绘制的指标（默认：all，可选：retrieval, reranking, end_to_end）

**命令5**: `compare-with-qwen-baseline`

**参数**:

- `--test-data`: 测试数据路径（默认：`data/qa_pairs/test_qa_pair_verify.json`）
- `--output-dir`: 对比结果输出目录（默认：`rag_test_reports/qwen_baseline_comparison/`）
- `--config`: 对比配置文件路径（默认：`config/evaluation/qwen_baseline_comparison.yaml`）
- `--sample-size`: 抽样测试样本数（默认：30）
- `--full-test`: 是否执行全量测试（默认：False）
- `--baseline-embedding-model`: 基线Embedding模型名称（覆盖配置文件，默认：`Qwen/Qwen3-Embedding-8B`）
- `--baseline-llm-model`: 基线LLM模型名称（覆盖配置文件，默认：`Qwen/Qwen3-32B`）
- `--baseline-siliconflow-api-key`: 基线系统siliconflow API密钥（覆盖配置文件）
- `--baseline-milvus-topk`: 基线系统向量检索数量（覆盖配置文件，默认：10）
- `--our-reranker`: 我们的Reranker模型路径（覆盖配置文件）
- `--our-llm-url`: 我们的LLM的vLLM服务地址（覆盖配置文件）
- `--our-llm-model`: 我们的LLM模型名称（覆盖配置文件）
- `--bm25-topk`: BM25检索数量（覆盖配置文件）
- `--milvus-topk`: Milvus检索数量（覆盖配置文件）
- `--reranker-topk`: 重排序数量（覆盖配置文件）
- `--use-ragas`: 是否使用RAGas框架评估（覆盖配置文件）

## 八、文件结构

```
src/evrag/evaluation/
├── __init__.py
├── rag_evaluator.py              # RAG评估核心模块
├── ragas_evaluator.py            # RAGas框架评估
├── metrics.py                    # 评估指标计算
├── ablation_study.py             # 消融实验
├── parameter_tuning.py           # 参数调优
├── visualization.py              # 可视化图表生成
├── report_generator.py           # 评估报告生成
├── rag_comparison.py             # 微调前后对比（已存在）
└── qwen_baseline_comparison.py   # 与Qwen基线系统对比（新增）

config/evaluation/
├── rag_evaluation.yaml           # RAG评估配置
├── ablation_configs.yaml         # 消融实验配置
├── parameter_tuning.yaml         # 参数调优配置
└── qwen_baseline_comparison.yaml # Qwen基线对比配置（新增）

rag_test_reports/
├── rag_evaluation/               # RAG评估结果
│   ├── baseline/
│   ├── finetuned/
│   ├── ablation/
│   ├── parameter_tuning/
│   └── comparison/
├── rag_comparison/               # 微调前后对比结果
│   ├── baseline/
│   ├── finetuned/
│   └── comparison/
└── qwen_baseline_comparison/     # Qwen基线对比结果
    ├── baseline/                 # Qwen基线系统评估结果
    │   ├── evaluation_results.json
    │   ├── evaluation_report.md
    │   └── plots/
    ├── our_system/               # 我们的系统评估结果
    │   ├── evaluation_results.json
    │   ├── evaluation_report.md
    │   └── plots/
    └── comparison/               # 对比分析结果
        ├── comparison_results.json
        ├── comparison_report.md
        └── plots/
```

## 九、测试数据集构造

### 9.1 测试集规模

- **统一测试集**: `data/qa_pairs/test_qa_pair_verify.json`（约750条，**统一使用此测试集**）
- **总测试集**: `data/qa_pairs/test_qa_pair.json`（约2000条，不使用，因为其中质量qa较多）
- **测试集来源**: 人工构造的问答对，覆盖Tesla Model 3用户手册的各个章节
- **数据分布**:
  - 正样本：约675条（90%）
  - 负样本（无答案）：约75条（10%）

### 9.2 测试集构造方法

**参考原项目**: `/remote-home/share/liangZhang/EVRAG/src/gen_qa/run.py`

**构造流程**:

1. **正样本构造**:
   - 从PDF文档中提取结构化内容
   - 基于文档内容生成问题-答案对
   - 使用NER模型提取实体关键词
   - 每个样本包含：`question`、`answer`、`keywords`、`unique_id`

2. **负样本构造**:
   - 从通用对话数据（`raw_general_chats.txt`）中采样
   - 答案设置为"无答案"
   - 关键词字段为空数组
   - 用于测试系统对无关问题的处理能力

3. **数据分布**:
   - 正样本：约1800条（90%）
   - 负样本：约200条（10%）
   - 覆盖车辆操作、维修保养、故障排除、安全提示等多个领域

### 9.3 测试数据格式

**测试数据格式** (`data/qa_pairs/test_qa_pair.json`):

```json
[
    {
        "unique_id": "aa36d1d7b03d307b974dae437b81b283",
        "question": "后电机的标识牌安装在车辆哪里？",
        "answer": "后电机标签位于后备箱。",
        "keywords": ["后电机标签"]
    },
    {
        "unique_id": "xxx...",
        "question": "通用对话问题",
        "answer": "无答案",
        "keywords": []
    },
    ...
]
```

**字段说明**:
- `unique_id`: 唯一标识符（MD5哈希）
- `question`: 用户问题
- `answer`: 标准答案（或"无答案"）
- `keywords`: 实体关键词列表（用于关键词匹配评分）

## 十、依赖和工具

### 10.1 新增依赖

- `ragas>=0.1.0`: RAGas评估框架
- `text2vec` 或 `sentence-transformers`: 语义相似度计算（已安装）
- `langchain-openai`: RAGas框架的LLM包装器（可能需要）

### 10.2 可选依赖

- `plotly`: 交互式图表（可选）
- `scikit-learn`: 用于PR曲线等（已安装）

## 十一、实现细节

### 11.1 RAG评估流程

1. **初始化组件**:

   - BM25检索器
   - Milvus检索器
   - Reranker（基线或微调）
   - LLM客户端（基线或微调，通过vLLM）

2. **执行评估**:

   - 对每个测试样本：
     - 记录开始时间
     - BM25检索（记录时间）
     - Milvus检索（记录时间）
     - 合并文档（记录时间）
     - 重排序（记录时间）
     - LLM生成（记录时间）
     - 后处理
     - 计算各项指标

3. **计算评估指标**:

   - 检索准确率
   - 重排序效果
   - 生成质量（BLEU、ROUGE、语义相似度）
   - 引用准确率
   - 响应时间

4. **RAGas评估**（可选）:

   - 构建评估数据集
   - 使用RAGas框架评估
   - 获取ContextRecall和ContextPrecision

### 11.2 消融实验流程

1. **定义组件组合**:

   - 配置不同的组件组合（BM25、Milvus、Reranker）

2. **执行评估**:

   - 对每个组合执行完整的评估流程
   - 记录所有指标

3. **生成对比报告**:

   - 对比不同组合的性能
   - 分析各组件的贡献

### 11.3 参数调优流程

1. **定义参数空间**:

   - 配置需要调优的参数及其取值范围

2. **网格搜索或随机搜索**:

   - 遍历所有参数组合
   - 对每个组合执行评估

3. **选择最优参数**:

   - 根据评估指标选择最优参数组合
   - 生成参数性能热力图

### 11.4 微调前后对比流程

1. **基线模型评估**:

   - 使用基线Reranker和基线LLM进行评估

2. **微调模型评估**:

   - 使用微调Reranker和微调LLM进行评估

3. **对比分析**:

   - 计算各项指标的改进
   - 生成对比报告和图表

### 11.5 综合准确率评估流程

1. **执行端到端RAG流程**:
   - 对test_qa_pair_verify中测试数据执行完整的RAG流程
   - 记录每个样本的生成答案

2. **计算语义相似度+关键词加权评分**:
   - 对每个样本计算：
     - 语义相似度得分（使用text2vec）
     - 实体关键词匹配得分（Jaccard相似度）
     - 加权综合得分：`0.2 * keyword_score + 0.8 * semantic_score`
   - 计算平均值：`comprehensive_accuracy = mean(all_scores)`

3. **RAGas框架评估**:
   - 构建RAGas评估数据集
   - 使用LLM评估ContextRecall和ContextPrecision
   - 计算平均得分

4. **生成评估报告**:
   - 综合准确率：语义相似度+关键词加权评分的平均值
   - RAGas指标：ContextRecall、ContextPrecision的平均值
   - 其他指标：BLEU、ROUGE、引用准确率等

### 11.6 与Qwen基线系统对比流程

1. **构建基线系统**:
   - 初始化Qwen3-Embedding-8B模型（通过siliconflow API调用）
   - 使用Qwen Embedding对知识库文档进行向量化（如果尚未构建）
   - **注意**: 使用与我们的系统相同的底层文档集合（保证对比公平性）
   - 构建Milvus向量索引
   - 配置Qwen3-32B的siliconflow API客户端

2. **执行基线评估**:
   - 对test_qa_pair_verify测试数据（750条）：
     - 使用Qwen Embedding对问题进行编码（通过siliconflow）
     - 使用向量检索获取Top-K文档（仅Milvus，无BM25混合，无重排序）
     - 使用Qwen3-32B生成答案（通过siliconflow，无后处理）
   - 计算评估指标（语义相似度+关键词评分、RAGas指标等）
   - **抽样测试**: 先抽样30条测试，确保程序执行正确后再全量测试

3. **执行我们的系统评估**:
   - 使用相同的750条测试数据
   - 执行完整的RAG流程：
     - BM25检索 + Milvus检索（混合检索）
     - 使用微调后的Reranker进行重排序
     - 使用微调后的LLM生成答案（通过vLLM服务）
     - 后处理（引用提取等）
   - 计算评估指标
   - **抽样测试**: 先抽样30条测试，确保程序执行正确后再全量测试

4. **计算性能提升**:
   - 主要指标提升：`improvement = (our_comprehensive_accuracy - baseline_comprehensive_accuracy) / baseline_comprehensive_accuracy * 100%`
   - 目标：≥ 18%相对提升（**理想目标**，允许作为阶段性结果）
   - 其他指标对比：ContextRecall、ContextPrecision、BLEU、ROUGE等

5. **生成对比报告**:
   - 指标对比表格
   - 性能提升分析
   - 组件贡献度分析（混合检索、重排序、微调LLM的贡献）
   - 响应时间对比

**注意**: 
- 基线系统使用siliconflow API调用Qwen3-32B和Qwen3-Embedding-8B（未微调）
- 我们的系统使用本地vLLM服务（微调后的LLM）和本地Reranker（微调后）
- 两者使用相同的底层文档集合，但Embedding模型和LLM都不同（这是我们要对比的场景）

## 十二、评估指标详细说明

### 12.1 语义相似度+实体关键词加权评分

**计算公式**:
```
final_score = 0.2 * keyword_score + 0.8 * semantic_score
```

**其中**:
- `semantic_score`: 使用text2vec计算的标准答案与生成答案的余弦相似度
- `keyword_score`: Jaccard相似度，计算公式为 `len(join_keywords) / (len(keywords) + 1e-6)`
  - `join_keywords`: 生成答案中包含的关键词列表
  - 如果 `keyword_score > 0.3`，返回1，否则返回0

**综合准确率计算**:

综合准确率需要结合两个部分：

1. **语义相似度+关键词加权得分**:
```
avg_semantic_keyword_score = mean([final_score for each sample in test_set])
```

2. **RAGas指标平均值**（使用2个指标，不启用AnswerRelevancy）:
```python
# 使用2个指标（默认，不启用AnswerRelevancy）
avg_ragas_score = (context_recall + context_precision) / 2.0
```

**RAGas指标说明**:
- **ContextRecall**: 评估检索到的上下文是否包含回答问题所需的所有相关信息
- **ContextPrecision**: 评估检索到的上下文中与问题相关的比例
- **AnswerRelevancy**: 不启用（简化评估流程）

3. **最终综合准确率**（默认权重0.7/0.3）:
```
comprehensive_accuracy = 0.7 * avg_semantic_keyword_score + 0.3 * avg_ragas_score
```

**目标**: `comprehensive_accuracy ≥ 0.90` (90%)（**理想目标**，允许作为阶段性结果）

**说明**: 
- 默认权重为0.7/0.3，可通过YAML配置文件调整
- 如果RAGas评估不可用，可以仅使用语义相似度+关键词加权得分
- 如果最终达不到目标值（如86%），需要在报告中给出解释，允许作为阶段性结果

### 12.2 RAGas评估指标

**ContextRecall（上下文召回率）**:
- 评估检索到的上下文是否包含回答问题所需的所有相关信息
- 使用LLM判断：给定问题和参考答案，检索到的上下文是否完整
- 计算公式：`ContextRecall = (包含完整信息的样本数) / (总样本数)`
- 得分范围：[0, 1]，值越大表示上下文越完整
- 目标值：≥ 0.85

**ContextPrecision（上下文精确率）**:
- 评估检索到的上下文中与问题相关的比例
- 使用LLM判断：给定问题、参考答案和生成答案，检索到的上下文的相关性
- 计算公式：`ContextPrecision = (相关上下文数) / (总检索上下文数)`
- 得分范围：[0, 1]，值越大表示上下文越相关
- 目标值：≥ 0.80

**AnswerRelevancy（答案相关性）**（不启用）:
- 此指标不启用，简化评估流程
- 仅使用ContextRecall和ContextPrecision两个指标

### 12.3 性能提升计算

**相对提升**:
```
relative_improvement = (our_score - baseline_score) / baseline_score * 100%
```

**目标**: 相对于Qwen基线系统，综合准确率提升 ≥ 18%

**示例计算**:
- 基线系统综合准确率：76%
- 我们的系统综合准确率：90%
- 相对提升：`(90 - 76) / 76 * 100% = 18.4%`

### 12.4 综合准确率计算实现细节

**实现位置**: `src/evrag/evaluation/rag_comparison.py`（**全新创建**）

**计算流程**:

1. **获取语义相似度+关键词加权得分**:
   ```python
   baseline_semantic = baseline_metrics.get("semantic_keyword_score", {}).get("final_score", {}).get("mean", 0.0)
   finetuned_semantic = finetuned_metrics.get("semantic_keyword_score", {}).get("final_score", {}).get("mean", 0.0)
   ```

2. **获取RAGas指标**:
   ```python
   # 从RAGas评估结果中获取（仅使用2个指标，不启用AnswerRelevancy）
   ragas_scores = {
       "context_recall": ragas_results.get("context_recall", 0.0),
       "context_precision": ragas_results.get("context_precision", 0.0),
   }
   ```

3. **计算RAGas平均得分**:
   ```python
   # 使用2个指标（默认，不启用AnswerRelevancy）
   avg_ragas_score = (context_recall + context_precision) / 2.0
   ```

4. **计算综合准确率**（默认权重0.7/0.3）:
   ```python
   comprehensive_accuracy = 0.7 * avg_semantic_keyword_score + 0.3 * avg_ragas_score
   ```
   - 权重可通过YAML配置文件调整，但默认使用0.7/0.3

**在对比报告中的展示**:
- 基线综合准确率 vs 微调综合准确率
- 绝对改进和相对改进百分比
- 组件得分分解（语义+关键词得分、RAGas得分）

### 12.5 评估指标优化说明

**已优化的指标**:

1. **exact_match（精确匹配）**:
   - 优化前：严格字符串比较，去除标点后仍可能失败
   - 优化后：去除标点符号和空白字符后比较，更宽松和准确
   - 实现：使用正则表达式 `re.sub(r'[。，、；：！？\s]+', '', text.strip())`

2. **semantic_similarity（语义相似度）**:
   - 优化前：当标准答案和生成答案都是"无答案"时返回0.0
   - 优化后：当两者都是"无答案"时返回1.0（逻辑上应该完全匹配）
   - 实现：在 `calculate_generation_quality` 中添加特殊处理

3. **ROUGE指标**:
   - 优化前：依赖 `rouge_score` 库，对中文支持不佳
   - 优化后：使用自定义的中文ROUGE计算函数（基于jieba分词）
   - 实现：`_calculate_rouge_chinese` 函数，支持中文文本的ROUGE-1、ROUGE-2、ROUGE-L计算

4. **RAGas指标**:
   - 新增：支持 `AnswerRelevancy` 指标（可选）
   - 实现：在 `RAGasEvaluator.evaluate` 中添加 `include_answer_relevancy` 参数

**测试验证**:
- 所有指标已通过真实测试用例验证（见 `test_metrics_with_real_cases.py`）
- 测试用例包括：完全匹配、部分匹配、高度相似、无答案情况等
- 所有测试用例均通过 ✓

## 十三、测试执行计划

### 13.1 测试阶段划分（里程碑拆分）

**Phase 5 拆分为以下里程碑**:

1. **里程碑1: 评估核心模块**（2-3小时）
   - 实现 `rag_evaluator.py`（RAG评估核心模块）
   - 实现 `metrics.py`（评估指标计算）
   - 实现基本的端到端评估流程
   - 抽样30条测试，验证程序执行正确性

2. **里程碑2: 评估指标完善**（2-3小时）
   - 完善所有评估指标（检索准确率、重排序效果、生成质量等）
   - 实现语义相似度+关键词加权评分
   - 实现无答案样本的单独统计
   - 抽样30条测试，验证指标计算正确性

3. **里程碑3: RAGas框架集成**（2-3小时）
   - 实现 `ragas_evaluator.py`
   - 集成RAGas框架（ContextRecall、ContextPrecision）
   - 配置Doubao/OpenAI兼容接口（通过siliconflow）
   - 抽样30条测试，验证RAGas评估正确性

4. **里程碑4: Qwen基线对比**（3-4小时）
   - 实现 `qwen_baseline_comparison.py`
   - 实现Qwen基线系统（Qwen3-32B + Qwen3-Embedding-8B，通过siliconflow）
   - 执行对比评估
   - 计算性能提升百分比

5. **里程碑5: 可视化图表**（2-3小时）
   - 实现 `visualization.py`
   - 生成优先级1图表（检索准确率对比图、重排序对比图、端到端性能热力图）
   - 生成评估报告

6. **里程碑6: 消融实验和参数调优**（2-3小时）
   - 实现 `ablation_study.py`
   - 实现 `parameter_tuning.py`
   - 执行消融实验和参数调优

7. **里程碑7: CLI接口和文档**（1-2小时）
   - 在 `main.py` 中添加所有评估命令
   - 创建配置文件模板
   - 完善文档

**总时间估算**: 14-21小时（约2-3个工作日，但目标是一天内完成）

**执行策略**:
- 每个里程碑完成后进行抽样测试（30条）
- 确保程序执行正确后再进行全量测试
- 优先完成核心功能，可选功能（如优先级2图表）可延后

### 13.2 测试数据准备

1. **确认测试集规模**:
   - 检查 `data/qa_pairs/test_qa_pair_verify.json` 
   - 验证数据格式和字段完整性

2. **数据质量检查**:
   - 检查是否有重复的unique_id
   - 检查关键词字段是否完整
   - 检查答案字段是否有效

3. **数据分布分析**:
   - 统计正样本和负样本数量
   - 分析问题类型分布
   - 分析关键词数量分布

## 十四、模块说明

### 14.1 对比模块区分

**现有模块** (`rag_comparison.py`):
- **用途**: 微调前后对比
- **对比内容**: 
  - 基线Reranker vs 微调Reranker
  - 基线LLM vs 微调LLM
- **输出路径**: `reports/rag_comparison/`

**新增模块** (`qwen_baseline_comparison.py`):
- **用途**: 与Qwen基线系统对比（外挂知识库方案）
- **对比内容**:
  - Qwen3-32B+Qwen3-Embedding-8B（仅向量检索，无重排序，无微调）
  - vs 我们的系统（混合检索+重排序+微调LLM）
- **输出路径**: `reports/qwen_baseline_comparison/`

### 14.2 基线系统配置说明

**Qwen基线系统**:
- LLM: `Qwen/Qwen3-32B`（未微调，通过vLLM服务）
- Embedding: `Qwen/Qwen3-Embedding-8B`（本地部署）
- 检索: 仅Milvus向量检索（Top-K=10）
- 重排序: 无
- 后处理: 无

**我们的系统**:
- LLM: `models/finetuned/qwen3_lora_sft`（微调后，通过vLLM服务）
- Embedding: 使用现有的Milvus向量库
- 检索: BM25 + Milvus混合检索
- 重排序: `models/finetuned/bge_reranker`（微调后）
- 后处理: 引用提取等

## 十五、部署脚本优化

### 15.1 vLLM显存优化配置

**问题**: vLLM服务占用显存过多（GPU 0和1各占用44-45GB），导致评估时显存不足，无法进行大量并发评估。

**优化方案**: 调整 `scripts/start_vllm_for_evaluation.sh`，添加显存优化参数。

**优化参数说明**:

1. **`--gpu-memory-utilization`** (默认: 0.75)
   - GPU显存利用率（0-1）
   - 默认vLLM使用0.9（90%），占用过多显存
   - 降低到0.75可以留出更多显存给KV cache，提高并发能力
   - **效果**: 减少约15%的显存占用，可以支持更多并发请求

2. **`--max-model-len`** (默认: 4096)
   - 最大模型长度，限制KV cache大小
   - Qwen3-8B默认8192，对于RAG任务4096通常足够
   - **效果**: 减少约50%的KV cache显存占用

3. **`--max-num-seqs`** (默认: 32)
   - 最大并发序列数
   - 从16增加到32，提高并发能力
   - **效果**: 支持更多并发请求，减少评估时间

4. **`--swap-space`** (默认: 4)
   - CPU交换空间（GiB）
   - 当GPU显存不足时使用CPU内存作为交换
   - **效果**: 提供额外的缓冲空间

**优化后的启动命令**:
```bash
CUDA_VISIBLE_DEVICES=$BASELINE_GPU_IDS nohup vllm serve "$BASELINE_MODEL_PATH" \
    --tensor-parallel-size $BASELINE_TP_SIZE \
    --port $BASELINE_PORT \
    --host 0.0.0.0 \
    --trust-remote-code \
    --served-model-name "$BASELINE_MODEL_NAME" \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.75 \
    --max-model-len 4096 \
    --swap-space 4 \
    > "$BASELINE_LOG" 2>&1 &
```

**预期效果**:
- GPU显存占用从44-45GB降低到约35-38GB
- 可以支持更多并发请求（从16增加到32）
- 评估时间显著减少（并发能力提升）

### 15.2 重新部署vLLM服务

**步骤**:
1. 停止当前vLLM服务：`./scripts/stop_vllm_for_evaluation.sh`
2. 确认服务已停止：`nvidia-smi` 查看GPU显存释放
3. 重新启动服务：`./scripts/start_vllm_for_evaluation.sh`
4. 验证服务正常：`curl http://localhost:8000/health` 和 `curl http://localhost:8001/health`
5. 检查显存使用：`nvidia-smi` 确认显存占用降低

**注意事项**:
- 如果评估时仍然出现显存不足，可以进一步降低 `gpu-memory-utilization` 到 0.7
- 如果评估任务不需要很长的上下文，可以进一步降低 `max-model-len` 到 2048
- 可以根据实际需求调整 `max-num-seqs`，但要注意显存限制

## 十六、CLI框架和配置文件

### 16.1 CLI框架

**使用现有框架**: 使用 `main.py` 中已有的 `typer` CLI框架。

**命令命名规范**: 使用 `kebab-case`（如 `evaluate-rag`、`ablation-study`）。

**参数优先级**: CLI参数 > YAML配置文件 > 默认值

### 16.2 配置文件结构

**配置文件位置**: `config/evaluation/`

**配置文件列表**:
- `rag_evaluation.yaml`: RAG评估配置
- `ablation_configs.yaml`: 消融实验配置
- `parameter_tuning.yaml`: 参数调优配置
- `qwen_baseline_comparison.yaml`: Qwen基线对比配置

**配置项示例** (`rag_evaluation.yaml`):
```yaml
# RAG评估配置
test_data: "data/qa_pairs/test_qa_pair_verify.json"
output_dir: "rag_test_reports/rag_evaluation"
sample_size: 30  # 抽样测试样本数

# 模型配置
baseline_reranker: "models/bge-reranker-v2-m3"
finetuned_reranker: "models/finetuned/bge_reranker"
baseline_llm_url: "http://localhost:8000/v1"
baseline_llm_model: "Qwen3-8B"
finetuned_llm_url: "http://localhost:8001/v1"
finetuned_llm_model: "qwen3_lora_sft"

# 检索参数
bm25_topk: 5
milvus_topk: 10
reranker_topk: 5

# 评估配置
use_ragas: true
use_semantic: true
comprehensive_accuracy_weights:
  semantic_keyword: 0.7
  ragas: 0.3

# 并发配置
max_workers: null  # null表示自动探测vLLM并发度上限
```

## 十七、参考文档

- 原项目评估脚本：`EVRAG/final_score.py`
- 原项目测试数据构造：`EVRAG/src/gen_qa/run.py`
- RAGas框架文档：https://docs.ragas.io/
- 当前项目RAG实现：`EvRAG/main.py:612-692` (infer命令)
- 当前项目工具函数：`EvRAG/src/evrag/tool_func.py`
- 当前项目评估模块：`EvRAG/src/evrag/evaluation/`（**全新创建**，目录下目前无文件）
- Qwen模型文档：https://github.com/QwenLM/Qwen
- SiliconFlow API文档：https://siliconflow.cn/