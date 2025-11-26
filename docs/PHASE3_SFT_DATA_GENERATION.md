# Phase 3: SFT数据生成阶段总结

## 一、阶段概述

本阶段实现了完整的SFT（Supervised Fine-Tuning）数据生成流程，包括：
1. **train_data.json** - RAG检索数据生成
2. **summary_data** - SFT训练数据生成
3. **rerank_data** - Reranker训练数据生成
4. **test_qa_pair_verify.json** 和 **test_qa_pair_pred.json** - 测试集验证和预测数据生成
5. **数据分析和验证** - 数据统计、可视化和格式验证

## 二、实现流程

### 2.1 核心模块

#### 1. SFTDataGenerator (`src/evrag/gen_qa/sft_data_generator.py`)

**主要功能**：
- 从 `train_qa_pair.json` 生成 RAG 检索数据
- 生成 SFT 训练数据（summary_data）
- 生成 Reranker 训练数据（rerank_data）
- 生成测试集验证和预测数据

**关键方法**：
- `generate_train_data()`: 生成 train_data.json（RAG检索+LLM生成）
- `generate_summary_data()`: 生成 summary_data（SFT训练数据）
- `generate_rerank_data()`: 生成 rerank_data（Reranker训练数据）
- `generate_test_verify_data()`: 从测试集筛选验证集
- `generate_test_pred_data()`: 对验证集进行RAG预测

#### 2. SFTDataAnalyzer (`src/evrag/gen_qa/sft_data_analyzer.py`)

**主要功能**：
- 数据统计分析
- 可视化图表生成

**关键方法**：
- `analyze_summary_data()`: 分析SFT数据统计
- `analyze_rerank_data()`: 分析Reranker数据统计
- `generate_visualizations()`: 生成可视化图表

#### 3. SFTDataValidator (`src/evrag/gen_qa/sft_data_analyzer.py`)

**主要功能**：
- 数据格式验证
- 数据完整性检查
- 数据平衡性验证

**关键方法**：
- `validate_all()`: 验证所有数据
- `validate_train_data()`: 验证train_data.json格式
- `validate_summary_data()`: 验证summary_data格式
- `validate_rerank_data()`: 验证rerank_data格式
- `validate_data_balance()`: 验证数据平衡性
- `validate_data_completeness()`: 验证数据完整性

### 2.2 数据生成流程

```
train_qa_pair.json
    ↓ [RAG检索 + LLM生成]
train_data.json
    ↓ [解析和格式化]
summary_data/
    ├── train.json (SFT训练集)
    └── test.json (SFT测试集)
    ↓ [生成Reranker样本]
rerank_data/
    ├── train.json (Reranker训练集)
    ├── dev.json (Reranker开发集)
    └── test.json (Reranker测试集)
```

### 2.3 测试数据生成流程

```
test_qa_pair.json
    ↓ [筛选10%]
test_qa_pair_verify.json (验证集)
    ↓ [RAG预测]
test_qa_pair_pred.json (预测结果，用于评估)
```

## 三、完成的工作

### 3.1 核心功能实现

#### ✅ 1. train_data.json 生成
- 实现RAG检索流程（BM25 + Milvus + Reranker）
- 使用Deepseek API生成LLM响应
- 支持并发处理（ThreadPoolExecutor）
- 实现错误重试机制（连接错误自动重试3次）
- 实现响应清理（去除思考过程、占位符等）

#### ✅ 2. summary_data 生成
- 解析LLM响应，提取答案和引用标记
- 格式化instruction和output
- 切分训练集和测试集（TEST_RATE=0.08）
- 处理"无答案"情况

#### ✅ 3. rerank_data 生成
- 生成正样本（label=2）、中等样本（label=1）、负样本（label=0）
- 从训练集中切分dev集（RERANK_DEV_SIZE=1000）
- 处理"无答案"情况的特殊逻辑

#### ✅ 4. 测试数据生成
- 实现 `generate_test_verify_data()`: 从test_qa_pair.json筛选验证集
- 实现 `generate_test_pred_data()`: 对验证集进行RAG预测，生成预测结果

#### ✅ 5. 数据分析和可视化
- 实现数据统计分析（SFT数据和Reranker数据）
- 生成可视化图表：
  - SFT数据分布图（样本数量、Context长度、Output长度、引用率等）
  - Reranker标签分布图（饼图展示正/负/中等样本比例）

#### ✅ 6. 数据验证
- 实现格式验证（JSON格式、字段类型、必需字段）
- 实现数据平衡性验证（正负样本比例检查）
- 实现数据完整性验证（无缺失字段检查）

### 3.2 命令行接口

在 `main.py` 中添加了以下命令：

#### `generate-sft-data`
生成微调训练数据，支持以下步骤：
- `train_data`: 生成train_data.json
- `summary`: 生成summary_data
- `rerank`: 生成rerank_data
- `test_verify`: 生成test_qa_pair_verify.json
- `test_pred`: 生成test_qa_pair_pred.json
- `all`: 执行完整流程

**使用示例**：
```bash
# 完整流程
conda run -n evrag python main.py generate-sft-data --step all --output-dir data

# 分步执行
conda run -n evrag python main.py generate-sft-data --step train_data --workers 10
conda run -n evrag python main.py generate-sft-data --step summary
conda run -n evrag python main.py generate-sft-data --step rerank
```

#### `analyze-sft-data`
分析和验证SFT数据，支持以下操作：
- `stats`: 数据统计分析
- `validate`: 数据验证
- `all`: 执行所有分析和验证

**使用示例**：
```bash
conda run -n evrag python main.py analyze-sft-data --output-dir data --action all
```

### 3.3 技术特性

#### 1. LLM服务集成
- **Deepseek API**: 用于生成QA数据（`OpenAIClient(service="deepseek")`）
- **SiliconFlow API**: 用于Reranker数据生成（`SiliconFlowReranker`）
- **本地vLLM**: 用于推理（支持Qwen3-8B，带思考模式控制）

#### 2. 思考模式控制
- 数据生成时禁用思考模式（`enable_thinking=False`）
- 推理时可配置启用思考模式（`--enable-thinking`）
- 自动清理响应中的思考过程标签（`<think>`）

#### 3. 错误处理
- 连接错误自动重试（最多3次，指数退避）
- 数据格式验证和错误报告
- 断点续传支持（checkpoint机制）

#### 4. 性能优化
- 并发处理（ThreadPoolExecutor）
- 进度条显示（tqdm）
- 文件锁（用于并发写入）

## 四、数据统计结果

### 4.1 SFT数据（Summary Data）

| 指标 | 训练集 | 测试集 |
|------|--------|--------|
| 样本数量 | 6,406 | 516 |
| 平均Context长度 | 1,131.9 | 1,147.3 |
| 平均Output长度 | 103.4 | 103.6 |
| 引用率 | 92.3% | 94.6% |
| 无答案率 | 7.7% | 5.4% |

### 4.2 Reranker数据

| 指标 | 训练集 | 开发集 | 测试集 |
|------|--------|--------|--------|
| 样本数量 | 17,131 | 1,000 | 503 |
| 正样本率 | 32.5% | 32.5% | 0.0%* |
| 中等样本率 | 32.5% | 32.5% | 0.0%* |
| 负样本率 | 35.0% | 35.0% | 0.0%* |

*注：测试集存在格式问题，缺少label字段

### 4.3 数据验证结果

**格式验证**：
- ✅ train_data.json: 6,922/6,922 条有效
- ✅ summary_data/train.json: 6,406/6,406 条有效
- ✅ summary_data/test.json: 516/516 条有效
- ✅ rerank_data/train.json: 17,131/17,131 条有效
- ✅ rerank_data/dev.json: 1,000/1,000 条有效
- ❌ rerank_data/test.json: 格式错误（缺少label字段）

**数据平衡性**：
- ✅ 训练集和开发集：数据平衡（正/中/负样本比例合理）
- ⚠️ 测试集：数据不平衡（缺少label字段）

**数据完整性**：
- ✅ 除 rerank_data/test.json 外，所有文件数据完整

## 五、遇到的问题及解决方案

### 5.1 思考模式问题

**问题**：
- Qwen3-8B模型在生成响应时输出了思考过程（`<think>`标签）
- 响应中包含占位符（`{答案}`、`{从...}`）
- 响应格式不一致（有时包含"答案内容："前缀）

**解决方案**：
1. 在 `ChatClient` 中禁用思考模式（`enable_thinking=False`）
2. 在 `LocalLLMClient` 中通过 `extra_body` 传递 `chat_template_kwargs`
3. 在 `OpenAIClient` 中过滤 `enable_thinking` 参数（Deepseek不支持）
4. 改进响应清理逻辑，去除思考过程、占位符和前缀

**相关代码**：
- `src/evrag/client/chat_client.py`: 更新LLM_CHAT_PROMPT，禁用思考模式
- `src/evrag/client/local_client.py`: 通过extra_body传递enable_thinking
- `src/evrag/client/openai_client.py`: 过滤enable_thinking参数
- `src/evrag/gen_qa/sft_data_generator.py`: 改进响应清理逻辑

### 5.2 模型ID不匹配问题

**问题**：
- vLLM服务注册的模型ID是完整路径（`/remote-home/share/liangZhang/EvRAG/models/Qwen3-8B`）
- 客户端请求的模型ID是简化名称（`Qwen3-8B`）
- 导致404错误：`The model 'Qwen3-8B' does not exist`

**解决方案**：
1. 在 `start_vllm.sh` 中添加 `--served-model-name` 参数
2. 在 `config.yaml` 中使用完整路径作为 `local_llm_model_name`

**相关文件**：
- `scripts/start_vllm.sh`: 添加 `--served-model-name "$MODEL_NAME"`
- `config/config.yaml`: 更新 `local_llm_model_name` 为完整路径

### 5.3 Tensor并行度问题

**问题**：
- Qwen3-8B有32个attention heads
- 初始配置使用3个GPU（tensor_parallel_size=3）
- 导致错误：`Total number of attention heads (32) must be divisible by tensor parallel size (3)`

**解决方案**：
- 将GPU数量从3个改为2个（`vllm_gpu_ids=[3, 5]`）
- 更新 `start_vllm.sh` 中的 `TENSOR_PARALLEL_SIZE=2`

**相关文件**：
- `config/config.yaml`: 更新 `vllm_gpu_ids` 为 `[3, 5]`
- `scripts/start_vllm.sh`: 更新 `TENSOR_PARALLEL_SIZE=2`

### 5.4 Reranker数据格式问题

**问题**：
- `rerank_data/test.json` 缺少 `label` 字段
- 导致验证失败和数据不平衡警告

**原因分析**：
- 测试集中如果答案为"无答案"，不生成rerank数据
- 但实际生成的test.json文件可能包含其他格式的数据

**待解决**：
- 需要检查 `generate_rerank_data()` 方法中测试集生成逻辑
- 确保测试集数据也包含正确的label字段

### 5.5 导入路径问题

**问题**：
- 初始实现中 `merge_docs` 的导入路径不正确
- 从 `src.evrag.utils` 导入失败

**解决方案**：
- 确认 `merge_docs` 在 `src/evrag/tool_func.py` 中
- 更新导入路径为 `from ..tool_func import merge_docs`

**相关文件**：
- `src/evrag/gen_qa/sft_data_generator.py`: 更新导入路径

## 六、生成的文件

### 6.1 数据文件

```
data/
├── qa_pairs/
│   ├── train_data.json          # RAG检索数据（6,922条）
│   ├── test_qa_pair_verify.json # 验证集（从test_qa_pair.json筛选10%）
│   └── test_qa_pair_pred.json   # 预测结果（用于评估）
├── summary_data/
│   ├── train.json               # SFT训练集（6,406条）
│   └── test.json                # SFT测试集（516条）
└── rerank_data/
    ├── train.json               # Reranker训练集（17,131条）
    ├── dev.json                 # Reranker开发集（1,000条）
    └── test.json                # Reranker测试集（503条，存在问题）
```

### 6.2 报告文件

```
data/logs/
├── sft_data_statistics.json     # 统计数据报告
├── sft_data_validation.json     # 验证结果报告
└── plots/
    ├── sft_data_distribution.png      # SFT数据分布图
    └── rerank_label_distribution.png  # Reranker标签分布图
```

## 七、使用指南

### 7.1 生成SFT数据

```bash
# 完整流程（推荐）
conda run -n evrag python main.py generate-sft-data \
    --step all \
    --output-dir data \
    --workers 10

# 分步执行
# 步骤1：生成train_data.json
conda run -n evrag python main.py generate-sft-data \
    --step train_data \
    --output-dir data \
    --workers 10

# 步骤2：生成summary_data
conda run -n evrag python main.py generate-sft-data \
    --step summary \
    --output-dir data

# 步骤3：生成rerank_data
conda run -n evrag python main.py generate-sft-data \
    --step rerank \
    --output-dir data

# 步骤4：生成测试验证集
conda run -n evrag python main.py generate-sft-data \
    --step test_verify \
    --output-dir data

# 步骤5：生成测试预测结果
conda run -n evrag python main.py generate-sft-data \
    --step test_pred \
    --output-dir data
```

### 7.2 分析和验证数据

```bash
# 完整分析和验证
conda run -n evrag python main.py analyze-sft-data \
    --output-dir data \
    --action all

# 仅统计分析
conda run -n evrag python main.py analyze-sft-data \
    --output-dir data \
    --action stats

# 仅数据验证
conda run -n evrag python main.py analyze-sft-data \
    --output-dir data \
    --action validate
```

## 八、后续工作建议

### 8.1 待修复问题

1. **rerank_data/test.json格式问题**
   - 检查测试集生成逻辑
   - 确保所有rerank数据都包含label字段

2. **测试集数据平衡性**
   - 确保测试集也包含正/负/中等样本
   - 调整生成逻辑，避免测试集为空

### 8.2 功能增强

1. **可视化增强**
   - 添加更多统计图表（长度分布直方图、引用标记分布等）
   - 支持交互式图表（使用plotly）

2. **性能优化**
   - 支持分布式处理（多机多卡）
   - 优化RAG检索性能（批量检索）

3. **质量评估**
   - 添加数据质量评分
   - 自动识别低质量样本

## 九、技术栈

- **Python 3.12**
- **LangChain**: 文档处理
- **vLLM**: 本地LLM推理
- **OpenAI Python SDK**: API调用
- **matplotlib**: 数据可视化
- **rich**: 终端美化输出
- **typer**: 命令行接口

## 十、参考文档

- 原项目实现：`/remote-home/share/liangZhang/EVRAG/generate_sft_data.py`
- 原项目评估：`/remote-home/share/liangZhang/EVRAG/final_score.py`
- QA数据处理文档：`docs/PHASE2_QA_DATA_PROCESSING.md`
- 数据生成计划：`docs/sft_data_generation_plan.md`

---

**阶段完成时间**: 2024年11月26日  
**主要贡献**: 实现了完整的SFT数据生成、分析和验证流程

