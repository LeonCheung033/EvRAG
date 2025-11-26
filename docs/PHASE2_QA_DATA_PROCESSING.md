# Phase 2: QA数据处理和训练测试集生成

## 一、概述

本阶段实现了完整的QA数据处理流程，从已生成的 `qa_pair.json` 开始，经过质量打分、过滤、问题改写、数据扩充、训练/测试集切分、关键词提取和负样本添加，最终生成 `train_qa_pair.json` 和 `test_qa_pair.json`。

## 二、数据生产流程

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
train_qa_pair.json, test_qa_pair.json (切分后的数据集，并进行了问题与答案的匹配)
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

### 详细步骤说明

#### Step 1: QA质量打分和过滤

**功能**：对QA对进行质量评分，过滤低质量数据

**实现**：
- 使用Deepseek API对每个QA对进行质量评分（1-5分）
- 过滤包含"无法准确"或"未提及"的答案
- 保留评分≥阈值的QA对（默认阈值：3）

**输入**：`data/qa_pairs/qa_pair.json`

**输出**：`data/qa_pairs/filtered_qa_pair.json`

**运行命令**：
```bash
conda run -n evrag python main.py process-qa \
    --step filter \
    --qa-pair-path data/qa_pairs/qa_pair.json \
    --output-dir data/qa_pairs \
    --quality-threshold 3 \
    --workers 5
```

**关键实现**：
- `QAProcessor.score_and_filter_qa_pairs()`: 执行质量打分和过滤
- `QAGenerator.score_qa_quality()`: 调用LLM进行质量评分
- 并发处理，支持断点续传

#### Step 2: 问题改写（生成同义问题）

**功能**：为每个问题生成5个同义问题，扩充数据集

**实现**：
- 从 `filtered_qa_pair.json` 加载过滤后的QA对
- 提取所有问题
- 使用Deepseek API为每个问题生成5个同义问题
- 保存到 `expand_qa_pair.json`（JSONL格式）

**输入**：`data/qa_pairs/filtered_qa_pair.json`

**输出**：`data/qa_pairs/expand_qa_pair.json`

**运行命令**：
```bash
conda run -n evrag python main.py process-qa \
    --step generalize \
    --output-dir data/qa_pairs \
    --workers 5
```

**关键实现**：
- `QAProcessor.generalize_and_expand_questions()`: 问题改写主方法
- `QAGenerator.generalize_questions()`: 调用LLM生成同义问题
- 格式：每行一个JSON，`{"unique_id": "问题文本", "raw_resp": "改写后的5个问题（换行分隔）"}`

#### Step 3: 答案匹配和扩充 + 训练/测试集切分

**功能**：将原始问题和改写问题与答案匹配，扩充数据集，并切分为训练集和测试集

**实现**：
- 从 `filtered_qa_pair.json` 加载原始QA对
- 从 `expand_qa_pair.json` 加载改写问题
- 为每个改写问题匹配原始答案，生成扩充后的QA对
- 随机切分为训练集（90%）和测试集（10%）

**输入**：
- `data/qa_pairs/filtered_qa_pair.json`
- `data/qa_pairs/expand_qa_pair.json`

**输出**：
- `data/qa_pairs/train_qa_pair.json`
- `data/qa_pairs/test_qa_pair.json`

**运行命令**：
```bash
conda run -n evrag python main.py process-qa \
    --step split \
    --output-dir data/qa_pairs \
    --train-ratio 0.9
```

**关键实现**：
- `QAProcessor.expand_qa_pairs()`: 答案匹配和扩充
- `QAProcessor.split_train_test()`: 训练/测试集切分
- 使用固定随机种子（42）确保可复现

#### Step 4: 测试集关键词提取

**功能**：为测试集中的每个唯一答案提取关键词

**实现**：
- 从 `test_qa_pair.json` 加载测试集
- 提取所有唯一的答案（去重）
- 使用Deepseek API为每个答案提取关键词
- 将关键词添加到测试集的每个QA对中（`keywords` 字段）
- 保存关键词映射到 `test_keywords_pair.json`

**输入**：`data/qa_pairs/test_qa_pair.json`

**输出**：
- `data/qa_pairs/test_keywords_pair.json`（关键词映射文件）
- `data/qa_pairs/test_qa_pair.json`（更新，添加keywords字段）

**运行命令**：
```bash
conda run -n evrag python main.py process-qa \
    --step keywords \
    --output-dir data/qa_pairs \
    --workers 5
```

**关键实现**：
- `QAProcessor.extract_keywords_for_test_set()`: 关键词提取主方法
- `QAGenerator.extract_keywords()`: 调用LLM提取关键词
- 过滤无效关键词（如"无"、"Model 3"等）

#### Step 5: 添加负样本

**功能**：从通用对话数据中添加负样本（答案="无答案"）

**实现**：
- 从 `raw_general_chats.txt` 读取通用对话数据
- 95% 加入训练集，5% 加入测试集
- 答案设置为"无答案"
- 测试集的负样本 `keywords` 字段为空数组

**输入**：
- `data/qa_pairs/train_qa_pair.json`
- `data/qa_pairs/test_qa_pair.json`
- `data/ut/raw_general_chats.txt`

**输出**：
- `data/qa_pairs/train_qa_pair.json`（更新，添加负样本）
- `data/qa_pairs/test_qa_pair.json`（更新，添加负样本）

**运行命令**：
```bash
conda run -n evrag python main.py process-qa \
    --step negative \
    --output-dir data/qa_pairs \
    --negative-samples-path data/ut/raw_general_chats.txt
```

**关键实现**：
- `QAProcessor.add_negative_samples()`: 添加负样本主方法
- 使用固定随机种子（42）确保可复现

#### Step 6: 保存最终数据（可选）

**功能**：验证和重新保存最终数据，确保格式正确

**运行命令**：
```bash
conda run -n evrag python main.py process-qa \
    --step save \
    --output-dir data/qa_pairs
```

## 三、完整流程执行

### 方式1：分步执行（推荐）

```bash
# Step 1: 质量打分和过滤
conda run -n evrag python main.py process-qa --step filter \
    --qa-pair-path data/qa_pairs/qa_pair.json \
    --output-dir data/qa_pairs \
    --quality-threshold 3 \
    --workers 5

# Step 2: 问题改写
conda run -n evrag python main.py process-qa --step generalize \
    --output-dir data/qa_pairs \
    --workers 5

# Step 3: 切分训练/测试集
conda run -n evrag python main.py process-qa --step split \
    --output-dir data/qa_pairs \
    --train-ratio 0.9

# Step 4: 提取关键词
conda run -n evrag python main.py process-qa --step keywords \
    --output-dir data/qa_pairs \
    --workers 5

# Step 5: 添加负样本
conda run -n evrag python main.py process-qa --step negative \
    --output-dir data/qa_pairs \
    --negative-samples-path data/ut/raw_general_chats.txt
```

### 方式2：一次性执行

```bash
conda run -n evrag python main.py process-qa --step all \
    --qa-pair-path data/qa_pairs/qa_pair.json \
    --output-dir data/qa_pairs \
    --negative-samples-path data/ut/raw_general_chats.txt \
    --quality-threshold 3 \
    --train-ratio 0.9 \
    --workers 5
```

## 四、输出文件说明

### 中间文件

- **`filtered_qa_pair.json`**: 过滤后的QA对（JSON数组格式）
  - 包含 `question`, `answer`, `quality_score`, `quality_reason` 字段

- **`expand_qa_pair.json`**: 问题改写结果（JSONL格式）
  - 每行一个JSON：`{"unique_id": "问题文本", "raw_resp": "改写后的5个问题（换行分隔）"}`

### 最终文件

- **`train_qa_pair.json`**: 训练集（JSON数组格式）
  - 包含正样本和负样本
  - 字段：`unique_id`, `question`, `answer`

- **`test_qa_pair.json`**: 测试集（JSON数组格式）
  - 包含正样本和负样本
  - 字段：`unique_id`, `question`, `answer`, `keywords`

- **`test_keywords_pair.json`**: 测试集关键词映射（JSONL格式）
  - 每行一个JSON：`{"unique_id": "答案文本", "raw_resp": "关键词（逗号分隔）"}`

## 五、关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `quality_threshold` | 3 | QA质量打分阈值（1-5分） |
| `train_ratio` | 0.9 | 训练集比例 |
| `negative_samples_train_ratio` | 0.95 | 负样本训练集比例 |
| `workers` | 20 | 最大并发工作线程数 |

## 六、实现细节

### 1. 原子性保证

每个步骤都保证原子性：
- 每个步骤从已保存的文件加载数据，不重复执行前面的步骤
- 每个步骤完成后保存中间结果，支持断点续传

### 2. 错误处理

- LLM调用失败自动重试（最多3次）
- JSON解析错误处理
- 文件读写错误处理
- API限流处理（降低并发数，添加延迟）

### 3. 性能优化

- 使用 `ThreadPoolExecutor` 进行并发处理
- 支持断点续传，避免重复处理
- 添加API调用延迟，避免限流

### 4. 数据一致性

- 使用固定随机种子（42）确保可复现
- 与原项目数据格式保持一致
- 支持数据对比验证（`compare_qa_data.py`）

## 七、数据对比验证

使用对比工具验证新项目和原项目的数据一致性：

```bash
conda run -n evrag python compare_qa_data.py
```

对比报告保存在 `logs/qa_data_comparison_*.json`

## 八、常见问题

### 1. API限流错误

**问题**：`Error code: 400 - Invalid Format`

**解决**：
- 降低并发数：`--workers 5`
- 代码已自动添加延迟和重试机制

### 2. top_p参数错误

**问题**：`Invalid top_p value, the valid range of top_p is (0, 1.0]`

**解决**：已修复，`top_p=0` 改为 `top_p=0.1`

### 3. 文件不存在错误

**问题**：`Filtered QA pairs file not found`

**解决**：按顺序执行步骤，确保前置步骤已完成

## 九、代码结构

### 核心模块

- **`src/evrag/gen_qa/qa_processor.py`**: QA数据处理主模块
  - `QAProcessor`: 主要处理类
  - `load_qa_pairs()`: 加载QA对
  - `score_and_filter_qa_pairs()`: 质量打分和过滤
  - `generalize_and_expand_questions()`: 问题改写
  - `expand_qa_pairs()`: 答案匹配和扩充
  - `split_train_test()`: 训练/测试集切分
  - `extract_keywords_for_test_set()`: 关键词提取
  - `add_negative_samples()`: 添加负样本
  - `save_final_data()`: 保存最终数据

- **`src/evrag/gen_qa/generator.py`**: QA生成器
  - `QAGenerator`: QA生成器类
  - `score_qa_quality()`: QA质量评分
  - `generalize_questions()`: 问题改写
  - `extract_keywords()`: 关键词提取

- **`main.py`**: 命令行接口
  - `process_qa` 命令：QA处理流程入口

### 工具脚本

- **`compare_qa_data.py`**: 数据对比工具

## 十、参考文档

- 原项目实现：`/remote-home/share/liangZhang/EVRAG/src/gen_qa/run.py`
- 详细计划：`docs/QA_DATA_PROCESSING_PLAN.md`
- **流程示例**：`docs/QA_PROCESSING_EXAMPLE.md` - 展示一个QA对在整个处理流程中的变化

