# QA数据处理流程示例

本文档展示一个QA对在整个处理流程中的变化过程。

## 示例QA对

**原始问题**：`TPMLM指的是什么？`

**原始答案**：`TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。`

---

## Step 1: 原始QA对 (qa_pair.json)

**文件格式**：JSONL（每行一个JSON对象）

```json
{
  "unique_id": "44253665fc8a815f05f67f40253dd1d3",
  "raw_resp": "[\n  {\n    \"question\": \"TPMLM指的是什么？\",\n    \"answer\": \"TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。\"\n  },\n  ...\n]"
}
```

**说明**：
- `unique_id`: 文档的唯一标识符
- `raw_resp`: JSON字符串，包含该文档生成的所有QA对（通常5个）

---

## Step 2: 质量打分和过滤 (filtered_qa_pair.json)

**文件格式**：JSON数组

```json
{
  "question": "TPMLM指的是什么？",
  "answer": "TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。",
  "quality_score": 5,
  "quality_reason": "问题明确询问TPMLM的定义，属于事实性询问；答案直接回应了问题，提供了准确的定义，没有引用无关内容或进行文本摘要。"
}
```

**变化说明**：
- ✅ 从原始文档中提取出单个QA对
- ✅ 添加了 `quality_score` 字段（1-5分，本例为5分）
- ✅ 添加了 `quality_reason` 字段（质量评分的原因）
- ✅ 过滤掉了包含"无法准确"或"未提及"的答案
- ✅ 过滤掉了评分低于阈值（默认3分）的QA对

---

## Step 3: 问题改写 (expand_qa_pair.json)

**文件格式**：JSONL（每行一个JSON对象）

```json
{
  "unique_id": "TPMLM指的是什么？",
  "raw_resp": "1. TPMLM具体是什么意思呢？\n2. 能不能解释一下TPMLM的含义？\n3. TPMLM这个缩写代表什么内容？\n4. 我想知道TPMLM的定义是什么？\n5. 请问TPMLM指的是什么概念？"
}
```

**变化说明**：
- ✅ 为原始问题生成了5个同义问题
- ✅ `unique_id` 字段存储原始问题文本
- ✅ `raw_resp` 字段存储改写后的5个问题（换行分隔，带序号）

**解析后的改写问题列表**：
1. TPMLM具体是什么意思呢？
2. 能不能解释一下TPMLM的含义？
3. TPMLM这个缩写代表什么内容？
4. 我想知道TPMLM的定义是什么？
5. 请问TPMLM指的是什么概念？

---

## Step 4: 答案匹配和扩充 + 训练/测试集切分

### 4.1 扩充后的QA对列表

原始问题 + 5个改写问题，每个问题都匹配相同的答案，生成6个QA对：

```json
[
  {
    "unique_id": "abc123...",
    "question": "TPMLM指的是什么？",
    "answer": "TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。"
  },
  {
    "unique_id": "def456...",
    "question": "TPMLM具体是什么意思呢？",
    "answer": "TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。"
  },
  {
    "unique_id": "ghi789...",
    "question": "能不能解释一下TPMLM的含义？",
    "answer": "TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。"
  },
  {
    "unique_id": "jkl012...",
    "question": "TPMLM这个缩写代表什么内容？",
    "answer": "TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。"
  },
  {
    "unique_id": "mno345...",
    "question": "我想知道TPMLM的定义是什么？",
    "answer": "TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。"
  },
  {
    "unique_id": "pqr678...",
    "question": "请问TPMLM指的是什么概念？",
    "answer": "TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。"
  }
]
```

**变化说明**：
- ✅ 原始问题 + 5个改写问题 = 6个QA对
- ✅ 每个问题都有唯一的 `unique_id`（基于问题文本的MD5哈希）
- ✅ 所有问题共享相同的答案

### 4.2 切分到训练集或测试集

假设其中3个QA对被分配到测试集（10%），3个被分配到训练集（90%）。

**训练集 (train_qa_pair.json)**：
```json
[
  {
    "unique_id": "abc123...",
    "question": "TPMLM指的是什么？",
    "answer": "TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。"
  },
  {
    "unique_id": "def456...",
    "question": "TPMLM具体是什么意思呢？",
    "answer": "TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。"
  },
  {
    "unique_id": "ghi789...",
    "question": "能不能解释一下TPMLM的含义？",
    "answer": "TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。"
  }
]
```

**测试集 (test_qa_pair.json)**：
```json
[
  {
    "unique_id": "jkl012...",
    "question": "TPMLM这个缩写代表什么内容？",
    "answer": "TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。"
  },
  {
    "unique_id": "mno345...",
    "question": "我想知道TPMLM的定义是什么？",
    "answer": "TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。"
  },
  {
    "unique_id": "pqr678...",
    "question": "请问TPMLM指的是什么概念？",
    "answer": "TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。"
  }
]
```

---

## Step 5: 测试集关键词提取

### 5.1 关键词映射文件 (test_keywords_pair.json)

**文件格式**：JSONL（每行一个JSON对象）

```json
{
  "unique_id": "TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。",
  "raw_resp": "TPMLM,最大允许总质量,车辆,乘客,液体,货物"
}
```

**说明**：
- ✅ 为测试集中的**唯一答案**提取关键词
- ✅ `unique_id` 字段存储答案文本
- ✅ `raw_resp` 字段存储逗号分隔的关键词列表

### 5.2 更新测试集 (test_qa_pair.json)

测试集中的每个QA对都添加了 `keywords` 字段：

```json
[
  {
    "unique_id": "jkl012...",
    "question": "TPMLM这个缩写代表什么内容？",
    "answer": "TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。",
    "keywords": ["TPMLM", "最大允许总质量", "车辆", "乘客", "液体", "货物"]
  },
  {
    "unique_id": "mno345...",
    "question": "我想知道TPMLM的定义是什么？",
    "answer": "TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。",
    "keywords": ["TPMLM", "最大允许总质量", "车辆", "乘客", "液体", "货物"]
  },
  {
    "unique_id": "pqr678...",
    "question": "请问TPMLM指的是什么概念？",
    "answer": "TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。",
    "keywords": ["TPMLM", "最大允许总质量", "车辆", "乘客", "液体", "货物"]
  }
]
```

**变化说明**：
- ✅ 为测试集中的每个QA对添加了 `keywords` 字段
- ✅ 相同答案的QA对共享相同的关键词列表
- ✅ 关键词用于后续的RAG评估

---

## Step 6: 添加负样本

### 6.1 负样本示例

从 `raw_general_chats.txt` 中读取的通用对话数据：

```
你好
今天天气怎么样？
吃饭了吗？
```

### 6.2 添加到训练集和测试集

**训练集 (train_qa_pair.json)** - 添加负样本（95%）：
```json
[
  {
    "unique_id": "abc123...",
    "question": "TPMLM指的是什么？",
    "answer": "TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。"
  },
  // ... 其他正样本 ...
  {
    "unique_id": "xyz999...",
    "question": "你好",
    "answer": "无答案"
  },
  {
    "unique_id": "uvw888...",
    "question": "今天天气怎么样？",
    "answer": "无答案"
  }
  // ... 更多负样本 ...
]
```

**测试集 (test_qa_pair.json)** - 添加负样本（5%）：
```json
[
  {
    "unique_id": "jkl012...",
    "question": "TPMLM这个缩写代表什么内容？",
    "answer": "TPMLM是指车辆的最大允许总质量，包括所有乘客、液体和货物。",
    "keywords": ["TPMLM", "最大允许总质量", "车辆", "乘客", "液体", "货物"]
  },
  // ... 其他正样本 ...
  {
    "unique_id": "rst777...",
    "question": "吃饭了吗？",
    "answer": "无答案",
    "keywords": []
  }
]
```

**变化说明**：
- ✅ 训练集添加了95%的负样本（答案="无答案"）
- ✅ 测试集添加了5%的负样本（答案="无答案"，keywords=[]）
- ✅ 负样本用于训练模型识别无关问题

---

## 完整流程总结

| 步骤 | 文件 | QA对数量 | 关键变化 |
|------|------|----------|----------|
| Step 1 | qa_pair.json | 1个（在文档中） | 原始QA对 |
| Step 2 | filtered_qa_pair.json | 1个 | 添加质量评分 |
| Step 3 | expand_qa_pair.json | 1个（改写问题） | 生成5个同义问题 |
| Step 4 | train_qa_pair.json<br>test_qa_pair.json | 6个（1原始+5改写） | 答案匹配，切分训练/测试 |
| Step 5 | test_qa_pair.json | 3个（测试集中） | 添加关键词字段 |
| Step 6 | train_qa_pair.json<br>test_qa_pair.json | 6个 + 负样本 | 添加负样本 |

## 数据扩充效果

- **原始QA对**：1个
- **扩充后QA对**：6个（1个原始问题 + 5个改写问题）
- **数据扩充倍数**：6倍

## 注意事项

1. **质量评分**：只有评分≥阈值的QA对才会进入后续流程
2. **问题改写**：每个问题生成5个同义问题，但实际可能少于5个（如果LLM生成失败）
3. **训练/测试切分**：使用固定随机种子（42）确保可复现
4. **关键词提取**：只为测试集中的唯一答案提取关键词，相同答案共享关键词
5. **负样本**：95%加入训练集，5%加入测试集，使用固定随机种子确保可复现

