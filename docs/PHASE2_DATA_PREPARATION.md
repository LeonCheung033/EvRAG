# 阶段二：数据准备和索引构建开发文档

## 概述

本文档详细说明阶段二（数据准备和索引构建）的完整开发流程，包括PDF解析、文档清洗、文档切分、索引构建、QA生成、数据分析和测试验证等所有步骤。

## 目录

1. [PDF解析和清洗](#pdf解析和清洗)
2. [文档切分和数据入库](#文档切分和数据入库)
3. [索引构建](#索引构建)
4. [QA对生成](#qa对生成)
5. [数据分析](#数据分析)
6. [测试验证](#测试验证)
7. [故障排查](#故障排查)

## PDF解析和清洗

### 1.1 PDF解析

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

**代码示例**:
```python
from src.evrag.parser import PDFParser

parser = PDFParser(
    pdf_path="data/Tesla_Manual.pdf",
    min_filter_pages=0,
    max_filter_pages=None,
    page_clip=50
)
raw_docs = parser.parse()
```

### 1.2 文档清洗

**模块**: `src/evrag/client/clean_client.py`

**功能**: 使用LLM批量清理和整理文档内容

**处理步骤**:
1. 初始化CleanClient（需要LLM客户端，默认使用豆包API）
2. 对每个原始文档：
   - 构建清理prompt（要求LLM让句子更通顺，按标题归类整理）
   - 调用LLM进行文档清理（temperature=0.001, top_p=0）
   - 保留原始metadata
3. 批量处理（默认20个并发线程）

**Prompt模板**:
```
你是一个专业的文档整理助手，负责对汽车用户手册中的内容进行整理和总结。请根据以下要求对文档进行处理：

1. **让句子变得更加通顺**：重新整合句子、段落，去除一些不必要的符号，例如换行符等。
2. **按标题归类整理**：按照文档的语义关系，把属于同一个标题下的文档做归类合并, 记住标题要用markdown的形式加粗，例如###。

请根据以下文档内容进行整理：
{文档内容}
整理后的输出：
```

**输出**:
- 文件: `data/processed_docs/clean_docs.pkl`
- 数据结构: `List[Document]`，每个Document包含清理后的内容

**代码示例**:
```python
from src.evrag.client import CleanClient, OpenAIClient

llm_client = OpenAIClient()  # 使用豆包API
clean_client = CleanClient(llm_client)
clean_docs = clean_client.clean_documents(raw_docs)
```

## 文档切分和数据入库

### 2.1 语义切分（父文档）

**模块**: `src/evrag/parser/document_splitter.py`

**功能**: 对清洗后的文档进行语义切分

**处理步骤**:
1. 调用语义切分服务（SemanticChunkClient）
   - 服务地址: `http://localhost:6000/v1/semantic-chunks`
   - 使用M3E-small模型进行语义聚类
   - 参数: `group_size=10`（每组目标最大句子数）
2. 将文档按语义相似性分组，生成多个语义块
3. 为每个语义块创建父Document：
   - `unique_id`: MD5哈希值（基于语义块内容）
   - `metadata`: 继承原始文档的metadata（**只设置unique_id，不设置parent_id**）
   - `page_content`: 语义块文本内容
4. 如果语义块长度 < 512字符，直接加入最终文档列表
5. **保存父文档到MongoDB**

### 2.2 句子级切分（子文档）

**处理步骤**:
1. 使用`RecursiveCharacterTextSplitter`进行句子级切分：
   - `chunk_size=256`（tokens，使用tiktoken编码）
   - `chunk_overlap=50`（tokens）
   - 分隔符优先级: `["\n\n", "\n"]`
2. 为每个子块创建子Document：
   - `unique_id`: MD5哈希值（基于子块内容）
   - `metadata`: 
     - 继承父文档的metadata
     - 添加`parent_id`: 父文档的unique_id（**指向父文档的unique_id**）
   - `page_content`: 子块文本内容
3. **保存子文档到MongoDB**
4. 将子文档加入最终文档列表

**配置参数**:
```python
_chunk_size = 256              # 句子级切分的块大小（tokens）
_chunk_overlap = 50            # 重叠大小（tokens）
_semantic_group_size = 10      # 语义分组大小
_max_parent_size = 512         # 最大父文档大小（字符数）
```

**输出**:
- 文件: `data/processed_docs/split_docs.pkl`
- MongoDB集合: `manual_text`
- 数据结构: `List[Document]`，包含所有父文档和子文档

**MongoDB文档结构**:
```json
{
  "unique_id": "文档的唯一ID（MD5哈希）",
  "page_content": "文档文本内容",
  "metadata": {
    "unique_id": "文档的唯一ID",
    "source": "PDF文件路径",
    "page": 页码,
    "images_info": [图片信息列表],
    "parent_id": "父文档ID（仅子文档有此字段）"
  }
}
```

### 2.3 数据入库

**模块**: `src/evrag/parser/document_splitter.py` 中的 `save_2_mongo` 函数

**功能**: 将切分后的文档保存到MongoDB数据库

**处理步骤**:
1. 连接MongoDB（使用MongoDBClient）
   - 默认连接: `localhost:27017`
   - 数据库: `evrag`（可配置）
   - 集合: `manual_text`（可配置）
2. 对每个文档使用`update_one`进行upsert操作（存在则更新，不存在则插入）
3. 关闭MongoDB连接

**注意事项**:
- 使用`upsert=True`确保文档的唯一性（基于unique_id）
- 父文档和子文档都会保存到同一个集合中
- 父文档只设置`unique_id`，不设置`parent_id`
- 子文档的`parent_id`指向父文档的`unique_id`

## 索引构建

### 3.1 BM25索引构建

**模块**: `src/evrag/retriever/bm25_retriever.py`

**处理步骤**:
1. 初始化BM25Retriever（`retrieve=False`表示构建新索引）
2. 对每个文档进行分词：
   - 使用jieba进行中文分词
   - 过滤停用词
3. 使用LangChain的BM25Retriever构建索引
4. 保存索引到pickle文件

**输出**:
- 文件: `data/saved_index/bm25retriever.pkl`

### 3.2 Milvus索引构建

**模块**: `src/evrag/retriever/milvus_retriever.py`

**处理步骤**:
1. 初始化MilvusRetriever（`retrieve=False`表示构建新索引）
2. 连接Milvus（本地文件模式）
3. 创建Collection（如果不存在）：
   - `text`字段：VARCHAR类型，max_length=512
   - `sparse_vector`字段：SPARSE_FLOAT_VECTOR类型
   - `dense_vector`字段：FLOAT_VECTOR类型（BGE-M3 dense维度）
4. 对每个文档计算embedding：
   - 使用BGE-M3模型
   - 生成dense vector和sparse vector
   - **如果文本长度超过512字符，自动截断**
5. 批量插入向量到Milvus（EMB_BATCH=50）
6. 创建索引（sparse和dense向量索引）
7. 加载Collection

**配置参数**:
```python
MAX_TEXT_LENGTH = 512  # 文本字段最大长度
EMB_BATCH = 50         # 批量插入大小
```

**输出**:
- 文件: `data/saved_index/milvus.db`

**注意事项**:
- 文本长度超过512字符的文档会被截断（作为安全措施）
- 使用GPU加速embedding计算（配置在`config.yaml`中）

## QA对生成

### 4.1 基本使用

**命令**: `python main.py gen-qa`

**功能**: 从文档生成问答对，使用Deepseek作为LLM

**基本命令**:
```bash
# 使用默认配置（从clean_docs.pkl生成）
python main.py gen-qa --config config/config.yaml

# 指定输入文件
python main.py gen-qa --input data/processed_docs/clean_docs.pkl --config config/config.yaml

# 指定输出文件
python main.py gen-qa --output data/qa_pairs/qa_pair.json --config config/config.yaml

# 调整并发数
python main.py gen-qa --workers 20 --config config/config.yaml
```

### 4.2 处理流程

1. **加载文档**：从pickle文件加载文档列表
2. **初始化QA生成器**：使用OpenAIClient（Deepseek API）
3. **生成QA对**：
   - 对每个文档构建prompt
   - 调用LLM生成QA对（temperature=0.85, top_p=0.95）
   - **清理markdown代码块标记**（自动移除```json和```）
   - 保存到输出文件（支持checkpoint机制）
4. **解析和统计**：解析生成的QA对，显示统计信息

### 4.3 Prompt模板

```
我会给你一段文本（<document></document>之间的部分），你需要阅读这段文本，分别针对这段文本生成5个问题，和基于这段文本对问题的回答，回答请保持完整，无须重复问题。

对问题、答案的要求：
1.问题：问题要与这段文本相关，不要询问类似"这个问题的答案在哪一章"这样的问题;
2.答案：回答请保持完整且简洁，无须重复问题。答案要能够独立回答问题，而不是引用其他章节和页码，例如答案内容不能出现请参阅xx页码;
3.5个问题里面至少要包含一个需要综合*大段*文本才能回答的问题，但不要问类似"这一段主要讲了什么内容"这样的问题;

对输出的要求：
1.返回结果以JSON形式组织，格式为[{"question": "...", "answer": "..."}, ...]。
2.如果当前文本主要是目录，或者是一些人名、地址、电子邮箱等没有办法生成有意义的问题时，可以返回[]。
```

### 4.4 输出格式

**文件格式**: JSONL（每行一个JSON对象）

```json
{"unique_id": "文档ID", "raw_resp": "[{\"question\": \"问题\", \"answer\": \"答案\"}, ...]"}
```

**解析后的QA对格式**:
```json
[
  {"question": "问题1", "answer": "答案1"},
  {"question": "问题2", "answer": "答案2"},
  ...
]
```

### 4.5 过滤机制

- **内容太短**：文档长度 < 100字符（`min_chunk_size`）会被过滤
- **空响应**：LLM返回空数组`[]`（主要是目录类内容）会保留在输出中

### 4.6 Checkpoint机制

如果输出文件已存在，会自动加载已有的QA对作为checkpoint，跳过已生成的文档，支持断点续传。

## 数据分析

### 5.1 基本使用

**命令**: `python main.py analyze-data`

```bash
# 分析数据质量（使用默认配置）
python main.py analyze-data --config config/config.yaml

# 指定输出报告路径
python main.py analyze-data --config config/config.yaml --output reports/data_quality.json
```

### 5.2 分析内容

#### 5.2.1 统计信息分析

- **文档数量**：各阶段的文档总数
- **长度统计**：平均/最小/最大字符数、词数、Token数
- **长度分布**：按长度范围统计文档分布
- **元数据统计**：包含parent_id、图片信息的文档数量

#### 5.2.2 质量检查

- **重复检查**：检查unique_id重复、内容重复
- **父子关系**：分析父文档和子文档的关系
- **切分效果**：计算文档扩展倍数、平均大小减少

#### 5.2.3 输出内容

**控制台输出**:
1. 各阶段统计表
2. 长度分布表
3. 示例文档
4. 重复检查结果
5. 父子关系分析
6. 阶段对比表

**JSON报告**（默认：`logs/data_quality_report.json`）:
```json
{
  "raw_stats": {...},
  "clean_stats": {...},
  "split_stats": {...},
  "duplicates": {...},
  "relationship": {
    "total_parent_docs": 100,
    "total_child_docs": 500,
    "avg_children_per_parent": 5.0
  },
  "expansion_ratio": 2.35,
  "avg_size_reduction": 45.2
}
```

### 5.3 质量评估标准

**优秀指标**:
1. **文档数量**：切分后文档数应该是清洗后的2-5倍
2. **长度分布**：大部分文档在100-512字符范围内
3. **无重复**：unique_id无重复，内容无重复
4. **父子关系**：每个父文档有2-10个子文档

**需要改进的情况**:
1. 切分后文档数过少（<1.5倍）或过多（>10倍）
2. 大量超短文档（<100字符）或超长文档（>1024字符）
3. 存在重复的unique_id或内容
4. 平均子文档数过少（<1）或过多（>20）

## 测试验证

### 6.1 数据质量测试

**验证文档解析完整性**:
```bash
# 检查文档数量
python -c "
import pickle
with open('data/processed_docs/raw_docs.pkl', 'rb') as f:
    raw_docs = pickle.load(f)
print(f'原始文档数: {len(raw_docs)}')
"
```

**验证清洗效果**:
```bash
# 对比清洗前后的文档
python main.py analyze-data --config config/config.yaml
```

**验证切分效果**:
```bash
# 检查MongoDB中的数据
python src/evrag/parser/check/check_mongodb.py

# 检查文档长度分布
python src/evrag/parser/check/check_text_length.py
```

### 6.2 索引测试

**测试BM25检索**:
```python
from src.evrag.retriever import BM25Retriever
import pickle

# 加载BM25检索器
with open('data/saved_index/bm25retriever.pkl', 'rb') as f:
    bm25_retriever = pickle.load(f)

# 测试检索
results = bm25_retriever.retrieve_topk("如何打开车门", topk=5)
for doc in results:
    print(doc.page_content)
```

**测试Milvus检索**:
```python
from src.evrag.retriever import MilvusRetriever

# 初始化检索器
milvus_retriever = MilvusRetriever(retrieve=True)

# 测试检索
results = milvus_retriever.retrieve_topk("如何打开车门", topk=5)
for doc in results:
    print(doc.page_content)
```

**测试混合检索**:
```bash
# 使用infer命令测试
python main.py infer "如何打开车门" --topk 5 --config config/config.yaml
```

### 6.3 QA质量测试

**检查生成的QA对**:
```bash
# 查看QA对统计
python -c "
import json
with open('data/qa_pairs/qa_pair.json', 'r') as f:
    qa_count = sum(1 for line in f if line.strip())
print(f'QA对总数: {qa_count}')
"
```

**抽样检查QA对质量**:
```python
import json
import random

# 随机抽样检查
with open('data/qa_pairs/qa_pair.json', 'r') as f:
    lines = [line for line in f if line.strip()]

# 随机选择5个
samples = random.sample(lines, min(5, len(lines)))
for line in samples:
    item = json.loads(line)
    qa_list = json.loads(item['raw_resp'])
    print(f"文档ID: {item['unique_id']}")
    for qa in qa_list[:2]:  # 只显示前2个
        print(f"  问题: {qa['question']}")
        print(f"  答案: {qa['answer'][:100]}...")
    print()
```

### 6.4 数据统计

**生成统计报告**:
```bash
# 运行数据分析
python main.py analyze-data --config config/config.yaml

# 查看报告
cat logs/data_quality_report.json | python -m json.tool
```

**检查关键指标**:
- 文档扩展倍数（理想值：2-5倍）
- 平均文档大小减少（理想值：30-60%）
- 父子关系（平均每个父文档2-10个子文档）
- 文档长度分布（大部分在100-512字符）

## 完整工作流

### 7.1 首次执行

```bash
cd /remote-home/share/liangZhang/EvRAG

# ========== 步骤1：数据准备 ==========
echo "步骤1: 数据准备..."
conda run -n evrag python main.py prepare-data --config config/config.yaml

# ========== 步骤2：索引构建 ==========
echo "步骤2: 索引构建..."
conda run -n evrag python main.py build-index --config config/config.yaml

# ========== 步骤3：数据分析 ==========
echo "步骤3: 数据分析..."
conda run -n evrag python main.py analyze-data --config config/config.yaml

# ========== 步骤4：生成QA对 ==========
echo "步骤4: 生成QA对..."
conda run -n evrag python main.py gen-qa --config config/config.yaml

# ========== 步骤5：验证 ==========
echo "步骤5: 验证数据..."
conda run -n evrag python src/evrag/parser/check/check_mongodb.py
conda run -n evrag python src/evrag/parser/check/check_text_length.py

echo "✅ 所有步骤完成！"
```

### 7.2 分步执行

如果某个步骤失败，可以单独重试：

```bash
# 如果文档清洗已完成，跳过清洗
conda run -n evrag python main.py prepare-data --config config/config.yaml --skip-clean

# 如果索引构建失败，只重新构建索引
conda run -n evrag python main.py build-index --config config/config.yaml --force

# 如果QA生成中断，支持checkpoint自动续传
conda run -n evrag python main.py gen-qa --config config/config.yaml
```

### 7.3 从MongoDB重新构建索引

如果MongoDB中有数据，但索引文件丢失：

```bash
conda run -n evrag python main.py build-index --config config/config.yaml --from-mongodb
```

## 故障排查

### 8.1 数据准备失败

**问题1：PDF解析失败**
- 检查PDF文件是否存在且可读
- 检查页面过滤配置是否正确

**问题2：文档清洗失败**
- 检查豆包API配置是否正确
- 检查网络连接
- 查看错误日志，失败时会跳过该文档继续处理

**问题3：文档切分失败**
- 检查语义切分服务是否运行（`curl http://localhost:6000/health`）
- 检查MongoDB是否运行
- 如果服务失败，会返回原始文本作为单个元素

### 8.2 索引构建失败

**问题1：BM25索引构建失败**
- 检查磁盘空间是否充足
- 检查文件权限

**问题2：Milvus索引构建失败**
- 检查Milvus连接
- 检查GPU配置（如果使用GPU）
- 检查文档是否为空
- 如果文本长度超过512字符，会自动截断

**错误示例**:
```
ParamError: length of string exceeds max length. length: 578, max length: 512
```
**解决方案**：已添加自动截断逻辑，如果仍出现此错误，检查代码是否正确更新。

### 8.3 QA生成失败

**问题1：LLM调用失败**
- 检查Deepseek API配置
- 检查网络连接
- 查看错误日志

**问题2：生成的QA对格式错误**
- 检查`raw_resp`字段是否包含markdown代码块标记
- 已添加自动清理逻辑，如果仍有问题，检查代码是否正确更新

**问题3：QA对数量少于文档数量**
- 部分文档可能因为内容太短被过滤（< 100字符）
- 部分文档可能返回空数组（主要是目录类内容）
- 这是正常行为，参考[阶段二数据修复总结](./PHASE2_DATA_FIXES_SUMMARY.md)中的分析

### 8.4 数据分析异常

**问题1：文件不存在**
```bash
# 先运行数据准备
conda run -n evrag python main.py prepare-data --config config/config.yaml
```

**问题2：分析结果异常**
- 检查切分参数配置
- 查看示例文档，判断切分是否合理
- 调整切分参数后重新运行

**问题3：发现重复**
- 检查切分逻辑
- 查看重复ID详情，分析重复原因
- 修复后重新运行数据准备

## 依赖服务

执行完整流程需要以下服务运行：

1. **MongoDB**: 默认 `localhost:27017`
   - 启动: `scripts/start_services.sh` 会自动启动
   
2. **语义切分服务**: 默认 `http://localhost:6000`
   - 启动: `scripts/start_services.sh` 会自动启动
   - 使用M3E-small模型进行语义聚类

3. **LLM服务**: 用于文档清洗和QA生成
   - 文档清洗：使用豆包API（配置在`config.yaml`中）
   - QA生成：使用Deepseek API（配置在`config.yaml`中）

## 性能优化

1. **并发处理**: 
   - 文档清洗使用20个并发线程
   - QA生成使用20个并发线程（可配置）
   - 语义切分通过HTTP服务异步处理

2. **批量操作**:
   - MongoDB使用upsert批量更新
   - Milvus使用批量插入（EMB_BATCH=50）

3. **缓存机制**:
   - 中间结果保存为pickle文件
   - 索引文件持久化，避免重复构建
   - QA生成支持checkpoint机制

4. **检查点机制**:
   - `raw_docs.pkl`存在 → 跳过PDF解析
   - `clean_docs.pkl`存在 → 跳过文档清洗
   - `split_docs.pkl`存在 → 跳过文档切分
   - QA生成支持checkpoint续传

## 相关文档

- [阶段二数据修复总结](./PHASE2_DATA_FIXES_SUMMARY.md) - 本阶段所有修复和问题分析
- [清空MongoDB并重新生成数据指南](./CLEAR_AND_REGEN.md) - 数据重新生成操作指南
- [阶段二工具配置和系统优化](./PHASE2_TOOLS_CONFIG.md) - GPU分配、LLM服务配置和性能监控

## 下一步

完成数据准备和索引构建后，可以：

1. **测试检索**：使用 `python main.py infer` 测试检索功能
2. **评估系统**：使用评估脚本测试检索准确率
3. **模型微调**：使用生成的QA对进行模型微调
4. **部署服务**：将系统部署到生产环境

