# 阶段二：数据修复和优化总结

## 概述

本文档总结了阶段二（数据准备阶段）中发现的所有问题、修复方案、验证步骤和操作指南。本阶段主要解决了文档切分、PDF解析、QA生成和索引构建中的多个关键问题。

## 修复内容

### 1. parent_id 问题修复

**问题描述**：
- 父文档被错误地设置了 `parent_id`，且 `parent_id` 等于自己的 `unique_id`
- 子文档的 `parent_id` 指向了父文档的 `parent_id` 而不是父文档的 `unique_id`

**修复位置**：`src/evrag/parser/document_splitter.py`

**修复内容**：
1. **父文档修复**：父文档只设置 `unique_id`，不设置 `parent_id`
   ```python
   # 修复后的代码（第140行）
   parent_metadata["unique_id"] = parent_id  # ✅ 父文档只设置 unique_id
   # 不再设置 parent_id
   ```

2. **子文档修复**：子文档的 `parent_id` 指向父文档的 `unique_id`
   ```python
   # 修复后的代码（第170行）
   child_metadata["parent_id"] = chunk.metadata["unique_id"]  # ✅ 指向父文档的 unique_id
   ```

**影响**：
- 修复了父子关系错误，确保子文档正确关联到父文档
- 修复后需要清空 MongoDB 并重新生成数据

### 2. PDF 解析页面过滤修复

**问题描述**：
- 新项目的 PDF 解析没有使用配置的页面过滤参数（min_filter_pages、max_filter_pages、page_clip）
- 导致新旧项目的 clean_docs.pkl 内容不一致

**修复位置**：
- `src/evrag/config.py`：添加 PDF 解析配置字段
- `main.py`：修改 PDFParser 初始化，传递配置参数

**修复内容**：
1. **配置字段**（`src/evrag/config.py`）：
   ```python
   pdf_min_filter_pages: int = Field(default=0, description="最小页码（从0开始）")
   pdf_max_filter_pages: Optional[int] = Field(default=None, description="最大页码")
   pdf_page_clip: int = Field(default=50, description="页面底部裁剪像素数")
   ```

2. **PDFParser 初始化**（`main.py` 第88-89行）：
   ```python
   parser = PDFParser(
       pdf_path=pdf_file,
       min_filter_pages=settings.pdf_min_filter_pages,
       max_filter_pages=settings.pdf_max_filter_pages,
       page_clip=settings.pdf_page_clip,
   )
   ```

**影响**：
- 确保新旧项目的 PDF 解析行为一致
- 可以灵活配置页面过滤范围

### 3. gen_qa 默认输入路径修复

**问题描述**：
- `gen_qa` 命令的默认输入路径错误地使用了 `split_docs.pkl`，应该使用 `clean_docs.pkl`

**修复位置**：`main.py`

**修复内容**：
- 第689行：将默认 `input_path` 从 `split_docs.pkl` 改为 `clean_docs.pkl`
- 第722行：更新帮助信息

**影响**：
- 确保 QA 生成使用正确的输入数据
- 与旧项目行为保持一致

### 4. QA 生成 markdown 代码块清理修复

**问题描述**：
- LLM 返回的响应包含 markdown 代码块标记（```json 和 ```）
- `raw_resp` 字段直接存储了包含这些标记的原始响应，导致 JSON 格式混乱

**修复位置**：`src/evrag/gen_qa/generator.py`

**修复内容**：
1. **新增清理方法**（第485-512行）：
   ```python
   @staticmethod
   def clean_markdown_code_blocks(text: str) -> str:
       """清理文本中的 markdown 代码块标记"""
       # 移除 ```json 和 ``` 标记
   ```

2. **存储前清理**（第279-280行）：
   ```python
   cleaned_resp = self.clean_markdown_code_blocks(result)
   item = {"unique_id": unique_id, "raw_resp": cleaned_resp}
   ```

**影响**：
- `raw_resp` 字段现在存储的是纯 JSON 格式，不再包含 markdown 标记
- 向后兼容：即使 LLM 返回无标记的响应，也能正常处理

### 5. Milvus 文本长度限制修复

**问题描述**：
- 部分文档的文本长度超过 Milvus schema 的 `max_length` 限制（512）
- 导致索引构建时出现 `ParamError: length of string exceeds max length`

**根本原因**：
- `RecursiveCharacterTextSplitter` 使用 `chunk_size=256`（tokens），不是字符数
- 中文 token/字符比例约为 1:1.2-1.5，可能出现 token < 256 但字符 > 512 的情况
- 当文档没有合适分隔符时，`RecursiveCharacterTextSplitter` 会保留整个文档

**修复位置**：`src/evrag/retriever/milvus_retriever.py`

**修复内容**：
1. **保持 MAX_TEXT_LENGTH = 512**（与旧项目一致）
2. **添加截断逻辑**（第173-182行）：
   ```python
   for doc in docs:
       text = doc.page_content
       if len(text) > MAX_TEXT_LENGTH:
           print(f"Warning: Document text length ({len(text)}) exceeds MAX_TEXT_LENGTH ({MAX_TEXT_LENGTH}). Truncating.")
           text = text[:MAX_TEXT_LENGTH]
       raw_texts.append(text)
   ```

**影响**：
- 确保所有文档都能成功插入 Milvus
- 作为安全措施，防止超过长度限制的文档导致错误

## 问题分析

### 1. 文档长度超过 512 字符的原因

**现象**：
- 新项目的 `split_docs.pkl` 中有 3 个文档超过 512 字符（578、572、684）
- 旧项目的 `split_docs.pkl` 中所有文档都不超过 512 字符（最大 507）

**根本原因**：
1. **Token vs 字符数**：
   - `RecursiveCharacterTextSplitter` 使用 `chunk_size=256`（tokens）
   - 中文 token/字符比例约为 1:1.2-1.5
   - 可能出现 token < 256 但字符 > 512 的情况

2. **分隔符问题**：
   - 当文档没有合适的分隔符（`\n\n` 或 `\n`）时，`RecursiveCharacterTextSplitter` 会保留整个文档
   - 即使 token 数超过 256，也可能因为找不到分隔符而无法切分

3. **版本差异**：
   - 新旧项目可能使用不同版本的 LangChain，导致切分行为不同
   - 或者 PDF 内容/清洗结果不同

**解决方案**：
- 在 Milvus 插入前添加截断逻辑（已实现）
- 未来可以考虑在切分阶段添加基于字符数的二次切分

### 2. QA 生成缺失原因分析

**现象**：
- `clean_docs.pkl` 有 244 条文档
- `qa_pair.json` 只有 242 条 QA 对
- 缺失 2 条，另有 1 条返回空数组

**分析结果**：

1. **被过滤的文档（2条）**：
   - 内容太短（< 100 字符），被 `min_chunk_size` 过滤
   - unique_id: `36bc9d9fc9f50dcac03da9b4b498fb56`（30 字符）
   - unique_id: `798d595b86303d36f16686de5d6d654f`（73 字符）

2. **空响应（1条）**：
   - unique_id: `f6b94d22a40b999c189439c4c0d3f930`
   - 内容：主要是外部部件说明，包含大量页码引用
   - LLM 判断该文档主要是目录/页码引用，无法生成有意义的 QA 对，返回了空数组 `[]`
   - 这是符合 prompt 要求的正常行为

**结论**：
- 缺失数量匹配：244 = 2（被过滤）+ 242（生成 QA），其中 242 = 241（有效）+ 1（空响应）
- 所有文档都已正确处理

## 验证步骤

### 1. 验证 parent_id 修复

**使用脚本**：`src/evrag/parser/check/check_mongodb.py`

```bash
cd /remote-home/share/liangZhang/EvRAG
conda run -n evrag python src/evrag/parser/check/check_mongodb.py
```

**预期结果**：
- ✅ 错误文档（parent_id == unique_id）: 0 个
- ✅ 父文档数（无 parent_id）: 1000+ 个
- ✅ 子文档数（有 parent_id）: 1000+ 个
- ✅ 所有子文档的 parent_id 都指向有效的父文档

### 2. 验证文档长度

**使用脚本**：`src/evrag/parser/check/check_text_length.py`

```bash
cd /remote-home/share/liangZhang/EvRAG
conda run -n evrag python src/evrag/parser/check/check_text_length.py
```

**预期结果**：
- 检查文档长度分布
- 识别超过 512 字符的文档（如果有）
- 显示长度统计信息

### 3. 验证 QA 生成

**检查文件**：`data/qa_pairs/qa_pair.json`

```bash
# 检查空响应
cd /remote-home/share/liangZhang/EvRAG
conda run -n evrag python -c "
import json
with open('data/qa_pairs/qa_pair.json', 'r') as f:
    empty = [line for line in f if '\"raw_resp\": \"[]\"' in line]
    print(f'空响应数量: {len(empty)}')
"
```

**预期结果**：
- `raw_resp` 字段不包含 markdown 代码块标记
- 空响应数量合理（主要是目录类内容）

## 操作指南

### 完整重新生成流程

如果修复后需要重新生成数据，按以下步骤操作：

```bash
cd /remote-home/share/liangZhang/EvRAG

# ========== 步骤1：清空 MongoDB ==========
echo "步骤1: 清空 MongoDB..."
conda run -n evrag python src/evrag/parser/check/clear_mongodb.py

# ========== 步骤2：删除旧的 split_docs.pkl ==========
echo "步骤2: 删除旧的 split_docs.pkl..."
rm -f data/processed_docs/split_docs.pkl

# ========== 步骤3：重新生成数据 ==========
echo "步骤3: 重新生成数据..."
conda run -n evrag python main.py prepare-data --config config/config.yaml --skip-clean

# ========== 步骤4：验证数据 ==========
echo "步骤4: 验证 MongoDB..."
conda run -n evrag python src/evrag/parser/check/check_mongodb.py

echo "步骤5: 检查文档长度..."
conda run -n evrag python src/evrag/parser/check/check_text_length.py

# ========== 步骤5：重新构建索引 ==========
echo "步骤6: 重新构建索引..."
conda run -n evrag python main.py build-index --config config/config.yaml --force

echo "✅ 所有步骤完成！"
```

### 故障排查

#### 如果验证失败

1. **检查 parent_id 错误**：
   - 如果 `parent_id == unique_id` 的错误文档 > 0，说明代码修复未生效
   - 需要检查代码是否正确修复，并重新生成数据

2. **检查文档长度**：
   - 如果大量文档超过 512 字符，检查切分逻辑
   - 确认 Milvus 截断逻辑正常工作

3. **检查 MongoDB 连接**：
   - 确保 MongoDB 服务正在运行
   - 检查配置文件中的 MongoDB 连接信息

#### 常见问题

**Q: 为什么有少量无效的 parent_id？**
A: 这是正常的。这些 parent_id 对应的父文档长度 >= `max_parent_size` (512)，因此没有被包含在 `split_docs.pkl` 中，但它们仍然保存在 MongoDB 中。

**Q: 为什么有些文档超过 512 字符？**
A: 由于 token/字符比例问题和切分器的限制，可能出现这种情况。已添加截断逻辑作为安全措施。

**Q: 为什么 QA 生成有缺失？**
A: 部分文档因为内容太短被过滤，或者 LLM 判断无法生成有意义的 QA 对。这是正常行为。

## 相关文档

- [清空 MongoDB 并重新生成数据指南](./CLEAR_AND_REGEN.md) - 详细的操作步骤
- [阶段二数据准备开发文档](./PHASE2_DATA_PREPARATION.md) - 完整的数据准备和索引构建开发文档（包含PDF解析、文档清洗、文档切分、索引构建、QA生成、数据分析和测试验证）

## 总结

本阶段修复了 5 个关键问题，分析了 2 个重要问题，并提供了完整的验证和操作指南。所有修复都已通过测试和验证，可以安全使用。

**修复统计**：
- 代码修复：5 处
- 问题分析：2 项
- 验证脚本：3 个
- 文档更新：1 个总结文档

**下一步**：
- 继续使用修复后的代码进行数据准备和索引构建
- 如遇到新问题，参考本文档的故障排查部分

