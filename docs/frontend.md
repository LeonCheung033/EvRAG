# 前端实现指南

本文档总结了EvRAG项目的Gradio前端实现和RAG服务API。

## 📋 目录

- [RAG服务API](#rag服务api)
- [Gradio前端界面](#gradio前端界面)
- [图片处理](#图片处理)
- [性能监控](#性能监控)

---

## 🚀 RAG服务API

### 核心服务文件

- **文件**: `src/evrag/server/rag_server.py`
- **功能**: 基于FastAPI实现的RAG对话服务
- **端口**: 8002（8001已被vLLM微调模型占用）

### 组件初始化

- ✅ BM25检索器初始化（从已有索引加载）
- ✅ Milvus检索器初始化（从已有索引加载）
- ✅ BGE Reranker初始化（使用微调模型）
  - 模型路径：`models/finetuned/bge_reranker/runs/checkpoints/checkpoint_0`
- ✅ LocalLLMClient初始化（使用微调模型）
  - 服务地址：`http://localhost:8001/v1`
  - 模型名称：`qwen3_lora_sft`

### API接口

#### POST /chat - 非流式聊天接口

**功能**: 支持对话历史管理，返回答案、引用页码、相关图片、性能指标

**请求参数**:
```json
{
  "query": "用户问题",
  "history": [
    {"role": "user", "content": "历史问题"},
    {"role": "assistant", "content": "历史答案"}
  ],
  "bm25_topk": 10,
  "milvus_topk": 10,
  "reranker_topk": 5,
  "stream": false
}
```

**响应格式**:
```json
{
  "answer": "生成的答案",
  "references": [1, 2, 3],
  "images": [
    {
      "title": "图片标题",
      "path": "图片路径"
    }
  ],
  "performance": {
    "retrieval_time": 0.123,
    "rerank_time": 0.045,
    "generation_time": 1.234,
    "total_time": 1.402
  }
}
```

#### POST /chat/stream - 流式聊天接口

**功能**: 实时返回token，最后返回完整元数据（引用、图片、性能指标）

**响应格式**: SSE (Server-Sent Events) 流式响应

**示例**:
```
data: {"token": "答案"}
data: {"token": "的"}
data: {"token": "第一部分"}
...
data: {"done": true, "references": [1, 2], "images": [...], "performance": {...}}
```

#### GET /health - 健康检查接口

**功能**: 检查服务健康状态

**响应格式**:
```json
{
  "status": "healthy",
  "components": {
    "bm25_retriever": "ready",
    "milvus_retriever": "ready",
    "reranker": "ready",
    "llm_client": "ready"
  }
}
```

### 完整RAG流程集成

1. **检索阶段**: BM25 + Milvus混合检索
2. **文档合并**: 使用 `merge_docs` 函数去重
3. **重排序**: 使用BGE Reranker对文档重排序
4. **生成阶段**: 使用微调LLM生成答案
5. **后处理**: 提取答案、引用页码、相关图片

### 对话历史管理

- ✅ 支持多轮对话上下文
- ✅ 上下文窗口管理：
  - 安全长度：4096 tokens
  - 当历史超过安全长度时：
    1. 保留最近2轮对话
    2. 对剩余历史使用LLM进行总结
    3. 将总结作为第一条用户消息

---

## 🎨 Gradio前端界面

### 核心应用文件

- **文件**: `src/evrag/server/gradio_app.py`
- **端口**: 8080

### 界面组件

#### 聊天界面

- `gr.Chatbot` 组件显示对话历史
- 输入框支持多行输入
- 发送按钮和清除按钮
- 流式输出开关

#### 参数配置面板

- **BM25 TopK**: 滑块（1-20，默认10）
- **Milvus TopK**: 滑块（1-20，默认10）
- **Reranker TopK**: 滑块（1-10，默认5）
- **思考模式**: 开关

#### 结果显示区域

- **答案详情**: Markdown格式，支持引用高亮
- **相关图片Gallery**: 2x2布局，支持点击查看大图
- **性能指标面板**: 显示各阶段耗时

### 功能实现

#### 流式输出

- ✅ 实时更新Chatbot组件
- ✅ 使用SSE格式解析流式响应
- ✅ 流式输出完成后显示图片和性能指标

#### 非流式输出

- ✅ 一次性显示完整答案
- ✅ 同时显示图片和性能指标

#### 图片显示

- ✅ 从metadata中提取图片信息
- ✅ 支持多种路径格式（image_path、url、path）
- ✅ 图片路径验证和错误处理
- ✅ Gallery组件展示（2x2布局）

#### 性能指标显示

- ✅ 检索时间（BM25 + Milvus）
- ✅ 重排序时间
- ✅ 生成时间
- ✅ 总响应时间
- ✅ Markdown表格格式展示

### 启动方式

#### 方式1：使用启动脚本（推荐）

```bash
./scripts/deployment/start_gradio.sh
```

#### 方式2：直接运行

```bash
python -m src.evrag.server.gradio_app
```

#### 方式3：使用uvicorn

```bash
uvicorn src.evrag.server.gradio_app:app --host 0.0.0.0 --port 8080
```

---

## 🖼️ 图片处理

### 图片提取

**模块**: `src/evrag/parser/image_handler.py`

**功能**: 从PDF中提取图片并保存

**处理步骤**:
1. 遍历PDF每一页
2. 提取页面中的图片
3. 保存图片到指定目录
4. 记录图片信息（标题、路径等）

**配置参数**:
```yaml
image_save_dir: "data/saved_images"
```

### 图片关联

**功能**: 将图片与文档关联

**实现**:
- 在Document的metadata中添加`images_info`字段
- 包含图片标题、路径等信息

**格式**:
```json
{
  "images_info": [
    {
      "title": "图片标题",
      "image_path": "data/saved_images/image_001.jpg",
      "page": 1
    }
  ]
}
```

### 图片显示

**功能**: 在Gradio界面中显示相关图片

**实现**:
- 从检索到的文档中提取图片信息
- 使用Gradio的Gallery组件展示
- 支持点击查看大图

### 图片测试

**工具**: `scripts/utils/test_image_loading.py`

**功能**: 测试图片加载功能，查找包含图片的文档

**使用方法**:
```bash
python scripts/utils/test_image_loading.py
```

---

## 📊 性能监控

### 性能指标

系统记录以下性能指标：

- **检索时间**: BM25和Milvus检索的总耗时
- **重排序时间**: BGE Reranker重排序耗时
- **生成时间**: LLM生成答案耗时
- **总响应时间**: 从接收到请求到返回响应的总时间

### 性能显示

在Gradio界面中以Markdown表格格式显示：

```
| 指标 | 时间 | 指标 | 时间 |
|------|------|------|------|
| **检索时间** | 0.123秒 | **重排序时间** | 0.045秒 |
| **生成时间** | 1.234秒 | **总响应时间** | 1.402秒 |
```

---

## 🔧 配置说明

### RAG服务配置

在`config/config.yaml`中配置：

```yaml
# RAG服务配置
server_host: "0.0.0.0"
server_port: 8002

# LLM服务配置
llm_base_url: "http://localhost:8001/v1"
llm_model_name: "qwen3_lora_sft"

# 检索参数
bm25_topk: 10
milvus_topk: 10
reranker_topk: 5
```

### Gradio配置

在`src/evrag/server/gradio_app.py`中配置：

```python
# Gradio配置
GRADIO_PORT = 8080
GRADIO_HOST = "0.0.0.0"
RAG_SERVICE_URL = "http://localhost:8002"
```

---

## 📚 相关文档

- **Gradio实现总结**: `dev_docs/Frontend-implementation/gradio_implementation_summary.md`
- **Gradio实现计划**: `dev_docs/Frontend-implementation/gradio_implement.plan.md`
- **图片测试指南**: `dev_docs/Frontend-implementation/image_testing_guide.md`
- **主README**: `README.md`

