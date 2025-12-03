# 环境配置指南

本文档总结了EvRAG项目的环境配置相关内容，包括环境设置、CUDA配置和工具配置。

## 📋 目录

- [环境设置](#环境设置)
- [CUDA配置](#cuda配置)
- [工具配置](#工具配置)

---

## 🔧 环境设置

### 前置要求

- **Conda**: 推荐使用 Miniconda 或 Anaconda
- **Python**: 3.12+
- **Git**: 版本控制工具
- **CUDA**: 可选，用于GPU加速

### 创建Conda环境

```bash
# 从environment.yml创建环境
conda env create -f environment.yml

# 激活环境
conda activate evrag
```

### 安装依赖

#### 基础依赖

基础依赖已在`environment.yml`中定义，创建环境时会自动安装，包括：
- Python 3.12
- 开发工具：pytest, ruff, mypy, pre-commit等

#### 生产依赖

```bash
pip install -r requirements.txt
```

#### 开发依赖

```bash
pip install -r requirements-dev.txt
```

### 环境变量配置

#### 创建.env文件

```bash
# 复制示例文件
cp .env.example .env

# 编辑.env文件，填入你的配置
vim .env
```

#### 必需的环境变量

- `LLM_API_KEY`: LLM API密钥（本地LLM可为"EMPTY"）
- `LLM_BASE_URL`: LLM API基础URL（默认：http://localhost:8000/v1）
- `LLM_MODEL_NAME`: LLM模型名称
- `MONGODB_HOST`: MongoDB主机地址（默认：localhost）
- `MONGODB_PORT`: MongoDB端口（默认：27017）
- `MONGODB_DATABASE`: MongoDB数据库名称（默认：evrag）

### 配置文件

项目支持YAML配置文件，示例文件位于`config/config.example.yaml`：

```bash
# 复制示例配置
cp config/config.example.yaml config/config.yaml

# 编辑配置文件
vim config/config.yaml
```

### 验证安装

```bash
# 运行测试
pytest tests/unit/test_config.py -v

# 检查代码质量
bash scripts/utils/check_code_quality.sh

# 查看CLI帮助
python main.py --help
```

---

## 🖥️ CUDA配置

### CUDA环境验证

#### 检查CUDA是否可用

```bash
# 检查PyTorch是否支持CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 检查CUDA版本
nvidia-smi

# 检查PyTorch CUDA版本
python -c "import torch; print(torch.version.cuda)"
```

### 安装CUDA版本的PyTorch

```bash
# 根据CUDA版本安装PyTorch
# 例如：CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU版本
pip install torch torchvision torchaudio
```

### CUDA升级注意事项

**重要提示**：
- PyTorch通常通过conda/pip安装CUDA运行时库，不需要系统级CUDA工具包
- 如果只需要编译FlashAttention等扩展，可以考虑使用conda安装CUDA工具包
- 不推荐升级系统级CUDA工具包，可能影响其他项目

---

## ⚙️ 工具配置

### GPU设备分配

系统推荐使用8张NVIDIA RTX A6000 GPU（每张49GB显存），分配方案如下：

| GPU ID | 显存使用 | 状态 | 分配用途 |
|--------|---------|------|---------|
| GPU 0  | ~1GB    | 空闲 | **语义切分服务** |
| GPU 1  | ~2-3GB  | 空闲 | **Milvus检索器** |
| GPU 2  | ~2-3GB  | 空闲 | **Reranker** |
| GPU 3-5| ~15GB   | 空闲 | **vLLM服务（3卡并行）** |

### 服务配置

#### 语义切分服务

- **GPU**: GPU 0
- **模型**: M3E-small
- **显存需求**: ~1GB
- **配置项**: `semantic_chunk_gpu_id: 0`
- **服务端口**: 6000

#### Milvus检索器

- **GPU**: GPU 1
- **模型**: BGE-M3
- **显存需求**: ~2-3GB
- **配置项**: `milvus_retriever_gpu_id: 1`

#### Reranker

- **GPU**: GPU 2
- **模型**: BGE-Reranker-v2-m3（微调版）
- **显存需求**: ~2-3GB
- **配置项**: `reranker_gpu_id: 2`

#### vLLM服务

- **GPU**: GPU 3, 4
- **模型**: Qwen3-8B（微调版）
- **显存需求**: ~15GB/卡
- **配置项**: `vllm_gpu_ids: [3, 4]`
- **服务端口**: 8001

### 性能监控

使用GPU监控脚本实时监控GPU使用情况：

```bash
# 启动GPU监控
./scripts/utils/monitor_gpu.sh

# 监控日志保存在 logs/gpu_monitor.log
```

---

## 🐛 常见问题

### 问题1：Conda环境创建失败

**解决方案**：
- 检查`environment.yml`中的包版本是否可用
- 尝试更新conda：`conda update conda`
- 清除conda缓存：`conda clean --all`
- 检查网络连接

### 问题2：CUDA不可用

**解决方案**：
- 确认已安装NVIDIA驱动
- 检查PyTorch是否安装了CUDA版本
- 参考PyTorch官方文档安装对应CUDA版本的PyTorch
- 如果不需要GPU，可以使用CPU版本的PyTorch

### 问题3：依赖冲突

**解决方案**：
```bash
# 查看冲突详情
pip check

# 使用pip-tools解决冲突
pip install pip-tools
pip-compile requirements.in
```

### 问题4：导入错误

**解决方案**：
```bash
# 确保在项目根目录
cd /path/to/EvRAG

# 检查Python路径
python -c "import sys; print(sys.path)"

# 重新安装包
pip install -e .
```

---

## 📚 相关文档

- **详细环境设置**: `dev_docs/environment/ENVIRONMENT.md`
- **CUDA升级分析**: `dev_docs/environment/CUDA_UPGRADE_ANALYSIS.md`
- **工具配置详情**: `dev_docs/environment/PHASE2_TOOLS_CONFIG.md`
- **主README**: `README.md`

