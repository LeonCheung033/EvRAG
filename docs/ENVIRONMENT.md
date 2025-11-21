# 环境设置指南

本文档介绍如何设置EvRAG项目的开发环境。

## 前置要求

- Conda (推荐使用 Miniconda 或 Anaconda)
- Python 3.12+
- Git
- CUDA (可选，用于GPU加速)

## 1. 创建Conda环境

```bash
# 从environment.yml创建环境
conda env create -f environment.yml

# 激活环境
conda activate evrag
```

## 2. 安装依赖

### 基础依赖

基础依赖已在`environment.yml`中定义，创建环境时会自动安装，包括：
- Python 3.12
- 开发工具：pytest, ruff, mypy, pre-commit等

### 生产依赖

```bash
# 安装生产依赖
pip install -r requirements.txt
```

### 开发依赖

```bash
# 安装开发依赖
pip install -r requirements-dev.txt
```

## 3. 开发过程中添加依赖

### 添加新依赖

1. **安装依赖**：
   ```bash
   pip install <package_name>
   ```

2. **更新requirements.txt**：
   ```bash
   # 使用pipreqs自动生成（推荐）
   pipreqs . --force
   
   # 或手动添加到requirements.txt
   echo "<package_name>==<version>" >> requirements.txt
   ```

3. **查看依赖树**：
   ```bash
   pipdeptree
   ```

4. **检查依赖冲突**：
   ```bash
   pip check
   ```

### 依赖管理最佳实践

- **生产依赖**：添加到`requirements.txt`
- **开发依赖**：添加到`requirements-dev.txt`
- **Conda依赖**：添加到`environment.yml`（仅限Python版本和基础工具）
- **版本锁定**：使用`pip freeze > requirements.txt`锁定版本（可选）

## 4. CUDA环境验证

### 检查CUDA是否可用

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

## 5. 环境变量配置

### 创建.env文件

```bash
# 复制示例文件
cp .env.example .env

# 编辑.env文件，填入你的配置
vim .env
```

### 必需的环境变量

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

## 6. 验证安装

```bash
# 运行测试
pytest tests/unit/test_config.py -v

# 检查代码质量
bash scripts/check_code_quality.sh

# 查看CLI帮助
python main.py --help
```

## 7. 常见问题

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
cd /remote-home/share/liangZhang/EvRAG

# 检查Python路径
python -c "import sys; print(sys.path)"

# 重新安装包
pip install -e .
```

## 8. 环境清理

```bash
# 删除Conda环境
conda env remove -n evrag

# 清理pip缓存
pip cache purge

# 清理conda缓存
conda clean --all
```

## 9. 开发工具配置

### Pre-commit hooks

```bash
# 安装pre-commit hooks
pre-commit install

# 手动运行
pre-commit run --all-files
```

### IDE配置

推荐使用VS Code或PyCharm，配置Python解释器指向conda环境：

```bash
# 查看conda环境路径
conda env list
```

## 10. 下一步

环境设置完成后，请参考：
- [开发指南](DEVELOPMENT.md) - 了解开发流程
- [测试指南](TESTING.md) - 了解测试规范
- [README.md](../README.md) - 了解项目概览

