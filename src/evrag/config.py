"""
配置管理模块

使用pydantic-settings管理配置,支持：
- 环境变量读取
- YAML配置文件
- 默认值设置
"""

from pathlib import Path
from typing import Optional, Any, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from watchdog.events import FileSystemEventHandler


class Settings(BaseSettings):
    """
    应用配置类

    使用pydantic-settings管理所有配置项。
    配置可以通过以下方式提供（优先级从高到低）：
    1. 环境变量
    2. YAML配置文件
    3. 默认值
    """

    # ========== 基础路径配置 ==========
    base_dir: Path = Field(
        # cwd是启动 Python 进程时所在的 shell 目录。
        default=Path.cwd(),
        description="项目根目录",
    )

    # ========== 数据路径配置 ==========
    data_dir: Path = Field(
        default=Path("data"),
        description="数据目录路径",
    )
    pdf_path: Optional[Path] = Field(
        default=None,
        description="PDF文件路径",
    )
    test_doc_path: Optional[Path] = Field(
        default=None,
        description="测试文档路径",
    )
    stopwords_path: Optional[Path] = Field(
        default=None,
        description="停用词路径",
    )
    image_save_dir: Path = Field(
        default=Path("data/saved_images"),
        description="图片保存目录",
    )
    raw_docs_path: Path = Field(
        default=Path("data/processed_docs/raw_docs.pkl"),
        description="原始文档路径",
    )
    clean_docs_path: Path = Field(
        default=Path("data/processed_docs/clean_docs.pkl"),
        description="清洗后文档pickle文件路径",
    )
    split_docs_path: Path = Field(
        default=Path("data/processed_docs/split_docs.pkl"),
        description="切分后文档pickle文件路径",
    )

    # ========== 索引路径配置 ==========
    index_dir: Path = Field(
        default=Path("data/saved_index"), description="索引保存目录"
    )

    bm25_pickle_path: Path = Field(
        default=Path("data/saved_index/bm25retriever.pkl"),
        description="BM25检索器pickle文件路径",
    )

    tfidf_pickle_path: Path = Field(
        default=Path("data/saved_index/tfidfretriever.pkl"),
        description="TFIDF检索器pickle文件路径",
    )

    milvus_db_path: Path = Field(
        default=Path("data/saved_index/milvus.db"), description="Milvus数据库路径"
    )

    faiss_db_path: Path = Field(
        default=Path("data/saved_index/faiss.db"), description="Faiss数据库路径"
    )

    faiss_qwen_db_path: Path = Field(
        default=Path("data/saved_index/faiss_qwen.db"),
        description="Qwen Faiss数据库路径",
    )

    # ========== 模型路径配置 ==========
    models_dir: Path = Field(default=Path("models"), description="模型目录路径")

    m3e_small_model_path: Optional[Path] = Field(
        default=None, description="M3E-small模型路径"
    )

    bge_m3_model_path: Optional[Path] = Field(
        default=None, description="BGE-M3模型路径"
    )

    qwen3_embedding_model_path: Optional[Path] = Field(
        default=None, description="Qwen3 Embedding模型路径"
    )

    qwen3_reranker_model_path: Optional[Path] = Field(
        default=None, description="Qwen3 Reranker模型路径"
    )

    bge_reranker_model_path: Optional[Path] = Field(
        default=None, description="BGE Reranker模型路径"
    )

    bge_reranker_tuned_model_path: Optional[Path] = Field(
        default=None, description="BGE Reranker微调模型路径"
    )

    # ========== LLM配置 ==========
    device: str = Field(default="cuda", description="设备类型")

    # 本地vLLM服务配置（基线模型）
    local_llm_model_name: str = Field(
        default="qwen3_8b", description="本地vLLM模型名称（基线模型）"
    )
    local_llm_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="本地vLLM API基础URL（基线模型）",
    )
    local_llm_api_key: str = Field(
        default="EMPTY", description="本地vLLM API密钥（通常为EMPTY，基线模型）"
    )

    # 微调模型vLLM服务配置
    finetuned_llm_model_name: str = Field(
        default="qwen3_lora_sft", description="微调模型vLLM模型名称"
    )
    finetuned_llm_base_url: str = Field(
        default="http://localhost:8001/v1", description="微调模型vLLM API基础URL"
    )
    finetuned_llm_api_key: str = Field(
        default="EMPTY", description="微调模型vLLM API密钥（通常为EMPTY）"
    )

    # 豆包API配置
    doubao_api_key: str = Field(default="", description="豆包API密钥")
    doubao_base_url: str = Field(
        default="https://ark.cn-beijing.volces.com/api/v3", description="豆包API基础URL"
    )
    doubao_model_name: str = Field(
        default="doubao-seed-1-6-lite-251015", description="豆包推理接入点ID"
    )

    # Deepseek API配置
    deepseek_api_key: str = Field(default="", description="Deepseek API密钥")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com", description="Deepseek API基础URL"
    )
    deepseek_model_name: str = Field(
        default="deepseek-chat", description="Deepseek模型名称"
    )

    # ========== SiliconFlow API配置（用于Reranker） ==========
    siliconflow_api_key: str = Field(default="", description="SiliconFlow API密钥")
    siliconflow_base_url: str = Field(
        default="https://api.siliconflow.cn/v1", description="SiliconFlow API基础URL"
    )
    siliconflow_reranker_model: str = Field(
        default="Qwen/Qwen3-Reranker-8B", description="SiliconFlow Reranker模型名称"
    )

    # 兼容旧配置（保留向后兼容）
    llm_model_name: str = Field(
        default="qwen3_lora_sft_int4",
        description="LLM模型名称（兼容字段，等同于local_llm_model_name）",
    )
    llm_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="LLM API基础URL（兼容字段，等同于local_llm_base_url）",
    )
    llm_api_key: str = Field(
        default="EMPTY", description="LLM API密钥（兼容字段，等同于local_llm_api_key）"
    )

    # ========== GPU设备分配配置 ==========
    semantic_chunk_gpu_id: int = Field(
        default=0, description="语义切分服务使用的GPU设备ID"
    )
    milvus_retriever_gpu_id: int = Field(
        default=1, description="Milvus检索器使用的GPU设备ID"
    )
    reranker_gpu_id: int = Field(default=2, description="Reranker使用的GPU设备ID")
    vllm_gpu_ids: List[int] = Field(
        default_factory=lambda: [3, 5, 6],
        description="vLLM服务使用的GPU设备ID列表（tensor parallelism）",
    )

    # ========== MongoDB配置 ==========
    mongodb_host: str = Field(default="localhost", description="MongoDB主机地址")

    mongodb_port: int = Field(default=27017, description="MongoDB端口")

    mongodb_database: str = Field(default="evrag", description="MongoDB数据库名称")

    # ========== PDF解析配置 ==========
    pdf_min_filter_pages: int = Field(
        default=4,
        description="最小页码（从0开始），小于此页码的页面将被跳过（用于跳过封面、目录等）",
    )
    pdf_max_filter_pages: Optional[int] = Field(
        default=247,
        description="最大页码（从0开始），大于此页码的页面将被跳过，None表示不限制",
    )
    pdf_page_clip: int = Field(
        default=50, description="页面底部裁剪像素数（用于去除页眉页脚）"
    )

    # ========== 服务配置 ==========
    server_host: str = Field(default="0.0.0.0", description="服务器监听地址")

    server_port: int = Field(default=8000, description="服务器端口")

    semantic_chunk_url: str = Field(
        default="http://0.0.0.0:6000/v1/semantic-chunks", description="语义分块服务URL"
    )

    # ========== Pydantic配置 ==========
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=False,  # 大小写不敏感
        extra="ignore",
    )

    def __init__(self, config_file: Optional[Path] = None, **kwargs):
        """
        初始化配置

        Args:
            config_file: YAML配置文件路径（可选）
            **kwargs: 其他配置参数（优先级最高，会覆盖配置文件和默认值）
        """
        # 如果提供了配置文件，从文件加载
        file_config: dict[str, Any] = {}
        if config_file and config_file.exists():
            import yaml

            with open(config_file, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f) or {}

        # pydantic-settings的优先级是：环境变量 > kwargs > 默认值
        # 但是，如果我们通过kwargs传入配置文件的值，环境变量可能无法覆盖
        # 解决方案：先调用父类（只传入kwargs，不传入配置文件的值）
        # 这样环境变量可以正确覆盖默认值
        # 然后，对于配置文件中的值，只更新那些没有被环境变量或kwargs设置的字段

        # 先处理环境变量和kwargs（不包含配置文件的值）
        super().__init__(**kwargs)

        # 然后，用配置文件的值更新那些使用默认值的字段
        # 这样可以保证优先级：环境变量 > kwargs > 配置文件 > 默认值
        if file_config:
            for key, file_value in file_config.items():
                if key in type(self).model_fields:
                    # 获取字段的默认值
                    field_info = type(self).model_fields[key]
                    default_value = field_info.default
                    current_value = getattr(self, key)

                    # 如果当前值等于默认值，说明没有被环境变量或kwargs覆盖
                    # 此时可以使用配置文件的值
                    if current_value == default_value:
                        # 直接设置，pydantic会自动进行类型验证和转换
                        try:
                            setattr(self, key, file_value)
                        except Exception:
                            # 如果设置失败（类型不匹配等），跳过
                            pass

        # 解析相对路径为绝对路径（基于base_dir）
        self._resolve_paths()

    def _resolve_paths(self) -> None:
        """
        解析所有路径为绝对路径

        如果路径是相对路径，则基于base_dir解析为绝对路径
        """
        # 先确保base_dir是Path类型
        if isinstance(self.base_dir, str):
            self.base_dir = Path(self.base_dir)
        base = Path(self.base_dir).resolve()

        # 处理所有Path类型的字段
        for field_name, field_info in type(self).model_fields.items():
            field_value = getattr(self, field_name)

            # 跳过None值
            if field_value is None:
                continue

            # 检查字段类型是否是Path
            field_type = field_info.annotation
            is_path_type = False

            # 处理Optional[Path]类型
            if hasattr(field_type, "__origin__"):
                # 获取泛型参数（例如Optional[Path] -> (Path, type(None))）
                args = getattr(field_type, "__args__", ())
                if Path in args:
                    is_path_type = True
            elif field_type == Path:
                is_path_type = True

            if is_path_type:
                # 第一步：如果是字符串，先转换为Path
                if isinstance(field_value, str):
                    field_value = Path(field_value)
                    setattr(self, field_name, field_value)

                # 第二步：如果是Path类型且是相对路径，解析为绝对路径
                if isinstance(field_value, Path) and not field_value.is_absolute():
                    # 如果是数据或模型相关路径，基于base_dir解析
                    if "data" in str(field_value) or "models" in str(field_value):
                        setattr(self, field_name, base / field_value)
                    else:
                        setattr(self, field_name, field_value.resolve())


# 全局配置实例（单例模式）
_settings: Optional[Settings] = None


def get_settings(config_file: Optional[Path] = None) -> Settings:
    """
    获取配置实例（单例模式）

    Args:
        config_file: 配置文件路径（可选）。如果为None，会自动查找以下位置的配置文件：
            - config/config.yaml（相对于当前工作目录）
            - 项目根目录下的config/config.yaml

    Returns:
        Settings实例
    """
    # Python 的设计哲学：只要在函数内有 = 赋值，就默认这个变量是局部的（除非声明 global）。
    # 所以这里我们可能要修改全局变量，所以需要声明 global
    global _settings
    if _settings is None:
        # 如果没有提供配置文件路径，尝试查找默认配置文件
        if config_file is None:
            # 首先尝试当前工作目录下的 config/config.yaml
            default_config = Path.cwd() / "config" / "config.yaml"
            if default_config.exists():
                config_file = default_config
            else:
                # 尝试项目根目录（通过查找包含 src 目录的父目录）
                current = Path.cwd()
                while current != current.parent:
                    potential_config = current / "config" / "config.yaml"
                    if potential_config.exists():
                        config_file = potential_config
                        break
                    current = current.parent

        _settings = Settings(config_file=config_file)
    return _settings


def reload_settings(config_file: Optional[Path] = None) -> Settings:
    """
    重新加载配置

    Args:
        config_file: 配置文件路径（可选）

    Returns:
        新的Settings实例
    """
    global _settings
    _settings = Settings(config_file=config_file)
    return _settings


# 热重载配置
class ConfigReloader(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith((".env", "config.yaml")):
            reload_settings()
