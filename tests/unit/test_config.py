"""
配置管理模块的单元测试

测试配置加载、环境变量覆盖、配置文件解析等功能。
"""

import os
import tempfile
from pathlib import Path
import yaml

from src.evrag.config import Settings, get_settings, reload_settings


class TestSettings:
    """Settings类的测试"""

    def test_default_values(self):
        """
        测试默认值

        验证：不提供任何配置时，使用默认值
        """
        # 临时清除可能影响测试的环境变量
        env_backup = {}
        env_keys_to_clear = ["LLM_API_KEY", "llm_api_key"]
        for key in env_keys_to_clear:
            if key in os.environ:
                env_backup[key] = os.environ.pop(key)

        try:
            # 重新加载配置，不使用.env文件
            settings = Settings(_env_file=None)  # 禁用.env文件加载

            # 验证默认值
            assert settings.llm_base_url == "http://localhost:8000/v1"
            assert settings.llm_api_key == "EMPTY"
            assert settings.mongodb_port == 27017
            assert settings.server_port == 8000

            # 验证路径类型
            assert isinstance(settings.data_dir, Path)
            assert isinstance(settings.index_dir, Path)
        finally:
            # 恢复环境变量
            for key, value in env_backup.items():
                os.environ[key] = value
            reload_settings()

    def test_environment_variable_override(self):
        """
        测试环境变量覆盖

        验证：环境变量可以覆盖默认值
        """
        # 设置环境变量
        os.environ["LLM_API_KEY"] = "test_api_key"
        os.environ["MONGODB_PORT"] = "27018"

        try:
            # 重新加载配置以读取环境变量
            settings = reload_settings()

            # 验证环境变量生效
            assert settings.llm_api_key == "test_api_key"
            assert settings.mongodb_port == 27018
        finally:
            # 清理环境变量
            os.environ.pop("LLM_API_KEY", None)
            os.environ.pop("MONGODB_PORT", None)
            reload_settings()  # 重新加载以清除环境变量影响

    def test_yaml_config_file(self):
        """
        测试YAML配置文件加载

        验证：可以从YAML文件加载配置
        """
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "llm_model_name": "test_model",
                "mongodb_port": 27019,
                "server_port": 9000,
            }
            yaml.dump(config_data, f)
            config_file = Path(f.name)

        try:
            # 从配置文件加载（禁用.env文件）
            settings = Settings(config_file=config_file, _env_file=None)

            # 验证配置生效
            assert settings.llm_model_name == "test_model"
            assert settings.mongodb_port == 27019
            assert settings.server_port == 9000
        finally:
            # 清理临时文件
            config_file.unlink()

    def test_path_resolution(self):
        """
        测试路径解析

        验证：相对路径被正确解析为绝对路径
        """
        # 创建临时目录作为base_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            settings = Settings(base_dir=str(base_dir), _env_file=None)

            # 验证路径被解析为绝对路径
            assert settings.data_dir.is_absolute()
            assert settings.data_dir == base_dir / "data"
            assert settings.index_dir.is_absolute()
            assert settings.index_dir == base_dir / "data/saved_index"

    def test_path_combinations(self):
        """
        测试路径组合

        验证：路径正确组合
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            settings = Settings(base_dir=str(base_dir), _env_file=None)

            # 验证完整路径
            expected_bm25_path = base_dir / "data/saved_index/bm25retriever.pkl"
            assert settings.bm25_pickle_path == expected_bm25_path

    def test_optional_fields(self):
        """
        测试可选字段

        验证：可选字段可以为None
        """
        settings = Settings(_env_file=None)

        # 验证可选字段可以为None
        assert settings.pdf_path is None or isinstance(settings.pdf_path, Path)
        assert settings.m3e_small_model_path is None or isinstance(
            settings.m3e_small_model_path, Path
        )

    def test_config_priority(self):
        """
        测试配置优先级

        验证：环境变量 > 配置文件 > 默认值
        """
        # 创建配置文件
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"mongodb_port": 27020}, f)
            config_file = Path(f.name)

        try:
            # 重要：在创建Settings之前设置环境变量
            # 因为pydantic-settings在初始化时读取环境变量
            os.environ["MONGODB_PORT"] = "27021"

            try:
                # 从配置文件加载，环境变量应该覆盖配置文件
                # 注意：环境变量在super().__init__时读取，所以必须在创建Settings之前设置
                settings = Settings(config_file=config_file, _env_file=None)

                # 环境变量应该覆盖配置文件
                assert settings.mongodb_port == 27021, (
                    f"Expected 27021 (from env var), got {settings.mongodb_port} "
                    f"(from config file: 27020). Environment variable should have higher priority."
                )
            finally:
                os.environ.pop("MONGODB_PORT", None)
                reload_settings()
        finally:
            config_file.unlink()


class TestGetSettings:
    """get_settings函数的测试（单例模式）"""

    def test_singleton_pattern(self):
        """
        测试单例模式

        验证：多次调用get_settings返回同一个实例
        """
        settings1 = get_settings()
        settings2 = get_settings()

        # 应该是同一个实例
        assert settings1 is settings2

    def test_reload_settings(self):
        """
        测试重新加载配置

        验证：reload_settings创建新实例
        """
        settings1 = get_settings()
        settings2 = reload_settings()

        # 应该是不同的实例
        assert settings1 is not settings2

        # 新的get_settings应该返回新实例
        settings3 = get_settings()
        assert settings3 is settings2


class TestConfigIntegration:
    """配置集成测试"""

    def test_full_config_loading(self):
        """
        测试完整配置加载流程

        验证：配置文件 + 环境变量的完整流程
        """
        # 创建完整配置文件
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "base_dir": ".",
                "data_dir": "test_data",
                "llm_model_name": "test_model",
                "mongodb_host": "test_host",
                "mongodb_port": 27022,
            }
            yaml.dump(config_data, f)
            config_file = Path(f.name)

        try:
            # 设置环境变量
            os.environ["LLM_API_KEY"] = "env_api_key"

            try:
                settings = Settings(config_file=config_file, _env_file=None)

                # 验证配置文件中的值
                assert settings.llm_model_name == "test_model"
                assert settings.mongodb_host == "test_host"
                assert settings.mongodb_port == 27022

                # 验证环境变量中的值
                assert settings.llm_api_key == "env_api_key"
            finally:
                os.environ.pop("LLM_API_KEY", None)
                reload_settings()
        finally:
            config_file.unlink()
