"""
阶段一：服务启动集成测试
"""

import pytest
from pathlib import Path
import urllib.request
import json

from src.evrag.config import get_settings
from src.evrag.client import MongoDBClient
from pymilvus import connections


class TestServicesIntegration:
    """服务集成测试"""

    def test_mongodb_connection(self):
        """测试MongoDB连接"""
        client = MongoDBClient()
        client.connect()
        # 测试ping
        result = client._client.admin.command("ping")
        assert result["ok"] == 1.0
        client.close()

    def test_milvus_connection(self):
        """测试Milvus连接"""
        settings = get_settings()
        db_path = Path(settings.milvus_db_path).absolute()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # 连接Milvus
        # list_connections() 返回元组列表 [(name, handler), ...]
        existing_names = [name for name, _ in connections.list_connections()]
        if "default" not in existing_names:
            connections.connect(alias="default", uri=str(db_path))

        # 验证连接
        current_names = [name for name, _ in connections.list_connections()]
        assert "default" in current_names

    def test_semantic_chunk_service_health(self):
        """测试语义切分服务健康检查"""
        try:
            response = urllib.request.urlopen("http://localhost:6000/health", timeout=5)
            data = json.loads(response.read().decode())
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
        except urllib.error.URLError:
            pytest.skip("语义切分服务未启动，跳过测试")
