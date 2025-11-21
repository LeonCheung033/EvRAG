"""
MongoDB客户端

管理MongoDB数据库连接和操作。
"""

from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, ConfigurationError

from ..config import get_settings


class MongoDBClient:
    """
    MongoDB客户端

    单例模式管理MongoDB连接。
    """

    _client: Optional[MongoClient] = None
    _db: Optional[Database] = None
    _settings = None

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        auth_source: str = "admin",
    ):
        """
        初始化MongoDB客户端

        Args:
            host: MongoDB主机地址
            port: MongoDB端口
            database: 数据库名称
            username: 用户名（可选）
            password: 密码（可选）
            auth_source: 认证数据库
        """
        settings = get_settings()

        self.host = host or settings.mongodb_host
        self.port = port or settings.mongodb_port
        self.database = database or settings.mongodb_database
        self.username = username
        self.password = password
        self.auth_source = auth_source

        # 连接参数
        self.max_pool_size = 100
        self.connect_timeout = 5000  # 毫秒
        self.socket_timeout = 3000  # 毫秒

    def _build_connection_uri(self) -> str:
        """构建MongoDB连接URI"""
        if self.username and self.password:
            return (
                f"mongodb://{self.username}:{self.password}@"
                f"{self.host}:{self.port}/?authSource={self.auth_source}"
            )
        return f"mongodb://{self.host}:{self.port}"

    def connect(self) -> None:
        """建立MongoDB连接"""
        if self._client is None:
            try:
                self._client = MongoClient(
                    self._build_connection_uri(),
                    maxPoolSize=self.max_pool_size,
                    connectTimeoutMS=self.connect_timeout,
                    socketTimeoutMS=self.socket_timeout,
                    serverSelectionTimeoutMS=5000,
                )

                # 验证连接
                self._client.admin.command("ping")
                self._db = self._client[self.database]
                print(f"Successfully connected to MongoDB at {self.host}:{self.port}")

            except ConfigurationError as e:
                raise RuntimeError(f"MongoDB configuration error: {str(e)}")
            except ConnectionFailure as e:
                raise RuntimeError(f"Failed to connect to MongoDB: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Unexpected MongoDB connection error: {str(e)}")

    def get_db(self) -> Database:
        """
        获取数据库实例

        Returns:
            MongoDB数据库实例
        """
        if self._client is None:
            self.connect()
        if self._db is None:
            raise RuntimeError("MongoDB connection not established")
        return self._db

    def get_collection(self, collection_name: str) -> Collection:
        """
        获取集合实例

        Args:
            collection_name: 集合名称

        Returns:
            MongoDB集合实例
        """
        return self.get_db()[collection_name]

    def close(self) -> None:
        """关闭MongoDB连接"""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            print("MongoDB connection closed")

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
