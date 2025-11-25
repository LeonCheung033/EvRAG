#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
获取 MongoDB 连接字符串
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.evrag.config import get_settings

def get_mongodb_connection_strings():
    """获取 MongoDB 连接字符串"""
    settings = get_settings()
    
    host = settings.mongodb_host
    port = settings.mongodb_port
    database = settings.mongodb_database
    
    print("=" * 80)
    print("MongoDB 连接信息")
    print("=" * 80)
    print(f"主机: {host}")
    print(f"端口: {port}")
    print(f"数据库: {database}")
    print()
    
    # 标准连接字符串（无认证）
    standard_uri = f"mongodb://{host}:{port}/{database}"
    print("【标准连接字符串】")
    print(standard_uri)
    print()
    
    # 标准连接字符串（带认证，如果需要）
    # 注意：如果 MongoDB 需要认证，需要添加用户名和密码
    # standard_uri_auth = f"mongodb://username:password@{host}:{port}/{database}?authSource=admin"
    # print("【标准连接字符串（带认证）】")
    # print(standard_uri_auth)
    # print()
    
    # SRV 连接字符串（通常用于 MongoDB Atlas 云服务）
    # 本地 MongoDB 通常不使用 SRV，但提供格式供参考
    print("【SRV 连接字符串格式（通常用于 MongoDB Atlas）】")
    print("mongodb+srv://<username>:<password>@<cluster>.mongodb.net/<database>?retryWrites=true&w=majority")
    print()
    print("注意：本地 MongoDB 通常不使用 SRV 连接字符串")
    print()
    
    # 用于 MongoDB 扩展的连接字符串
    print("【用于 MongoDB 扩展的连接字符串】")
    print("=" * 80)
    print("在 VS Code 的 MongoDB 扩展中使用以下连接字符串：")
    print()
    print(f"mongodb://{host}:{port}")
    print()
    print("或者如果指定数据库：")
    print(f"mongodb://{host}:{port}/{database}")
    print()
    
    # 提供不同格式的连接字符串
    print("【其他常用格式】")
    print("-" * 80)
    
    # 1. 仅主机和端口（用于连接后选择数据库）
    print("1. 仅主机和端口：")
    print(f"   mongodb://{host}:{port}")
    print()
    
    # 2. 包含数据库
    print("2. 包含数据库：")
    print(f"   mongodb://{host}:{port}/{database}")
    print()
    
    # 3. 包含选项（推荐）
    print("3. 包含常用选项（推荐）：")
    print(f"   mongodb://{host}:{port}/{database}?retryWrites=true&w=majority")
    print()
    
    # 4. 如果 MongoDB 需要认证
    print("4. 如果需要认证（替换 username 和 password）：")
    print(f"   mongodb://username:password@{host}:{port}/{database}?authSource=admin")
    print()
    
    print("=" * 80)
    print("提示：")
    print("- 如果 MongoDB 没有启用认证，使用格式 1 或 2 即可")
    print("- 如果使用 MongoDB 扩展，通常使用格式 1 或 2")
    print("- 连接后可以在扩展中选择要查看的数据库")
    print("=" * 80)

if __name__ == "__main__":
    get_mongodb_connection_strings()

