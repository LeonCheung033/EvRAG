#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清空 MongoDB 中的文档数据，用于重新生成
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.evrag.client import MongoDBClient
from src.evrag.config import get_settings


def clear_mongodb_collection(collection_name: str = "manual_text"):
    """
    清空指定的 MongoDB 集合

    Args:
        collection_name: 要清空的集合名称
    """
    settings = get_settings()

    print("=" * 80)
    print("清空 MongoDB 集合")
    print("=" * 80)
    print(f"数据库: {settings.mongodb_database}")
    print(f"集合: {collection_name}")
    print()

    mongodb_client = MongoDBClient(
        host=settings.mongodb_host,
        port=settings.mongodb_port,
        database=settings.mongodb_database,
    )

    try:
        mongodb_client.connect()
        collection = mongodb_client.get_collection(collection_name)

        # 统计删除前的文档数量
        count_before = collection.count_documents({})
        print(f"删除前文档数量: {count_before}")

        if count_before == 0:
            print("集合已经是空的，无需清空")
            return

        # 确认操作（可以通过环境变量跳过确认）
        import os

        skip_confirm = os.environ.get("SKIP_CONFIRM", "false").lower() == "true"

        if not skip_confirm:
            print(f"\n⚠️  警告: 即将删除 {count_before} 个文档")
            confirm = input("确认删除？(yes/no): ").strip().lower()

            if confirm not in ["yes", "y"]:
                print("操作已取消")
                return

        # 删除所有文档
        result = collection.delete_many({})
        print(f"\n✅ 成功删除 {result.deleted_count} 个文档")

        # 验证
        count_after = collection.count_documents({})
        print(f"删除后文档数量: {count_after}")

        if count_after == 0:
            print("✅ 集合已清空")
        else:
            print(f"⚠️  警告: 仍有 {count_after} 个文档未删除")

    except Exception as e:
        print(f"❌ 错误: {e}")
        raise
    finally:
        mongodb_client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="清空 MongoDB 集合")
    parser.add_argument(
        "--collection",
        "-c",
        default="manual_text",
        help="要清空的集合名称（默认: manual_text）",
    )

    args = parser.parse_args()
    clear_mongodb_collection(args.collection)
