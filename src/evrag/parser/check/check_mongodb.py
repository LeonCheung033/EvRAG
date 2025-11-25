#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查 MongoDB 中的数据是否正确
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.evrag.client import MongoDBClient
from src.evrag.config import get_settings

def check_mongodb():
    settings = get_settings()
    
    print("=" * 80)
    print("检查 MongoDB 中的数据")
    print("=" * 80)
    print(f"数据库: {settings.mongodb_database}")
    print(f"集合: manual_text")
    print()
    
    mongodb_client = MongoDBClient(
        host=settings.mongodb_host,
        port=settings.mongodb_port,
        database=settings.mongodb_database,
    )
    
    try:
        mongodb_client.connect()
        collection = mongodb_client.get_collection("manual_text")
        
        # 统计总数
        total_count = collection.count_documents({})
        print(f"总文档数: {total_count}")
        
        # 统计父文档（无 parent_id）
        parent_count = collection.count_documents({"metadata.parent_id": {"$exists": False}})
        print(f"父文档数（无 parent_id）: {parent_count}")
        
        # 统计子文档（有 parent_id）
        child_count = collection.count_documents({"metadata.parent_id": {"$exists": True}})
        print(f"子文档数（有 parent_id）: {child_count}")
        
        # 检查错误的文档（parent_id == unique_id）
        # 这个查询比较复杂，需要检查每个文档
        print("\n检查错误的文档（parent_id == unique_id）...")
        error_count = 0
        error_samples = []
        
        for doc in collection.find({"metadata.parent_id": {"$exists": True}}).limit(1000):
            unique_id = doc.get("metadata", {}).get("unique_id")
            parent_id = doc.get("metadata", {}).get("parent_id")
            if unique_id and parent_id and unique_id == parent_id:
                error_count += 1
                if len(error_samples) < 3:
                    error_samples.append({
                        "unique_id": unique_id[:16] + "...",
                        "parent_id": parent_id[:16] + "..."
                    })
        
        if error_count == 0:
            print("✅ 没有发现 parent_id == unique_id 的错误文档")
        else:
            print(f"❌ 发现 {error_count} 个错误文档（parent_id == unique_id）")
            print("示例:")
            for sample in error_samples:
                print(f"  - unique_id={sample['unique_id']}, parent_id={sample['parent_id']}")
        
        # 检查父子关系
        print("\n检查父子关系...")
        # 获取所有父文档的 unique_id
        parent_ids = set()
        for doc in collection.find({"metadata.parent_id": {"$exists": False}}):
            unique_id = doc.get("metadata", {}).get("unique_id")
            if unique_id:
                parent_ids.add(unique_id)
        
        # 检查子文档的 parent_id 是否指向有效的父文档
        child_parent_ids = set()
        invalid_count = 0
        for doc in collection.find({"metadata.parent_id": {"$exists": True}}).limit(1000):
            parent_id = doc.get("metadata", {}).get("parent_id")
            if parent_id:
                child_parent_ids.add(parent_id)
                if parent_id not in parent_ids:
                    invalid_count += 1
        
        print(f"父文档 unique_id 数量: {len(parent_ids)}")
        print(f"子文档引用的 parent_id 数量: {len(child_parent_ids)}")
        valid_parent_ids = child_parent_ids & parent_ids
        print(f"有效的 parent_id: {len(valid_parent_ids)}")
        
        if invalid_count > 0:
            print(f"⚠️  有 {invalid_count} 个子文档的 parent_id 在父文档中未找到（可能是父文档长度超过阈值）")
        else:
            print("✅ 所有子文档的 parent_id 都指向有效的父文档")
        
        # 显示示例
        print("\n【示例文档】")
        # 父文档示例
        parent_sample = collection.find_one({"metadata.parent_id": {"$exists": False}})
        if parent_sample:
            unique_id = parent_sample.get("metadata", {}).get("unique_id", "")
            print(f"父文档示例: unique_id={unique_id[:16]}..., 无 parent_id")
        
        # 子文档示例
        child_sample = collection.find_one({"metadata.parent_id": {"$exists": True}})
        if child_sample:
            unique_id = child_sample.get("metadata", {}).get("unique_id", "")
            parent_id = child_sample.get("metadata", {}).get("parent_id", "")
            print(f"子文档示例: unique_id={unique_id[:16]}..., parent_id={parent_id[:16]}...")
            if parent_id in parent_ids:
                print("  ✓ parent_id 指向有效的父文档")
            else:
                print("  ✗ parent_id 未找到对应的父文档")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        mongodb_client.close()
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    check_mongodb()

