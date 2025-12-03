#!/usr/bin/env python3
"""
测试图片加载功能
查找哪些文档包含图片，以及哪些问题可能触发图片显示
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from src.evrag.retriever import BM25Retriever, MilvusRetriever
from src.evrag.config import get_settings

settings = get_settings()


def find_documents_with_images():
    """查找包含图片的文档"""
    print("=" * 80)
    print("查找包含图片的文档...")
    print("=" * 80)
    
    # 初始化检索器（这会加载索引）
    print("\n正在加载检索器...")
    bm25_retriever = BM25Retriever(docs=None, retrieve=True)
    milvus_retriever = MilvusRetriever(docs=None, retrieve=True)
    
    # 从BM25检索器获取所有文档（如果可能）
    # 注意：BM25Retriever可能不直接暴露所有文档，我们需要通过检索来测试
    
    # 测试一些可能包含图片的问题
    test_queries = [
        "仪表盘",
        "中控屏幕",
        "触摸屏",
        "按钮",
        "方向盘",
        "座椅",
        "充电接口",
        "显示屏",
        "界面",
        "图标",
        "指示灯",
        "控制面板",
        "操作界面",
        "设置界面",
        "菜单",
        "车辆外观",
        "内饰",
        "配置",
    ]
    
    print(f"\n测试 {len(test_queries)} 个查询，查找包含图片的文档...\n")
    
    documents_with_images = []
    seen_docs = set()
    
    for query in test_queries:
        print(f"查询: {query}")
        
        # BM25检索
        bm25_docs = bm25_retriever.retrieve_topk(query, topk=10)
        
        # Milvus检索
        milvus_docs = milvus_retriever.retrieve_topk(query, topk=10)
        
        # 合并去重
        all_docs = {}
        for doc in bm25_docs + milvus_docs:
            unique_id = doc.metadata.get("unique_id")
            if unique_id and unique_id not in seen_docs:
                seen_docs.add(unique_id)
                all_docs[unique_id] = doc
        
        # 检查每个文档是否包含图片
        for doc in all_docs.values():
            metadata = doc.metadata
            if "images_info" in metadata:
                images = metadata["images_info"]
                if isinstance(images, list) and len(images) > 0:
                    # 检查是否有带标题的图片
                    images_with_title = [
                        img for img in images 
                        if isinstance(img, dict) and img.get("title")
                    ]
                    if images_with_title:
                        doc_info = {
                            "unique_id": metadata.get("unique_id"),
                            "page": metadata.get("page"),
                            "content_preview": doc.page_content[:100] + "...",
                            "images_count": len(images_with_title),
                            "images": images_with_title,
                            "query": query,
                        }
                        documents_with_images.append(doc_info)
                        print(f"  ✓ 找到包含图片的文档 (页码: {doc_info['page']}, 图片数: {doc_info['images_count']})")
    
    return documents_with_images


def check_image_paths(documents_with_images):
    """检查图片路径是否存在"""
    print("\n" + "=" * 80)
    print("检查图片路径...")
    print("=" * 80)
    
    image_save_dir = Path(settings.image_save_dir)
    print(f"\n图片保存目录: {image_save_dir}")
    print(f"目录是否存在: {image_save_dir.exists()}")
    
    if image_save_dir.exists():
        image_files = list(image_save_dir.glob("*.jpg")) + list(image_save_dir.glob("*.jpeg"))
        print(f"目录中的图片文件数: {len(image_files)}")
        if image_files:
            print(f"示例文件: {image_files[0].name}")
    
    valid_images = []
    invalid_images = []
    
    for doc_info in documents_with_images:
        for img_info in doc_info["images"]:
            img_path = img_info.get("image_path") or img_info.get("url") or img_info.get("path", "")
            if not img_path:
                continue
            
            # 检查路径
            path_obj = Path(img_path)
            if path_obj.exists():
                valid_images.append({
                    "path": str(path_obj),
                    "title": img_info.get("title", ""),
                    "page": doc_info["page"],
                })
            else:
                # 尝试相对路径
                project_root = Path(__file__).parent.parent
                relative_path = project_root / img_path
                if relative_path.exists():
                    valid_images.append({
                        "path": str(relative_path),
                        "title": img_info.get("title", ""),
                        "page": doc_info["page"],
                    })
                else:
                    # 尝试从image_save_dir查找
                    if image_save_dir.exists():
                        image_file = image_save_dir / Path(img_path).name
                        if image_file.exists():
                            valid_images.append({
                                "path": str(image_file),
                                "title": img_info.get("title", ""),
                                "page": doc_info["page"],
                            })
                        else:
                            invalid_images.append({
                                "path": img_path,
                                "title": img_info.get("title", ""),
                                "page": doc_info["page"],
                            })
                    else:
                        invalid_images.append({
                            "path": img_path,
                            "title": img_info.get("title", ""),
                            "page": doc_info["page"],
                        })
    
    print(f"\n有效图片路径: {len(valid_images)}")
    print(f"无效图片路径: {len(invalid_images)}")
    
    if invalid_images:
        print("\n无效图片路径示例:")
        for img in invalid_images[:5]:
            print(f"  - {img['path']} (页码: {img['page']}, 标题: {img['title']})")
    
    return valid_images, invalid_images


def suggest_test_queries(documents_with_images):
    """根据找到的文档建议测试问题"""
    print("\n" + "=" * 80)
    print("建议的测试问题")
    print("=" * 80)
    
    if not documents_with_images:
        print("\n未找到包含图片的文档。")
        print("建议尝试以下类型的问题：")
        print("  - 关于车辆配置、仪表盘、中控屏幕的问题")
        print("  - 关于操作界面、按钮、菜单的问题")
        print("  - 关于车辆外观、内饰的问题")
        return
    
    # 按页码分组
    pages_with_images = {}
    for doc_info in documents_with_images:
        page = doc_info["page"]
        if page not in pages_with_images:
            pages_with_images[page] = []
        pages_with_images[page].append(doc_info)
    
    print(f"\n找到 {len(documents_with_images)} 个包含图片的文档，分布在 {len(pages_with_images)} 页")
    print("\n建议的测试问题（基于找到的文档内容）：")
    
    # 提取文档内容的关键词
    suggested_queries = set()
    for doc_info in documents_with_images[:10]:  # 只取前10个
        content = doc_info["content_preview"].lower()
        query = doc_info["query"]
        suggested_queries.add(query)
        
        # 从内容中提取可能的查询词
        if "仪表" in content or "屏幕" in content:
            suggested_queries.add("仪表盘")
            suggested_queries.add("中控屏幕")
        if "按钮" in content or "控制" in content:
            suggested_queries.add("按钮")
            suggested_queries.add("控制面板")
        if "设置" in content or "菜单" in content:
            suggested_queries.add("设置")
            suggested_queries.add("菜单")
        if "充电" in content:
            suggested_queries.add("充电")
        if "座椅" in content:
            suggested_queries.add("座椅")
    
    print("\n可以尝试以下问题：")
    for i, query in enumerate(sorted(suggested_queries), 1):
        print(f"  {i}. {query}")
    
    print("\n提示：")
    print("  - 这些问题应该能检索到包含图片的文档")
    print("  - 如果LLM在回答时引用了这些文档（使用【1】等标记），图片就会显示")
    print("  - 确保图片文件存在于 data/saved_images/ 目录中")


def main():
    """主函数"""
    print("图片加载功能测试工具")
    print("=" * 80)
    
    # 1. 查找包含图片的文档
    documents_with_images = find_documents_with_images()
    
    if not documents_with_images:
        print("\n未找到包含图片的文档。")
        print("可能的原因：")
        print("  1. PDF解析时没有提取到图片")
        print("  2. 图片没有标题（只有带标题的图片才会被包含）")
        print("  3. 索引中不包含这些文档")
        return
    
    print(f"\n总共找到 {len(documents_with_images)} 个包含图片的文档")
    
    # 2. 检查图片路径
    valid_images, invalid_images = check_image_paths(documents_with_images)
    
    # 3. 建议测试问题
    suggest_test_queries(documents_with_images)
    
    # 4. 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"包含图片的文档数: {len(documents_with_images)}")
    print(f"有效图片路径: {len(valid_images)}")
    print(f"无效图片路径: {len(invalid_images)}")
    
    if valid_images:
        print("\n✓ 图片加载功能应该可以正常工作")
        print("  建议在Gradio界面中尝试上述建议的问题")
    else:
        print("\n⚠ 警告：没有找到有效的图片路径")
        print("  请检查：")
        print("  1. 图片文件是否存在于 data/saved_images/ 目录")
        print("  2. 图片路径配置是否正确")


if __name__ == "__main__":
    main()

