"""
工具函数模块

包含文档合并、后处理等工具函数。
"""

import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from .client import MongoDBClient


def merge_docs(
    docs1: List[Document],
    docs2: List[Document],
    mongodb_client: Optional[MongoDBClient] = None,
    collection_name: str = "manual_text",
) -> List[Document]:
    """
    合并两个文档列表，去重

    如果文档有parent_id，则从MongoDB获取父文档。
    基于unique_id去重。

    Args:
        docs1: 第一个文档列表
        docs2: 第二个文档列表
        mongodb_client: MongoDB客户端（可选，如果需要获取父文档）
        collection_name: MongoDB集合名称

    Returns:
        合并并去重后的文档列表
    """
    merged_docs = []
    merged_ids = set()
    candidate_docs = docs1 + docs2

    for doc in candidate_docs:
        parent_id = doc.metadata.get("parent_id")

        if parent_id and mongodb_client:
            # 如果有parent_id，从MongoDB获取父文档
            try:
                collection = mongodb_client.get_collection(collection_name)
                parent_mg = collection.find_one({"unique_id": parent_id})

                if parent_mg:
                    unique_id = parent_mg.get("unique_id")
                    if unique_id and unique_id not in merged_ids:
                        merged_ids.add(unique_id)
                        parent_doc = Document(
                            page_content=parent_mg.get("page_content", ""),
                            metadata=parent_mg.get("metadata", {}),
                        )
                        merged_docs.append(parent_doc)
            except Exception as e:
                # 如果MongoDB查询失败，使用原文档
                print(f"Warning: Failed to fetch parent document {parent_id}: {e}")
                unique_id = doc.metadata.get("unique_id")
                if unique_id and unique_id not in merged_ids:
                    merged_ids.add(unique_id)
                    merged_docs.append(doc)
        else:
            # 没有parent_id，直接使用原文档
            unique_id = doc.metadata.get("unique_id")
            if unique_id and unique_id not in merged_ids:
                merged_ids.add(unique_id)
                merged_docs.append(doc)

    return merged_docs


def post_processing(
    response: str,
    docs: List[Document],
) -> Dict[str, Any]:
    """
    后处理RAG响应

    从响应中提取：
    - 答案（去除引用标记）
    - 引用页码
    - 相关图片

    Args:
        response: LLM返回的原始响应
        docs: 检索到的文档列表

    Returns:
        包含answer、cite_pages、related_images的字典
    """
    # 提取所有引用标记，格式：【1, 2, 3】或【1】【2】
    all_cites = re.findall(r"[【](.*?)[】]", response)
    cites = []

    for cite in all_cites:
        # 清理引用文本
        cite = re.sub(r"[{} 【】]", "", cite)
        cite = cite.replace(",", "，")
        # 提取数字
        cite_numbers = [int(k) for k in cite.split("，") if k.strip().isdigit()]
        cites.extend(cite_numbers)

    # 去重
    cites = list(set(cites))

    # 提取答案（去除引用标记）
    answer = re.sub(r"[【](.*?)[】]", "", response)
    answer = re.sub(r"[{}【】]", "", answer)
    answer = answer.strip()

    # 提取相关图片和页码
    related_images = []
    pages = []

    for index in cites:
        # 索引从1开始，转换为列表索引（从0开始）
        if index > len(docs) or index < 1:
            continue

        doc = docs[index - 1]
        metadata = doc.metadata

        # 提取页码
        if "page" in metadata:
            pages.append(metadata["page"])

        # 提取图片信息
        if "images_info" in metadata:
            images = metadata["images_info"]
            if isinstance(images, list):
                for image in images:
                    if isinstance(image, dict) and image.get("title"):
                        related_images.append(image)

    # 去重并排序页码
    pages = sorted(list(set(pages)))

    return {"answer": answer, "cite_pages": pages, "related_images": related_images}
