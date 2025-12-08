"""
文档切分模块

实现语义切分和句子级切分，并将文档保存到MongoDB。
"""

import copy
import hashlib
from typing import List, Optional
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

from ..config import get_settings
from ..client import MongoDBClient, SemanticChunkClient

# 全局配置
_chunk_size = 256
_chunk_overlap = 50
_semantic_group_size = 10
_max_parent_size = 512
encoding = tiktoken.get_encoding("cl100k_base")

# TextSplitter 设置
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_chunk_size,
    chunk_overlap=_chunk_overlap,
    # 按这个优先级递归切
    separators=["\n\n", "\n"],
    length_function=lambda text: len(encoding.encode(text)),
)


def save_2_mongo(
    split_docs: List[Document],
    collection_name: str = "manual_text",
    mongodb_client: Optional[MongoDBClient] = None,
) -> None:
    """
    将文档保存到MongoDB

    Args:
        split_docs: 要保存的文档列表
        collection_name: MongoDB集合名称
        mongodb_client: MongoDB客户端实例（可选，如果提供则复用连接）
    """
    # 如果没有提供客户端，创建新的（兼容旧代码）
    should_close = False
    if mongodb_client is None:
        settings = get_settings()
        mongodb_client = MongoDBClient(
            host=settings.mongodb_host,
            port=settings.mongodb_port,
            database=settings.mongodb_database,
        )
        should_close = True

    try:
        # 如果客户端未连接，则连接
        if mongodb_client._client is None:
            mongodb_client.connect()

        collection = mongodb_client.get_collection(collection_name)

        for doc in split_docs:
            metadata = doc.metadata

            unique_id = metadata.get("unique_id")
            if not unique_id:
                continue

            # 创建文档记录对象
            doc_record = {
                "unique_id": unique_id,
                "page_content": doc.page_content,
                "metadata": metadata,
            }

            # 更新数据库操作
            collection.update_one(
                {"unique_id": unique_id},
                {"$set": doc_record},
                upsert=True,
            )
    finally:
        # 只有在我们创建的客户端时才关闭
        if should_close:
            mongodb_client.close()


def texts_split(
    raw_docs: List[Document],
    semantic_chunk_client: SemanticChunkClient,
    collection_name: str = "manual_text",
    semantic_group_size: int = None,
    max_parent_size: int = None,
) -> List[Document]:
    """
    句子级 + 语义感知切分

    Args:
        raw_docs: 原始文档列表
        semantic_chunk_client: 语义切分客户端
        collection_name: MongoDB集合名称
        semantic_group_size: 语义分组大小（默认使用全局配置）
        max_parent_size: 最大父文档大小（默认使用全局配置）

    Returns:
        split_docs: 切分后的文档列表
    """
    if semantic_group_size is None:
        semantic_group_size = _semantic_group_size
    if max_parent_size is None:
        max_parent_size = _max_parent_size

    all_split_docs = []

    # 在函数开始时连接MongoDB，结束时关闭，避免频繁连接/断开
    settings = get_settings()
    mongodb_client = MongoDBClient(
        host=settings.mongodb_host,
        port=settings.mongodb_port,
        database=settings.mongodb_database,
    )

    try:
        mongodb_client.connect()

        for doc in tqdm(raw_docs, desc="Splitting documents"):
            # 1. 语义分组：将原文按语义连贯性切分为若干语义块
            grouped_chunks = semantic_chunk_client.chunk(
                sentences=doc.page_content,
                group_size=semantic_group_size,
            )

            # 2. 构建父文档：每个语义块作为"父文档"，赋予唯一 unique_id
            parent_docs = []
            for group in grouped_chunks:
                parent_id = hashlib.md5(group.encode("utf-8")).hexdigest()
                parent_metadata = copy.deepcopy(doc.metadata)
                parent_metadata["unique_id"] = (
                    parent_id  # 父文档只设置unique_id，不设置parent_id
                )
                parent_doc = Document(
                    page_content=group,
                    metadata=parent_metadata,
                )
                parent_docs.append(parent_doc)
                # 若父文档长度未超过阈值，也作为有效片段保留
                if len(group) < max_parent_size:
                    all_split_docs.append(parent_doc)

            # 3. 持久化父文档到 MongoDB（用于后续父子关联检索）
            save_2_mongo(
                parent_docs,
                collection_name=collection_name,
                mongodb_client=mongodb_client,
            )

            # 4. 子文档切分：对每个父文档进行带重叠的句子级细粒度切分
            for chunk in parent_docs:
                # 使用metadatas参数（复数形式），兼容新版本LangChain
                split_docs = text_splitter.create_documents(
                    [chunk.page_content],
                    metadatas=[chunk.metadata],
                )
                reid_split_docs = []
                for child_doc in split_docs:
                    # 跳过与父文档完全相同的内容（避免冗余）
                    if child_doc.page_content == chunk.page_content:
                        continue

                    # 5. 为子文档生成唯一 ID 并建立父子关系
                    child_id = hashlib.md5(
                        child_doc.page_content.encode("utf-8")
                    ).hexdigest()
                    child_metadata = copy.deepcopy(chunk.metadata)
                    child_metadata["unique_id"] = child_id
                    child_metadata["parent_id"] = chunk.metadata[
                        "unique_id"
                    ]  # 子文档的parent_id指向父文档的unique_id
                    reid_child_doc = Document(
                        page_content=child_doc.page_content,
                        metadata=child_metadata,
                    )
                    reid_split_docs.append(reid_child_doc)

                # 6. 持久化子文档到 MongoDB
                save_2_mongo(
                    reid_split_docs,
                    collection_name=collection_name,
                    mongodb_client=mongodb_client,
                )

                # 7. 累加所有子文档到最终结果列表
                all_split_docs.extend(reid_split_docs)

    finally:
        # 在函数结束时关闭MongoDB连接
        mongodb_client.close()

    return all_split_docs
