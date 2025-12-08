"""
语义切分服务
基于原项目实现，使用M3E-small模型进行语义聚类
"""

import gc
import re
import math
import torch
import pandas as pd
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException  # pyright: ignore[reportMissingImports]
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from ..config import get_settings

# 全局变量
settings = get_settings()
# 根据配置指定GPU设备ID
if torch.cuda.is_available() and settings.device == "cuda":
    device = torch.device(f"cuda:{settings.semantic_chunk_gpu_id}")
else:
    device = torch.device("cpu")
_min_chunk_size = 50
_min_doc_size = 256
embedding_model: Optional[SentenceTransformer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    启动时加载模型，关闭时清理GPU内存
    """
    global embedding_model

    # 启动时加载模型
    if settings.m3e_small_model_path and settings.m3e_small_model_path.exists():
        embedding_model = SentenceTransformer(str(settings.m3e_small_model_path))
        embedding_model = embedding_model.to(device)
        print(f"✓ 语义切分模型已加载: {settings.m3e_small_model_path}")
        print(f"✓ 模型部署在设备: {device}")
    else:
        raise RuntimeError(f"模型路径不存在: {settings.m3e_small_model_path}")

    yield

    # 关闭时清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


app = FastAPI(
    title="EvRAG Semantic Chunk Service",
    description="语义切分服务，将文本按语义相似性分组",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SemanticRequest(BaseModel):
    """语义切分请求"""

    sentences: str
    group_size: Optional[int] = 5


class ChunkResponse(BaseModel):
    """语义切分响应"""

    chunks: List[str]


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model_loaded": embedding_model is not None,
        "device": str(device),
    }


@app.post("/v1/semantic-chunks", response_model=ChunkResponse)
async def create_semantic_chunks(request: SemanticRequest):
    """
    将句子按语义相似性分组

    Args:
        request: 包含sentences和group_size的请求

    Returns:
        合并后的分组文本列表

    Raises:
        HTTPException: 当输入参数不合法或处理失败时
    """
    global embedding_model

    if embedding_model is None:
        raise HTTPException(status_code=500, detail="模型未加载")

    if request.group_size is None or request.group_size < 1:
        raise HTTPException(status_code=400, detail="组大小必须大于0")

    if len(request.sentences) < _min_chunk_size:
        return ChunkResponse(chunks=[request.sentences])

    # 考虑文档标题（以###分割）
    split_docs = re.split(r"(###)", request.sentences)
    split_docs = [k for k in split_docs if k.strip()]

    if len(split_docs) > 1:
        if split_docs[0] == "###":
            split_docs = [
                "".join(split_docs[i : i + 2]) for i in range(0, len(split_docs), 2)
            ]
        else:
            split_docs = [split_docs[0]] + [
                "".join(split_docs[i : i + 2]) for i in range(1, len(split_docs), 2)
            ]

        if len(split_docs) > 1:
            return ChunkResponse(chunks=split_docs)

    # 按双换行符分割
    split_docs = request.sentences.split("\n\n")

    if len(split_docs) <= request.group_size:
        return ChunkResponse(chunks=split_docs)

    # 计算合理的聚类数量（向上取整）
    n_clusters = max(1, math.ceil(len(split_docs) / request.group_size))

    try:
        # 生成嵌入向量（已自动使用GPU加速）
        embeddings = embedding_model.encode(split_docs, show_progress_bar=False)

        # 使用余弦相似度的层次聚类
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",  # 使用余弦距离
            linkage="average",  # 使用平均链接算法
            compute_full_tree="auto",
        )

        labels = clustering.fit_predict(embeddings)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

    # 按标签分组
    df = pd.DataFrame({"sentence": split_docs, "label": labels})
    result = (
        df.groupby("label", sort=True)["sentence"].agg(lambda x: " ".join(x)).to_dict()
    )

    # 合并误切分的small chunks
    docs = list(result.values())
    merged_docs = []
    index = 0
    while index < len(docs):
        cur_doc = docs[index]
        plus = 1
        for sub_idx in range(index + 1, len(docs)):
            if len(docs[sub_idx]) < _min_chunk_size:
                cur_doc += docs[sub_idx]
                plus += 1
            else:
                break
        index += plus
        merged_docs.append(cur_doc)

    return ChunkResponse(chunks=merged_docs)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6000, workers=1)
