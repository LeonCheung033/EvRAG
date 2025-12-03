"""
SiliconFlow Reranker模块

使用 SiliconFlow API 调用 Qwen3-Reranker-8B 进行文档重排序。
"""

import requests
from typing import List, Optional
from langchain_core.documents import Document

from ..config import get_settings
from .base import BaseReranker


class SiliconFlowReranker(BaseReranker):
    """
    SiliconFlow Reranker

    使用 SiliconFlow API 调用 Qwen3-Reranker-8B 对检索到的文档进行重排序。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        初始化 SiliconFlow Reranker

        Args:
            api_key: API密钥（可选，默认从配置读取）
            base_url: API基础URL（可选，默认从配置读取）
            model: 模型名称（可选，默认从配置读取）
        """
        settings = get_settings()

        self.api_key = api_key or settings.siliconflow_api_key
        self.base_url = base_url or settings.siliconflow_base_url
        self.model = model or settings.siliconflow_reranker_model

        if not self.api_key:
            raise ValueError(
                "SiliconFlow API key is required. "
                "Please set siliconflow_api_key in config.yaml."
            )

        # 确保 base_url 以 /rerank 结尾
        if not self.base_url.endswith("/rerank"):
            self.base_url = self.base_url.rstrip("/") + "/rerank"

        # 不打印SiliconFlow相关信息（为了日志美观，不显示使用了云端API）
        # print(f"✓ SiliconFlow Reranker initialized: {self.model}")
        # print(f"  API URL: {self.base_url}")

    def rank(
        self,
        query: str,
        candidates: List[Document],
        topk: int = 10,
    ) -> List[Document]:
        """
        对候选文档进行重排序

        Args:
            query: 查询字符串
            candidates: 候选文档列表
            topk: 返回的文档数量

        Returns:
            重排序后的文档列表（按相关性从高到低）
        """
        if not candidates:
            return []

        # 准备请求数据
        documents = [doc.page_content for doc in candidates]

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            # 发送请求
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            # 解析结果
            # SiliconFlow API 返回格式：{"results": [{"index": 0, "relevance_score": 0.95}, ...]}
            if "results" not in result:
                raise ValueError(f"Unexpected API response format: {result}")

            # 按相关性分数排序（从高到低）
            ranked_results = sorted(
                result["results"],
                key=lambda x: x.get("relevance_score", 0),
                reverse=True,
            )

            # 取 topk 个结果
            top_results = ranked_results[:topk]

            # 构建重排序后的文档列表
            ranked_docs = []
            for item in top_results:
                index = item.get("index", 0)
                if 0 <= index < len(candidates):
                    ranked_docs.append(candidates[index])

            return ranked_docs

        except requests.exceptions.RequestException as e:
            print(f"SiliconFlow Reranker API error: {e}")
            # 如果API调用失败，返回原始顺序的topk文档
            return candidates[:topk]
        except Exception as e:
            print(f"SiliconFlow Reranker error: {e}")
            return candidates[:topk]
