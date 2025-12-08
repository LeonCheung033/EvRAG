"""
语义分块客户端

调用外部语义分块服务API，将文本按语义相似性分组。
"""

import json
from typing import List, Optional
import requests

from ..config import get_settings


class SemanticChunkClient:
    """
    语义分块客户端

    调用外部HTTP API服务进行语义分块。
    """

    def __init__(
        self,
        url: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        初始化语义分块客户端

        Args:
            url: 语义分块服务URL（可选，默认从配置读取）
            timeout: 请求超时时间（秒）
        """
        settings = get_settings()
        self.url = url or settings.semantic_chunk_url
        self.timeout = timeout

    def chunk(
        self,
        sentences: str,
        group_size: int,
    ) -> List[str]:
        """
        将文本按语义相似性分组

        Args:
            sentences: 待分组的文本内容
            group_size: 每组的目标最大句子数

        Returns:
            分组后的文本列表，失败时返回原始文本（作为单个元素）
        """
        headers = {"Content-Type": "application/json"}

        payload = json.dumps({"sentences": sentences, "group_size": group_size})

        try:
            response = requests.post(
                self.url, headers=headers, data=payload, timeout=self.timeout
            )
            response.raise_for_status()
            res = response.json()
            return res["chunks"]
        except Exception as e:
            print(f"Semantic chunk API call failed: {e}")
            # 失败时返回原始文本作为单个元素
            return [sentences]
