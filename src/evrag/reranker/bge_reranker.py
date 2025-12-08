"""
BGE-M3重排序器模块

使用 BGE (BAAI General Embedding) 模型进行文档重排序。
支持 BGE-Reranker 系列模型。
"""

import torch
from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import get_settings
from .base import BaseReranker


class BGEReranker(BaseReranker):
    """
    BGE 重排序器

    使用 BGE-Reranker 模型对检索到的文档进行重排序。
    模型使用交叉编码器架构，能够更准确地评估查询-文档相关性。
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        max_length: int = 4096,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the BGE reranker.
        """
        settings = get_settings()

        if model_path is None:
            model_path = (
                settings.bge_reranker_tuned_model_path
                or settings.bge_reranker_model_path
            )
            if model_path is None:
                raise ValueError(
                    "BGE reranker model path not specified. "
                    "Please set bge_reranker_model_path or bge_reranker_tuned_model_path in config."
                )
        # 保证跨平台路径格式正确
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"BGE reranker model not found at: {self.model_path}"
            )
        # 确定设备
        if device is None:
            if settings.device == "cuda":
                # 使用配置中指定的GPU设备ID
                self.device_id = settings.reranker_gpu_id
                self.device = f"cuda:{self.device_id}"
            else:
                self.device_id = None
                self.device = "cpu"
        else:
            # 如果手动指定了设备，解析设备ID
            if device.startswith("cuda:"):
                self.device_id = int(device.split(":")[1])
                self.device = device
            elif device == "cuda":
                self.device_id = settings.reranker_gpu_id
                self.device = f"cuda:{self.device_id}"
            else:
                self.device_id = None
                self.device = "cpu"

        self.max_length = max_length

        self._load_model()

    def _load_model(self) -> None:
        """
        Load the BGE reranker model.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
            )
            # 根据设备类型移动模型
            if self.device.startswith("cuda:") and torch.cuda.is_available():
                # 使用指定的GPU设备ID
                self.model = self.model.half().to(self.device)  # 使用半精度以节省显存
                print(f"✓ Reranker模型已加载到设备: {self.device}")
            elif self.device == "cuda" and torch.cuda.is_available():
                # 兼容旧配置（只指定cuda，不指定ID）
                self.model = self.model.half().cuda()
            else:
                self.model = self.model.float().cpu()
                self.device = "cpu"
                self.device_id = None

        except Exception as e:
            raise ValueError(
                f"Failed to load BGE reranker model from {self.model_path}: {e}"
            ) from e

    def rank(
        self,
        query: str,
        candidates_docs: List[Document],
        topk: int = 10,
    ) -> List[Document]:
        """
        Rank the candidates based on the query.
        """
        if not candidates_docs:
            return []

        if not query.strip():
            return candidates_docs[:topk]

        # 构建查询-文档对
        pairs = [(query, doc.page_content) for doc in candidates_docs]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        if self.device.startswith("cuda:") and torch.cuda.is_available():
            # 使用指定的GPU设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        elif self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # 计算相关性分数
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(dim=-1)

        scores = scores.detach().cpu().numpy()

        if scores.ndim == 0:
            scores = scores.reshape(1)

        scored_docs = list(zip(scores, candidates_docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # 返回前topk个文档
        reranked_docs = [doc for _, doc in scored_docs[:topk]]
        return reranked_docs
