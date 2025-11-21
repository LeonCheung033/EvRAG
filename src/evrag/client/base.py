"""
LLM客户端基类

定义所有LLM客户端必须实现的接口。
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Iterator


class BaseLLMClient(ABC):
    """
    LLM客户端基类

    所有LLM客户端实现必须继承此类并实现抽象方法。
    """

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        发送聊天请求

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            model: 模型名称（可选，使用客户端默认模型）
            temperature: 温度参数
            max_tokens: 最大生成token数
            stream: 是否流式返回
            **kwargs: 其他参数

        Returns:
            模型返回的文本内容（非流式）或生成器（流式）
        """
        pass

    @abstractmethod
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        流式发送聊天请求

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成token数
            **kwargs: 其他参数

        Returns:
            生成器，每次yield一个token或chunk
        """
        pass
