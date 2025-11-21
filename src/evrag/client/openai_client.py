"""
OpenAI API客户端

连接到远程OpenAI兼容的API服务（如豆包、通义千问等）。
"""

from typing import List, Dict, Optional, Any, Iterator
from openai import OpenAI

from .base import BaseLLMClient
from ..config import get_settings


class OpenAIClient(BaseLLMClient):
    """
    OpenAI兼容API客户端

    连接到远程OpenAI兼容的API服务。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        初始化OpenAI客户端

        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 默认模型名称
        """
        settings = get_settings()

        self.api_key = api_key or settings.llm_api_key
        self.base_url = base_url or settings.llm_base_url
        self.model = model or settings.llm_model_name

        if not self.api_key or self.api_key == "EMPTY":
            raise ValueError(
                "OpenAI API key is required. "
                "Please set LLM_API_KEY environment variable or configure it in settings."
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

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
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成token数
            stream: 是否流式返回
            **kwargs: 其他参数

        Returns:
            模型返回的文本内容
        """
        if stream:
            return self.chat_stream(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

        completion = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            **kwargs,
        )

        return completion.choices[0].message.content

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

        Yields:
            每个chunk的文本内容
        """
        stream = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
