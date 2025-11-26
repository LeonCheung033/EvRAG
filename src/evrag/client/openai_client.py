"""
OpenAI API客户端

连接到远程OpenAI兼容的API服务（如豆包、通义千问等）。
"""

import os
from typing import List, Dict, Optional, Any, Iterator
from openai import OpenAI

from .base import BaseLLMClient
from ..config import get_settings


class OpenAIClient(BaseLLMClient):
    """
    OpenAI兼容API客户端

    连接到远程OpenAI兼容的API服务（如豆包、Deepseek等）。
    支持通过service参数选择不同的服务商。
    """

    def __init__(
        self,
        service: str = "doubao",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        初始化OpenAI客户端

        Args:
            service: 服务商名称，可选值：'doubao', 'deepseek', 'custom'
            api_key: API密钥（可选，如果提供则覆盖配置）
            base_url: API基础URL（可选，如果提供则覆盖配置）
            model: 默认模型名称（可选，如果提供则覆盖配置）
        """
        settings = get_settings()

        # 根据service选择配置
        if service == "doubao":
            self.api_key = api_key or settings.doubao_api_key
            self.base_url = base_url or settings.doubao_base_url
            self.model = model or settings.doubao_model_name
        elif service == "deepseek":
            self.api_key = api_key or settings.deepseek_api_key
            self.base_url = base_url or settings.deepseek_base_url
            self.model = model or settings.deepseek_model_name
        elif service == "custom":
            # 自定义服务，必须提供所有参数
            if not api_key or not base_url or not model:
                raise ValueError(
                    "For custom service, api_key, base_url, and model must be provided."
                )
            self.api_key = api_key
            self.base_url = base_url
            self.model = model
        else:
            raise ValueError(
                f"Unknown service: {service}. Supported services: 'doubao', 'deepseek', 'custom'"
            )

        if not self.api_key or self.api_key == "EMPTY" or self.api_key == "":
            raise ValueError(
                f"{service.capitalize()} API key is required. "
                f"Please configure {service}_api_key in config.yaml."
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

        # 过滤掉 enable_thinking 参数（Deepseek 等 API 不支持）
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "enable_thinking"}
        
        completion = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            **filtered_kwargs,
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
        # 过滤掉 enable_thinking 参数（Deepseek 等 API 不支持）
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "enable_thinking"}
        
        stream = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **filtered_kwargs,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
