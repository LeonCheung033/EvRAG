"""
本地LLM客户端

连接到本地部署的LLM服务（如vLLM、Ollama等）。
"""

from typing import List, Dict, Optional, Any, Iterator
from openai import OpenAI

from .base import BaseLLMClient
from ..config import get_settings


class LocalLLMClient(BaseLLMClient):
    """
    本地LLM客户端

    连接到本地部署的LLM服务，使用OpenAI兼容的API。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        初始化本地LLM客户端

        Args:
            api_key: API密钥（通常为"EMPTY"）
            base_url: API基础URL
            model: 默认模型名称
        """

        settings = get_settings()

        # 优先使用新的local_llm配置，兼容旧的llm配置
        self.api_key = api_key or settings.local_llm_api_key or settings.llm_api_key
        self.base_url = base_url or settings.local_llm_base_url or settings.llm_base_url
        self.model = model or settings.local_llm_model_name or settings.llm_model_name

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
            模型返回的文本内容（非流式）或生成器（流式）
        """
        if stream:
            # 如果stream=True，返回生成器
            return self.chat_stream(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

        # 处理extra_body参数（用于传递chat_template_kwargs等，参考官方文档）
        # 从kwargs中提取enable_thinking（如果存在），默认False
        enable_thinking = kwargs.pop("enable_thinking", False)
        
        # 处理extra_body
        if "extra_body" in kwargs:
            extra_body = kwargs.pop("extra_body")
        else:
            extra_body = {}
        
        # 设置chat_template_kwargs（如果不存在或需要更新enable_thinking）
        if "chat_template_kwargs" not in extra_body:
            extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        else:
            # 如果已存在，更新enable_thinking（允许外部覆盖）
            extra_body["chat_template_kwargs"]["enable_thinking"] = enable_thinking
        
        # 设置top_k（非思考模式推荐值）
        if "top_k" not in extra_body:
            extra_body["top_k"] = 20
        
        completion = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            extra_body=extra_body,
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
        # 处理extra_body参数（用于传递chat_template_kwargs等）
        enable_thinking = kwargs.pop("enable_thinking", False)
        
        if "extra_body" in kwargs:
            extra_body = kwargs.pop("extra_body")
        else:
            extra_body = {}
        
        if "chat_template_kwargs" not in extra_body:
            extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        else:
            extra_body["chat_template_kwargs"]["enable_thinking"] = enable_thinking
        
        if "top_k" not in extra_body:
            extra_body["top_k"] = 20
        
        stream = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            extra_body=extra_body,
            **kwargs,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
