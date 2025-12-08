"""
RAG问答客户端

基于LLM客户端实现RAG问答功能，使用特定的prompt模板。
"""

from typing import Optional
from .base import BaseLLMClient


# RAG问答的prompt模板
LLM_CHAT_PROMPT = """
### 信息
{context}

### 任务
你是特斯拉电动汽车Model 3车型的用户手册问答系统，你具备{{信息}}中的知识。
请回答问题"{query}"，答案需要精准，语句通顺。

**输出要求：**
1. 直接输出答案内容，不要输出任何思考过程、推理过程或解释性文字
2. 答案末尾必须包含引用标记，格式为：【引用编号1, 引用编号2, ...】
3. 如果无法从中得到答案，请说 "无答案"
4. 不允许在答案中添加编造成分

**输出格式示例：**
{{答案内容}}【1, 2, 3】
"""


class ChatClient:
    """
    RAG问答客户端

    基于LLM客户端实现RAG问答功能。
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_template: Optional[str] = None,
        system_message: Optional[str] = None,
    ):
        """
        初始化RAG问答客户端

        Args:
            llm_client: LLM客户端实例
            prompt_template: 自定义prompt模板（可选）
            system_message: 系统消息（可选）
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template or LLM_CHAT_PROMPT
        self.system_message = system_message or "你是一个有用的人工智能助手."

    def chat(
        self,
        query: str,
        context: str,
        stream: bool = False,
        enable_thinking: bool = False,
        **kwargs,
    ) -> str:
        """
        发送RAG问答请求

        Args:
            query: 用户问题
            context: 检索到的上下文信息
            stream: 是否流式返回
            enable_thinking: 是否启用思考模式（默认False，数据生成任务禁用；问答任务可启用）
            **kwargs: 其他LLM参数

        Returns:
            模型返回的答案
        """
        prompt = self.prompt_template.format(context=context, query=query)

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]

        # 设置默认参数
        # 根据enable_thinking选择不同的推荐参数
        if enable_thinking:
            # 思考模式推荐参数：Temperature=0.6, TopP=0.95
            default_kwargs = {
                "max_tokens": 4096,
                "temperature": 0.6,
                "top_p": 0.95,
            }
        else:
            # 非思考模式推荐参数：Temperature=0.7, TopP=0.8
            default_kwargs = {
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.8,
            }

        default_kwargs["enable_thinking"] = enable_thinking
        default_kwargs.update(kwargs)

        if stream:
            # 流式返回
            return self.llm_client.chat_stream(messages=messages, **default_kwargs)
        else:
            # 非流式返回
            return self.llm_client.chat(
                messages=messages, stream=False, **default_kwargs
            )
