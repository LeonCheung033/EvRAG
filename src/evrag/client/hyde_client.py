"""
HYDE客户端

实现Hypothetical Document Embeddings (HYDE)功能，
通过生成假设文档来改进检索效果。
"""

from typing import Optional
from .base import BaseLLMClient


# HYDE的prompt模板
LLM_HYDE_PROMPT = """
你是一位Tesla汽车专家，现在请你结合Model 3车辆和新能源电动汽车相关知识回答下列问题.
请给出用户问题的使用方法，详细分析问题原因，返回有用的内容。
{query}
最终的回答请尽可能的精简, 不超过100字:
"""


class HydeClient:
    """
    HYDE客户端

    通过生成假设文档来改进检索效果。
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_template: Optional[str] = None,
        system_message: Optional[str] = None,
    ):
        """
        初始化HYDE客户端

        Args:
            llm_client: LLM客户端实例
            prompt_template: 自定义prompt模板（可选）
            system_message: 系统消息（可选）
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template or LLM_HYDE_PROMPT
        self.system_message = system_message or "你是一个有用的人工智能助手."

    def generate_hypothetical_document(self, query: str, **kwargs) -> str:
        """
        生成假设文档

        Args:
            query: 用户查询
            **kwargs: 其他LLM参数

        Returns:
            生成的假设文档内容
        """
        prompt = self.prompt_template.format(query=query)

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]

        # 设置默认参数
        default_kwargs = {
            "temperature": 0.001,
            "top_p": 0,
        }
        default_kwargs.update(kwargs)

        result = self.llm_client.chat(messages=messages, stream=False, **default_kwargs)

        return result
