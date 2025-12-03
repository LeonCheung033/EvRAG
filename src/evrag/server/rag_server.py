"""
RAG对话服务
基于FastAPI实现RAG对话服务，支持流式和非流式输出
"""

import time
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Any, Iterator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..config import get_settings
from ..retriever import BM25Retriever, MilvusRetriever
from ..reranker import BGEReranker
from ..client import LocalLLMClient, ChatClient
from ..tool_func import merge_docs, post_processing

# 全局变量
settings = get_settings()

# 全局组件（在启动时初始化）
bm25_retriever: Optional[BM25Retriever] = None
milvus_retriever: Optional[MilvusRetriever] = None
reranker: Optional[BGEReranker] = None
llm_client: Optional[LocalLLMClient] = None
chat_client: Optional[ChatClient] = None

# 默认参数
DEFAULT_BM25_TOPK = 10
DEFAULT_MILVUS_TOPK = 10
DEFAULT_RERANKER_TOPK = 5
DEFAULT_CONTEXT_WINDOW = 4096  # tokens
DEFAULT_RECENT_ROUNDS = 2  # 保留最近N轮对话


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    启动时加载模型，关闭时清理资源
    """
    global bm25_retriever, milvus_retriever, reranker, llm_client, chat_client

    print("正在初始化RAG服务组件...")

    # 初始化检索器
    print("加载检索器...")
    bm25_retriever = BM25Retriever(docs=None, retrieve=True)
    milvus_retriever = MilvusRetriever(docs=None, retrieve=True)
    print("✓ 检索器已加载")

    # 初始化重排序器（使用微调模型）
    print("加载重排序器...")
    reranker_path = settings.bge_reranker_tuned_model_path
    if not reranker_path or not reranker_path.exists():
        raise RuntimeError(f"微调Reranker模型路径不存在: {reranker_path}")
    reranker = BGEReranker(model_path=reranker_path)
    print(f"✓ 重排序器已加载: {reranker_path}")

    # 初始化LLM客户端（使用微调模型）
    print("初始化LLM客户端...")
    llm_url = settings.finetuned_llm_base_url
    llm_model = settings.finetuned_llm_model_name
    if not llm_url or not llm_model:
        raise RuntimeError("微调LLM服务URL或模型名称未配置")
    llm_client = LocalLLMClient(base_url=llm_url, model=llm_model)
    chat_client = ChatClient(llm_client)
    print(f"✓ LLM客户端已初始化: {llm_url}, 模型: {llm_model}")

    print("✓ RAG服务初始化完成")

    yield

    # 关闭时清理（如果需要）
    print("正在关闭RAG服务...")


app = FastAPI(
    title="EvRAG Chat Service",
    description="RAG对话服务，支持检索增强生成",
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


# 请求/响应模型
class ChatMessage(BaseModel):
    """聊天消息"""

    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    """聊天请求"""

    query: str
    history: Optional[List[ChatMessage]] = []
    stream: bool = False
    bm25_topk: Optional[int] = DEFAULT_BM25_TOPK
    milvus_topk: Optional[int] = DEFAULT_MILVUS_TOPK
    reranker_topk: Optional[int] = DEFAULT_RERANKER_TOPK
    enable_thinking: bool = False


class ChatResponse(BaseModel):
    """聊天响应"""

    answer: str
    cite_pages: List[int]
    related_images: List[Dict[str, Any]]
    performance: Dict[str, float]


def estimate_tokens(text: str) -> int:
    """粗略估算文本的token数（中文约1.5字符/token，英文约4字符/token）"""
    # 简单估算：中文字符数 * 1.5 + 英文字符数 / 4
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + other_chars / 4)


def summarize_history(
    history: List[ChatMessage],
    llm_client: LocalLLMClient,
    max_tokens: int = DEFAULT_CONTEXT_WINDOW,
    recent_rounds: int = DEFAULT_RECENT_ROUNDS,
) -> List[ChatMessage]:
    """
    总结历史对话

    当历史对话超过安全长度时：
    1. 保留最近的N轮对话
    2. 对剩余的历史对话使用LLM进行总结
    3. 将总结作为第一条用户消息

    Args:
        history: 历史对话列表
        llm_client: LLM客户端
        max_tokens: 最大token数
        recent_rounds: 保留最近N轮对话

    Returns:
        处理后的历史对话列表
    """
    if not history:
        return []

    # 计算总token数
    total_tokens = sum(estimate_tokens(msg.content) for msg in history)

    # 如果未超过限制，直接返回
    if total_tokens <= max_tokens:
        return history

    # 保留最近的N轮对话（每轮包含user和assistant两条消息）
    recent_messages = (
        history[-recent_rounds * 2 :] if len(history) >= recent_rounds * 2 else history
    )
    old_messages = (
        history[: -len(recent_messages)] if len(history) > len(recent_messages) else []
    )

    if not old_messages:
        return recent_messages

    # 构建总结prompt
    old_conversation = "\n".join(
        [
            f"{'用户' if msg.role == 'user' else '助手'}: {msg.content}"
            for msg in old_messages
        ]
    )

    summary_prompt = f"""请总结以下对话历史的关键信息，保留重要的事实、决策和上下文信息。总结要简洁但完整：

{old_conversation}

请用一段话总结上述对话的关键信息："""

    # 调用LLM进行总结
    try:
        summary_messages = [
            {"role": "system", "content": "你是一个有用的助手，擅长总结对话历史。"},
            {"role": "user", "content": summary_prompt},
        ]
        summary = llm_client.chat(
            messages=summary_messages, stream=False, temperature=0.3
        )

        # 将总结作为第一条用户消息
        summarized_history = [
            ChatMessage(role="user", content=f"[历史对话总结] {summary}")
        ] + recent_messages

        return summarized_history
    except Exception as e:
        print(f"警告: 历史对话总结失败: {e}，将只保留最近对话")
        return recent_messages


def build_context_from_history(history: List[ChatMessage], query: str) -> str:
    """从历史对话构建上下文字符串"""
    context_parts = []
    for msg in history:
        if msg.role == "user":
            context_parts.append(f"用户: {msg.content}")
        elif msg.role == "assistant":
            context_parts.append(f"助手: {msg.content}")

    if query:
        context_parts.append(f"用户: {query}")

    return "\n".join(context_parts)


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "components": {
            "bm25_retriever": bm25_retriever is not None,
            "milvus_retriever": milvus_retriever is not None,
            "reranker": reranker is not None,
            "llm_client": llm_client is not None,
        },
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    非流式聊天接口

    Args:
        request: 聊天请求

    Returns:
        聊天响应，包含答案、引用、图片和性能指标
    """
    if not bm25_retriever or not milvus_retriever or not reranker or not chat_client:
        raise HTTPException(status_code=500, detail="服务未初始化")

    start_time = time.time()
    performance = {}

    try:
        # 处理历史对话
        history = request.history or []
        history = summarize_history(history, llm_client)

        # 1. 检索阶段
        retrieval_start = time.time()
        bm25_docs = bm25_retriever.retrieve_topk(
            request.query, topk=request.bm25_topk or DEFAULT_BM25_TOPK
        )
        milvus_docs = milvus_retriever.retrieve_topk(
            request.query, topk=request.milvus_topk or DEFAULT_MILVUS_TOPK
        )
        retrieval_time = time.time() - retrieval_start
        performance["retrieval_time"] = retrieval_time

        # 2. 合并文档
        merged_docs = merge_docs(bm25_docs, milvus_docs)

        # 3. 重排序
        rerank_start = time.time()
        ranked_docs = reranker.rank(
            request.query,
            merged_docs,
            topk=request.reranker_topk or DEFAULT_RERANKER_TOPK,
        )
        rerank_time = time.time() - rerank_start
        performance["rerank_time"] = rerank_time

        # 4. 构建上下文
        context = "\n".join(
            [f"【{idx + 1}】{doc.page_content}" for idx, doc in enumerate(ranked_docs)]
        )

        # 5. 生成答案
        generation_start = time.time()
        response = chat_client.chat(
            query=request.query,
            context=context,
            stream=False,
            enable_thinking=request.enable_thinking,
        )
        generation_time = time.time() - generation_start
        performance["generation_time"] = generation_time

        # 6. 后处理
        result = post_processing(response, ranked_docs)

        total_time = time.time() - start_time
        performance["total_time"] = total_time

        return ChatResponse(
            answer=result["answer"],
            cite_pages=result["cite_pages"],
            related_images=result["related_images"],
            performance=performance,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


def generate_stream_response(
    query: str,
    history: List[ChatMessage],
    bm25_topk: int,
    milvus_topk: int,
    reranker_topk: int,
    enable_thinking: bool,
) -> Iterator[str]:
    """
    生成流式响应

    Yields:
        SSE格式的数据块
    """
    if not bm25_retriever or not milvus_retriever or not reranker or not chat_client:
        yield f"data: {__import__('json').dumps({'error': '服务未初始化'})}\n\n"
        return

    try:
        start_time = time.time()
        performance = {}

        # 处理历史对话
        history = summarize_history(history, llm_client)

        # 1. 检索阶段
        retrieval_start = time.time()
        bm25_docs = bm25_retriever.retrieve_topk(query, topk=bm25_topk)
        milvus_docs = milvus_retriever.retrieve_topk(query, topk=milvus_topk)
        retrieval_time = time.time() - retrieval_start
        performance["retrieval_time"] = retrieval_time

        # 2. 合并文档
        merged_docs = merge_docs(bm25_docs, milvus_docs)

        # 3. 重排序
        rerank_start = time.time()
        ranked_docs = reranker.rank(query, merged_docs, topk=reranker_topk)
        rerank_time = time.time() - rerank_start
        performance["rerank_time"] = rerank_time

        # 4. 构建上下文
        context = "\n".join(
            [f"【{idx + 1}】{doc.page_content}" for idx, doc in enumerate(ranked_docs)]
        )

        # 5. 流式生成答案
        generation_start = time.time()
        full_response = ""
        for chunk in chat_client.chat(
            query=query, context=context, stream=True, enable_thinking=enable_thinking
        ):
            full_response += chunk
            # 发送token数据
            yield f"data: {__import__('json').dumps({'type': 'token', 'content': chunk})}\n\n"

        generation_time = time.time() - generation_start
        performance["generation_time"] = generation_time

        # 6. 后处理
        result = post_processing(full_response, ranked_docs)

        total_time = time.time() - start_time
        performance["total_time"] = total_time

        # 发送最终结果（答案、引用、图片、性能指标）
        yield f"data: {
            __import__('json').dumps(
                {
                    'type': 'final',
                    'answer': result['answer'],
                    'cite_pages': result['cite_pages'],
                    'related_images': result['related_images'],
                    'performance': performance,
                }
            )
        }\n\n"

        # 发送结束标记
        yield "data: [DONE]\n\n"

    except Exception as e:
        yield f"data: {__import__('json').dumps({'error': str(e)})}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天接口（SSE格式）

    Args:
        request: 聊天请求

    Returns:
        SSE流式响应
    """
    return StreamingResponse(
        generate_stream_response(
            query=request.query,
            history=request.history or [],
            bm25_topk=request.bm25_topk or DEFAULT_BM25_TOPK,
            milvus_topk=request.milvus_topk or DEFAULT_MILVUS_TOPK,
            reranker_topk=request.reranker_topk or DEFAULT_RERANKER_TOPK,
            enable_thinking=request.enable_thinking,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    # FastAPI服务使用8002端口（8001已被vLLM微调模型占用）
    uvicorn.run(app, host="0.0.0.0", port=8002, workers=1)
