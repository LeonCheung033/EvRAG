"""
Gradio前端界面
实现RAG对话的Web界面
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import gradio as gr
import requests

from ..config import get_settings

settings = get_settings()

# 默认配置
# RAG服务使用8002端口（8001已被vLLM微调模型占用）
DEFAULT_RAG_SERVER_URL = "http://localhost:8002"
DEFAULT_BM25_TOPK = 10
DEFAULT_MILVUS_TOPK = 10
DEFAULT_RERANKER_TOPK = 5


def get_rag_server_url() -> str:
    """从配置获取RAG服务地址"""
    # 可以从config.yaml读取，这里先使用默认值
    return DEFAULT_RAG_SERVER_URL


def check_image_path(image_path: str) -> Optional[str]:
    """
    检查图片路径是否存在，返回可用的路径
    
    Args:
        image_path: 图片路径（可能是相对路径或绝对路径）
    
    Returns:
        可用的图片路径，如果不存在则返回None
    """
    if not image_path:
        return None
    
    project_root = Path(__file__).parent.parent.parent.parent
    
    # 尝试作为绝对路径
    if os.path.isabs(image_path):
        path_obj = Path(image_path)
        if path_obj.exists():
            return str(path_obj.resolve())
    
    # 尝试作为相对路径（相对于项目根目录）
    relative_path = project_root / image_path
    if relative_path.exists():
        return str(relative_path.resolve())
    
    # 尝试从image_save_dir查找
    if hasattr(settings, 'image_save_dir') and settings.image_save_dir:
        image_dir = Path(settings.image_save_dir)
        if image_dir.exists():
            # 尝试完整路径
            image_file = image_dir / image_path
            if image_file.exists():
                return str(image_file.resolve())
            # 尝试只使用文件名
            image_file = image_dir / Path(image_path).name
            if image_file.exists():
                return str(image_file.resolve())
    
    return None


def format_performance_metrics(performance: Dict[str, float]) -> str:
    """格式化性能指标为Markdown文本（横向两排展示）"""
    if not performance:
        return "暂无性能数据"
    
    # 使用表格格式，两排横向展示
    lines = []
    lines.append("| 指标 | 时间 | 指标 | 时间 |")
    lines.append("|------|------|------|------|")
    lines.append(
        f"| **检索时间** | {performance.get('retrieval_time', 0):.3f}秒 | "
        f"**重排序时间** | {performance.get('rerank_time', 0):.3f}秒 |"
    )
    lines.append(
        f"| **生成时间** | {performance.get('generation_time', 0):.3f}秒 | "
        f"**总响应时间** | {performance.get('total_time', 0):.3f}秒 |"
    )
    
    return "\n".join(lines)


def chat_non_stream(
    query: str,
    history: List[List[str]],
    bm25_topk: int,
    milvus_topk: int,
    reranker_topk: int,
    enable_thinking: bool,
) -> Tuple[List[List[str]], str, List[Tuple[str, str]], str]:
    """
    非流式聊天处理
    
    Returns:
        (更新后的历史, 答案, 图片列表, 性能指标)
    """
    if not query.strip():
        return history, "", [], ""
    
    # 构建历史消息
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    # 准备请求
    rag_url = get_rag_server_url()
    request_data = {
        "query": query,
        "history": messages,
        "stream": False,
        "bm25_topk": bm25_topk,
        "milvus_topk": milvus_topk,
        "reranker_topk": reranker_topk,
        "enable_thinking": enable_thinking,
    }
    
    try:
        response = requests.post(
            f"{rag_url}/chat",
            json=request_data,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        
        # 更新历史
        new_history = history + [[query, result["answer"]]]
        
        # 处理图片
        images = []
        related_images = result.get("related_images", [])
        print(f"DEBUG: 收到 {len(related_images)} 个图片信息")
        for img_info in related_images:
            if isinstance(img_info, dict):
                # 图片路径可能在 image_path、url 或 path 字段中
                img_path = img_info.get("image_path") or img_info.get("url") or img_info.get("path", "")
                print(f"DEBUG: 处理图片信息 - {img_info}, 路径: {img_path}")
                if img_path:
                    checked_path = check_image_path(img_path)
                    if checked_path:
                        title = img_info.get("title", "图片")
                        images.append((checked_path, title))
                        print(f"DEBUG: 成功添加图片 - 路径: {checked_path}, 标题: {title}")
                    else:
                        print(f"DEBUG: 图片路径无效 - {img_path}")
                else:
                    print(f"DEBUG: 图片信息中没有路径字段")
        print(f"DEBUG: 最终图片列表长度: {len(images)}")
        
        # 格式化性能指标
        performance_text = format_performance_metrics(result.get("performance", {}))
        
        # 格式化答案（添加引用信息）
        answer = result["answer"]
        cite_pages = result.get("cite_pages", [])
        if cite_pages:
            answer += f"\n\n**引用页码**: {', '.join(map(str, cite_pages))}"
        
        return new_history, answer, images, performance_text
    
    except requests.exceptions.RequestException as e:
        error_msg = f"请求失败: {str(e)}"
        return history, error_msg, [], ""
    except Exception as e:
        error_msg = f"处理错误: {str(e)}"
        return history, error_msg, [], ""


def chat_stream(
    query: str,
    history: List[List[str]],
    bm25_topk: int,
    milvus_topk: int,
    reranker_topk: int,
    enable_thinking: bool,
):
    """
    流式聊天处理（生成器函数）
    
    Yields:
        (更新后的历史, 当前答案, 图片列表, 性能指标)
    """
    if not query.strip():
        yield history, "", [], ""
        return
    
    # 构建历史消息
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    # 准备请求
    rag_url = get_rag_server_url()
    request_data = {
        "query": query,
        "history": messages,
        "stream": True,
        "bm25_topk": bm25_topk,
        "milvus_topk": milvus_topk,
        "reranker_topk": reranker_topk,
        "enable_thinking": enable_thinking,
    }
    
    accumulated_answer = ""
    images = []
    performance_text = ""
    cite_pages = []
    
    try:
        response = requests.post(
            f"{rag_url}/chat/stream",
            json=request_data,
            stream=True,
            timeout=120,
            headers={"Accept": "text/event-stream"}
        )
        response.raise_for_status()
        
        # 手动解析SSE流式响应
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            
            # SSE格式: "data: {...}\n\n"
            if line.startswith("data: "):
                data_str = line[6:]  # 移除 "data: " 前缀
                
                if data_str == "[DONE]":
                    break
                
                try:
                    data = json.loads(data_str)
                    
                    if data.get("type") == "token":
                        # 流式token
                        token = data.get("content", "")
                        accumulated_answer += token
                        # 更新历史（临时答案）
                        new_history = history + [[query, accumulated_answer]]
                        yield new_history, accumulated_answer, images, performance_text
                    
                    elif data.get("type") == "final":
                        # 最终结果
                        final_answer = data.get("answer", accumulated_answer)
                        cite_pages = data.get("cite_pages", [])
                        
                        # 处理图片
                        images = []
                        for img_info in data.get("related_images", []):
                            if isinstance(img_info, dict):
                                # 图片路径可能在 image_path、url 或 path 字段中
                                img_path = img_info.get("image_path") or img_info.get("url") or img_info.get("path", "")
                                if img_path:
                                    checked_path = check_image_path(img_path)
                                    if checked_path:
                                        title = img_info.get("title", "图片")
                                        images.append((checked_path, title))
                        
                        # 格式化性能指标
                        performance_text = format_performance_metrics(data.get("performance", {}))
                        
                        # 格式化答案（添加引用信息）
                        formatted_answer = final_answer
                        if cite_pages:
                            formatted_answer += f"\n\n**引用页码**: {', '.join(map(str, cite_pages))}"
                        
                        # 更新历史（最终答案）
                        new_history = history + [[query, formatted_answer]]
                        yield new_history, formatted_answer, images, performance_text
                
                except json.JSONDecodeError:
                    # 如果JSON解析失败，跳过这一行
                    continue
                except Exception as e:
                    print(f"处理SSE事件时出错: {e}")
                    continue
    
    except requests.exceptions.RequestException as e:
        error_msg = f"请求失败: {str(e)}"
        yield history, error_msg, [], ""
    except Exception as e:
        error_msg = f"处理错误: {str(e)}"
        yield history, error_msg, [], ""


def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="EvRAG 对话系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# EvRAG 对话系统")
        gr.Markdown("基于检索增强生成(RAG)的智能问答系统")
        
        with gr.Row():
            with gr.Column(scale=2):
                # 聊天界面
                chatbot = gr.Chatbot(
                    label="对话历史",
                    height=500,
                    show_copy_button=True,
                )
                
                with gr.Row():
                    query_input = gr.Textbox(
                        label="输入问题",
                        placeholder="请输入您的问题...",
                        lines=2,
                        scale=4,
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)
                    clear_btn = gr.Button("清除", scale=1)
                
                # 流式输出开关
                stream_checkbox = gr.Checkbox(
                    label="启用流式输出",
                    value=True,
                )
            
            with gr.Column(scale=1):
                # 性能监控（移到最上面）
                gr.Markdown("### 性能指标")
                performance_display = gr.Markdown(label="性能", value="")
                
                gr.Markdown("---")  # 分隔线
                
                # 参数配置
                gr.Markdown("### 检索参数")
                bm25_topk = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=DEFAULT_BM25_TOPK,
                    step=1,
                    label="BM25 TopK",
                )
                milvus_topk = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=DEFAULT_MILVUS_TOPK,
                    step=1,
                    label="Milvus TopK",
                )
                reranker_topk = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=DEFAULT_RERANKER_TOPK,
                    step=1,
                    label="Reranker TopK",
                )
                enable_thinking = gr.Checkbox(
                    label="启用思考模式",
                    value=False,
                )
                
                gr.Markdown("---")  # 分隔线
                
                # 结果显示区域
                gr.Markdown("### 答案详情")
                answer_display = gr.Markdown(label="答案", value="")
                
                # 图片展示
                gr.Markdown("### 相关图片")
                image_gallery = gr.Gallery(
                    label="图片",
                    show_label=False,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height="auto",
                    type="filepath",  # 明确指定使用文件路径类型
                )
        
        # 事件处理
        def process_query(
            query: str,
            history: List[List[str]],
            stream: bool,
            bm25_topk_val: int,
            milvus_topk_val: int,
            reranker_topk_val: int,
            enable_thinking_val: bool,
        ):
            """处理查询"""
            if stream:
                # 流式处理
                for result in chat_stream(
                    query, history, bm25_topk_val, milvus_topk_val,
                    reranker_topk_val, enable_thinking_val
                ):
                    # 返回结果 + 清空输入框
                    yield result[0], result[1], result[2], result[3], ""
            else:
                # 非流式处理
                result = chat_non_stream(
                    query, history, bm25_topk_val, milvus_topk_val,
                    reranker_topk_val, enable_thinking_val
                )
                # 返回结果 + 清空输入框
                yield result[0], result[1], result[2], result[3], ""
        
        # 绑定事件
        submit_btn.click(
            fn=process_query,
            inputs=[
                query_input,
                chatbot,
                stream_checkbox,
                bm25_topk,
                milvus_topk,
                reranker_topk,
                enable_thinking,
            ],
            outputs=[chatbot, answer_display, image_gallery, performance_display, query_input],
        )
        
        query_input.submit(
            fn=process_query,
            inputs=[
                query_input,
                chatbot,
                stream_checkbox,
                bm25_topk,
                milvus_topk,
                reranker_topk,
                enable_thinking,
            ],
            outputs=[chatbot, answer_display, image_gallery, performance_display, query_input],
        )
        
        clear_btn.click(
            fn=lambda: ([], "", [], ""),
            outputs=[chatbot, answer_display, image_gallery, performance_display],
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
    )

