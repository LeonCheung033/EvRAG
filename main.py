"""
EvRAG主入口

提供命令行接口，支持构建索引、推理和QA生成等功能。
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from src.evrag.config import get_settings, reload_settings

# 初始化typer应用
app = typer.Typer(
    name="evrag",
    help="EvRAG - Enhanced RAG System",
    add_completion=False,
)

# 初始化rich console
console = Console()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("evrag")

@app.command()
def build_index(
    pdf_path: Optional[Path] = typer.Option(None, "--pdf-path", "-p", help="PDF文件路径"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
):
    """
    构建索引
    
    从PDF文件解析文档，构建BM25和Milvus索引。
    """
    console.print("[bold green]Building index...[/bold green]")
    
    # 重新加载配置
    if config_file:
        reload_settings(config_file)
    settings = get_settings()
    
    # 使用配置文件中的pdf_path或命令行参数
    pdf_file = pdf_path or settings.pdf_path
    if not pdf_file or not pdf_file.exists():
        console.print(f"[bold red]Error: PDF file not found: {pdf_file}[/bold red]")
        raise typer.Exit(1)
    
    try:
        from src.evrag.parser import PDFParser
        
        console.print(f"Parsing PDF: {pdf_file}")
        parser = PDFParser()
        documents = parser.parse(pdf_file)
        console.print(f"Parsed {len(documents)} documents")
        
        # 构建BM25索引
        console.print("Building BM25 index...")
        from src.evrag.retriever import BM25Retriever
        bm25_retriever = BM25Retriever(docs=documents, retrieve=False)
        console.print("[bold green]✓[/bold green] BM25 index built")
        
        # 构建Milvus索引
        console.print("Building Milvus index...")
        from src.evrag.retriever import MilvusRetriever
        milvus_retriever = MilvusRetriever(docs=documents, retrieve=False)
        console.print("[bold green]✓[/bold green] Milvus index built")
        
        console.print("[bold green]Index building completed![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error building index: {e}[/bold red]")
        logger.exception("Index building failed")
        raise typer.Exit(1)
    

@app.command()
def infer(
    query: str = typer.Argument(..., help="查询问题"),
    topk: int = typer.Option(5, "--topk", "-k", help="返回的文档数量"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    stream: bool = typer.Option(False, "--stream", "-s", help="流式输出"),
):
    """
    推理/问答
    
    使用RAG系统回答问题。
    """
    console.print(f"[bold green]Query:[/bold green] {query}")
    
    # 重新加载配置
    if config_file:
        reload_settings(config_file)
    settings = get_settings()

    try:
        # 初始化检索器
        from src.evrag.retriever import BM25Retriever, MilvusRetriever
        from src.evrag.reranker import BGEReranker
        from src.evrag.client import LocalLLMClient, ChatClient
        from src.evrag.utils import merge_docs, post_processing

        console.print("Loading retrievers...")
        bm25_retriever = BM25Retriever(docs=None, retrieve=True)
        milvus_retriever = MilvusRetriever(docs=None, retrieve=True)
        
        # BM25检索
        console.print("Retrieving with BM25...")
        bm25_docs = bm25_retriever.retrieve_topk(query, topk=topk * 2)
        
        # Milvus检索
        console.print("Retrieving with Milvus...")
        milvus_docs = milvus_retriever.retrieve_topk(query, topk=topk * 2)
        
        # 合并文档
        console.print("Merging documents...")
        merged_docs = merge_docs(bm25_docs, milvus_docs)
        
        # 重排序
        console.print("Reranking documents...")
        reranker = BGEReranker()
        ranked_docs = reranker.rank(query, merged_docs, topk=topk)
        
        # 构建上下文
        context = "\n".join([
            f"【{idx+1}】{doc.page_content}"
            for idx, doc in enumerate(ranked_docs)
        ])

        # 生成答案
        console.print("Generating answer...")
        llm_client = LocalLLMClient()
        chat_client = ChatClient(llm_client)

        if stream:
            console.print("\n[bold cyan]Answer:[/bold cyan]")
            response = ""
            for chunk in chat_client.chat(query=query, context=context, stream=True):
                console.print(chunk, end="")
                response += chunk
            console.print("\n")
        else:
            response = chat_client.chat(query=query, context=context, stream=False)
            console.print(f"\n[bold cyan]Answer:[/bold cyan]\n{response}\n")

        # 后处理
        result = post_processing(response, ranked_docs)
        console.print(f"[bold green]Processed Answer:[/bold green] {result['answer']}")
        if result['cite_pages']:
            console.print(f"[bold green]Cited Pages:[/bold green] {result['cite_pages']}")
        if result['related_images']:
            console.print(f"[bold green]Related Images:[/bold green] {len(result['related_images'])} images")

    except Exception as e:
        console.print(f"[bold red]Error during inference: {e}[/bold red]")
        logger.exception("Inference failed")
        raise typer.Exit(1)

@app.command()
def gen_qa(
    input_path: Path = typer.Argument(..., help="输入文档pickle文件路径"),
    output_path: Path = typer.Option(Path("data/qa_pairs/qa_pair.json"), "--output", "-o", help="输出文件路径"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    max_workers: int = typer.Option(20, "--workers", "-w", help="最大并发工作线程数"),
):
    """
    生成QA对
    
    从文档生成问答对。
    """
    console.print("[bold green]Generating QA pairs...[/bold green]")
    
    # 重新加载配置
    if config_file:
        reload_settings(config_file)
    
    if not input_path.exists():
        console.print(f"[bold red]Error: Input file not found: {input_path}[/bold red]")
        raise typer.Exit(1)
    
    try:
        import pickle
        from src.evrag.gen_qa import QAGenerator
        from src.evrag.client import LocalLLMClient
        
        # 加载文档
        console.print(f"Loading documents from {input_path}...")
        with open(input_path, 'rb') as f:
            documents = pickle.load(f)
        console.print(f"Loaded {len(documents)} documents")
        
        # 初始化QA生成器
        llm_client = LocalLLMClient()
        generator = QAGenerator(llm_client, max_workers=max_workers)
        
        # 生成QA对
        console.print("Generating QA pairs...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        qa_dict = generator.generate_qa_from_documents(
            documents,
            output_file=output_path
        )
        
        console.print(f"[bold green]Generated {len(qa_dict)} QA pairs[/bold green]")
        console.print(f"Output saved to: {output_path}")

    except Exception as e:
        console.print(f"[bold red]Error generating QA pairs: {e}[/bold red]")
        logger.exception("QA generation failed")
        raise typer.Exit(1)


@app.command()
def version():
    """显示版本信息"""
    from src.evrag import __version__
    console.print(f"[bold green]EvRAG version:[/bold green] {__version__}")


def main():
    """主函数"""
    app()

if __name__ == "__main__":
    main()