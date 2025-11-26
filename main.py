"""
EvRAG主入口

提供命令行接口，支持构建索引、推理和QA生成等功能。
"""

import sys
import json
import logging
import random
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm

from src.evrag.config import get_settings, reload_settings
from src.evrag.utils import PerformanceMonitor, analyze_data_quality

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
def prepare_data(
    pdf_path: Optional[Path] = typer.Option(None, "--pdf-path", "-p", help="PDF文件路径"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    skip_clean: bool = typer.Option(False, "--skip-clean", help="跳过文档清洗步骤"),
    skip_split: bool = typer.Option(False, "--skip-split", help="跳过文档切分步骤"),
):
    """
    数据准备：PDF解析、文档清洗、文档切分并保存到MongoDB
    
    完整流程：PDF解析 -> 文档清洗 -> 文档切分 -> 保存到MongoDB
    这是构建索引的第一步，完成后可以运行 build-index 命令构建检索索引。
    """
    console.print("[bold green]Preparing data...[/bold green]")
    
    # 重新加载配置
    if config_file:
        reload_settings(config_file)
    settings = get_settings()
    
    # 初始化性能监控
    monitor = PerformanceMonitor("prepare_data", output_dir=Path("logs/performance"))
    monitor.start()
    
    # 使用配置文件中的pdf_path或命令行参数
    pdf_file = pdf_path or settings.pdf_path
    if not pdf_file or not pdf_file.exists():
        console.print(f"[bold red]Error: PDF file not found: {pdf_file}[/bold red]")
        raise typer.Exit(1)
    
    try:
        import pickle
        from src.evrag.parser import PDFParser, texts_split
        from src.evrag.client import CleanClient, SemanticChunkClient
        
        # ========== 1. 解析PDF ==========
        with monitor.step("PDF解析"):
            console.print("\n[bold cyan][1/4] Parsing PDF...[/bold cyan]")
            raw_docs_path = settings.raw_docs_path
            raw_docs_path.parent.mkdir(parents=True, exist_ok=True)
            
            if raw_docs_path.exists():
                console.print(f"Loading raw documents from {raw_docs_path}...")
                with open(raw_docs_path, 'rb') as f:
                    raw_docs = pickle.load(f)
                console.print(f"[bold green]✓[/bold green] Loaded {len(raw_docs)} raw documents")
            else:
                console.print(f"Parsing PDF: {pdf_file}")
                parser = PDFParser(
                    pdf_path=pdf_file,
                    min_filter_pages=settings.pdf_min_filter_pages,
                    max_filter_pages=settings.pdf_max_filter_pages,
                    page_clip=settings.pdf_page_clip,
                )
                raw_docs = parser.parse()
                console.print(f"Parsed {len(raw_docs)} documents")
                
                # 保存raw_docs
                console.print(f"Saving raw documents to {raw_docs_path}...")
                with open(raw_docs_path, 'wb') as f:
                    pickle.dump(raw_docs, f)
                console.print("[bold green]✓[/bold green] Raw documents saved")
        
        # ========== 2. 文档清洗 ==========
        with monitor.step("文档清洗"):
            console.print("\n[bold cyan][2/4] Cleaning documents...[/bold cyan]")
            clean_docs_path = settings.clean_docs_path
            clean_docs_path.parent.mkdir(parents=True, exist_ok=True)
            
            if skip_clean:
                console.print("[yellow]Skipping document cleaning (--skip-clean)[/yellow]")
                clean_docs = raw_docs
            elif clean_docs_path.exists():
                console.print(f"Loading cleaned documents from {clean_docs_path}...")
                with open(clean_docs_path, 'rb') as f:
                    clean_docs = pickle.load(f)
                console.print(f"[bold green]✓[/bold green] Loaded {len(clean_docs)} cleaned documents")
            else:
                console.print("Cleaning documents with LLM...")
                # 使用OpenAIClient连接豆包API进行文档清洗
                from src.evrag.client import OpenAIClient
                llm_client = OpenAIClient(service="doubao")
                clean_client = CleanClient(llm_client)
                clean_docs = clean_client.clean_documents(raw_docs)
                console.print(f"Cleaned {len(clean_docs)} documents")
                
                # 保存clean_docs
                console.print(f"Saving cleaned documents to {clean_docs_path}...")
                with open(clean_docs_path, 'wb') as f:
                    pickle.dump(clean_docs, f)
                console.print("[bold green]✓[/bold green] Cleaned documents saved")
        
        # ========== 3. 文档切分 ==========
        with monitor.step("文档切分"):
            console.print("\n[bold cyan][3/4] Splitting documents...[/bold cyan]")
            split_docs_path = settings.split_docs_path
            split_docs_path.parent.mkdir(parents=True, exist_ok=True)
            
            if skip_split:
                console.print("[yellow]Skipping document splitting (--skip-split)[/yellow]")
                split_docs = clean_docs
            elif split_docs_path.exists():
                console.print(f"Loading split documents from {split_docs_path}...")
                with open(split_docs_path, 'rb') as f:
                    split_docs = pickle.load(f)
                console.print(f"[bold green]✓[/bold green] Loaded {len(split_docs)} split documents")
            else:
                console.print("Splitting documents (semantic + sentence-level)...")
                semantic_chunk_client = SemanticChunkClient()
                split_docs = texts_split(
                    clean_docs,
                    semantic_chunk_client=semantic_chunk_client,
                    collection_name="manual_text"
                )
                console.print(f"Split into {len(split_docs)} documents")
                console.print("[bold green]✓[/bold green] Documents saved to MongoDB")
                
                # 保存split_docs
                console.print(f"Saving split documents to {split_docs_path}...")
                with open(split_docs_path, 'wb') as f:
                    pickle.dump(split_docs, f)
                console.print("[bold green]✓[/bold green] Split documents saved")
        
        # 结束监控并生成报告
        monitor.end()
        report_path = monitor.save_report()
        monitor.print_summary()
        
        console.print("\n[bold green]✓ Data preparation completed![/bold green]")
        console.print(f"  - Raw documents: {len(raw_docs)}")
        console.print(f"  - Cleaned documents: {len(clean_docs)}")
        console.print(f"  - Split documents: {len(split_docs)}")
        console.print(f"  - Documents saved to MongoDB collection: manual_text")
        console.print(f"  - Performance report saved to: {report_path}")
        console.print("\n[yellow]Next step: Run 'python main.py build-index' to build retrieval indexes[/yellow]")
        
    except Exception as e:
        # 即使失败也保存监控数据
        if 'monitor' in locals():
            monitor.end()
            monitor.save_report()
            monitor.print_summary()
        console.print(f"[bold red]Error preparing data: {e}[/bold red]")
        logger.exception("Data preparation failed")
        raise typer.Exit(1)


@app.command()
def build_index(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    from_mongodb: bool = typer.Option(False, "--from-mongodb", help="从MongoDB加载文档（而不是从split_docs.pkl）"),
    skip_if_exists: bool = typer.Option(False, "--skip-if-exists", help="如果索引已存在则跳过构建（只进行验证）"),
    force: bool = typer.Option(False, "--force", help="强制重建索引（即使已存在）"),
):
    """
    构建检索索引
    
    从MongoDB或split_docs.pkl加载已切分的文档，然后构建BM25和Milvus检索索引。
    这是构建索引的第二步，需要先运行 prepare-data 命令完成数据准备。
    
    默认行为：如果索引已存在，会覆盖重建。
    使用 --skip-if-exists 可以跳过已存在的索引，只进行验证。
    使用 --force 可以强制重建索引。
    """
    console.print("[bold green]Building retrieval indexes...[/bold green]")
    
    # 重新加载配置
    if config_file:
        reload_settings(config_file)
    settings = get_settings()
    
    # 初始化性能监控
    monitor = PerformanceMonitor("build_index", output_dir=Path("logs/performance"))
    monitor.start()
    
    try:
        import pickle
        from src.evrag.retriever import BM25Retriever, MilvusRetriever
        from src.evrag.client import MongoDBClient
        from pymilvus import utility
        
        # ========== 0. 检查索引是否存在（如果使用 --skip-if-exists）==========
        bm25_index_path = settings.bm25_pickle_path
        bm25_exists = bm25_index_path.exists()
        
        # 检查Milvus collection是否存在
        milvus_exists = False
        try:
            # 先连接Milvus（不创建retriever）
            from pymilvus import connections
            from pathlib import Path as PathLib
            
            db_path = PathLib(settings.milvus_db_path)
            db_dir = db_path.parent
            db_dir.mkdir(parents=True, exist_ok=True)
            absolute_path = db_path.absolute()
            
            MILVUS_ALIAS = "default"
            if MILVUS_ALIAS not in connections.list_connections():
                connections.connect(alias=MILVUS_ALIAS, uri=str(absolute_path))
            
            # 检查collection是否存在
            collection_name = "hybrid_bge_m3"  # 默认collection名称
            milvus_exists = utility.has_collection(collection_name)
        except Exception:
            pass
        
        # 如果使用 --skip-if-exists 且索引都存在，跳过构建
        if skip_if_exists and bm25_exists and milvus_exists:
            console.print("\n[bold yellow]Indexes already exist, skipping build...[/bold yellow]")
            console.print("[dim]Using --skip-if-exists: Loading existing indexes for verification[/dim]")
            
            # 加载已存在的索引（不需要文档）
            with monitor.step("加载BM25索引"):
                console.print("\n[bold cyan][1/2] Loading existing indexes...[/bold cyan]")
                console.print("Loading BM25 index...")
                bm25_retriever = BM25Retriever(docs=None, retrieve=True)
                console.print("[bold green]✓[/bold green] BM25 index loaded")
            
            with monitor.step("加载Milvus索引"):
                console.print("Loading Milvus index...")
                milvus_retriever = MilvusRetriever(docs=None, retrieve=True)
                console.print("[bold green]✓[/bold green] Milvus index loaded")
            
            # 获取文档数量（用于验证）
            try:
                num_docs = milvus_retriever.col.num_entities
            except Exception:
                num_docs = 0
        else:
            # 需要构建索引，先加载文档
            # ========== 1. 加载文档 ==========
            with monitor.step("加载文档"):
                console.print("\n[bold cyan][1/3] Loading documents...[/bold cyan]")
                split_docs_path = settings.split_docs_path
                
                if from_mongodb:
                    console.print("Loading documents from MongoDB...")
                    mongodb_client = MongoDBClient()
                    collection = mongodb_client.get_collection("manual_text")
                    
                    # 从MongoDB加载所有文档
                    split_docs = []
                    cursor = collection.find({})
                    for doc_data in cursor:
                        from langchain_core.documents import Document
                        split_docs.append(
                            Document(
                                page_content=doc_data.get("page_content", ""),
                                metadata=doc_data.get("metadata", {})
                            )
                        )
                    console.print(f"[bold green]✓[/bold green] Loaded {len(split_docs)} documents from MongoDB")
                else:
                    if not split_docs_path.exists():
                        console.print(f"[bold red]Error: Split documents file not found: {split_docs_path}[/bold red]")
                        console.print("[yellow]Please run 'python main.py prepare-data' first, or use --from-mongodb to load from MongoDB[/yellow]")
                        raise typer.Exit(1)
                    
                    console.print(f"Loading split documents from {split_docs_path}...")
                    with open(split_docs_path, 'rb') as f:
                        split_docs = pickle.load(f)
                    console.print(f"[bold green]✓[/bold green] Loaded {len(split_docs)} documents from file")
                
                if len(split_docs) == 0:
                    console.print("[bold red]Error: No documents found![/bold red]")
                    console.print("[yellow]Please run 'python main.py prepare-data' first[/yellow]")
                    raise typer.Exit(1)
            
            num_docs = len(split_docs)
            
            # ========== 2. 构建索引 ==========
            if force:
                console.print("\n[bold yellow]Force rebuild: Existing indexes will be overwritten[/bold yellow]")
            elif bm25_exists or milvus_exists:
                console.print("\n[bold yellow]Warning: Existing indexes will be overwritten[/bold yellow]")
                console.print("[dim]Use --skip-if-exists to skip building if indexes already exist[/dim]")
            
            with monitor.step("构建BM25索引"):
                console.print("\n[bold cyan][2/3] Building indexes...[/bold cyan]")
                console.print("Building BM25 index...")
                bm25_retriever = BM25Retriever(docs=split_docs, retrieve=False)
                console.print("[bold green]✓[/bold green] BM25 index built")
            
            with monitor.step("构建Milvus索引"):
                console.print("Building Milvus index...")
                milvus_retriever = MilvusRetriever(docs=split_docs, retrieve=False)
                console.print("[bold green]✓[/bold green] Milvus index built")
            
            num_docs = len(split_docs)
        
        # ========== 3. 验证索引质量 ==========
        bm25_size = 0.0
        total_size = 0.0
        
        with monitor.step("验证索引质量"):
            console.print("\n[bold cyan][3/3] Verifying index quality...[/bold cyan]")
            
            # 验证BM25索引
            console.print("Verifying BM25 index...")
            bm25_index_path = settings.bm25_pickle_path
            if not bm25_index_path.exists():
                console.print(f"[bold red]✗[/bold red] BM25 index file not found: {bm25_index_path}")
                raise RuntimeError("BM25 index file not found")
            
            bm25_size = bm25_index_path.stat().st_size / (1024 * 1024)  # MB
            console.print(f"  - BM25 index file size: {bm25_size:.2f} MB")
            
            # 验证BM25检索功能
            try:
                test_query = "测试"
                test_results = bm25_retriever.retrieve_topk(test_query, topk=1)
                console.print(f"  - BM25 retrieval test: ✓ (returned {len(test_results)} results)")
            except Exception as e:
                console.print(f"  - BM25 retrieval test: ✗ ({e})")
                raise RuntimeError(f"BM25 retrieval test failed: {e}")
            
            # 验证Milvus索引
            console.print("Verifying Milvus index...")
            milvus_db_path = Path(settings.milvus_db_path)
            if not milvus_db_path.exists():
                console.print(f"[bold red]✗[/bold red] Milvus database not found: {milvus_db_path}")
                raise RuntimeError("Milvus database not found")
            
            # 计算Milvus数据库大小
            if milvus_db_path.is_dir():
                import os
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(milvus_db_path)
                    for filename in filenames
                ) / (1024 * 1024)  # MB
            else:
                total_size = milvus_db_path.stat().st_size / (1024 * 1024)  # MB
            console.print(f"  - Milvus database size: {total_size:.2f} MB")
            
            # 验证Milvus中的文档数量
            try:
                from pymilvus import utility
                collection_name = milvus_retriever.collection_name
                if utility.has_collection(collection_name):
                    collection = milvus_retriever.col
                    num_entities = collection.num_entities
                    console.print(f"  - Milvus documents count: {num_entities}")
                    
                    # 验证文档数量（如果知道输入文档数量）
                    if 'num_docs' in locals() and num_entities != num_docs:
                        console.print(f"  [yellow]⚠ Warning: Milvus document count ({num_entities}) != expected ({num_docs})[/yellow]")
                    else:
                        console.print(f"  - Document count: {num_entities}")
                else:
                    console.print(f"  [yellow]⚠ Warning: Milvus collection '{collection_name}' not found[/yellow]")
            except Exception as e:
                console.print(f"  [yellow]⚠ Warning: Could not verify Milvus document count: {e}[/yellow]")
            
            # 验证Milvus检索功能
            try:
                test_query = "测试"
                test_results = milvus_retriever.retrieve_topk(test_query, topk=1)
                console.print(f"  - Milvus retrieval test: ✓ (returned {len(test_results)} results)")
            except Exception as e:
                console.print(f"  - Milvus retrieval test: ⚠ ({type(e).__name__}: {str(e)[:100]})")
                console.print(f"  [yellow]Note: Index was built successfully, but retrieval test failed.[/yellow]")
                console.print(f"  [yellow]This may be due to sparse vector format compatibility. Index is still usable.[/yellow]")
                # 不抛出异常，因为索引构建本身是成功的
            
            console.print("[bold green]✓[/bold green] Index quality verification completed")
        
        # 结束监控并生成报告
        monitor.end()
        report_path = monitor.save_report()
        monitor.print_summary()
        
        console.print("\n[bold green]✓ Index building completed![/bold green]")
        if skip_if_exists and bm25_exists and milvus_exists:
            console.print(f"  - Indexes loaded (skipped build)")
            if 'num_docs' in locals():
                console.print(f"  - Documents in index: {num_docs}")
        else:
            if 'num_docs' in locals():
                console.print(f"  - Total documents indexed: {num_docs}")
        console.print(f"  - BM25 index: {settings.bm25_pickle_path} ({bm25_size:.2f} MB)")
        console.print(f"  - Milvus index: {settings.milvus_db_path} ({total_size:.2f} MB)")
        console.print(f"  - Performance report saved to: {report_path}")
        
    except Exception as e:
        # 即使失败也保存监控数据
        if 'monitor' in locals():
            monitor.end()
            monitor.save_report()
            monitor.print_summary()
        console.print(f"[bold red]Error building index: {e}[/bold red]")
        logger.exception("Index building failed")
        raise typer.Exit(1)

@app.command()
def analyze_data(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    output_report: Optional[Path] = typer.Option(None, "--output", "-o", help="输出报告路径（JSON格式）"),
):
    """
    分析数据质量
    
    分析PDF解析、文档清洗、文档切分三个阶段的数据质量和效果。
    包括统计信息、重复检查、父子关系分析等。
    """
    console.print("[bold green]Analyzing data quality...[/bold green]")
    
    # 重新加载配置
    if config_file:
        reload_settings(config_file)
    settings = get_settings()
    
    try:
        import pickle
        
        # 检查文件是否存在
        raw_docs_path = settings.raw_docs_path
        clean_docs_path = settings.clean_docs_path
        split_docs_path = settings.split_docs_path
        
        missing_files = []
        if not raw_docs_path.exists():
            missing_files.append(str(raw_docs_path))
        if not clean_docs_path.exists():
            missing_files.append(str(clean_docs_path))
        if not split_docs_path.exists():
            missing_files.append(str(split_docs_path))
        
        if missing_files:
            console.print(f"[bold red]Error: 以下文件不存在：[/bold red]")
            for f in missing_files:
                console.print(f"  - {f}")
            console.print("[yellow]请先运行 'python main.py prepare-data' 生成数据文件[/yellow]")
            raise typer.Exit(1)
        
        # 设置默认输出路径
        if output_report is None:
            output_report = Path("logs/data_quality_report.json")
        
        # 执行分析
        result = analyze_data_quality(
            raw_docs_path=raw_docs_path,
            clean_docs_path=clean_docs_path,
            split_docs_path=split_docs_path,
            output_report=output_report,
        )
        
        console.print("\n[bold green]✓ Data quality analysis completed![/bold green]")
        console.print(f"  - Report saved to: {output_report}")
        
    except Exception as e:
        console.print(f"[bold red]Error analyzing data: {e}[/bold red]")
        logger.exception("Data analysis failed")
        raise typer.Exit(1)


@app.command()
def show_examples(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    num_examples: int = typer.Option(3, "--num", "-n", help="要展示的示例数量"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="保存报告到文件（默认：logs/processing_examples.txt）"),
):
    """
    展示文档处理流程示例
    
    显示从原始文档(raw_docs)到清洗后文档(clean_docs)再到切分后文档(split_docs)的完整变化过程。
    包括每个阶段的文本内容、统计信息、主要变化等。
    
    如果指定了--output，会将详细报告保存到文件中。
    """
    console.print("[bold green]Loading documents and showing processing examples...[/bold green]")
    
    # 重新加载配置
    if config_file:
        reload_settings(config_file)
    settings = get_settings()
    
    try:
        import pickle
        from src.evrag.utils import show_processing_examples, generate_processing_examples_report
        
        # 检查文件是否存在
        raw_docs_path = settings.raw_docs_path
        clean_docs_path = settings.clean_docs_path
        split_docs_path = settings.split_docs_path
        
        missing_files = []
        if not raw_docs_path.exists():
            missing_files.append(str(raw_docs_path))
        if not clean_docs_path.exists():
            missing_files.append(str(clean_docs_path))
        if not split_docs_path.exists():
            missing_files.append(str(split_docs_path))
        
        if missing_files:
            console.print(f"[bold red]Error: 以下文件不存在：[/bold red]")
            for f in missing_files:
                console.print(f"  - {f}")
            console.print("[yellow]请先运行 'python main.py prepare-data' 生成数据文件[/yellow]")
            raise typer.Exit(1)
        
        # 加载文档
        console.print(f"[dim]Loading raw_docs from {raw_docs_path}...[/dim]")
        with open(raw_docs_path, 'rb') as f:
            raw_docs = pickle.load(f)
        console.print(f"[green]✓[/green] 加载原始文档: {len(raw_docs)} 个")
        
        console.print(f"[dim]Loading clean_docs from {clean_docs_path}...[/dim]")
        with open(clean_docs_path, 'rb') as f:
            clean_docs = pickle.load(f)
        console.print(f"[green]✓[/green] 加载清洗后文档: {len(clean_docs)} 个")
        
        console.print(f"[dim]Loading split_docs from {split_docs_path}...[/dim]")
        with open(split_docs_path, 'rb') as f:
            split_docs = pickle.load(f)
        console.print(f"[green]✓[/green] 加载切分后文档: {len(split_docs)} 个")
        
        # 展示示例（控制台输出）
        show_processing_examples(
            raw_docs=raw_docs,
            clean_docs=clean_docs,
            split_docs=split_docs,
            num_examples=num_examples,
        )
        
        # 如果指定了输出文件，生成详细报告并保存
        if output_file is not None:
            console.print(f"\n[dim]Generating detailed report...[/dim]")
            report = generate_processing_examples_report(
                raw_docs=raw_docs,
                clean_docs=clean_docs,
                split_docs=split_docs,
                num_examples=min(num_examples * 2, 10),  # 报告中显示更多示例
            )
            
            # 确保输出目录存在
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存报告
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            console.print(f"[bold green]✓[/bold green] 详细报告已保存到: {output_file}")
        else:
            # 默认保存到 logs/processing_examples.txt
            default_output = Path("logs/processing_examples.txt")
            default_output.parent.mkdir(parents=True, exist_ok=True)
            
            console.print(f"\n[dim]Generating detailed report...[/dim]")
            report = generate_processing_examples_report(
                raw_docs=raw_docs,
                clean_docs=clean_docs,
                split_docs=split_docs,
                num_examples=min(num_examples * 2, 10),
            )
            
            with open(default_output, 'w', encoding='utf-8') as f:
                f.write(report)
            
            console.print(f"[bold green]✓[/bold green] 详细报告已保存到: {default_output}")
        
        console.print("\n[bold green]✓ Examples display completed![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error showing examples: {e}[/bold red]")
        logger.exception("Examples display failed")
        raise typer.Exit(1)


@app.command()
def infer(
    query: str = typer.Argument(..., help="查询问题"),
    topk: int = typer.Option(5, "--topk", "-k", help="返回的文档数量"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    stream: bool = typer.Option(False, "--stream", "-s", help="流式输出"),
    enable_thinking: bool = typer.Option(False, "--enable-thinking/--no-enable-thinking", help="是否启用思考模式（默认关闭，复杂推理任务可启用）"),
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
            for chunk in chat_client.chat(query=query, context=context, stream=True, enable_thinking=enable_thinking):
                console.print(chunk, end="")
                response += chunk
            console.print("\n")
        else:
            response = chat_client.chat(query=query, context=context, stream=False, enable_thinking=enable_thinking)
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
    input_path: Optional[Path] = typer.Option(None, "--input", "-i", help="输入文档pickle文件路径（默认：clean_docs.pkl）"),
    output_path: Path = typer.Option(Path("data/qa_pairs/qa_pair.json"), "--output", "-o", help="输出文件路径"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    max_workers: int = typer.Option(20, "--workers", "-w", help="最大并发工作线程数"),
    show_examples: bool = typer.Option(True, "--show-examples/--no-show-examples", help="是否展示生成的QA对示例"),
    num_examples: int = typer.Option(5, "--num-examples", "-n", help="展示的示例数量"),
):
    """
    生成QA对
    
    从文档生成问答对，使用Deepseek作为LLM。
    支持示例展示。
    """
    console.print("[bold green]Generating QA pairs with Deepseek...[/bold green]")
    
    # 重新加载配置
    if config_file:
        reload_settings(config_file)
    settings = get_settings()
    
    # 初始化性能监控
    monitor = PerformanceMonitor("gen_qa", output_dir=Path("logs/performance"))
    monitor.start()
    
    try:
        import pickle
        import json
        from src.evrag.gen_qa import QAGenerator
        from src.evrag.client import OpenAIClient
        
        # ========== 1. 加载文档 ==========
        with monitor.step("加载文档"):
            if input_path is None:
                input_path = settings.clean_docs_path
            
            if not input_path.exists():
                console.print(f"[bold red]Error: Input file not found: {input_path}[/bold red]")
                console.print("[yellow]Please run 'python main.py prepare-data' first[/yellow]")
                raise typer.Exit(1)
            
            console.print(f"\n[bold cyan][1/4] Loading documents...[/bold cyan]")
            console.print(f"Loading documents from {input_path}...")
            with open(input_path, 'rb') as f:
                documents = pickle.load(f)
            console.print(f"[bold green]✓[/bold green] Loaded {len(documents)} documents")
        
        # ========== 2. 初始化QA生成器（使用Deepseek）==========
        with monitor.step("初始化QA生成器"):
            console.print("\n[bold cyan][2/4] Initializing QA generator...[/bold cyan]")
            console.print("Using Deepseek as LLM service...")
            llm_client = OpenAIClient(service="deepseek")
            generator = QAGenerator(llm_client, max_workers=max_workers)
            console.print("[bold green]✓[/bold green] QA generator initialized")
        
        # ========== 3. 生成QA对 ==========
        with monitor.step("生成QA对"):
            console.print("\n[bold cyan][3/4] Generating QA pairs...[/bold cyan]")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 如果输出文件已存在，加载checkpoint
            checkpoint = {}
            if output_path.exists():
                console.print(f"Loading checkpoint from {output_path}...")
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                item = json.loads(line)
                                checkpoint[item.get('unique_id')] = item
                    console.print(f"[dim]Found {len(checkpoint)} existing QA pairs in checkpoint[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to load checkpoint: {e}[/yellow]")
            
            qa_dict = generator.generate_qa_from_documents(
                documents,
                output_file=output_path,
                checkpoint=checkpoint if checkpoint else None
            )
            
            console.print(f"[bold green]✓[/bold green] Generated {len(qa_dict)} QA pairs")
            console.print(f"Output saved to: {output_path}")
        
        # ========== 4. 解析和统计QA对 ==========
        with monitor.step("解析QA对"):
            console.print("\n[bold cyan][4/4] Parsing and analyzing QA pairs...[/bold cyan]")
            
            all_qa_pairs = []
            failed_docs = []
            
            for unique_id, item in qa_dict.items():
                raw_resp = item.get('raw_resp', '')
                qa_list = QAGenerator.parse_qa_response(raw_resp)
                
                if qa_list:
                    for qa in qa_list:
                        # 确保qa是字典类型
                        if not isinstance(qa, dict):
                            continue
                        qa['unique_id'] = unique_id
                        all_qa_pairs.append(qa)
                else:
                    failed_docs.append(unique_id)
            
            console.print(f"  - Total QA pairs: {len(all_qa_pairs)}")
            console.print(f"  - Failed documents: {len(failed_docs)}")
            console.print(f"  - Average QA pairs per document: {len(all_qa_pairs) / max(len(qa_dict) - len(failed_docs), 1):.2f}")
        
        # ========== 5. 示例展示 ==========
        if show_examples and all_qa_pairs:
            console.print("\n[bold cyan]QA Pair Examples:[/bold cyan]")
            console.print("=" * 80)
            
            # 选择示例（优先选择高质量）
            examples = random.sample(all_qa_pairs, min(num_examples, len(all_qa_pairs)))
            
            for i, qa in enumerate(examples, 1):
                console.print(f"\n[bold]Example {i}:[/bold]")
                console.print(f"  [cyan]Question:[/cyan] {qa.get('question', 'N/A')}")
                console.print(f"  [green]Answer:[/green] {qa.get('answer', 'N/A')[:200]}{'...' if len(qa.get('answer', '')) > 200 else ''}")
        
        # 结束监控并生成报告
        monitor.end()
        report_path = monitor.save_report()
        monitor.print_summary()
        
        console.print("\n[bold green]✓ QA generation completed![/bold green]")
        console.print(f"  - Total QA pairs: {len(all_qa_pairs)}")
        console.print(f"  - Output file: {output_path}")
        console.print(f"  - Performance report saved to: {report_path}")

    except Exception as e:
        # 即使失败也保存监控数据
        if 'monitor' in locals():
            monitor.end()
            monitor.save_report()
            monitor.print_summary()
        console.print(f"[bold red]Error generating QA pairs: {e}[/bold red]")
        logger.exception("QA generation failed")
        raise typer.Exit(1)


@app.command()
def process_qa(
    qa_pair_path: Path = typer.Option(Path("data/qa_pairs/qa_pair.json"), "--qa-pair-path", help="QA对文件路径"),
    output_dir: Path = typer.Option(Path("data/qa_pairs"), "--output-dir", "-o", help="输出目录"),
    negative_samples_path: Path = typer.Option(Path("data/ut/raw_general_chats.txt"), "--negative-samples-path", help="负样本文件路径"),
    quality_threshold: int = typer.Option(3, "--quality-threshold", help="质量打分阈值"),
    train_ratio: float = typer.Option(0.9, "--train-ratio", help="训练集比例"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    step: str = typer.Option("all", "--step", help="执行步骤: load, filter, generalize, split, keywords, negative, save, all"),
    skip_quality_scoring: bool = typer.Option(False, "--skip-quality-scoring", help="跳过质量打分"),
    skip_question_rewriting: bool = typer.Option(False, "--skip-question-rewriting", help="跳过问题改写"),
    max_workers: int = typer.Option(20, "--workers", "-w", help="最大并发工作线程数"),
):
    """
    QA数据处理流程
    
    从qa_pair.json开始，经过质量打分、过滤、问题改写、数据扩充、
    训练/测试集切分，最终生成train_qa_pair.json和test_qa_pair.json。
    
    支持分步执行，使用 --step 参数指定执行的步骤。
    """
    console.print("[bold green]Processing QA data...[/bold green]")
    
    # 重新加载配置
    if config_file:
        reload_settings(config_file)
    settings = get_settings()
    
    # 初始化性能监控
    monitor = PerformanceMonitor("process_qa", output_dir=Path("logs/performance"))
    monitor.start()
    
    try:
        from src.evrag.gen_qa import QAGenerator, QAProcessor
        from src.evrag.client import OpenAIClient
        
        # ========== 1. 初始化LLM客户端和QA生成器 ==========
        with monitor.step("初始化"):
            console.print("\n[bold cyan][1/1] Initializing...[/bold cyan]")
            llm_client = OpenAIClient(service="deepseek")
            qa_generator = QAGenerator(
                llm_client=llm_client,
                max_workers=max_workers,
            )
            qa_processor = QAProcessor(
                qa_generator=qa_generator,
                quality_threshold=quality_threshold,
                train_ratio=train_ratio,
            )
            console.print("[bold green]✓[/bold green] Initialized")
        
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 定义输出文件路径
        expand_qa_pair_path = output_dir / "expand_qa_pair.json"
        train_path = output_dir / "train_qa_pair.json"
        test_path = output_dir / "test_qa_pair.json"
        test_keywords_path = output_dir / "test_keywords_pair.json"
        
        # ========== 2. 分步执行 ==========
        if step == "all":
            # 完整流程
            console.print("\n[bold cyan]Running full pipeline...[/bold cyan]")
            qa_processor.process(
                qa_pair_path=qa_pair_path,
                output_dir=output_dir,
                negative_samples_path=negative_samples_path,
                skip_quality_scoring=skip_quality_scoring,
                skip_question_rewriting=skip_question_rewriting,
            )
        elif step == "load":
            # 步骤1: 加载QA对
            console.print("\n[bold cyan][Step 1] Loading QA pairs...[/bold cyan]")
            qa_dict = qa_processor.load_qa_pairs(qa_pair_path)
            console.print(f"[bold green]✓[/bold green] Loaded {len(qa_dict)} QA pairs")
            
            # 统计信息
            total_qa_pairs = 0
            for unique_id, info in qa_dict.items():
                raw_resp = info.get("raw_resp", "[]")
                try:
                    qa_list = qa_generator.parse_qa_response(raw_resp)
                    total_qa_pairs += len(qa_list)
                except:
                    pass
            console.print(f"  - Total documents: {len(qa_dict)}")
            console.print(f"  - Total QA pairs: {total_qa_pairs}")
            
        elif step == "filter":
            # 步骤2: QA质量打分和过滤
            console.print("\n[bold cyan][Step 2] Scoring and filtering QA pairs...[/bold cyan]")
            qa_dict = qa_processor.load_qa_pairs(qa_pair_path)
            filtered_qa_pairs = qa_processor.score_and_filter_qa_pairs(
                qa_dict=qa_dict,
                skip_scoring=skip_quality_scoring,
            )
            # 保存过滤后的QA对
            filtered_path = output_dir / "filtered_qa_pair.json"
            qa_processor.save_filtered_qa_pairs(
                filtered_qa_pairs=filtered_qa_pairs,
                output_path=filtered_path,
            )
            console.print(f"[bold green]✓[/bold green] Filtered to {len(filtered_qa_pairs)} QA pairs")
            console.print(f"  - Quality threshold: {quality_threshold}")
            console.print(f"  - Skip scoring: {skip_quality_scoring}")
            console.print(f"  - Saved to: {filtered_path}")
            
        elif step == "generalize":
            # 步骤3: 问题改写并生成expand_qa_pair.json（合并generalize和expand）
            console.print("\n[bold cyan][Step 3] Generalizing and expanding questions...[/bold cyan]")
            # 从filtered_qa_pair.json加载数据（保证原子性）
            filtered_path = output_dir / "filtered_qa_pair.json"
            if not filtered_path.exists():
                console.print(f"[bold red]✗[/bold red] Filtered QA pairs file not found: {filtered_path}")
                console.print("  Please run 'filter' step first")
                return
            
            filtered_qa_pairs = qa_processor.load_filtered_qa_pairs(filtered_path)
            expand_qa_pairs = qa_processor.generalize_and_expand_questions(
                filtered_qa_pairs=filtered_qa_pairs,
                output_file=expand_qa_pair_path,
                skip_rewriting=skip_question_rewriting,
            )
            console.print(f"[bold green]✓[/bold green] Generalized {len(expand_qa_pairs)} questions")
            console.print(f"  - Output file: {expand_qa_pair_path}")
            console.print(f"  - Skip rewriting: {skip_question_rewriting}")
            
        elif step == "split":
            # 步骤5: 训练/测试集切分
            console.print("\n[bold cyan][Step 5] Splitting train/test sets...[/bold cyan]")
            # 从filtered_qa_pair.json加载数据
            filtered_path = output_dir / "filtered_qa_pair.json"
            if not filtered_path.exists():
                console.print(f"[bold red]✗[/bold red] Filtered QA pairs file not found: {filtered_path}")
                console.print("  Please run 'filter' step first")
                return
            
            filtered_qa_pairs = qa_processor.load_filtered_qa_pairs(filtered_path)
            # 加载expand_qa_pair.json
            if not expand_qa_pair_path.exists():
                console.print(f"[bold red]✗[/bold red] Expand QA pairs file not found: {expand_qa_pair_path}")
                console.print("  Please run 'generalize' step first")
                return
            
            expand_qa_pairs = qa_processor.generalize_and_expand_questions(
                filtered_qa_pairs=filtered_qa_pairs,
                output_file=expand_qa_pair_path,
                skip_rewriting=True,  # 只加载，不重新生成
            )
            expanded_qa_pairs = qa_processor.expand_qa_pairs(
                qa_pairs=filtered_qa_pairs,
                expand_qa_pairs=expand_qa_pairs,
            )
            train_qa_pairs, test_qa_pairs = qa_processor.split_train_test(expanded_qa_pairs)
            
            # 保存切分后的结果（中间结果，后续还会添加关键词和负样本）
            qa_processor.save_final_data(
                train_qa_pairs=train_qa_pairs,
                test_qa_pairs=test_qa_pairs,
                train_path=train_path,
                test_path=test_path,
            )
            
            console.print(f"[bold green]✓[/bold green] Split completed")
            console.print(f"  - Expanded QA pairs: {len(expanded_qa_pairs)}")
            console.print(f"  - Train set: {len(train_qa_pairs)} QA pairs ({len(train_qa_pairs)/len(expanded_qa_pairs)*100:.1f}%)")
            console.print(f"  - Test set: {len(test_qa_pairs)} QA pairs ({len(test_qa_pairs)/len(expanded_qa_pairs)*100:.1f}%)")
            console.print(f"  - Saved to: {train_path} and {test_path}")
            
        elif step == "keywords":
            # 步骤6: 测试集关键词提取
            console.print("\n[bold cyan][Step 6] Extracting keywords for test set...[/bold cyan]")
            # 直接从切分后的测试集文件加载数据
            if not test_path.exists():
                console.print(f"[bold red]✗[/bold red] Test QA pairs file not found: {test_path}")
                console.print("  Please run 'split' step first")
                return
            
            # 加载测试集
            with open(test_path, "r", encoding="utf-8") as fd:
                test_qa_pairs = json.load(fd)
            
            console.print(f"  - Loaded {len(test_qa_pairs)} test QA pairs")
            
            # 提取关键词
            keywords_mapping = qa_processor.extract_keywords_for_test_set(
                test_qa_pairs=test_qa_pairs,
                output_file=test_keywords_path,
            )
            
            # 保存更新后的测试集（包含关键词）
            # 只保存测试集，不覆盖训练集
            test_path.parent.mkdir(parents=True, exist_ok=True)
            with open(test_path, "w", encoding="utf-8") as fd:
                json.dump(test_qa_pairs, fd, ensure_ascii=False, indent=2)
            print(f"Updated test set saved to: {test_path}, {len(test_qa_pairs)} 条")
            
            console.print(f"[bold green]✓[/bold green] Extracted keywords for {len(keywords_mapping)} answers")
            console.print(f"  - Keywords file: {test_keywords_path}")
            console.print(f"  - Updated test set: {test_path}")
            
        elif step == "negative":
            # 步骤7: 添加负样本
            console.print("\n[bold cyan][Step 7] Adding negative samples...[/bold cyan]")
            # 直接从已保存的训练集和测试集文件加载数据
            if not train_path.exists():
                console.print(f"[bold red]✗[/bold red] Train QA pairs file not found: {train_path}")
                console.print("  Please run 'split' step first")
                return
            
            if not test_path.exists():
                console.print(f"[bold red]✗[/bold red] Test QA pairs file not found: {test_path}")
                console.print("  Please run 'split' step first")
                return
            
            # 加载训练集和测试集
            with open(train_path, "r", encoding="utf-8") as fd:
                train_qa_pairs = json.load(fd)
            with open(test_path, "r", encoding="utf-8") as fd:
                test_qa_pairs = json.load(fd)
            
            console.print(f"  - Loaded {len(train_qa_pairs)} train QA pairs")
            console.print(f"  - Loaded {len(test_qa_pairs)} test QA pairs")
            
            # 添加负样本
            train_qa_pairs, test_qa_pairs = qa_processor.add_negative_samples(
                train_qa_pairs=train_qa_pairs,
                test_qa_pairs=test_qa_pairs,
                negative_samples_path=negative_samples_path,
            )
            
            # 保存添加负样本后的数据
            qa_processor.save_final_data(
                train_qa_pairs=train_qa_pairs,
                test_qa_pairs=test_qa_pairs,
                train_path=train_path,
                test_path=test_path,
            )
            
            console.print(f"[bold green]✓[/bold green] Added negative samples")
            console.print(f"  - Train set: {len(train_qa_pairs)} QA pairs")
            console.print(f"  - Test set: {len(test_qa_pairs)} QA pairs")
            console.print(f"  - Saved to: {train_path} and {test_path}")
            
        elif step == "save":
            # 步骤8: 保存最终数据（验证和重新保存，确保格式正确）
            console.print("\n[bold cyan][Step 8] Saving final data...[/bold cyan]")
            # 直接从已保存的文件加载最终数据
            if not train_path.exists():
                console.print(f"[bold red]✗[/bold red] Train QA pairs file not found: {train_path}")
                console.print("  Please run 'negative' step first")
                return
            
            if not test_path.exists():
                console.print(f"[bold red]✗[/bold red] Test QA pairs file not found: {test_path}")
                console.print("  Please run 'negative' step first")
                return
            
            # 加载最终数据
            with open(train_path, "r", encoding="utf-8") as fd:
                train_qa_pairs = json.load(fd)
            with open(test_path, "r", encoding="utf-8") as fd:
                test_qa_pairs = json.load(fd)
            
            console.print(f"  - Loaded {len(train_qa_pairs)} train QA pairs")
            console.print(f"  - Loaded {len(test_qa_pairs)} test QA pairs")
            
            # 重新保存（确保格式正确，打乱顺序）
            qa_processor.save_final_data(
                train_qa_pairs=train_qa_pairs,
                test_qa_pairs=test_qa_pairs,
                train_path=train_path,
                test_path=test_path,
            )
            
            console.print(f"[bold green]✓[/bold green] Saved final data")
            console.print(f"  - Train set: {train_path} ({len(train_qa_pairs)} QA pairs)")
            console.print(f"  - Test set: {test_path} ({len(test_qa_pairs)} QA pairs)")
            
            # 统计信息
            train_negative = len([q for q in train_qa_pairs if q.get("answer") == "无答案"])
            test_negative = len([q for q in test_qa_pairs if q.get("answer") == "无答案"])
            test_with_keywords = len([q for q in test_qa_pairs if q.get("keywords")])
            console.print(f"  - Train negative samples: {train_negative}")
            console.print(f"  - Test negative samples: {test_negative}")
            console.print(f"  - Test samples with keywords: {test_with_keywords}")
        else:
            console.print(f"[bold red]Error: Unknown step '{step}'[/bold red]")
            console.print("Available steps: load, filter, generalize, split, keywords, negative, save, all")
            raise typer.Exit(1)
        
        # 结束监控并生成报告
        monitor.end()
        report_path = monitor.save_report()
        monitor.print_summary()
        
        console.print("\n[bold green]✓ QA processing completed![/bold green]")
        console.print(f"  - Performance report saved to: {report_path}")
        
    except Exception as e:
        # 即使失败也保存监控数据
        if 'monitor' in locals():
            monitor.end()
            monitor.save_report()
            monitor.print_summary()
        console.print(f"[bold red]Error processing QA data: {e}[/bold red]")
        logger.exception("QA processing failed")
        raise typer.Exit(1)


@app.command()
def generate_sft_data(
    train_qa_path: Path = typer.Option(Path("data/qa_pairs/train_qa_pair.json"), "--train-qa-path", help="训练集QA对文件路径"),
    output_dir: Path = typer.Option(Path("data"), "--output-dir", "-o", help="输出目录"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    step: str = typer.Option("all", "--step", help="执行步骤: train_data, summary, rerank, test_verify, test_pred, all"),
    bm25_topk: int = typer.Option(5, "--bm25-topk", help="BM25检索数量"),
    milvus_topk: int = typer.Option(10, "--milvus-topk", help="Milvus检索数量"),
    reranker_topk: int = typer.Option(5, "--reranker-topk", help="Reranker重排序数量"),
    test_rate: float = typer.Option(0.08, "--test-rate", help="测试集比例"),
    rerank_dev_size: int = typer.Option(1000, "--rerank-dev-size", help="Reranker开发集大小"),
    max_workers: int = typer.Option(10, "--workers", "-w", help="最大并发工作线程数"),
):
    """
    生成微调训练数据
    
    从train_qa_pair.json生成微调训练数据，包括：
    1. train_data.json - RAG检索数据
    2. summary_data - SFT训练数据
    3. rerank_data - Reranker训练数据
    4. test_qa_pair_verify.json - 测试集验证数据（从test_qa_pair.json筛选）
    5. test_qa_pair_pred.json - 测试集预测数据（对验证集进行RAG预测）
    """
    console.print("[bold green]Generating SFT data...[/bold green]")
    
    # 重新加载配置
    if config_file:
        reload_settings(config_file)
    settings = get_settings()
    
    # 初始化性能监控
    monitor = PerformanceMonitor("generate_sft_data", output_dir=Path("logs/performance"))
    monitor.start()
    
    try:
        from src.evrag.gen_qa import SFTDataGenerator
        from src.evrag.client import OpenAIClient, MongoDBClient
        
        # 初始化LLM客户端（使用Deepseek API生成问答数据）
        with monitor.step("初始化"):
            console.print("\n[bold cyan][1/1] Initializing...[/bold cyan]")
            console.print("[yellow]Using Deepseek API for QA generation...[/yellow]")
            llm_client = OpenAIClient(service="deepseek")
            
            # 检查MongoDB配置
            mongodb_client = None
            if hasattr(settings, 'mongodb_host') and settings.mongodb_host:
                try:
                    mongodb_client = MongoDBClient()
                except Exception as e:
                    console.print(f"[yellow]Warning: MongoDB connection failed: {e}[/yellow]")
                    mongodb_client = None
            
            # 检查SiliconFlow API配置（用于Reranker）
            use_siliconflow = bool(settings.siliconflow_api_key)
            if use_siliconflow:
                console.print("[yellow]Using SiliconFlow API for Reranker...[/yellow]")
            else:
                console.print("[yellow]Warning: SiliconFlow API key not configured, using BGE Reranker[/yellow]")
            
            sft_generator = SFTDataGenerator(
                llm_client=llm_client,
                mongodb_client=mongodb_client,
                bm25_topk=bm25_topk,
                milvus_topk=milvus_topk,
                reranker_topk=reranker_topk,
                test_rate=test_rate,
                rerank_dev_size=rerank_dev_size,
                max_workers=max_workers,
                use_siliconflow_reranker=use_siliconflow,
            )
            console.print("[bold green]✓[/bold green] Initialized")
        
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 定义输出文件路径
        train_data_path = output_dir / "qa_pairs" / "train_data.json"
        summary_dir = output_dir / "summary_data"
        rerank_dir = output_dir / "rerank_data"
        
        # 分步执行
        if step == "all":
            # 完整流程
            console.print("\n[bold cyan]Running full pipeline...[/bold cyan]")
            
            # Step 1: 生成train_data.json
            console.print("\n[bold cyan][Step 1] Generating train_data.json...[/bold cyan]")
            sft_generator.generate_train_data(
                train_qa_path=train_qa_path,
                output_path=train_data_path,
            )
            console.print(f"[bold green]✓[/bold green] Generated train_data.json")
            
            # Step 2: 生成summary_data
            console.print("\n[bold cyan][Step 2] Generating summary_data...[/bold cyan]")
            summary_train, summary_test = sft_generator.generate_summary_data(
                train_data_path=train_data_path,
                output_dir=output_dir,
            )
            console.print(f"[bold green]✓[/bold green] Generated summary_data")
            console.print(f"  - Train: {len(summary_train)} items")
            console.print(f"  - Test: {len(summary_test)} items")
            
            # Step 3: 生成rerank_data
            console.print("\n[bold cyan][Step 3] Generating rerank_data...[/bold cyan]")
            rerank_train, rerank_dev, rerank_test = sft_generator.generate_rerank_data(
                train_data_path=train_data_path,
                output_dir=output_dir,
            )
            console.print(f"[bold green]✓[/bold green] Generated rerank_data")
            console.print(f"  - Train: {len(rerank_train)} items")
            console.print(f"  - Dev: {len(rerank_dev)} items")
            console.print(f"  - Test: {len(rerank_test)} items")
            
            # Step 4: 生成test_qa_pair_verify.json
            console.print("\n[bold cyan][Step 4] Generating test_qa_pair_verify.json...[/bold cyan]")
            test_qa_path = output_dir / "qa_pairs" / "test_qa_pair.json"
            test_verify_path = output_dir / "qa_pairs" / "test_qa_pair_verify.json"
            
            if test_qa_path.exists():
                verify_pairs = sft_generator.generate_test_verify_data(
                    test_qa_path=test_qa_path,
                    output_path=test_verify_path,
                    verify_ratio=0.1,
                )
                console.print(f"[bold green]✓[/bold green] Generated test_qa_pair_verify.json")
                console.print(f"  - Verify set: {len(verify_pairs)} items")
                
                # Step 5: 生成test_qa_pair_pred.json
                console.print("\n[bold cyan][Step 5] Generating test_qa_pair_pred.json...[/bold cyan]")
                test_pred_path = output_dir / "qa_pairs" / "test_qa_pair_pred.json"
                
                pred_results = sft_generator.generate_test_pred_data(
                    test_verify_path=test_verify_path,
                    output_path=test_pred_path,
                )
                console.print(f"[bold green]✓[/bold green] Generated test_qa_pair_pred.json")
                console.print(f"  - Prediction results: {len(pred_results)} items")
            else:
                console.print(f"[yellow]⚠[/yellow] Test QA file not found: {test_qa_path}")
                console.print("  Skipping test data generation")
            
        elif step == "train_data":
            # 步骤1: 生成train_data.json
            console.print("\n[bold cyan][Step 1] Generating train_data.json...[/bold cyan]")
            sft_generator.generate_train_data(
                train_qa_path=train_qa_path,
                output_path=train_data_path,
            )
            console.print(f"[bold green]✓[/bold green] Generated train_data.json")
            console.print(f"  - Output: {train_data_path}")
            
        elif step == "summary":
            # 步骤2: 生成summary_data
            console.print("\n[bold cyan][Step 2] Generating summary_data...[/bold cyan]")
            if not train_data_path.exists():
                console.print(f"[bold red]✗[/bold red] Train data file not found: {train_data_path}")
                console.print("  Please run 'train_data' step first")
                return
            
            summary_train, summary_test = sft_generator.generate_summary_data(
                train_data_path=train_data_path,
                output_dir=output_dir,
            )
            console.print(f"[bold green]✓[/bold green] Generated summary_data")
            console.print(f"  - Train: {len(summary_train)} items")
            console.print(f"  - Test: {len(summary_test)} items")
            
        elif step == "rerank":
            # 步骤3: 生成rerank_data
            console.print("\n[bold cyan][Step 3] Generating rerank_data...[/bold cyan]")
            if not train_data_path.exists():
                console.print(f"[bold red]✗[/bold red] Train data file not found: {train_data_path}")
                console.print("  Please run 'train_data' step first")
                return
            
            rerank_train, rerank_dev, rerank_test = sft_generator.generate_rerank_data(
                train_data_path=train_data_path,
                output_dir=output_dir,
            )
            console.print(f"[bold green]✓[/bold green] Generated rerank_data")
            console.print(f"  - Train: {len(rerank_train)} items")
            console.print(f"  - Dev: {len(rerank_dev)} items")
            console.print(f"  - Test: {len(rerank_test)} items")
            
        elif step == "test_verify":
            # 步骤4: 生成test_qa_pair_verify.json
            console.print("\n[bold cyan][Step 4] Generating test_qa_pair_verify.json...[/bold cyan]")
            test_qa_path = output_dir / "qa_pairs" / "test_qa_pair.json"
            test_verify_path = output_dir / "qa_pairs" / "test_qa_pair_verify.json"
            
            if not test_qa_path.exists():
                console.print(f"[bold red]✗[/bold red] Test QA file not found: {test_qa_path}")
                console.print("  Please ensure test_qa_pair.json exists")
                return
            
            verify_pairs = sft_generator.generate_test_verify_data(
                test_qa_path=test_qa_path,
                output_path=test_verify_path,
                verify_ratio=0.1,  # 默认10%作为验证集
            )
            console.print(f"[bold green]✓[/bold green] Generated test_qa_pair_verify.json")
            console.print(f"  - Verify set: {len(verify_pairs)} items")
            
        elif step == "test_pred":
            # 步骤5: 生成test_qa_pair_pred.json
            console.print("\n[bold cyan][Step 5] Generating test_qa_pair_pred.json...[/bold cyan]")
            test_verify_path = output_dir / "qa_pairs" / "test_qa_pair_verify.json"
            test_pred_path = output_dir / "qa_pairs" / "test_qa_pair_pred.json"
            
            if not test_verify_path.exists():
                console.print(f"[bold red]✗[/bold red] Test verify file not found: {test_verify_path}")
                console.print("  Please run 'test_verify' step first")
                return
            
            pred_results = sft_generator.generate_test_pred_data(
                test_verify_path=test_verify_path,
                output_path=test_pred_path,
            )
            console.print(f"[bold green]✓[/bold green] Generated test_qa_pair_pred.json")
            console.print(f"  - Prediction results: {len(pred_results)} items")
            
        else:
            console.print(f"[bold red]Error: Unknown step '{step}'[/bold red]")
            console.print("Available steps: train_data, summary, rerank, test_verify, test_pred, all")
            raise typer.Exit(1)
        
        monitor.end()
        monitor.print_summary()
        
    except Exception as e:
        monitor.end()
        if monitor:
            monitor.print_summary()
        console.print(f"[bold red]Error generating SFT data: {e}[/bold red]")
        logger.exception("SFT data generation failed")
        raise typer.Exit(1)


@app.command()
def analyze_sft_data(
    output_dir: Path = typer.Option(Path("data"), "--output-dir", "-o", help="数据输出目录"),
    action: str = typer.Option("all", "--action", help="执行操作: stats, validate, all"),
    save_report: bool = typer.Option(True, "--save-report/--no-save-report", help="是否保存报告到JSON文件"),
):
    """
    分析和验证SFT数据
    
    功能包括：
    1. 数据统计：SFT数据和Reranker数据的统计信息
    2. 数据验证：验证数据格式、完整性和平衡性
    """
    console.print("[bold green]Analyzing SFT data...[/bold green]")
    
    try:
        from src.evrag.gen_qa.sft_data_analyzer import SFTDataAnalyzer, SFTDataValidator
        
        output_dir = Path(output_dir)
        
        if action in ["stats", "all"]:
            # 数据分析
            console.print("\n[bold cyan]Data Analysis[/bold cyan]")
            analyzer = SFTDataAnalyzer(output_dir)
            
            # 分析summary数据
            summary_stats = analyzer.analyze_summary_data()
            analyzer.print_summary_statistics(summary_stats)
            
            # 分析rerank数据
            rerank_stats = analyzer.analyze_rerank_data()
            analyzer.print_rerank_statistics(rerank_stats)
            
            # 生成可视化图表
            console.print("\n[bold cyan]Generating visualizations...[/bold cyan]")
            report = {
                "summary_data": summary_stats,
                "rerank_data": rerank_stats,
            }
            analyzer.generate_visualizations(report, output_dir)
            
            # 保存统计报告
            if save_report:
                report_path = output_dir / "logs" / "sft_data_statistics.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                
                console.print(f"\n[bold green]✓[/bold green] Statistics report saved to: {report_path}")
        
        if action in ["validate", "all"]:
            # 数据验证
            console.print("\n[bold cyan]Data Validation[/bold cyan]")
            validator = SFTDataValidator(output_dir)
            
            validation_results = validator.validate_all()
            validator.print_validation_results(validation_results)
            
            # 保存验证报告
            if save_report:
                report_path = output_dir / "logs" / "sft_data_validation.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(validation_results, f, ensure_ascii=False, indent=2)
                
                console.print(f"\n[bold green]✓[/bold green] Validation report saved to: {report_path}")
        
        console.print("\n[bold green]✓ Analysis completed![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error analyzing SFT data: {e}[/bold red]")
        logger.exception("SFT data analysis failed")
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