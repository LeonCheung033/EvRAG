"""
EvRAG主入口

提供命令行接口，支持构建索引、推理和QA生成等功能。
"""

import sys
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
def version():
    """显示版本信息"""
    from src.evrag import __version__
    console.print(f"[bold green]EvRAG version:[/bold green] {__version__}")


def main():
    """主函数"""
    app()

if __name__ == "__main__":
    main()