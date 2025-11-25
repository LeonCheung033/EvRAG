"""
数据分析工具

用于分析文档处理各阶段的数据质量和效果。
"""

import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
import tiktoken

from langchain_core.documents import Document
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# 用于计算token数量
encoding = tiktoken.get_encoding("cl100k_base")


def calculate_text_stats(text: str) -> Dict[str, Any]:
    """计算文本统计信息"""
    char_count = len(text)
    word_count = len(text.split())
    token_count = len(encoding.encode(text))
    line_count = text.count('\n') + 1
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "token_count": token_count,
        "line_count": line_count,
    }


def analyze_documents(
    docs: List[Document],
    stage_name: str,
    show_samples: bool = True,
    num_samples: int = 3,
) -> Dict[str, Any]:
    """
    分析文档列表
    
    Args:
        docs: 文档列表
        stage_name: 阶段名称（用于显示）
        show_samples: 是否显示示例
        num_samples: 显示的示例数量
    
    Returns:
        分析结果字典
    """
    if not docs:
        return {"error": "文档列表为空"}
    
    # 统计信息
    total_docs = len(docs)
    char_counts = []
    word_counts = []
    token_counts = []
    page_numbers = []
    has_parent_id = 0
    has_images = 0
    
    for doc in docs:
        stats = calculate_text_stats(doc.page_content)
        char_counts.append(stats["char_count"])
        word_counts.append(stats["word_count"])
        token_counts.append(stats["token_count"])
        
        metadata = doc.metadata
        if "page" in metadata:
            page_numbers.append(metadata["page"])
        if "parent_id" in metadata:
            has_parent_id += 1
        if "images_info" in metadata and metadata["images_info"]:
            has_images += 1
    
    # 计算统计值
    avg_char = sum(char_counts) / len(char_counts) if char_counts else 0
    avg_word = sum(word_counts) / len(word_counts) if word_counts else 0
    avg_token = sum(token_counts) / len(token_counts) if token_counts else 0
    
    min_char = min(char_counts) if char_counts else 0
    max_char = max(char_counts) if char_counts else 0
    min_token = min(token_counts) if token_counts else 0
    max_token = max(token_counts) if token_counts else 0
    
    # 长度分布
    char_distribution = {
        "0-100": sum(1 for c in char_counts if 0 <= c < 100),
        "100-256": sum(1 for c in char_counts if 100 <= c < 256),
        "256-512": sum(1 for c in char_counts if 256 <= c < 512),
        "512-1024": sum(1 for c in char_counts if 512 <= c < 1024),
        "1024+": sum(1 for c in char_counts if c >= 1024),
    }
    
    token_distribution = {
        "0-50": sum(1 for t in token_counts if 0 <= t < 50),
        "50-100": sum(1 for t in token_counts if 50 <= t < 100),
        "100-200": sum(1 for t in token_counts if 100 <= t < 200),
        "200-500": sum(1 for t in token_counts if 200 <= t < 500),
        "500+": sum(1 for t in token_counts if t >= 500),
    }
    
    result = {
        "stage_name": stage_name,
        "total_docs": total_docs,
        "avg_char_count": round(avg_char, 2),
        "avg_word_count": round(avg_word, 2),
        "avg_token_count": round(avg_token, 2),
        "min_char_count": min_char,
        "max_char_count": max_char,
        "min_token_count": min_token,
        "max_token_count": max_token,
        "char_distribution": char_distribution,
        "token_distribution": token_distribution,
        "has_parent_id": has_parent_id,
        "has_images": has_images,
        "page_range": (min(page_numbers), max(page_numbers)) if page_numbers else None,
        "unique_pages": len(set(page_numbers)) if page_numbers else 0,
    }
    
    # 显示统计信息
    table = Table(title=f"{stage_name} 统计信息", show_header=True, header_style="bold magenta")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")
    
    table.add_row("文档总数", str(total_docs))
    table.add_row("平均字符数", f"{avg_char:.2f}")
    table.add_row("平均词数", f"{avg_word:.2f}")
    table.add_row("平均Token数", f"{avg_token:.2f}")
    table.add_row("字符数范围", f"{min_char} - {max_char}")
    table.add_row("Token数范围", f"{min_token} - {max_token}")
    table.add_row("包含parent_id", f"{has_parent_id} ({has_parent_id/total_docs*100:.1f}%)" if total_docs > 0 else "0")
    table.add_row("包含图片", f"{has_images} ({has_images/total_docs*100:.1f}%)" if total_docs > 0 else "0")
    if page_numbers:
        table.add_row("页码范围", f"{min(page_numbers)} - {max(page_numbers)}")
        table.add_row("唯一页码数", str(len(set(page_numbers))))
    
    console.print(table)
    
    # 显示长度分布
    dist_table = Table(title=f"{stage_name} 长度分布", show_header=True, header_style="bold blue")
    dist_table.add_column("长度范围（字符）", style="cyan")
    dist_table.add_column("文档数量", style="green")
    dist_table.add_column("百分比", style="yellow")
    
    for range_name, count in char_distribution.items():
        percentage = (count / total_docs * 100) if total_docs > 0 else 0
        dist_table.add_row(range_name, str(count), f"{percentage:.1f}%")
    
    console.print(dist_table)
    
    # 显示示例
    if show_samples:
        console.print(f"\n[bold cyan]{stage_name} 示例文档：[/bold cyan]")
        for i, doc in enumerate(docs[:num_samples], 1):
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            metadata_str = json.dumps(doc.metadata, ensure_ascii=False, indent=2)
            
            panel_content = f"[bold]内容预览：[/bold]\n{content_preview}\n\n[bold]元数据：[/bold]\n{metadata_str}"
            console.print(Panel(panel_content, title=f"示例 {i}", border_style="green"))
    
    return result


def check_duplicates(docs: List[Document]) -> Dict[str, Any]:
    """检查重复文档"""
    unique_ids = [doc.metadata.get("unique_id") for doc in docs if doc.metadata.get("unique_id")]
    id_counts = Counter(unique_ids)
    duplicates = {id: count for id, count in id_counts.items() if count > 1}
    
    # 检查内容重复
    content_hashes = {}
    content_duplicates = []
    for doc in docs:
        content_hash = hash(doc.page_content)
        if content_hash in content_hashes:
            content_duplicates.append({
                "content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                "unique_id": doc.metadata.get("unique_id"),
            })
        else:
            content_hashes[content_hash] = doc
    
    return {
        "total_unique_ids": len(set(unique_ids)),
        "duplicate_ids": len(duplicates),
        "duplicate_id_details": dict(list(duplicates.items())[:10]),  # 只显示前10个
        "content_duplicates": len(content_duplicates),
        "content_duplicate_samples": content_duplicates[:5],  # 只显示前5个
    }


def analyze_parent_child_relationship(split_docs: List[Document]) -> Dict[str, Any]:
    """分析父子文档关系"""
    parent_docs = []
    child_docs = []
    
    for doc in split_docs:
        unique_id = doc.metadata.get("unique_id")
        parent_id = doc.metadata.get("parent_id")
        
        if parent_id:
            # 如果unique_id等于parent_id，说明是父文档
            if unique_id == parent_id:
                parent_docs.append(doc)
            # 如果unique_id不等于parent_id（且存在），说明是子文档
            elif unique_id and unique_id != parent_id:
                child_docs.append(doc)
            else:
                # 有parent_id但没有unique_id，可能是数据问题，暂时归类为父文档
                parent_docs.append(doc)
        else:
            # 没有parent_id，可能是独立文档或父文档（旧数据格式）
            parent_docs.append(doc)
    
    # 统计父子关系
    parent_ids = {doc.metadata.get("parent_id") for doc in child_docs if doc.metadata.get("parent_id")}
    children_per_parent = Counter([doc.metadata.get("parent_id") for doc in child_docs])
    
    return {
        "total_parent_docs": len(parent_docs),
        "total_child_docs": len(child_docs),
        "unique_parent_ids": len(parent_ids),
        "avg_children_per_parent": sum(children_per_parent.values()) / len(children_per_parent) if children_per_parent else 0,
        "max_children_per_parent": max(children_per_parent.values()) if children_per_parent else 0,
        "min_children_per_parent": min(children_per_parent.values()) if children_per_parent else 0,
    }


def compare_stages(
    raw_docs: List[Document],
    clean_docs: List[Document],
    split_docs: List[Document],
) -> Dict[str, Any]:
    """对比三个阶段的数据"""
    
    console.print("\n[bold yellow]=" * 60)
    console.print("[bold yellow]文档处理阶段对比分析[/bold yellow]")
    console.print("[bold yellow]=" * 60)
    
    # 分析各阶段
    raw_stats = analyze_documents(raw_docs, "原始文档 (Raw)", show_samples=True, num_samples=2)
    clean_stats = analyze_documents(clean_docs, "清洗后文档 (Clean)", show_samples=True, num_samples=2)
    split_stats = analyze_documents(split_docs, "切分后文档 (Split)", show_samples=True, num_samples=3)
    
    # 检查重复
    console.print("\n[bold cyan]重复检查：[/bold cyan]")
    raw_duplicates = check_duplicates(raw_docs)
    clean_duplicates = check_duplicates(clean_docs)
    split_duplicates = check_duplicates(split_docs)
    
    console.print(f"原始文档 - 唯一ID数: {raw_duplicates['total_unique_ids']}, 重复ID: {raw_duplicates['duplicate_ids']}")
    console.print(f"清洗后文档 - 唯一ID数: {clean_duplicates['total_unique_ids']}, 重复ID: {clean_duplicates['duplicate_ids']}")
    console.print(f"切分后文档 - 唯一ID数: {split_duplicates['total_unique_ids']}, 重复ID: {split_duplicates['duplicate_ids']}")
    
    if split_duplicates['duplicate_ids'] > 0:
        console.print(f"[yellow]警告: 发现 {split_duplicates['duplicate_ids']} 个重复的unique_id[/yellow]")
        if split_duplicates['duplicate_id_details']:
            console.print("重复ID示例:", split_duplicates['duplicate_id_details'])
    
    # 分析父子关系
    console.print("\n[bold cyan]父子文档关系分析：[/bold cyan]")
    relationship = analyze_parent_child_relationship(split_docs)
    
    rel_table = Table(title="父子文档关系", show_header=True, header_style="bold magenta")
    rel_table.add_column("指标", style="cyan")
    rel_table.add_column("值", style="green")
    
    rel_table.add_row("父文档数量", str(relationship["total_parent_docs"]))
    rel_table.add_row("子文档数量", str(relationship["total_child_docs"]))
    rel_table.add_row("唯一父文档ID数", str(relationship["unique_parent_ids"]))
    rel_table.add_row("平均每个父文档的子文档数", f"{relationship['avg_children_per_parent']:.2f}")
    rel_table.add_row("最大子文档数", str(relationship["max_children_per_parent"]))
    rel_table.add_row("最小子文档数", str(relationship["min_children_per_parent"]))
    
    console.print(rel_table)
    
    # 对比统计
    console.print("\n[bold cyan]阶段对比：[/bold cyan]")
    compare_table = Table(title="三个阶段对比", show_header=True, header_style="bold blue")
    compare_table.add_column("指标", style="cyan")
    compare_table.add_column("原始文档", style="white")
    compare_table.add_column("清洗后", style="yellow")
    compare_table.add_column("切分后", style="green")
    
    compare_table.add_row("文档总数", str(raw_stats["total_docs"]), str(clean_stats["total_docs"]), str(split_stats["total_docs"]))
    compare_table.add_row("平均字符数", f"{raw_stats['avg_char_count']:.0f}", f"{clean_stats['avg_char_count']:.0f}", f"{split_stats['avg_char_count']:.0f}")
    compare_table.add_row("平均Token数", f"{raw_stats['avg_token_count']:.0f}", f"{clean_stats['avg_token_count']:.0f}", f"{split_stats['avg_token_count']:.0f}")
    compare_table.add_row("最大字符数", str(raw_stats["max_char_count"]), str(clean_stats["max_char_count"]), str(split_stats["max_char_count"]))
    compare_table.add_row("最小字符数", str(raw_stats["min_char_count"]), str(clean_stats["min_char_count"]), str(split_stats["min_char_count"]))
    
    console.print(compare_table)
    
    # 计算切分效果
    expansion_ratio = split_stats["total_docs"] / clean_stats["total_docs"] if clean_stats["total_docs"] > 0 else 0
    avg_size_reduction = (1 - split_stats["avg_char_count"] / clean_stats["avg_char_count"]) * 100 if clean_stats["avg_char_count"] > 0 else 0
    
    console.print(f"\n[bold green]切分效果：[/bold green]")
    console.print(f"  - 文档扩展倍数: {expansion_ratio:.2f}x (从 {clean_stats['total_docs']} 个文档扩展到 {split_stats['total_docs']} 个)")
    console.print(f"  - 平均文档大小减少: {avg_size_reduction:.1f}%")
    
    return {
        "raw_stats": raw_stats,
        "clean_stats": clean_stats,
        "split_stats": split_stats,
        "duplicates": {
            "raw": raw_duplicates,
            "clean": clean_duplicates,
            "split": split_duplicates,
        },
        "relationship": relationship,
        "expansion_ratio": expansion_ratio,
        "avg_size_reduction": avg_size_reduction,
    }


def analyze_data_quality(
    raw_docs_path: Path,
    clean_docs_path: Path,
    split_docs_path: Path,
    output_report: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    分析数据质量
    
    Args:
        raw_docs_path: 原始文档路径
        clean_docs_path: 清洗后文档路径
        split_docs_path: 切分后文档路径
        output_report: 输出报告路径（可选）
    
    Returns:
        分析结果字典
    """
    console.print("[bold green]加载文档数据...[/bold green]")
    
    # 加载文档
    with open(raw_docs_path, 'rb') as f:
        raw_docs = pickle.load(f)
    console.print(f"[green]✓[/green] 加载原始文档: {len(raw_docs)} 个")
    
    with open(clean_docs_path, 'rb') as f:
        clean_docs = pickle.load(f)
    console.print(f"[green]✓[/green] 加载清洗后文档: {len(clean_docs)} 个")
    
    with open(split_docs_path, 'rb') as f:
        split_docs = pickle.load(f)
    console.print(f"[green]✓[/green] 加载切分后文档: {len(split_docs)} 个")
    
    # 对比分析
    result = compare_stages(raw_docs, clean_docs, split_docs)
    
    # 保存报告
    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        console.print(f"\n[bold green]✓[/bold green] 分析报告已保存到: {output_report}")
    
    return result


def show_processing_examples(
    raw_docs: List[Document],
    clean_docs: List[Document],
    split_docs: List[Document],
    num_examples: int = 3,
) -> None:
    """
    展示文档处理的完整流程示例
    
    显示从原始文档到切分后文档的完整变化过程。
    
    Args:
        raw_docs: 原始文档列表
        clean_docs: 清洗后文档列表
        split_docs: 切分后文档列表
        num_examples: 要展示的示例数量
    """
    console.print("\n[bold yellow]=" * 80)
    console.print("[bold yellow]文档处理流程示例[/bold yellow]")
    console.print("[bold yellow]=" * 80)
    
    # 确保三个列表长度一致
    min_len = min(len(raw_docs), len(clean_docs))
    if min_len == 0:
        console.print("[bold red]错误: 没有可用的文档数据[/bold red]")
        return
    
    # 选择几个有代表性的文档（选择不同长度的文档）
    indices = []
    if min_len >= num_examples:
        # 选择开头、中间、结尾的文档
        step = max(1, min_len // num_examples)
        indices = [0, min_len // 2, min_len - 1][:num_examples]
    else:
        indices = list(range(min_len))
    
    for idx, example_idx in enumerate(indices, 1):
        console.print(f"\n[bold cyan]{'=' * 80}[/bold cyan]")
        console.print(f"[bold cyan]示例 {idx}/{len(indices)}: 文档 #{example_idx + 1}[/bold cyan]")
        console.print(f"[bold cyan]{'=' * 80}[/bold cyan]")
        
        raw_doc = raw_docs[example_idx]
        clean_doc = clean_docs[example_idx] if example_idx < len(clean_docs) else None
        
        # 显示原始文档
        console.print("\n[bold green]1. 原始文档 (Raw)[/bold green]")
        raw_stats = calculate_text_stats(raw_doc.page_content)
        console.print(f"   - 来源: {raw_doc.metadata.get('source', 'N/A')}")
        console.print(f"   - 页码: {raw_doc.metadata.get('page', 'N/A')}")
        console.print(f"   - 字符数: {raw_stats['char_count']}")
        console.print(f"   - Token数: {raw_stats['token_count']}")
        console.print(f"   - 行数: {raw_stats['line_count']}")
        console.print(f"   - unique_id: {raw_doc.metadata.get('unique_id', 'N/A')[:16]}...")
        if raw_doc.metadata.get('images_info'):
            console.print(f"   - 图片数: {len(raw_doc.metadata.get('images_info', []))}")
        
        # 显示内容预览（前300字符）
        content_preview = raw_doc.page_content[:300]
        if len(raw_doc.page_content) > 300:
            content_preview += "..."
        console.print(f"\n   [dim]内容预览:[/dim]")
        console.print(Panel(content_preview, border_style="green", title="原始文档内容"))
        
        # 显示清洗后文档
        if clean_doc:
            console.print("\n[bold yellow]2. 清洗后文档 (Clean)[/bold yellow]")
            clean_stats = calculate_text_stats(clean_doc.page_content)
            console.print(f"   - 字符数: {clean_stats['char_count']} ({clean_stats['char_count'] - raw_stats['char_count']:+d})")
            console.print(f"   - Token数: {clean_stats['token_count']} ({clean_stats['token_count'] - raw_stats['token_count']:+d})")
            console.print(f"   - 行数: {clean_stats['line_count']} ({clean_stats['line_count'] - raw_stats['line_count']:+d})")
            console.print(f"   - unique_id: {clean_doc.metadata.get('unique_id', 'N/A')[:16]}...")
            
            # 计算变化
            char_reduction = ((raw_stats['char_count'] - clean_stats['char_count']) / raw_stats['char_count'] * 100) if raw_stats['char_count'] > 0 else 0
            console.print(f"   - 字符减少: {char_reduction:.1f}%")
            
            # 显示内容预览
            content_preview = clean_doc.page_content[:300]
            if len(clean_doc.page_content) > 300:
                content_preview += "..."
            console.print(f"\n   [dim]内容预览:[/dim]")
            console.print(Panel(content_preview, border_style="yellow", title="清洗后文档内容"))
            
            # 显示主要变化
            if raw_doc.page_content != clean_doc.page_content:
                console.print("\n   [dim]主要变化:[/dim]")
                # 简单的差异检测（显示前100个字符的差异）
                raw_start = raw_doc.page_content[:100]
                clean_start = clean_doc.page_content[:100]
                if raw_start != clean_start:
                    console.print(f"   - 开头部分已修改")
                # 检查是否有明显的格式变化
                if raw_doc.page_content.count('\n') != clean_doc.page_content.count('\n'):
                    console.print(f"   - 换行符数量变化: {raw_doc.page_content.count('\\n')} → {clean_doc.page_content.count('\\n')}")
        
        # 显示切分后文档（找到相关的子文档）
        console.print("\n[bold blue]3. 切分后文档 (Split)[/bold blue]")
        raw_unique_id = raw_doc.metadata.get('unique_id')
        
        # 查找相关的切分文档（通过parent_id或unique_id匹配）
        related_split_docs = []
        for split_doc in split_docs:
            split_parent_id = split_doc.metadata.get('parent_id')
            split_unique_id = split_doc.metadata.get('unique_id')
            # 如果split_doc的parent_id等于某个父文档的unique_id，或者split_doc本身就是父文档
            if split_parent_id and split_doc.metadata.get('source') == raw_doc.metadata.get('source'):
                # 检查是否来自同一页
                if split_doc.metadata.get('page') == raw_doc.metadata.get('page'):
                    related_split_docs.append(split_doc)
            elif split_unique_id == raw_unique_id:
                related_split_docs.append(split_doc)
        
        # 如果没有找到，尝试通过source和page匹配
        if not related_split_docs:
            for split_doc in split_docs:
                if (split_doc.metadata.get('source') == raw_doc.metadata.get('source') and
                    split_doc.metadata.get('page') == raw_doc.metadata.get('page')):
                    related_split_docs.append(split_doc)
                    if len(related_split_docs) >= 5:  # 最多显示5个子文档
                        break
        
        if related_split_docs:
            console.print(f"   - 切分后文档数: {len(related_split_docs)}")
            total_chars = sum(len(doc.page_content) for doc in related_split_docs)
            avg_chars = total_chars / len(related_split_docs) if related_split_docs else 0
            console.print(f"   - 总字符数: {total_chars} (平均: {avg_chars:.1f})")
            console.print(f"   - 扩展倍数: {len(related_split_docs):.1f}x")
            
            # 显示前3个子文档的预览
            for i, split_doc in enumerate(related_split_docs[:3], 1):
                split_stats = calculate_text_stats(split_doc.page_content)
                is_parent = split_doc.metadata.get('unique_id') == split_doc.metadata.get('parent_id')
                doc_type = "父文档" if is_parent else "子文档"
                
                console.print(f"\n   [bold]子文档 {i} ({doc_type}):[/bold]")
                console.print(f"   - 字符数: {split_stats['char_count']}")
                console.print(f"   - Token数: {split_stats['token_count']}")
                console.print(f"   - unique_id: {split_doc.metadata.get('unique_id', 'N/A')[:16]}...")
                if split_doc.metadata.get('parent_id'):
                    console.print(f"   - parent_id: {split_doc.metadata.get('parent_id', 'N/A')[:16]}...")
                
                content_preview = split_doc.page_content[:200]
                if len(split_doc.page_content) > 200:
                    content_preview += "..."
                console.print(f"   - 内容: {content_preview}")
            
            if len(related_split_docs) > 3:
                console.print(f"\n   ... 还有 {len(related_split_docs) - 3} 个子文档")
        else:
            console.print("   [yellow]未找到相关的切分文档[/yellow]")
        
        # 总结
        console.print("\n[bold magenta]处理总结:[/bold magenta]")
        if clean_doc:
            console.print(f"   - 清洗: 字符数减少 {char_reduction:.1f}%")
        if related_split_docs:
            console.print(f"   - 切分: 从1个文档切分为{len(related_split_docs)}个片段")
            if clean_doc:
                expansion = len(related_split_docs)
                console.print(f"   - 扩展倍数: {expansion:.1f}x")
    
    console.print(f"\n[bold yellow]{'=' * 80}[/bold yellow]")
    console.print("[bold green]示例展示完成[/bold green]")


def generate_processing_examples_report(
    raw_docs: List[Document],
    clean_docs: List[Document],
    split_docs: List[Document],
    num_examples: int = 5,
) -> str:
    """
    生成文档处理流程示例的文本报告
    
    返回一个格式化的文本字符串，可以保存到文件。
    
    Args:
        raw_docs: 原始文档列表
        clean_docs: 清洗后文档列表
        split_docs: 切分后文档列表
        num_examples: 要展示的示例数量
    
    Returns:
        格式化的文本报告字符串
    """
    lines = []
    lines.append("=" * 80)
    lines.append("文档处理流程示例报告")
    lines.append("=" * 80)
    lines.append("")
    
    # 确保三个列表长度一致
    min_len = min(len(raw_docs), len(clean_docs))
    if min_len == 0:
        lines.append("错误: 没有可用的文档数据")
        return "\n".join(lines)
    
    # 选择几个有代表性的文档（选择不同长度的文档）
    indices = []
    if min_len >= num_examples:
        # 选择开头、中间、结尾的文档
        step = max(1, min_len // num_examples)
        indices = [0, min_len // 4, min_len // 2, min_len * 3 // 4, min_len - 1][:num_examples]
    else:
        indices = list(range(min_len))
    
    for idx, example_idx in enumerate(indices, 1):
        lines.append("=" * 80)
        lines.append(f"示例 {idx}/{len(indices)}: 文档 #{example_idx + 1}")
        lines.append("=" * 80)
        lines.append("")
        
        raw_doc = raw_docs[example_idx]
        clean_doc = clean_docs[example_idx] if example_idx < len(clean_docs) else None
        
        # 显示原始文档
        lines.append("1. 原始文档 (Raw)")
        raw_stats = calculate_text_stats(raw_doc.page_content)
        lines.append(f"   - 来源: {raw_doc.metadata.get('source', 'N/A')}")
        lines.append(f"   - 页码: {raw_doc.metadata.get('page', 'N/A')}")
        lines.append(f"   - 字符数: {raw_stats['char_count']}")
        lines.append(f"   - Token数: {raw_stats['token_count']}")
        lines.append(f"   - 行数: {raw_stats['line_count']}")
        lines.append(f"   - unique_id: {raw_doc.metadata.get('unique_id', 'N/A')}")
        if raw_doc.metadata.get('images_info'):
            lines.append(f"   - 图片数: {len(raw_doc.metadata.get('images_info', []))}")
        lines.append("")
        lines.append("   内容:")
        lines.append("   " + "-" * 76)
        # 显示完整内容（或前500字符）
        content = raw_doc.page_content
        if len(content) > 1000:
            content = content[:1000] + "\n   ... (内容过长，已截断)"
        for line in content.split('\n'):
            lines.append(f"   {line}")
        lines.append("   " + "-" * 76)
        lines.append("")
        
        # 显示清洗后文档
        if clean_doc:
            lines.append("2. 清洗后文档 (Clean)")
            clean_stats = calculate_text_stats(clean_doc.page_content)
            char_diff = clean_stats['char_count'] - raw_stats['char_count']
            token_diff = clean_stats['token_count'] - raw_stats['token_count']
            lines.append(f"   - 字符数: {clean_stats['char_count']} ({char_diff:+d})")
            lines.append(f"   - Token数: {clean_stats['token_count']} ({token_diff:+d})")
            lines.append(f"   - 行数: {clean_stats['line_count']} ({clean_stats['line_count'] - raw_stats['line_count']:+d})")
            lines.append(f"   - unique_id: {clean_doc.metadata.get('unique_id', 'N/A')}")
            
            # 计算变化
            char_reduction = ((raw_stats['char_count'] - clean_stats['char_count']) / raw_stats['char_count'] * 100) if raw_stats['char_count'] > 0 else 0
            lines.append(f"   - 字符变化: {char_reduction:.1f}%")
            lines.append("")
            lines.append("   内容:")
            lines.append("   " + "-" * 76)
            content = clean_doc.page_content
            if len(content) > 1000:
                content = content[:1000] + "\n   ... (内容过长，已截断)"
            for line in content.split('\n'):
                lines.append(f"   {line}")
            lines.append("   " + "-" * 76)
            lines.append("")
            
            # 显示主要变化
            if raw_doc.page_content != clean_doc.page_content:
                lines.append("   主要变化:")
                raw_start = raw_doc.page_content[:100]
                clean_start = clean_doc.page_content[:100]
                if raw_start != clean_start:
                    lines.append("   - 开头部分已修改")
                if raw_doc.page_content.count('\n') != clean_doc.page_content.count('\n'):
                    lines.append(f"   - 换行符数量变化: {raw_doc.page_content.count(chr(10))} → {clean_doc.page_content.count(chr(10))}")
                lines.append("")
        
        # 显示切分后文档（找到相关的子文档）
        lines.append("3. 切分后文档 (Split)")
        raw_unique_id = raw_doc.metadata.get('unique_id')
        
        # 查找相关的切分文档
        related_split_docs = []
        for split_doc in split_docs:
            split_parent_id = split_doc.metadata.get('parent_id')
            split_unique_id = split_doc.metadata.get('unique_id')
            if split_parent_id and split_doc.metadata.get('source') == raw_doc.metadata.get('source'):
                if split_doc.metadata.get('page') == raw_doc.metadata.get('page'):
                    related_split_docs.append(split_doc)
            elif split_unique_id == raw_unique_id:
                related_split_docs.append(split_doc)
        
        # 如果没有找到，尝试通过source和page匹配
        if not related_split_docs:
            for split_doc in split_docs:
                if (split_doc.metadata.get('source') == raw_doc.metadata.get('source') and
                    split_doc.metadata.get('page') == raw_doc.metadata.get('page')):
                    related_split_docs.append(split_doc)
                    if len(related_split_docs) >= 10:  # 最多显示10个子文档
                        break
        
        if related_split_docs:
            lines.append(f"   - 切分后文档数: {len(related_split_docs)}")
            total_chars = sum(len(doc.page_content) for doc in related_split_docs)
            avg_chars = total_chars / len(related_split_docs) if related_split_docs else 0
            lines.append(f"   - 总字符数: {total_chars} (平均: {avg_chars:.1f})")
            lines.append(f"   - 扩展倍数: {len(related_split_docs):.1f}x")
            lines.append("")
            
            # 显示所有子文档的详细信息
            for i, split_doc in enumerate(related_split_docs, 1):
                split_stats = calculate_text_stats(split_doc.page_content)
                is_parent = split_doc.metadata.get('unique_id') == split_doc.metadata.get('parent_id')
                doc_type = "父文档" if is_parent else "子文档"
                
                lines.append(f"   子文档 {i} ({doc_type}):")
                lines.append(f"   - 字符数: {split_stats['char_count']}")
                lines.append(f"   - Token数: {split_stats['token_count']}")
                lines.append(f"   - unique_id: {split_doc.metadata.get('unique_id', 'N/A')}")
                if split_doc.metadata.get('parent_id'):
                    lines.append(f"   - parent_id: {split_doc.metadata.get('parent_id', 'N/A')}")
                lines.append("   - 内容:")
                lines.append("   " + "-" * 76)
                content = split_doc.page_content
                if len(content) > 500:
                    content = content[:500] + "\n   ... (内容过长，已截断)"
                for line in content.split('\n'):
                    lines.append(f"   {line}")
                lines.append("   " + "-" * 76)
                lines.append("")
            
            if len(related_split_docs) > 10:
                lines.append(f"   ... 还有 {len(related_split_docs) - 10} 个子文档未显示")
                lines.append("")
        else:
            lines.append("   未找到相关的切分文档")
            lines.append("")
        
        # 总结
        lines.append("处理总结:")
        if clean_doc:
            lines.append(f"   - 清洗: 字符数变化 {char_reduction:.1f}%")
        if related_split_docs:
            lines.append(f"   - 切分: 从1个文档切分为{len(related_split_docs)}个片段")
            if clean_doc:
                expansion = len(related_split_docs)
                lines.append(f"   - 扩展倍数: {expansion:.1f}x")
        lines.append("")
        lines.append("")
    
    lines.append("=" * 80)
    lines.append("报告生成完成")
    lines.append("=" * 80)
    
    return "\n".join(lines)

