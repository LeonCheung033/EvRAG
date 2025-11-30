#!/usr/bin/env python3
"""
计算PDF文件的token数量

用于评估PDF文件是否可以直接作为上下文输入到LLM模型中。
"""

import sys
from pathlib import Path
import tiktoken
import pymupdf as fitz
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# 常见模型的上下文窗口大小（token数）
MODEL_CONTEXT_WINDOWS = {
    "GPT-4o": 128000,
    "GPT-4 Turbo": 128000,
    "GPT-4": 8192,
    "GPT-3.5 Turbo": 16385,
    "Claude 3.5 Sonnet": 200000,
    "Claude 3 Opus": 200000,
    "Claude 3 Sonnet": 200000,
    "Claude 3 Haiku": 200000,
    "Gemini 1.5 Pro": 2097152,  # 2M tokens
    "Gemini 1.5 Flash": 1048576,  # 1M tokens
    "Qwen2.5-72B": 32768,
    "Qwen2.5-32B": 32768,
    "Qwen2.5-7B": 32768,
    "Llama 3.1 405B": 131072,
    "Llama 3.1 70B": 131072,
    "Llama 3.1 8B": 131072,
}

# 使用cl100k_base编码（OpenAI GPT-4等模型使用的编码）
encoding = tiktoken.get_encoding("cl100k_base")


def extract_pdf_text(pdf_path: Path, include_all_pages: bool = True) -> str:
    """
    从PDF文件中提取所有文本
    
    Args:
        pdf_path: PDF文件路径
        include_all_pages: 是否包含所有页面（True表示不跳过任何页面）
    
    Returns:
        提取的文本内容
    """
    try:
        pdf = fitz.open(str(pdf_path))
    except Exception as e:
        raise RuntimeError(f"无法打开PDF文件 {pdf_path}: {e}") from e
    
    all_text = []
    total_pages = len(pdf)
    
    console.print(f"[cyan]正在提取PDF文本...[/cyan]")
    console.print(f"[dim]总页数: {total_pages}[/dim]")
    
    try:
        for page_num in range(total_pages):
            try:
                page = pdf.load_page(page_num)
                # 提取页面文本（不裁剪，包含所有内容）
                text = page.get_text()
                if text.strip():
                    all_text.append(text)
            except Exception as e:
                console.print(f"[yellow]警告: 处理第 {page_num + 1} 页时出错: {e}[/yellow]")
                continue
    finally:
        pdf.close()
    
    # 合并所有文本，页面之间用双换行分隔
    full_text = "\n\n".join(all_text)
    
    console.print(f"[green]✓[/green] 成功提取 {len(all_text)} 页文本")
    
    return full_text


def count_tokens(text: str) -> int:
    """
    计算文本的token数量
    
    Args:
        text: 文本内容
    
    Returns:
        token数量
    """
    return len(encoding.encode(text))


def format_number(num: int) -> str:
    """格式化数字，添加千位分隔符"""
    return f"{num:,}"


def calculate_pdf_tokens(pdf_path: Path) -> dict:
    """
    计算PDF文件的token统计信息
    
    Args:
        pdf_path: PDF文件路径
    
    Returns:
        包含统计信息的字典
    """
    # 提取PDF文本
    text = extract_pdf_text(pdf_path, include_all_pages=True)
    
    # 计算统计信息
    char_count = len(text)
    word_count = len(text.split())
    token_count = count_tokens(text)
    line_count = text.count('\n') + 1
    page_count = text.count('\n\n') + 1  # 粗略估算页面数
    
    # 计算平均每页的统计
    avg_chars_per_page = char_count / page_count if page_count > 0 else 0
    avg_tokens_per_page = token_count / page_count if page_count > 0 else 0
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "token_count": token_count,
        "line_count": line_count,
        "page_count": page_count,
        "avg_chars_per_page": avg_chars_per_page,
        "avg_tokens_per_page": avg_tokens_per_page,
    }


def display_results(stats: dict, pdf_path: Path):
    """
    显示统计结果
    
    Args:
        stats: 统计信息字典
        pdf_path: PDF文件路径
    """
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]PDF Token统计报告[/bold cyan]")
    console.print("=" * 80)
    
    # 基本信息
    info_table = Table(title="文件基本信息", show_header=True, header_style="bold magenta")
    info_table.add_column("项目", style="cyan", width=30)
    info_table.add_column("值", style="green")
    
    info_table.add_row("文件路径", str(pdf_path))
    info_table.add_row("文件大小", f"{pdf_path.stat().st_size / 1024 / 1024:.2f} MB")
    info_table.add_row("页数（估算）", format_number(int(stats["page_count"])))
    
    console.print(info_table)
    
    # 文本统计
    text_table = Table(title="文本统计", show_header=True, header_style="bold blue")
    text_table.add_column("指标", style="cyan", width=30)
    text_table.add_column("数量", style="green", justify="right")
    
    text_table.add_row("总字符数", format_number(stats["char_count"]))
    text_table.add_row("总词数", format_number(stats["word_count"]))
    text_table.add_row("总Token数", format_number(stats["token_count"]))
    text_table.add_row("总行数", format_number(stats["line_count"]))
    text_table.add_row("平均每页字符数", f"{stats['avg_chars_per_page']:.0f}")
    text_table.add_row("平均每页Token数", f"{stats['avg_tokens_per_page']:.0f}")
    
    console.print("\n")
    console.print(text_table)
    
    # 模型上下文窗口对比
    model_table = Table(
        title="模型上下文窗口对比", 
        show_header=True, 
        header_style="bold yellow"
    )
    model_table.add_column("模型", style="cyan", width=25)
    model_table.add_column("上下文窗口", style="white", justify="right")
    model_table.add_column("是否适合", style="green", justify="center")
    model_table.add_column("使用率", style="yellow", justify="right")
    
    token_count = stats["token_count"]
    
    for model_name, context_window in sorted(
        MODEL_CONTEXT_WINDOWS.items(), 
        key=lambda x: x[1], 
        reverse=True
    ):
        fits = token_count <= context_window
        usage_percent = (token_count / context_window * 100) if context_window > 0 else 0
        
        status = "✓ 适合" if fits else "✗ 超出"
        status_style = "green" if fits else "red"
        
        model_table.add_row(
            model_name,
            format_number(context_window),
            f"[{status_style}]{status}[/{status_style}]",
            f"{usage_percent:.2f}%"
        )
    
    console.print("\n")
    console.print(model_table)
    
    # 总结和建议
    console.print("\n")
    
    # 找到最适合的模型（上下文窗口刚好大于token数的）
    suitable_models = [
        (name, window) 
        for name, window in MODEL_CONTEXT_WINDOWS.items() 
        if window >= token_count
    ]
    
    if suitable_models:
        # 选择最小的适合的模型
        best_model = min(suitable_models, key=lambda x: x[1])
        best_model_name, best_model_window = best_model
        usage = (token_count / best_model_window * 100) if best_model_window > 0 else 0
        
        summary = f"""
[bold green]✓ 该PDF可以直接作为上下文输入[/bold green]

[bold]推荐模型:[/bold] {best_model_name}
[bold]上下文窗口:[/bold] {format_number(best_model_window)} tokens
[bold]使用率:[/bold] {usage:.2f}%
[bold]剩余空间:[/bold] {format_number(best_model_window - token_count)} tokens

[bold]其他适合的模型:[/bold]
"""
        for name, window in sorted(suitable_models, key=lambda x: x[1])[:5]:
            usage_pct = (token_count / window * 100) if window > 0 else 0
            summary += f"  • {name}: {format_number(window)} tokens (使用率: {usage_pct:.2f}%)\n"
    else:
        summary = f"""
[bold red]✗ 该PDF无法直接作为上下文输入[/bold red]

[bold]当前Token数:[/bold] {format_number(token_count)}
[bold]最大上下文窗口:[/bold] {format_number(max(MODEL_CONTEXT_WINDOWS.values()))} tokens (Gemini 1.5 Pro)

[bold]建议:[/bold]
  • 考虑使用RAG（检索增强生成）方法
  • 将PDF切分为多个较小的chunk
  • 使用文档摘要或关键信息提取
  • 等待支持更大上下文窗口的模型
"""
    
    console.print(Panel(summary.strip(), title="总结和建议", border_style="cyan"))


def main():
    """主函数"""
    # PDF文件路径
    pdf_path = Path("/remote-home/share/liangZhang/EvRAG/data/Tesla_Manual.pdf")
    
    if not pdf_path.exists():
        console.print(f"[bold red]错误: PDF文件不存在: {pdf_path}[/bold red]")
        sys.exit(1)
    
    try:
        # 计算token统计
        stats = calculate_pdf_tokens(pdf_path)
        
        # 显示结果
        display_results(stats, pdf_path)
        
    except Exception as e:
        console.print(f"[bold red]错误: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()


