"""
PDF解析器模块

从PDF文件中提取文本和图片，生成Document对象。
"""

import hashlib
from pathlib import Path
from typing import List, Optional
import pymupdf as fitz  # PyMuPDF
from tqdm import tqdm
from langchain_core.documents import Document

from ..config import get_settings
from .image_handler import ImageHandler


class PDFParser:
    """
    PDF解析器

    从PDF文件中提取文本和图片，生成Document对象列表。
    """

    def __init__(
        self,
        pdf_path: Optional[Path] = None,
        min_filter_pages: int = 0,
        max_filter_pages: Optional[int] = None,
        page_clip: int = 50,
        image_handler: Optional[ImageHandler] = None,
    ) -> None:
        """
        初始化PDF解析器

        Args:
            pdf_path: PDF文件路径，如果为None则从配置中读取
            min_filter_pages: 最小页码（从0开始），小于此页码的页面将被跳过
            max_filter_pages: 最大页码（从0开始），大于此页码的页面将被跳过，None表示不限制
            page_clip: 页面底部裁剪像素数（用于去除页眉页脚）
            image_handler: 图片处理器，如果为None则创建默认的ImageHandler
        """
        settings = get_settings()

        self.pdf_path = pdf_path or settings.pdf_path
        if self.pdf_path is None:
            raise ValueError(
                "PDF path not specified. Please set pdf_path in config or pass it as argument."
            )

        self.pdf_path = Path(self.pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        self.min_filter_pages = min_filter_pages
        self.max_filter_pages = max_filter_pages
        self.page_clip = page_clip
        self.image_handler = image_handler or ImageHandler()

    def load_pdf(
        self,
        show_progress: bool = True,
    ) -> List[Document]:
        """
        从PDF文件中加载文档

        Args:
            show_progress: 是否显示进度条

        Returns:
            Document对象列表，每个Document代表一页PDF
        """
        try:
            pdf = fitz.open(str(self.pdf_path))
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF file {self.pdf_path}: {e}") from e

        raw_docs = []
        total_pages = len(pdf)

        # 确定要处理的页面范围
        start_page = max(0, self.min_filter_pages)
        end_page = (
            min(total_pages, self.max_filter_pages + 1)
            if self.max_filter_pages is not None
            else total_pages
        )

        page_range = range(start_page, end_page)
        if show_progress:
            page_range = tqdm(page_range, desc="Loading PDF pages")

        try:
            for page_num in page_range:
                try:
                    page = pdf.load_page(page_num)

                    # 裁剪页面（去除底部页眉页脚）
                    crop = fitz.Rect(
                        0, 0, page.rect.width, page.rect.height - self.page_clip
                    )
                    text = page.get_text(clip=crop)

                    # 提取图片
                    images = page.get_images(full=True)
                    images_info = []

                    for img_index, img in enumerate(images):
                        image_info = self.image_handler.handle_image(
                            img, img_index, page
                        )
                        if image_info:
                            images_info.append(image_info)

                    # 如果页面有文本，创建Document
                    if text.strip():
                        unique_id = hashlib.md5(text.encode("utf-8")).hexdigest()
                        metadata = {
                            "unique_id": unique_id,
                            "source": str(self.pdf_path),
                            "page": page_num + 1,
                            "images_info": images_info,
                        }

                        raw_docs.append(Document(page_content=text, metadata=metadata))
                except Exception as e:
                    # 如果某页处理失败，记录错误但继续处理其他页
                    print(f"Warning: Failed to process page {page_num + 1}: {e}")
                    continue
        finally:
            pdf.close()

        return raw_docs

    def parse(
        self,
        show_progress: bool = True,
    ) -> List[Document]:
        """
        解析PDF文件（load_pdf的别名，保持API一致性）

        Args:
            show_progress: 是否显示进度条

        Returns:
            Document对象列表
        """
        return self.load_pdf(show_progress=show_progress)
