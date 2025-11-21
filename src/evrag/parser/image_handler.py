"""
图片处理器模块

处理PDF中的图片，提取图片并关联相关文本块和标题。
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import pymupdf as fitz
from ..config import get_settings

# 标题判断配置
TITLE_PROPERTIES = {
    "min_size": 10,  # 字号 ≥10 才可能是标题
    "max_lines": 3,  # 最多3行
    "max_length": 30,  # 总字符数 ≤30
    "bold_weight": 0.7,  # （未使用？可忽略）
    "page_clip": 50,  # 页面底部保留50pt不搜索（防页脚干扰）
    "bottom_size": -200,  # 向上扩展200pt（负值表示向上）
}


class ImageHandler:
    """
    Image处理器类

    处理PDF中的图片，提取图片并关联相关文本块和标题。
    """

    def __init__(
        self,
        image_save_dir: Optional[Path] = None,
        title_properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the ImageHandler.

        Args:
            image_save_dir: The directory to save the images.
            title_properties: The properties to use for title detection.
        """
        self.image_save_dir = image_save_dir or get_settings().image_save_dir
        self.title_properties = title_properties or TITLE_PROPERTIES
        # 创建图片保存目录
        self.image_save_dir.mkdir(parents=True, exist_ok=True)

    def handle_image(
        self,
        img: Tuple[int, int, int, int, int, str, int, str],
        img_index: int,
        page: fitz.Page,
    ) -> None:
        """
        Handle the images in the PDF.
        Args:
            img: The image tuple.
            img_index: The index of the image.
            page: The page object.
        """
        xref = img[0]
        try:
            base_image = page.parent.extract_image(xref)
        except Exception:
            return None

        # 跳过小图标和PNG格式（通常是装饰性图标）
        if base_image.get("ext") == "png" or base_image.get("width", 0) <= 34:
            return None

        # 保存图片并获取路径
        image_path = self.save_image(base_image, img_index, page.number)

        # 获取扩展后的图片区域
        try:
            img_rect = page.get_image_bbox(img)
            expanded_rect = self._get_expanded_rect(img_rect, page.rect)

            # 获取关联文本块
            related_blocks = self._get_related_text_blocks(
                page, expanded_rect, img_rect.y0
            )
            title_blocks = [text for is_title, text in related_blocks if is_title]
        except Exception:
            # 如果提取文本块失败，仍然返回图片信息
            title_blocks = []

        return {
            "image_path": str(image_path),
            "page": page.number + 1,
            "title": "\n".join(title_blocks) if title_blocks else "",
        }

    def save_image(
        self, base_image: Dict[str, Any], img_index: int, page_number: int
    ) -> Path:
        """
        保存图片到本地并返回路径

        Args:
            base_image: 图片数据字典
            img_index: 图片索引
            page_number: 页码（从0开始）

        Returns:
            保存的图片文件路径
        """
        ext = base_image.get("ext", "jpg")
        image_name = f"page{page_number + 1}_img{img_index + 1}.{ext}"
        image_path = self.image_save_dir / image_name

        with open(image_path, "wb") as f:
            f.write(base_image["image"])

        return image_path

    def _get_expanded_rect(
        self, img_rect: fitz.Rect, page_rect: fitz.Rect
    ) -> fitz.Rect:
        """
        获取扩展后的搜索区域（用于查找图片上方的标题）

        Args:
            img_rect: 图片矩形区域
            page_rect: 页面矩形区域

        Returns:
            扩展后的矩形区域
        """
        bottom_size = self.title_properties["bottom_size"]
        expanded = img_rect + (0, bottom_size, 0, img_rect.height * 3)
        expanded[3] = min(
            expanded[3], page_rect[3] - self.title_properties["page_clip"]
        )
        return expanded.intersect(page_rect)

    def _get_related_text_blocks(
        self, page: fitz.Page, rect: fitz.Rect, img_y: float
    ) -> List[Tuple[bool, str]]:
        """
        获取与图片相关的文本块

        Args:
            page: PDF页面对象
            rect: 搜索区域矩形
            img_y: 图片的y坐标

        Returns:
            文本块列表，每个元素为 (是否为标题, 文本内容) 的元组
        """
        related_blocks = []
        try:
            blocks = page.get_text("blocks")
            for block in blocks:
                if len(block) < 5:
                    continue

                block_rect = fitz.Rect(block[:4])
                if not block_rect.intersects(rect):
                    continue

                block_text = block[4].strip() if isinstance(block[4], str) else ""
                if not block_text:
                    continue

                above = block_rect.y1 < img_y
                is_title_block = self._is_title_block_candidate(page, block, above)
                related_blocks.append((is_title_block, block_text))
        except Exception:
            # 如果提取失败，返回空列表
            pass

        return related_blocks

    def _is_title_block_candidate(
        self, page: fitz.Page, block: Tuple, above: bool
    ) -> bool:
        """
        判断是否为标题候选块

        Args:
            page: PDF页面对象
            block: 文本块元组
            above: 是否在图片上方

        Returns:
            是否为标题块
        """
        # 检查块的基本属性
        if len(block) < 7 or block[6] != 0:
            return False

        block_text = block[4].strip() if isinstance(block[4], str) else ""
        if not block_text:
            return False

        try:
            # 获取文本的字体信息
            page_dict = page.get_text("dict")
            block_num = block[5] if len(block) > 5 else 0

            if (
                block_num < len(page_dict.get("blocks", []))
                and "lines" in page_dict["blocks"][block_num]
                and len(page_dict["blocks"][block_num]["lines"]) > 0
                and "spans" in page_dict["blocks"][block_num]["lines"][0]
                and len(page_dict["blocks"][block_num]["lines"][0]["spans"]) > 0
            ):
                span = page_dict["blocks"][block_num]["lines"][0]["spans"][0]
                font_size = span.get("size", 0)
                font_name = span.get("font", "").lower()
                is_bold = "bold" in font_name
            else:
                return False
        except (IndexError, KeyError, TypeError):
            return False

        # 排除带句尾标点的文本
        if block_text.endswith((".", "。", "!", "！")):
            return False

        # 计分规则
        score = 0
        if font_size >= self.title_properties["min_size"]:
            score += 2
        if is_bold:
            score += 1
        if (block_text.count("\n") + 1) <= self.title_properties["max_lines"]:
            score += 0.5
        if len(block_text) <= self.title_properties["max_length"]:
            score += 0.5
        if above:
            score += 2
        else:
            score -= 1

        return score >= 3
