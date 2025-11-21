"""
解析器模块单元测试
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import fitz

from src.evrag.parser.pdf_parser import PDFParser
from src.evrag.parser.image_handler import ImageHandler


class TestImageHandler:
    """测试图片处理器"""

    @pytest.fixture
    def temp_image_dir(self, tmp_path):
        """临时图片目录"""
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        return image_dir

    @pytest.fixture
    def image_handler(self, temp_image_dir):
        """图片处理器实例"""
        return ImageHandler(image_save_dir=temp_image_dir)

    @pytest.fixture
    def mock_page(self):
        """模拟PDF页面"""
        page = Mock()
        page.number = 0
        page.rect = fitz.Rect(0, 0, 800, 1200)
        page.parent = Mock()
        return page

    def test_init_with_custom_dir(self, temp_image_dir):
        """测试：使用自定义目录初始化"""
        handler = ImageHandler(image_save_dir=temp_image_dir)
        assert handler.image_save_dir == temp_image_dir
        assert temp_image_dir.exists()

    def test_init_from_config(self):
        """测试：从配置中读取目录"""
        handler = ImageHandler()
        assert handler.image_save_dir is not None

    def test_save_image(self, image_handler, temp_image_dir):
        """测试：保存图片"""
        base_image = {
            "ext": "jpg",
            "image": b"fake_image_data",
            "width": 100,
            "height": 100,
        }

        image_path = image_handler.save_image(base_image, 0, 0)

        assert image_path.exists()
        assert image_path.name == "page1_img1.jpg"
        assert image_path.read_bytes() == b"fake_image_data"

    def test_handle_image_skips_small_images(self, image_handler, mock_page):
        """测试：跳过小图片"""
        # 模拟小图片
        mock_page.parent.extract_image.return_value = {
            "ext": "png",
            "width": 20,
            "height": 20,
            "image": b"data",
        }

        img = (1, 0, 100, 100, 8, "DeviceRGB", 0, "")
        result = image_handler.handle_image(img, 0, mock_page)

        assert result is None

    def test_handle_image_extracts_large_images(
        self, image_handler, mock_page, temp_image_dir
    ):
        """测试：提取大图片"""
        # 模拟大图片
        mock_page.parent.extract_image.return_value = {
            "ext": "jpg",
            "width": 200,
            "height": 200,
            "image": b"image_data",
        }

        # 模拟图片边界框
        mock_page.get_image_bbox.return_value = fitz.Rect(100, 100, 300, 300)
        mock_page.get_text.return_value = []

        img = (1, 0, 200, 200, 8, "DeviceRGB", 0, "")
        result = image_handler.handle_image(img, 0, mock_page)

        assert result is not None
        assert "image_path" in result
        assert result["page"] == 1
        assert Path(result["image_path"]).exists()


class TestPDFParser:
    """测试PDF解析器"""

    @pytest.fixture
    def temp_pdf_path(self, tmp_path):
        """创建临时PDF文件"""
        pdf_path = tmp_path / "test.pdf"
        # 创建一个简单的PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Test PDF Content")
        doc.save(str(pdf_path))
        doc.close()
        return pdf_path

    @pytest.fixture
    def pdf_parser(self, temp_pdf_path):
        """PDF解析器实例"""
        return PDFParser(pdf_path=temp_pdf_path)

    def test_init_with_pdf_path(self, temp_pdf_path):
        """测试：使用PDF路径初始化"""
        parser = PDFParser(pdf_path=temp_pdf_path)
        assert parser.pdf_path == temp_pdf_path

    def test_init_from_config(self):
        """测试：从配置中读取PDF路径"""
        with patch("src.evrag.parser.pdf_parser.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.pdf_path = Path("/fake/path.pdf")
            mock_get_settings.return_value = mock_settings

            with pytest.raises(FileNotFoundError):
                PDFParser()

    def test_init_without_pdf_path_raises_error(self):
        """测试：没有PDF路径时抛出错误"""
        with patch("src.evrag.parser.pdf_parser.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.pdf_path = None
            mock_get_settings.return_value = mock_settings

            with pytest.raises(ValueError, match="PDF path not specified"):
                PDFParser()

    def test_init_with_nonexistent_pdf_raises_error(self, tmp_path):
        """测试：PDF文件不存在时抛出错误"""
        nonexistent_path = tmp_path / "nonexistent.pdf"

        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            PDFParser(pdf_path=nonexistent_path)

    def test_load_pdf(self, pdf_parser):
        """测试：加载PDF"""
        docs = pdf_parser.load_pdf(show_progress=False)

        assert isinstance(docs, list)
        assert len(docs) > 0
        assert all(hasattr(doc, "page_content") for doc in docs)
        assert all(hasattr(doc, "metadata") for doc in docs)

    def test_load_pdf_with_page_filtering(self, temp_pdf_path):
        """测试：页面过滤"""
        # 创建多页PDF
        doc = fitz.open()
        for i in range(5):
            page = doc.new_page()
            page.insert_text((50, 50), f"Page {i + 1}")
        pdf_path = temp_pdf_path.parent / "multi_page.pdf"
        doc.save(str(pdf_path))
        doc.close()

        parser = PDFParser(
            pdf_path=pdf_path,
            min_filter_pages=1,
            max_filter_pages=3,
        )

        docs = parser.load_pdf(show_progress=False)
        # 应该只加载第2、3、4页（索引1、2、3）
        assert len(docs) == 3

    def test_load_pdf_extracts_metadata(self, pdf_parser):
        """测试：提取元数据"""
        docs = pdf_parser.load_pdf(show_progress=False)

        if docs:
            metadata = docs[0].metadata
            assert "unique_id" in metadata
            assert "source" in metadata
            assert "page" in metadata
            assert "images_info" in metadata

    def test_parse_alias(self, pdf_parser):
        """测试：parse方法是load_pdf的别名"""
        docs1 = pdf_parser.load_pdf(show_progress=False)
        docs2 = pdf_parser.parse(show_progress=False)

        assert len(docs1) == len(docs2)

    def test_load_pdf_handles_errors_gracefully(self, temp_pdf_path):
        """测试：优雅处理错误"""
        parser = PDFParser(pdf_path=temp_pdf_path)

        # 即使某页处理失败，也应该继续处理其他页
        with patch.object(parser, "image_handler") as mock_handler:
            mock_handler.handle_image.side_effect = Exception("Image error")
            docs = parser.load_pdf(show_progress=False)
            # 应该仍然能提取文本
            assert len(docs) > 0
