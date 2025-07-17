import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from unittest.mock import patch, MagicMock
from backend.app.services.document_processor import DocumentProcessor


@pytest.fixture
def processor():
    return DocumentProcessor()


def test_extract_txt(tmp_path, processor):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello world!\nThis is a test.")
    result = processor.extract_text(str(txt_file))
    assert "Hello world!" in result
    assert "This is a test." in result


@patch("os.path.exists", return_value=True)
@patch("pypdf.PdfReader")
def test_extract_pdf_text(mock_pdfreader, mock_exists, tmp_path, processor):
    # Mock a PDF with extractable text
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "PDF page text"
    mock_pdfreader.return_value.pages = [mock_page, mock_page]
    with patch.object(processor, "ocr_service") as mock_ocr:
        mock_ocr.is_image_pdf.return_value = False
        result = processor.extract_text("backend/app/services/dummy.pdf")
        assert "PDF page text" in result


@patch("os.path.exists", return_value=True)
@patch("pypdf.PdfReader")
@patch("backend.app.services.ocr_service.OCRService")
def test_extract_pdf_image_based(
    mock_ocrservice, mock_pdfreader, mock_exists, tmp_path, processor
):
    # Mock a PDF with no extractable text (image-based)
    mock_ocr = mock_ocrservice.return_value
    mock_ocr.is_image_pdf.return_value = True
    mock_ocr.pdf_to_images.return_value = [MagicMock(), MagicMock()]
    mock_ocr.run_ocr.side_effect = [("Scanned text 1", 0.9), ("Scanned text 2", 0.9)]
    result = processor._extract_pdf("backend/app/services/dummy.pdf")
    assert "Scanned text 1" in result
    assert "Scanned text 2" in result


@patch("os.path.exists", return_value=True)
@patch("docx.Document")
def test_extract_docx(mock_docx, mock_exists, tmp_path, processor):
    # Patch the instance's magic_detector
    processor.magic_detector.from_buffer = MagicMock(
        return_value="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    # Mock a DOCX file
    mock_doc = MagicMock()
    mock_doc.paragraphs = [MagicMock(text="Paragraph 1"), MagicMock(text="Paragraph 2")]
    mock_docx.return_value = mock_doc
    result = processor.extract_text("backend/app/services/dummy.docx")
    assert "Paragraph 1" in result
    assert "Paragraph 2" in result


def test_extract_scanned_pdf(processor):
    # This test uses the real scanned PDF file and real OCR
    text = processor.extract_text("backend/app/services/dummy_scanned.pdf")
    assert "Scanned text 1" in text
    assert "Scanned text 2" in text
