import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from backend.app.models.chunk_models import ChunkingConfig
from backend.app.services.document_processor import DocumentProcessor
from backend.app.services.chunking_service import ChunkingService


@pytest.fixture
def processor():
    return DocumentProcessor()


@pytest.mark.asyncio
async def test_per_document_type_config_selection():
    pdf_config = DocumentProcessor.get_chunking_config_for_type("application/pdf")
    assert isinstance(pdf_config, ChunkingConfig)
    assert pdf_config.chunk_size == 500
    txt_config = DocumentProcessor.get_chunking_config_for_type("text/plain")
    assert txt_config.strategy == "syntactic"
    unknown_config = DocumentProcessor.get_chunking_config_for_type("unknown/type")
    assert isinstance(unknown_config, ChunkingConfig)


@pytest.mark.asyncio
async def test_async_text_extraction_and_chunking(tmp_path):
    # Create a sample TXT file
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello world!\nThis is a test.\nAnother line.")
    processor = DocumentProcessor()
    config = DocumentProcessor.get_chunking_config_for_type("text/plain")
    text = await processor._extract_text_from_file(str(txt_file), "text/plain")
    assert "Hello world!" in text
    chunker = ChunkingService(config)
    chunks = await chunker.hybrid_chunk(text, metadata={})
    assert len(chunks) > 0
    assert any("Hello world!" in c.text for c in chunks)


@pytest.mark.asyncio
async def test_chunk_storage(tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("A. B. C. D. E. F. G.")
    processor = DocumentProcessor()
    config = DocumentProcessor.get_chunking_config_for_type("text/plain")
    text = await processor._extract_text_from_file(str(txt_file), "text/plain")
    chunker = ChunkingService(config)
    chunks = await chunker.hybrid_chunk(text, metadata={})
    doc_id = "test-doc-id"
    processor._store_chunks(doc_id, chunks)
    # Check Redis for stored chunks
    import inspect

    stored = processor.redis.lrange(f"doc_chunks:{doc_id}", 0, -1)
    if inspect.isawaitable(stored):
        stored = await stored
    # Now stored is always a list
    assert len(stored) == len(chunks)
    assert any("A." in s for s in stored)


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
    # Patch the ocr_service attribute directly
    mock_ocr = MagicMock()
    mock_ocr.is_image_pdf.return_value = True
    mock_ocr.pdf_to_images.return_value = [MagicMock(), MagicMock()]
    mock_ocr.run_ocr.side_effect = [("Scanned text 1", 0.9), ("Scanned text 2", 0.9)]
    processor.ocr_service = mock_ocr
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
