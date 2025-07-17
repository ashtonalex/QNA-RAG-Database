"""
Service for OCR processing: image detection, Tesseract integration, preprocessing, and language detection.
"""

from typing import Tuple, Optional
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader
import logging


class OCRService:
    def __init__(self):
        # You can add Tesseract config here if needed
        pass

    def is_image_pdf(self, filepath: str) -> bool:
        """
        Detect if PDF contains images (scanned PDF).
        Returns True if most pages have no extractable text.
        """
        try:
            reader = PdfReader(filepath)
            image_pages = 0
            total_pages = len(reader.pages)
            for page in reader.pages:
                text = page.extract_text()
                if not text or text.strip() == "":
                    image_pages += 1
            # Heuristic: if more than half the pages have no text, treat as image-based
            return image_pages > (total_pages // 2)
        except Exception as e:
            logging.error(f"Failed to check if PDF is image-based: {e}")
            return False

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image (deskew, denoise) before OCR. (Stub)
        """
        # For now, just return the image as-is
        return image

    def run_ocr(
        self, image: Image.Image, lang: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Run Tesseract OCR, return extracted text and confidence score (stub for confidence).
        """
        try:
            image = self.preprocess_image(image)
            config = ""
            if lang:
                config += f"-l {lang} "
            text = pytesseract.image_to_string(image, config=config)
            # Confidence is not directly available from pytesseract for image_to_string
            return text, 0.9  # Stub confidence
        except Exception as e:
            logging.error(f"OCR failed: {e}")
            return "", 0.0

    def pdf_to_images(self, filepath: str) -> list:
        """
        Convert PDF pages to images using pdf2image.
        """
        try:
            images = convert_from_path(filepath)
            return images
        except Exception as e:
            logging.error(f"Failed to convert PDF to images: {e}")
            return []

    def detect_language(self, text: str) -> str:
        """
        Detect language of extracted text. (Stub)
        """
        # You can use langdetect or similar here
        return "unknown"
