"""
Service for OCR processing: image detection, Tesseract integration, preprocessing, and language detection.
"""

from typing import Tuple, Optional
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader
import logging
import cv2
import numpy as np


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
        Preprocess image (deskew, denoise) before OCR.
        """
        # Convert PIL Image to OpenCV format
        img = np.array(image)
        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=30)
        # Deskew
        coords = np.column_stack(np.where(denoised > 0))
        angle = 0
        if coords.size > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
        (h, w) = denoised.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(
            denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        # Convert back to PIL Image
        return Image.fromarray(deskewed)

    def run_ocr(
        self, image: Image.Image, lang: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Run Tesseract OCR, return extracted text and average confidence score.
        """
        try:
            image = self.preprocess_image(image)
            config = ""
            if lang:
                config += f"-l {lang} "
            # Get OCR data for confidence
            ocr_data = pytesseract.image_to_data(
                image, config=config, output_type=pytesseract.Output.DICT
            )
            text = " ".join([word for word in ocr_data["text"] if word.strip()])
            confidences = [
                int(conf)
                for conf in ocr_data["conf"]
                if (isinstance(conf, int) and conf > 0)
                or (isinstance(conf, str) and conf.isdigit() and int(conf) > 0)
            ]
            avg_conf = float(np.mean(confidences)) / 100 if confidences else 0.0
            return text, avg_conf
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
