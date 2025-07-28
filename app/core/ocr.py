# ocr.py
import easyocr
import logging
from typing import Optional, List
import re
from PIL import Image
import io
import os
import tempfile
from pdf2image import convert_from_path
from app.config import settings


logger = logging.getLogger(__name__)

# OCR Configuration
CONFIG = {
    'OCR_SETTINGS': {
        'LANGUAGES': ['en'],
        'GPU': True,
        'DETAILED_OUTPUT': True,
        'PARAGRAPH': True,
        'ROTATE_PAGES': True,
        'MIN_TEXT_CONFIDENCE': 0.8
    }
}

class OCRProcessor:
    def __init__(self):
        try:
            # Initialize EasyOCR reader with GPU if available
            self.reader = easyocr.Reader(
                CONFIG['OCR_SETTINGS']['LANGUAGES'],
                gpu=CONFIG['OCR_SETTINGS']['GPU'],
                detect_network='craft',
                model_storage_directory=os.path.join(tempfile.gettempdir(), 'easyocr'),
                download_enabled=True
            )
            logger.info("OCR processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OCR processor: {str(e)}")
            raise RuntimeError("OCR initialization failed")

    def is_scanned_pdf(self, file_path: str) -> bool:
        """Check if PDF is scanned (image-based) with multiple detection methods"""
        try:
            # Method 1: Check for fonts in PDF
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    if '/Font' in page['/Resources']:
                        return False

                # Method 2: Check text extraction yield
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                if len(text.strip()) > 100:  # Arbitrary threshold
                    return False

                return True
        except Exception as e:
            logger.warning(f"PDF scan check failed: {str(e)}")
            return True  # Assume scanned if check fails

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image file using OCR with confidence filtering"""
        try:
            result = self.reader.readtext(
                image_path,
                detail=CONFIG['OCR_SETTINGS']['DETAILED_OUTPUT'],
                paragraph=CONFIG['OCR_SETTINGS']['PARAGRAPH'],
                min_size=10,
                rotation_info=CONFIG['OCR_SETTINGS']['ROTATE_PAGES']
            )

            # Filter results by confidence and combine
            text_blocks = [
                entry[1] for entry in result
                if len(entry) > 2 and entry[2] > CONFIG['OCR_SETTINGS']['MIN_TEXT_CONFIDENCE']
            ]

            return "\n".join(text_blocks)
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {str(e)}")
            return ""

    def convert_pdf_to_images(self, pdf_path: str) -> List[str]:
        """Convert PDF pages to images for OCR processing with optimization"""
        try:
            images = convert_from_path(
                pdf_path,
                dpi=300,
                grayscale=True,
                thread_count=4,
                fmt='jpeg',
                use_pdftocairo=True
            )

            temp_files = []
            for i, image in enumerate(images):
                temp_file = os.path.join(tempfile.gettempdir(), f"pdf_page_{i}.jpg")
                image.save(temp_file, 'JPEG', quality=80)
                temp_files.append(temp_file)

            return temp_files
        except Exception as e:
            logger.error(f"PDF to image conversion failed: {str(e)}")
            return []

    def process_scanned_pdf(self, pdf_path: str) -> str:
        """Process scanned PDF through OCR with parallel page processing"""
        try:
            image_files = self.convert_pdf_to_images(pdf_path)
            if not image_files:
                return ""

            full_text = []
            for img_file in image_files:
                try:
                    page_text = self.extract_text_from_image(img_file)
                    if page_text:
                        full_text.append(page_text)
                finally:
                    try:
                        os.remove(img_file)
                    except:
                        pass

            return "\n\n".join(full_text)
        except Exception as e:
            logger.error(f"Scanned PDF processing failed: {str(e)}")
            return ""
